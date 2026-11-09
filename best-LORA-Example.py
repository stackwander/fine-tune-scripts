"""
Built & Shared by Burhan Saadi for making LoRA/QLORA fine-tuning 
Fine tuning with LoRA/QLORA notes document avaiable in personal documents.
Three was to do this 

a) Programming using HG PEFT library
b) Unsloth UI 
c) Google AI Studio

This file covers programming using HG PEFT library

Step1: Install required packages, may create requirements.txt file
a.	PIP install transformers
b.	PIP install PEFT  (Hugging face Package)
c.	PIP install datasets
d.	PIP install accelerate 
e.	PIP install torch

This code is for LLM fine-tuning with LoRA
Pick LLM model to finetune e.g, GPT-2, Llama 2, Mistral, BERT, RoBERTa, T5 

"""

# =====================================================================================
# ADDED NOTES
# The original notes (the docstring above and the "#" notes inside the sections below)
# are kept word for word. Notes added afterwards start with "NOTE:".
# =====================================================================================
# NOTE: BERT / RoBERTa are encoder-only and T5 is encoder-decoder, so they are not causal
#       LMs and AutoModelForCausalLM cannot train them. For those use
#       AutoModelForSequenceClassification + TaskType.SEQ_CLS (see lora_finetuning.py in this
#       folder) or AutoModelForSeq2SeqLM + TaskType.SEQ_2_SEQ_LM. This file is for decoder-only
#       LLMs: GPT-2, Llama, Mistral, Qwen, Gemma, Phi-3, SmolLM, ...
# NOTE: optional extra   f. pip install bitsandbytes   (QLoRA only: --use_qlora, NVIDIA GPU + CUDA)
#       Tested with torch 2.12.1, transformers 5.12.1, peft 0.19.1, datasets 5.0.0, accelerate 1.14.0.
#       Needs transformers >= 4.56 for the `dtype=` argument (`torch_dtype=` is its deprecated old name).
#       Windows: the one-time huggingface_hub "cache-system uses symlinks" warning on a first download is
#       harmless (enable Developer Mode, or set HF_HUB_DISABLE_SYMLINKS_WARNING=1 to hide it).
#
# NOTE: PIPELINE - one function per step; main() at the bottom runs them in this order
#   1. Data  : load (Hub / local files / save_to_disk folder / streaming) -> detect the layout
#              -> clean + de-duplicate -> train/validation split -> prompt or chat template
#              -> tokenize with label masking -> optional packing -> DataLoaders
#   2. Model : tokenizer (+ pad token) -> base model (fp32 / bf16 / fp16, or 4-bit for QLoRA)
#              -> LoraConfig -> get_peft_model
#   3. Train : AdamW on the LoRA weights only -> warmup + LR schedule -> mixed precision
#              -> gradient accumulation + clipping -> validation loss / perplexity
#              -> early stopping -> best adapter restored
#   Then     : save adapter -> reload with AutoPeftModelForCausalLM -> generate (base vs LoRA)
#              -> optional merge of the adapter into the base weights for deployment
#
# NOTE: RUN (every option can also be changed in ScriptConfig below)
#   python best-LORA-Example.py                    # GPT-2 on Dolly-15k instructions (CPU ok)
#   python best-LORA-Example.py --help             # list every option
#   python best-LORA-Example.py --train_file data/train.jsonl --validation_file data/val.jsonl
#   python best-LORA-Example.py --train_file "data/*.parquet" --dataset_format text --packing
#   python best-LORA-Example.py --dataset_name HuggingFaceH4/no_robots --model_name HuggingFaceTB/SmolLM2-135M-Instruct
#   python best-LORA-Example.py --dataset_name HuggingFaceFW/fineweb-edu --dataset_config_name sample-10BT
#          ... plus --streaming --packing --max_steps 500   (streams a 10B-token corpus, nothing downloaded upfront)
#   python best-LORA-Example.py --do_train false --test_prompt "Explain LoRA in one sentence."
#   python best-LORA-Example.py my_run.json        # all options from a JSON (or .yaml) file
# NOTE: QLoRA: add --use_qlora --gradient_checkpointing --lora_target_modules all-linear.
#       Use --use_qlora for inference-only runs too (--do_train false) to keep the base in 4-bit.
#       --save_merged_model explicitly reloads an unquantized base for export and needs more memory.
#
# NOTE: DATA FILE FORMATS - the datasets builder is picked from the extension (compressed .gz .bz2 .xz
#       files work as they are; .zst needs `pip install zstandard`)
#   .csv .tsv -> "csv"   .json .jsonl -> "json" (--json_field when the records sit under one key)
#   .parquet -> "parquet"   .txt -> "text" (one example per line)   .arrow -> "arrow"
#   Paths may be globs ("data/train-*.parquet") or http(s) URLs. A Hub repo id, or a local
#   folder of such files, goes in --dataset_name; a folder from save_to_disk() goes in
#   --load_from_disk_path.
#
# NOTE: DATA LAYOUTS - auto-detected from the column names, or forced with --dataset_format
#   text        {"text": "..."}                                          loss on every token
#   instruction {"instruction": "...", "input": "...", "output": "..."}   loss on the answer only
#               (also accepted: prompt/question/query, context, response/completion/answer)
#   chat        {"messages": [{"role": "user", "content": "..."},
#                             {"role": "assistant", "content": "..."}]}  loss on assistant turns only
#               (ShareGPT style {"conversations": [{"from": "human", "value": "..."}]} also works)
#
# NOTE: deliberately NOT in this single-file template (use HF Trainer / TRL SFTTrainer + accelerate
#       when you need them): resuming a crashed run from optimizer/scheduler state, multi-GPU (DDP /
#       FSDP), TensorBoard / W&B logging, task metrics beyond loss and perplexity (ROUGE, exact match,
#       LLM-as-judge on held-out prompts).

import contextlib
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from itertools import chain
from typing import Literal, Optional

import torch
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk
from jinja2 import TemplateError
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    load_peft_weights,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    get_scheduler,
    set_seed,
)
from transformers.pytorch_utils import Conv1D

IGNORE_INDEX = -100  # label value skipped by the cross-entropy loss (prompt and padding positions)

SPLIT_ALIASES = ("validation", "valid", "val", "dev")  # "test" is never used for training decisions

COLUMN_ALIASES = {  # first match wins; explicit --*_column flags override these
    "messages": ("messages", "conversations", "conversation"),
    "instruction": ("instruction", "prompt", "question", "query"),
    "input": ("input", "context"),
    "response": ("output", "response", "completion", "answer"),
    "text": ("text", "content"),
}

ROLE_ALIASES = {  # OpenAI style roles + ShareGPT style "from" values
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
}

FILE_BUILDERS = {  # file extension -> (packaged `datasets` builder, builder kwargs)
    ".csv": ("csv", {}),
    ".tsv": ("csv", {"sep": "\t"}),
    ".json": ("json", {}),
    ".jsonl": ("json", {}),
    ".ndjson": ("json", {}),
    ".parquet": ("parquet", {}),
    ".txt": ("text", {}),
    ".arrow": ("arrow", {}),
}
COMPRESSION_SUFFIXES = (".gz", ".bz2", ".xz", ".zst")

# Prompt for the instruction layout. format_instruction_prompt() builds it for training AND for
# generation, so the fine-tuned model sees exactly the same pattern at inference time.
PROMPT_WITH_INPUT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
PROMPT_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n"

# Chat template used only when the tokenizer has none (base models such as GPT-2). The blank line
# sits BEFORE each header, so an assistant turn ends exactly with EOS (the model learns to stop).
DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if not loop.first %}{{ '\n\n' }}{% endif %}"
    "{% if message['role'] == 'system' %}{{ '### System:\n' + message['content'] }}"
    "{% elif message['role'] == 'user' %}{{ '### User:\n' + message['content'] }}"
    "{% elif message['role'] == 'assistant' %}{{ '### Assistant:\n' + message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{% if messages %}{{ '\n\n' }}{% endif %}{{ '### Assistant:\n' }}{% endif %}"
)


# =====================================================================================
# NOTE: Configuration - every knob in one place. Edit the defaults here, or override them
#       on the command line (--name value) or with a .json / .yaml file (see RUN above).
#       `python best-LORA-Example.py --help` prints every option with its help text and default.
# =====================================================================================
def option(default, help_text: str):
    """A config field whose help text and default are shown by --help."""
    return field(default=default, metadata={"help": help_text})


@dataclass
class ScriptConfig:
    # ---- Model ---------------------------------------------------------------------
    model_name: str = option("gpt2", "Decoder-only (causal) LM: a Hub id such as Qwen/Qwen2.5-0.5B, or a folder.")
    model_revision: Optional[str] = option(None, "Pin a model commit hash / tag for reproducible runs.")
    trust_remote_code: bool = option(False, "Allow models that ship their own code (e.g. Baichuan); read it first.")
    attn_implementation: Optional[str] = option(
        None, "None = library default (sdpa); eager; flash_attention_2 (needs the flash-attn package).")
    model_dtype: Literal["auto", "float32", "bfloat16", "float16"] = option(
        "auto", "Dtype of the frozen base weights. auto = bf16 / fp16 on CUDA, fp32 on CPU / MPS.")

    # ---- Data source: if several are set, load_from_disk_path > train_file > dataset_name ----
    dataset_name: Optional[str] = option(
        "databricks/databricks-dolly-15k", "Hub dataset repo id, or a local folder of data files.")
    dataset_config_name: Optional[str] = option(
        None, "Hub subset / config name, e.g. wikitext-2-raw-v1 for Salesforce/wikitext.")
    dataset_revision: Optional[str] = option(None, "Pin a dataset commit / tag / branch.")
    train_file: Optional[str] = option(
        None, "Path, glob or URL: .csv .tsv .json .jsonl .ndjson .parquet .txt .arrow (may be compressed).")
    validation_file: Optional[str] = option(None, "Optional validation file in the same format as --train_file.")
    load_from_disk_path: Optional[str] = option(None, "Folder written by Dataset / DatasetDict.save_to_disk().")
    json_field: Optional[str] = option(None, "JSON only: key that holds the list of records, e.g. data.")
    train_split: str = option("train", "Name of the training split.")
    eval_split: Optional[str] = option(
        None, "Validation split. None = validation / valid / val / dev if present, else carved from train.")
    streaming: bool = option(
        False, "Read rows lazily instead of downloading everything (huge datasets). Needs --max_steps.")
    cache_dir: Optional[str] = option(None, "Downloads + processed Arrow caches (default ~/.cache/huggingface).")

    # ---- Data layout (see DATA LAYOUTS above) --------------------------------------
    dataset_format: Literal["auto", "text", "instruction", "chat"] = option(
        "auto", "auto = detect from the column names (see COLUMN_ALIASES).")
    text_column: Optional[str] = option(None, "Text column (text layout). None = auto-detect.")
    instruction_column: Optional[str] = option(None, "Question / instruction column. None = auto-detect.")
    input_column: Optional[str] = option(None, "Optional extra context column (Dolly: context). None = auto-detect.")
    response_column: Optional[str] = option(None, "Answer column. None = auto-detect.")
    messages_column: Optional[str] = option(None, "Chat column (list of role/content turns). None = auto-detect.")

    # ---- Preprocessing -------------------------------------------------------------
    max_seq_length: int = option(256, "Tokens per example (GPT-2 limit 1024). Longer examples are truncated.")
    packing: bool = option(False, "Concatenate examples into full max_seq_length blocks (best for raw text).")
    deduplicate: bool = option(True, "Drop exact duplicates BEFORE splitting (no train / validation leakage).")
    validation_split_percentage: float = option(
        5.0, "Percent of train held out when the source has no validation split.")
    max_train_samples: int = option(1000, "Seeded random subset for quick runs. 0 = use every example.")
    max_eval_samples: int = option(
        200, "Validation subset. 0 = every example. Streaming: rows taken from the stream for validation.")
    group_by_length: bool = option(True, "Batch similar lengths together: far less padding (not for streaming).")
    preprocessing_num_workers: Optional[int] = option(None, "Processes for .map / .filter on big datasets.")
    overwrite_cache: bool = option(False, "Ignore the cached processed datasets and recompute them.")
    shuffle_buffer_size: int = option(10_000, "Streaming only: rows held in the shuffle buffer.")

    # ---- LoRA / QLoRA --------------------------------------------------------------
    lora_r: int = option(16, "LoRA rank r.")
    lora_alpha: int = option(32, "LoRA alpha (the update is scaled by alpha / r).")
    lora_dropout: float = option(0.1, "Dropout on the LoRA branch.")
    lora_target_modules: Optional[list[str]] = option(
        None, "Layer names, e.g. q_proj k_proj v_proj o_proj, or all-linear. None = PEFT default for the model.")
    use_qlora: bool = option(
        False, "Load the base in 4-bit NF4 for training and inference (this template uses CUDA + bitsandbytes).")
    gradient_checkpointing: bool = option(
        False, "Recompute activations in the backward pass: much less memory, roughly 30 percent slower.")

    # ---- Training ------------------------------------------------------------------
    output_dir: str = option("./my_lora_adapter", "Where the adapter, tokenizer and run_summary.json are saved.")
    do_train: bool = option(True, "false = skip training; only load and test the adapter in --output_dir.")
    num_epochs: int = option(3, "Passes over the training data.")
    max_steps: int = option(0, "> 0 overrides num_epochs (required with --streaming).")
    per_device_train_batch_size: int = option(8, "Sequences per training micro-batch.")
    per_device_eval_batch_size: int = option(8, "Sequences per validation batch.")
    gradient_accumulation_steps: int = option(
        1, "Micro-batches per optimizer step: effective batch = batch size x accumulation steps.")
    learning_rate: float = option(5e-4, "Peak learning rate (LoRA tolerates higher values than full fine-tuning).")
    weight_decay: float = option(0.01, "AdamW weight decay (not applied to 1-D tensors).")
    lr_scheduler_type: str = option("linear", "linear, cosine, constant_with_warmup, ... (transformers schedulers).")
    warmup_ratio: float = option(0.1, "Share of all optimizer steps used to warm the LR up from 0.")
    max_grad_norm: float = option(1.0, "Gradient clipping max norm. 0 = off.")
    mixed_precision: Literal["auto", "no", "bf16", "fp16"] = option(
        "auto", "auto = bf16 (fp16 on older GPUs) on CUDA, off on CPU / MPS.")
    eval_strategy: Literal["epoch", "steps", "no"] = option(
        "epoch", "When to compute the validation loss (no = never; nothing is held out then).")
    eval_steps: int = option(50, "Evaluate every N optimizer steps when --eval_strategy steps.")
    early_stopping_patience: int = option(2, "Stop after N evaluations in a row without improvement. 0 = off.")
    early_stopping_threshold: float = option(0.0, "Minimum eval-loss decrease that resets the patience counter.")
    logging_steps: int = option(10, "Print the training loss every N optimizer steps.")
    dataloader_num_workers: int = option(0, "DataLoader worker processes (0 is fastest on Windows for this data).")
    seed: int = option(42, "Seed for splits, shuffling, LoRA init, dropout and sampling.")

    # ---- Inference / test ----------------------------------------------------------
    test_prompt: str = option("The future of artificial intelligence", "Prompt used to test the adapter.")
    max_new_tokens: int = option(100, "Tokens to generate (the prompt is not counted).")
    temperature: float = option(0.7, "Sampling temperature.")
    top_p: float = option(0.9, "Nucleus sampling: keep the smallest token set with this total probability.")
    do_sample: bool = option(True, "Sample (true) or greedy decoding (false).")
    repetition_penalty: float = option(1.1, "> 1.0 discourages repeating tokens.")
    compare_with_base: bool = option(True, "Also generate with the adapter switched off (= the base model).")
    save_merged_model: bool = option(
        False, "Also save a merged model in <output_dir>-merged. QLoRA: reloads an unquantized base (more memory).")


def parse_config() -> ScriptConfig:
    """Defaults from ScriptConfig, overridden by --flags or by a single .json / .yaml file argument."""
    parser = HfArgumentParser(ScriptConfig)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (cfg,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        (cfg,) = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        (cfg,) = parser.parse_args_into_dataclasses()
    validate_config(cfg)
    return cfg


def validate_config(cfg: ScriptConfig) -> None:
    """Fail fast with a clear message instead of crashing half-way through a long run."""
    if not (cfg.load_from_disk_path or cfg.train_file or cfg.dataset_name):
        raise ValueError("Set a data source: --dataset_name, --train_file or --load_from_disk_path")
    if cfg.validation_file and not cfg.train_file:
        raise ValueError("--validation_file needs --train_file")
    if cfg.streaming:
        if cfg.load_from_disk_path:
            raise ValueError("--streaming works with --dataset_name / --train_file, not --load_from_disk_path")
        if cfg.max_steps <= 0:
            raise ValueError("--streaming needs --max_steps > 0 (a stream has no length to count epochs with)")
        if cfg.eval_strategy != "no" and cfg.max_eval_samples <= 0:
            raise ValueError("--streaming needs --max_eval_samples > 0 for the validation set (or --eval_strategy no)")
    if cfg.eval_strategy == "steps" and cfg.eval_steps <= 0:
        raise ValueError("--eval_strategy steps needs --eval_steps > 0")
    if not 0 <= cfg.validation_split_percentage < 100:
        raise ValueError("--validation_split_percentage must be in [0, 100)")
    if cfg.gradient_accumulation_steps < 1 or cfg.max_seq_length < 2 or cfg.logging_steps < 1:
        raise ValueError("--gradient_accumulation_steps and --logging_steps must be >= 1, --max_seq_length >= 2")


#=====================================================================================
#1.data loading and preprocessing section for PEFT Hugging face
#=====================================================================================
# NOTE: 1a load raw splits -> 1b detect layout + columns -> 1c clean + de-duplicate
#       -> 1d validation split + subsample -> 1e template + tokenize (label masking, packing)
#       -> 1f collator + DataLoaders.
#       Every .map() / .filter() result is cached as Arrow files next to the downloaded data,
#       so a second run with the same data and settings skips the work (--overwrite_cache redoes it).
#       Local files are fingerprinted too: editing a CSV/JSONL invalidates its cache automatically.
#       After the first download a run also works offline with HF_HUB_OFFLINE=1.
# NOTE: other sources end up as the same Dataset object and can go through the same steps:
#       Dataset.from_pandas(df), Dataset.from_list(rows), Dataset.from_sql(query, con), and several
#       sources can be mixed with concatenate_datasets([a, b]) or
#       interleave_datasets([a, b], probabilities=[0.7, 0.3], seed=42) (from datasets).


def infer_file_builder(path: str) -> tuple[str, dict]:
    """Pick the packaged `datasets` builder ("csv", "json", ...) from a file name, glob or URL."""
    name = path.lower().split("?")[0]
    for suffix in COMPRESSION_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    extension = os.path.splitext(name)[1]
    if extension not in FILE_BUILDERS:
        raise ValueError(f"Unsupported file type '{extension}' in {path}. Supported: {sorted(FILE_BUILDERS)}")
    builder, builder_kwargs = FILE_BUILDERS[extension]
    return builder, dict(builder_kwargs)


def load_raw_splits(cfg: ScriptConfig):
    """1a. Load the raw train split (+ the validation split when the source has one)."""
    if cfg.load_from_disk_path:
        # Arrow folder written earlier with save_to_disk() (e.g. after custom cleaning in a notebook)
        raw = load_from_disk(cfg.load_from_disk_path)
        if isinstance(raw, Dataset):
            raw = DatasetDict({cfg.train_split: raw})
        source = f"save_to_disk folder '{cfg.load_from_disk_path}'"
        train_name, eval_name = cfg.train_split, cfg.eval_split
    elif cfg.train_file:
        builder, builder_kwargs = infer_file_builder(cfg.train_file)
        data_files = {"train": cfg.train_file}
        if cfg.validation_file:
            if infer_file_builder(cfg.validation_file)[0] != builder:
                raise ValueError("--train_file and --validation_file must have the same format")
            data_files["validation"] = cfg.validation_file
        if cfg.json_field:
            builder_kwargs["field"] = cfg.json_field
        raw = load_dataset(builder, data_files=data_files, cache_dir=cfg.cache_dir, streaming=cfg.streaming,
                           **builder_kwargs)
        source = f"{builder} file(s) {data_files}"
        train_name, eval_name = "train", ("validation" if cfg.validation_file else None)
    else:
        builder_kwargs = {"field": cfg.json_field} if cfg.json_field else {}
        raw = load_dataset(cfg.dataset_name, cfg.dataset_config_name, revision=cfg.dataset_revision,
                           cache_dir=cfg.cache_dir, streaming=cfg.streaming, **builder_kwargs)
        source = f"Hub/local dataset '{cfg.dataset_name}'"
        if cfg.dataset_config_name:
            source += f" [{cfg.dataset_config_name}]"
        train_name, eval_name = cfg.train_split, cfg.eval_split

    if train_name not in raw:
        raise ValueError(f"Split '{train_name}' not found in {source}. Available splits: {list(raw)}")
    if eval_name is None:
        eval_name = next((split for split in SPLIT_ALIASES if split in raw), None)
    elif eval_name not in raw:
        raise ValueError(f"Split '{eval_name}' not found in {source}. Available splits: {list(raw)}")
    if cfg.eval_strategy == "no":
        eval_text = "none (eval_strategy no)"
    else:
        eval_text = repr(eval_name) if eval_name else "carved from train"
    print(f"[data] source: {source} | splits: {list(raw)} | train: '{train_name}' | validation: {eval_text}")
    return raw[train_name], (raw[eval_name] if eval_name else None)


def get_column_names(ds) -> list[str]:
    """Column names of a Dataset, or of a streamed IterableDataset (peeks at the first row if unknown)."""
    if ds.column_names is not None:
        return list(ds.column_names)
    return list(next(iter(ds)).keys())


def resolve_layout(cfg: ScriptConfig, column_names: list[str]) -> tuple[str, dict]:
    """1b. Decide text / instruction / chat and which column holds what (explicit flags win)."""

    def pick(role: str) -> Optional[str]:
        explicit = getattr(cfg, f"{role}_column")
        if explicit:
            if explicit not in column_names:
                raise ValueError(f"--{role}_column '{explicit}' not found. Columns: {column_names}")
            return explicit
        return next((name for name in COLUMN_ALIASES[role] if name in column_names), None)

    found = {role: pick(role) for role in COLUMN_ALIASES}
    fmt = cfg.dataset_format
    if fmt == "auto":
        if found["messages"]:
            fmt = "chat"
        elif found["instruction"] and found["response"]:
            fmt = "instruction"
        elif found["text"]:
            fmt = "text"
        else:
            raise ValueError(f"Cannot detect the data layout from columns {column_names}. "
                             "Set --dataset_format and the matching --*_column flags.")
    required = {"chat": ("messages",), "instruction": ("instruction", "response"), "text": ("text",)}[fmt]
    missing = [role for role in required if not found[role]]
    if missing:
        raise ValueError(f"Layout '{fmt}' needs a column for {missing}; columns are {column_names}. "
                         f"Use --{missing[0]}_column.")
    cols = {role: found[role] for role in required}
    if fmt == "instruction":
        cols["input"] = found["input"]  # may be None: the input/context column is optional
    print(f"[data] layout: {fmt} | columns: {cols}")
    return fmt, cols


def as_text(value) -> str:
    """Cells can be None / NaN (empty CSV cells) or numbers: turn them into plain strings."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def normalize_messages(raw_messages) -> Optional[list[dict]]:
    """Chat rows -> [{"role", "content"}]. Accepts OpenAI style and ShareGPT {"from", "value"} turns."""
    if not isinstance(raw_messages, list) or not raw_messages:
        return None
    messages = []
    for turn in raw_messages:
        if not isinstance(turn, dict):
            return None
        role = ROLE_ALIASES.get(str(turn.get("role") or turn.get("from") or "").strip().lower())
        content = turn.get("content") if turn.get("content") is not None else turn.get("value")
        if role is None or not isinstance(content, str):
            return None
        messages.append({"role": role, "content": content})
    return messages


def is_valid_batch(batch: dict, fmt: str, cols: dict) -> list[bool]:
    """1c. Keep only rows that can become a training example (no empty prompts / answers / texts)."""
    if fmt == "text":
        return [bool(as_text(text).strip()) for text in batch[cols["text"]]]
    if fmt == "instruction":
        return [bool(as_text(question).strip()) and bool(as_text(answer).strip())
                for question, answer in zip(batch[cols["instruction"]], batch[cols["response"]])]
    keep = []
    for raw_messages in batch[cols["messages"]]:
        messages = normalize_messages(raw_messages)
        keep.append(bool(messages) and any(m["role"] == "assistant" and m["content"].strip() for m in messages))
    return keep


def hash_batch(batch: dict, key_columns: list[str]) -> dict:
    rows = zip(*(batch[column] for column in key_columns))
    return {"_hash": [hashlib.sha256(json.dumps(row, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
                      for row in rows]}


def deduplicate(ds: Dataset, cols: dict, cfg: ScriptConfig) -> Dataset:
    """1c. Drop exact duplicates of the columns used for training (keeps the first copy)."""
    key_columns = [column for column in cols.values() if column]
    hashes = ds.map(hash_batch, batched=True, fn_kwargs={"key_columns": key_columns},
                    remove_columns=ds.column_names, num_proc=cfg.preprocessing_num_workers,
                    load_from_cache_file=not cfg.overwrite_cache, desc="Hashing rows for de-duplication")["_hash"]
    seen, keep = set(), []
    for index, digest in enumerate(hashes):
        if digest not in seen:
            seen.add(digest)
            keep.append(index)
    return ds.select(keep) if len(keep) < len(ds) else ds


def split_and_subsample(train_ds: Dataset, eval_ds: Optional[Dataset], cfg: ScriptConfig):
    """1d. Hold out a validation set when the source has none, then take seeded random subsets."""
    if eval_ds is None and cfg.validation_split_percentage > 0 and cfg.eval_strategy != "no":
        parts = train_ds.train_test_split(test_size=cfg.validation_split_percentage / 100, shuffle=True,
                                          seed=cfg.seed)
        train_ds, eval_ds = parts["train"], parts["test"]
    if 0 < cfg.max_train_samples < len(train_ds):
        train_ds = train_ds.shuffle(seed=cfg.seed).select(range(cfg.max_train_samples))
    if eval_ds is not None and 0 < cfg.max_eval_samples < len(eval_ds):
        eval_ds = eval_ds.shuffle(seed=cfg.seed).select(range(cfg.max_eval_samples))
    return train_ds, eval_ds


def split_and_subsample_streaming(train_ds: IterableDataset, eval_ds: Optional[IterableDataset],
                                  cfg: ScriptConfig):
    """1d (streaming). Materialize a small validation set from the stream; training data stays lazy.
    A stream cannot be sampled at random without reading all of it, so the held-out rows are the FIRST
    max_eval_samples rows. If the source is sorted (by topic, date, ...), prefer a real validation split."""
    n_eval = max(cfg.max_eval_samples, 0) if cfg.eval_strategy != "no" else 0
    if eval_ds is None:
        eval_rows = list(train_ds.take(n_eval))
        train_ds = train_ds.skip(n_eval)  # skip BEFORE shuffling, so train and validation never overlap
    else:
        eval_rows = list(eval_ds.take(n_eval))
    if cfg.max_train_samples > 0:
        train_ds = train_ds.take(cfg.max_train_samples)
    train_ds = train_ds.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer_size)
    return train_ds, (Dataset.from_list(eval_rows) if eval_rows else None)


def bos_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """[BOS] for tokenizers that normally prepend it (Llama, Mistral, Gemma), [] otherwise (GPT-2, Qwen)."""
    probe = tokenizer("a")["input_ids"]
    bos_id = tokenizer.bos_token_id
    return [bos_id] if bos_id is not None and probe[:1] == [bos_id] else []


def format_instruction_prompt(instruction: str, context: str = "") -> str:
    """Prompt part of an instruction example - shared by training and generation."""
    if context:
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=context)
    return PROMPT_NO_INPUT.format(instruction=instruction)


def chat_segments(messages: list[dict], tokenizer: PreTrainedTokenizerBase) -> Optional[list[tuple[str, bool]]]:
    """Render a conversation with the chat template and cut it into (text, trainable) pieces:
    every assistant reply is trainable, everything else is context. Works with templates that
    render earlier turns unchanged when later turns are appended (ChatML/Qwen, Llama 3, Mistral,
    Phi-3, Gemma and the fallback template); other rows are dropped instead of being mis-masked."""
    segments, rendered = [], ""
    try:
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            prompt = tokenizer.apply_chat_template(messages[:index], tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(messages[: index + 1], tokenize=False)
            if not (prompt.startswith(rendered) and full.startswith(prompt)):
                return None
            segments += [(prompt[len(rendered):], False), (full[len(prompt):], True)]
            rendered = full
    except TemplateError:  # e.g. the template forbids this role order or a system message
        return None
    return segments or None


def build_segments(example: dict, fmt: str, cols: dict, tokenizer: PreTrainedTokenizerBase):
    """1e. One raw row -> [(text, is_trainable), ...] in reading order (None = unusable row)."""
    if fmt == "text":
        return [(as_text(example[cols["text"]]), True)]
    if fmt == "instruction":
        context = as_text(example[cols["input"]]).strip() if cols.get("input") else ""
        prompt = format_instruction_prompt(as_text(example[cols["instruction"]]).strip(), context)
        return [(prompt, False), (as_text(example[cols["response"]]).strip(), True)]
    messages = normalize_messages(example[cols["messages"]])
    return chat_segments(messages, tokenizer) if messages else None


def tokenize_batch(batch: dict, fmt: str, cols: dict, tokenizer: PreTrainedTokenizerBase, bos: list[int],
                   max_seq_length: int, truncate: bool) -> dict:
    """1e. Raw rows -> input_ids + labels.

    labels = a copy of input_ids where the model must learn, and -100 everywhere else (BOS, prompt,
    user turns). The loss then ignores the prompt: the model is trained to ANSWER, not to repeat the
    question. EOS is appended (and trained) so the model learns where an answer ends."""
    n_rows = len(next(iter(batch.values())))
    rows = [build_segments({key: batch[key][i] for key in batch}, fmt, cols, tokenizer) for i in range(n_rows)]
    # Tokenize every piece of the whole batch in ONE call: the fast (Rust) tokenizer runs it in parallel
    pieces = [text for segments in rows if segments for text, _ in segments]
    piece_ids = iter(tokenizer(pieces, add_special_tokens=False, verbose=False)["input_ids"] if pieces else [])
    eos = [tokenizer.eos_token_id] if fmt != "chat" and tokenizer.eos_token_id is not None else []
    prefix = bos if fmt != "chat" else []  # chat templates already contain their own BOS / control tokens

    all_ids, all_labels = [], []
    for segments in rows:
        if not segments:
            all_ids.append([])
            all_labels.append([])
            continue
        ids, labels = list(prefix), [IGNORE_INDEX] * len(prefix)
        for _, trainable in segments:
            piece = next(piece_ids)
            ids += piece
            labels += piece if trainable else [IGNORE_INDEX] * len(piece)
        ids += eos
        labels += eos
        if truncate:
            ids, labels = ids[:max_seq_length], labels[:max_seq_length]
        all_ids.append(ids)
        all_labels.append(labels)
    return {"input_ids": all_ids, "labels": all_labels}


def has_trainable_tokens(batch: dict) -> list[bool]:
    # Position 0 is never predicted (it has no previous token), so only labels[1:] can carry loss
    return [any(label != IGNORE_INDEX for label in labels[1:]) for labels in batch["labels"]]


def pack_batch(batch: dict, block_size: int) -> dict:
    """1e (packing). Concatenate sequences and cut equal blocks of block_size tokens: zero padding.
    The tail of each 1000-row map batch that does not fill a whole block is dropped."""
    ids = list(chain.from_iterable(batch["input_ids"]))
    labels = list(chain.from_iterable(batch["labels"]))
    usable = (len(ids) // block_size) * block_size
    return {"input_ids": [ids[i:i + block_size] for i in range(0, usable, block_size)],
            "labels": [labels[i:i + block_size] for i in range(0, usable, block_size)]}


def _map(ds, function, cfg: ScriptConfig, desc: str, **kwargs):
    """Dataset.map with caching + optional multiprocessing, or the lazy IterableDataset.map when streaming."""
    if isinstance(ds, IterableDataset):
        return ds.map(function, **kwargs)
    return ds.map(function, num_proc=cfg.preprocessing_num_workers, load_from_cache_file=not cfg.overwrite_cache,
                  desc=desc, **kwargs)


def _filter(ds, function, cfg: ScriptConfig, desc: str, **kwargs):
    if isinstance(ds, IterableDataset):
        return ds.filter(function, **kwargs)
    return ds.filter(function, num_proc=cfg.preprocessing_num_workers,
                     load_from_cache_file=not cfg.overwrite_cache, desc=desc, **kwargs)


def preprocess(ds, fmt: str, cols: dict, tokenizer: PreTrainedTokenizerBase, cfg: ScriptConfig, name: str):
    """1e. Template + tokenize + label masking (+ packing). Drops rows left without trainable tokens."""
    fn_kwargs = {"fmt": fmt, "cols": cols, "tokenizer": tokenizer, "bos": bos_ids(tokenizer),
                 "max_seq_length": cfg.max_seq_length, "truncate": not cfg.packing}
    n_rows = None if isinstance(ds, IterableDataset) else len(ds)
    ds = _map(ds, tokenize_batch, cfg, f"Tokenizing {name}", batched=True,
              remove_columns=get_column_names(ds), fn_kwargs=fn_kwargs)
    ds = _filter(ds, has_trainable_tokens, cfg, f"Dropping {name} rows with nothing to learn", batched=True)
    if n_rows is not None and len(ds) < n_rows:
        print(f"[data] {name}: dropped {n_rows - len(ds):,} of {n_rows:,} rows with no trainable token left "
              f"(prompt alone fills max_seq_length={cfg.max_seq_length}, or a chat the template cannot mask)")
    if cfg.packing:
        ds = _map(ds, pack_batch, cfg, f"Packing {name} into {cfg.max_seq_length}-token blocks", batched=True,
                  fn_kwargs={"block_size": cfg.max_seq_length})
        ds = _filter(ds, has_trainable_tokens, cfg, f"Dropping {name} blocks with nothing to learn", batched=True)
    return ds


def describe_tokenized(name: str, ds, tokenizer: PreTrainedTokenizerBase, show_example: bool = False) -> None:
    """Sanity check: size, length stats, share of trained tokens and one decoded example."""
    if isinstance(ds, IterableDataset):
        print(f"[data] {name}: streaming (tokenized on the fly while training)")
        return
    sample = ds.select(range(min(len(ds), 2000)))
    lengths = [len(ids) for ids in sample["input_ids"]]
    label_rows = list(sample["labels"])
    trained = sum(sum(1 for label in labels[1:] if label != IGNORE_INDEX) for labels in label_rows)
    positions = sum(max(len(labels) - 1, 0) for labels in label_rows)
    print(f"[data] {name}: {len(ds):,} sequences | tokens per sequence: mean {sum(lengths) / len(lengths):.0f}, "
          f"max {max(lengths)} | {100 * trained / max(positions, 1):.0f}% of tokens trained (rest masked -100)")
    if show_example:
        ids, labels = ds[0]["input_ids"], ds[0]["labels"]
        pieces, start = [], 0
        for end in range(1, len(ids) + 1):  # group consecutive tokens with the same masked/trained state
            if end == len(ids) or (labels[end] != IGNORE_INDEX) != (labels[start] != IGNORE_INDEX):
                text = tokenizer.decode(ids[start:end])
                pieces.append(f"[[{text}]]" if labels[start] != IGNORE_INDEX else text)
                start = end
        rendered = "".join(pieces)
        print("[data] training example 0 - the loss is computed only on the [[...]] parts:")
        print(rendered[:700] + (" ..." if len(rendered) > 700 else ""))


def load_and_prepare_data(cfg: ScriptConfig, tokenizer: PreTrainedTokenizerBase):
    """Runs 1a-1e and returns (train_ds, eval_ds or None, layout, columns)."""
    train_raw, eval_raw = load_raw_splits(cfg)  # 1a
    if cfg.eval_strategy == "no":
        eval_raw = None  # no validation wanted -> nothing is held out, every row is used for training
    fmt, cols = resolve_layout(cfg, get_column_names(train_raw))  # 1b
    if fmt == "chat" and tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE  # saved with the tokenizer -> reused at inference
        print("[data] the tokenizer has no chat template -> using the plain '### User / ### Assistant' template")
    streaming = isinstance(train_raw, IterableDataset)

    # 1c. clean + de-duplicate
    n_rows = None if streaming else len(train_raw)
    clean_kwargs = {"batched": True, "fn_kwargs": {"fmt": fmt, "cols": cols}}
    train_raw = _filter(train_raw, is_valid_batch, cfg, "Removing empty / malformed rows", **clean_kwargs)
    if eval_raw is not None:
        eval_raw = _filter(eval_raw, is_valid_batch, cfg, "Removing empty / malformed rows", **clean_kwargs)
    if not streaming:
        print(f"[data] cleaning kept {len(train_raw):,} of {n_rows:,} train rows")
        if cfg.deduplicate:
            n_before = len(train_raw)
            train_raw = deduplicate(train_raw, cols, cfg)
            print(f"[data] de-duplication removed {n_before - len(train_raw):,} exact duplicate rows")

    # 1d. validation split + subsampling
    if streaming:
        train_raw, eval_raw = split_and_subsample_streaming(train_raw, eval_raw, cfg)
    else:
        train_raw, eval_raw = split_and_subsample(train_raw, eval_raw, cfg)
        print(f"[data] using {len(train_raw):,} train rows and {len(eval_raw) if eval_raw is not None else 0:,} "
              "validation rows")

    # 1e. template + tokenize + label masking (+ packing)
    train_ds = preprocess(train_raw, fmt, cols, tokenizer, cfg, "train")
    eval_ds = preprocess(eval_raw, fmt, cols, tokenizer, cfg, "validation") if eval_raw is not None else None
    if not streaming and len(train_ds) == 0:
        raise ValueError("No training examples left after preprocessing - check the layout, the columns "
                         "and --max_seq_length (answers longer than the limit are cut off).")
    if eval_ds is not None and len(eval_ds) == 0:
        eval_ds = None
    describe_tokenized("train", train_ds, tokenizer, show_example=True)
    if eval_ds is not None:
        describe_tokenized("validation", eval_ds, tokenizer)
    return train_ds, eval_ds, fmt, cols


@dataclass
class CausalLMCollator:
    """1f. Dynamic padding: each batch is padded (on the right) only to its own longest sequence.
    input_ids <- pad_token_id, attention_mask <- 0, labels <- -100 on the padding.
    NOTE: labels are padded with -100 explicitly. GPT-2 uses EOS as its pad token, so a collator
    that masks "labels == pad_token_id" (DataCollatorForLanguageModeling) would also hide every real
    EOS, and the model would never learn to stop - and it would overwrite the prompt masking."""

    pad_token_id: int
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            max_length = math.ceil(max_length / self.pad_to_multiple_of) * self.pad_to_multiple_of
        input_ids = torch.full((len(features), max_length), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(features), max_length), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_length), dtype=torch.long)
        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            labels[row, :length] = torch.tensor(feature["labels"], dtype=torch.long)
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class LengthGroupedBatchSampler(Sampler[list[int]]):
    """1f. Random batches made of similar-length sequences -> far less padding (big win on CPU).
    Shuffle all indices, sort by length inside "mega-batches" of batch_size * mega_batch_mult,
    cut into batches, then shuffle the batch order. New permutation every epoch (seed + epoch)."""

    def __init__(self, lengths: list[int], batch_size: int, seed: int, mega_batch_mult: int = 50):
        super().__init__()
        self.lengths, self.batch_size, self.seed, self.mega_batch_mult = lengths, batch_size, seed, mega_batch_mult
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        indices = torch.randperm(len(self.lengths), generator=generator).tolist()
        mega_size = self.batch_size * self.mega_batch_mult
        batches = []
        for start in range(0, len(indices), mega_size):
            mega = sorted(indices[start:start + mega_size], key=self.lengths.__getitem__, reverse=True)
            batches += [mega[i:i + self.batch_size] for i in range(0, len(mega), self.batch_size)]
        order = torch.randperm(len(batches), generator=generator).tolist()
        yield from (batches[i] for i in order)


def build_dataloaders(train_ds, eval_ds, tokenizer: PreTrainedTokenizerBase, cfg: ScriptConfig,
                      device: torch.device):
    """1f. DataLoaders: shuffled (or length-grouped) training batches, ordered validation batches."""
    collator = CausalLMCollator(tokenizer.pad_token_id, pad_to_multiple_of=8 if device.type == "cuda" else None)
    loader_kwargs = {"collate_fn": collator, "num_workers": cfg.dataloader_num_workers,
                     "pin_memory": device.type == "cuda", "persistent_workers": cfg.dataloader_num_workers > 0}
    if isinstance(train_ds, IterableDataset):
        train_loader = DataLoader(train_ds, batch_size=cfg.per_device_train_batch_size, **loader_kwargs)
    elif cfg.group_by_length and not cfg.packing:
        lengths = [len(ids) for batch in train_ds.select_columns(["input_ids"]).iter(batch_size=1000)
                   for ids in batch["input_ids"]]
        sampler = LengthGroupedBatchSampler(lengths, cfg.per_device_train_batch_size, cfg.seed)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.per_device_train_batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(cfg.seed), **loader_kwargs)
    eval_loader = None
    if eval_ds is not None:
        eval_loader = DataLoader(eval_ds, batch_size=cfg.per_device_eval_batch_size, shuffle=False, **loader_kwargs)
    return train_loader, eval_loader


#=====================================================================================
#2. Load base model & setup configuration
#=====================================================================================
# NOTE: main() loads the tokenizer first because section 1 needs it to tokenize the data.


def load_tokenizer(cfg: ScriptConfig) -> PreTrainedTokenizerBase:
    model_name = cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=cfg.model_revision,
                                              trust_remote_code=cfg.trust_remote_code)
    # NOTE: GPT-2 / Llama tokenizers have no pad token. Re-using EOS for padding is safe here because
    #       padded positions are excluded by attention_mask and by labels = -100 (see CausalLMCollator).
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"{model_name} tokenizer has neither a pad nor an EOS token - add one with "
                             "tokenizer.add_special_tokens(...) and model.resize_token_embeddings(len(tokenizer))")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # training pads on the right; batched generation would use "left"
    return tokenizer


def check_max_seq_length(cfg: ScriptConfig) -> None:
    """Never tokenize longer than the model's position embeddings allow (GPT-2: 1024)."""
    config = AutoConfig.from_pretrained(cfg.model_name, revision=cfg.model_revision,
                                        trust_remote_code=cfg.trust_remote_code)
    limit = getattr(config, "max_position_embeddings", None)
    if limit and cfg.max_seq_length > limit:
        print(f"[setup] max_seq_length {cfg.max_seq_length} > model limit {limit} -> using {limit}")
        cfg.max_seq_length = limit


def uses_conv1d(model) -> bool:
    """GPT-2 style models implement their projections with transformers' Conv1D (weights stored
    transposed, in x out); LoRA must then be told fan_in_fan_out=True. (A 4-bit QLoRA load turns
    those layers into ordinary Linear4bit layers, so this correctly returns False there.)"""
    return any(isinstance(module, Conv1D) for module in model.modules())


def qlora_load_kwargs(device: torch.device, model_dtype: torch.dtype) -> dict:
    """Shared 4-bit settings for this template's single-CUDA-device QLoRA path."""
    if device.type != "cuda":
        raise SystemExit("This template's QLoRA path (--use_qlora) needs an NVIDIA GPU with CUDA. "
                         "Run without it on CPU / MPS.")
    if find_spec("bitsandbytes") is None:
        raise SystemExit("QLoRA needs bitsandbytes:  pip install bitsandbytes")
    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=model_dtype,
        ),
        "device_map": {"": torch.cuda.current_device()},
    }


def load_base_model(cfg: ScriptConfig, device: torch.device, model_dtype: torch.dtype):
    model_name = cfg.model_name
    load_kwargs = {"revision": cfg.model_revision, "trust_remote_code": cfg.trust_remote_code}
    if cfg.attn_implementation:
        load_kwargs["attn_implementation"] = cfg.attn_implementation

    if cfg.use_qlora:
        # NOTE: QLoRA = the frozen base weights are stored in 4-bit NF4 (about 4x less memory), the LoRA
        #       weights stay in 16/32-bit and are the only thing trained. compute_dtype = precision of
        #       the matmuls. prepare_model_for_kbit_training freezes the base, upcasts norms/embeddings
        #       to fp32 for stability and (optionally) enables gradient checkpointing.
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=model_dtype,
            **qlora_load_kwargs(device, model_dtype),
            **load_kwargs,
        )
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=model_dtype, **load_kwargs)
        if cfg.gradient_checkpointing:
            base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model.config.use_cache = False  # NOTE: the generation KV-cache is useless in training and costs memory
    return base_model


def apply_lora(base_model, cfg: ScriptConfig) -> PeftModel:
    #set lora configuration, check 007_fine_tuning_in_practice.docx for details
    # r is 16/32 common can go up to 128 but more memory
    # lora_alpha is r or 2r , how much new weights impact the original weights, how much lora influences output

    # target modules are the layers to apply LoRA to, they can be different for different base model types
    # for GPT-2: ["c_attn"] (attention layers)
    # for Llama: ["q_proj", "k_proj", "v_proj", "o_proj"] (attention layers)    , you can select first two or all four
    # for Mistral: ["q_proj", "k_proj", "v_proj", "o_proj"] (attention layers)
    # for Qwen: ["q_proj", "k_proj", "v_proj", "o_proj"] (attention layers)
    # for Google Gemma: ["q_proj", "k_proj", "v_proj", "o_proj"] (attention layers) , edge cases native attention
    # for Phi-3: ["q_proj", "k_proj", "v_proj", "o_proj"] (attention layers)
    # for Baichuan: ["W_pack"] (attention layers)
    # NOTE: Phi-3 / Phi-3.5 fuse q, k, v into one layer named "qkv_proj", so there use ["qkv_proj", "o_proj"].
    # NOTE: lora_target_modules = None (the default) uses PEFT's per-architecture default, which is
    #       exactly ["c_attn"] for GPT-2 and ["q_proj", "v_proj"] for Llama / Mistral / Qwen2 / Gemma.
    #       --lora_target_modules q_proj k_proj v_proj o_proj picks layers by name;
    #       --lora_target_modules all-linear = every linear layer except the output head (attention + MLP),
    #       usually the best quality per parameter. print(model) shows a model's real layer names.

    # lora dropout is regularization, helps prevent overfitting, 0.05 to 0.1 is common, range 0 to 1
    # Bias is used for fine-tuning, can be "none", "all", or "lora_only"
    # NOTE: the LoRA update is scaled by lora_alpha / r, so with alpha = 2r the scale is 2 for any r.

    target_modules = cfg.lora_target_modules
    if target_modules == ["all-linear"]:
        target_modules = "all-linear"  # PEFT expects this keyword as a plain string, not a list

    lora_config = LoraConfig(
        r=cfg.lora_r,                       # Rank of low-rank matrices (16 is a common choice) Higher r more capacity but more memory
        lora_alpha=cfg.lora_alpha,          # Scaling factor (32 is a common choice) 2r
        target_modules=target_modules,      # Which layers to apply LoRA (c_attn is the attention layer in GPT-2)
        lora_dropout=cfg.lora_dropout,      # Dropout for regularization (0.1 is a common choice)
        bias="none",                # Bias training strategy (none means no bias training)
        fan_in_fan_out=uses_conv1d(base_model),  # NOTE: True only for GPT-2 style Conv1D layers
        task_type=TaskType.CAUSAL_LM # Type of task (CAUSAL_LM is for causal language modeling)
    )

    # Apply configuration to model
    # Inside get_peft_model() : base model all params trainable -> Freeze base 0 trainable -> Inject Lora Add adapters -> PEFT mode <1% trainable params
    # freeze base model params, Inject low rank matrices into target modules, makes lora params trainable, preserve original model architecture
    # NOTE: LoRA B starts at zero, so right after this call the model still behaves exactly like the base.
    #       revision is stored in adapter_config.json so the matching base commit is reloaded later.

    # NOTE: with GPT-2, transformers v5 prints a one-time "`loss_type=None` ... Using the default loss:
    #       `ForCausalLMLoss`" message at the first forward pass. It is harmless: that IS the right loss
    #       (the class name GPT2LMHeadModel just does not contain "ForCausalLM").

    model = get_peft_model(base_model, lora_config, revision=cfg.model_revision)
    model.print_trainable_parameters()
    print(f"[model] LoRA injected into: {sorted(model.peft_config['default'].target_modules)}")
    return model


#=====================================================================================
#3. Training Section
#=====================================================================================


def get_device() -> torch.device:
    # NOTE: Apple-silicon "mps" added as a fallback before the CPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(device)


def resolve_precision(cfg: ScriptConfig, device: torch.device) -> tuple[torch.dtype, Optional[torch.dtype]]:
    """Returns (dtype for the frozen base weights, autocast dtype or None = full fp32)."""
    bf16_ok = device.type == "cuda" and torch.cuda.is_bf16_supported()
    mode = cfg.mixed_precision
    if mode == "auto":
        mode = ("bf16" if bf16_ok else "fp16") if device.type == "cuda" else "no"
    if mode == "fp16" and device.type != "cuda":
        raise ValueError("fp16 mixed precision needs a CUDA GPU (use --mixed_precision bf16 or no)")
    if mode == "bf16" and device.type == "cuda" and not bf16_ok:
        raise ValueError("This GPU has no bfloat16 support - use --mixed_precision fp16")
    if mode == "bf16" and device.type == "cpu":
        print("[setup] WARNING: bf16 on a CPU is only fast with native bf16 units (AMX / AVX512-BF16, e.g. recent "
              "Xeons). On laptop CPUs (AVX2) it is emulated and can be 100x slower - use --mixed_precision no.")
    amp_dtype = {"no": None, "bf16": torch.bfloat16, "fp16": torch.float16}[mode]
    if cfg.model_dtype != "auto":
        return getattr(torch, cfg.model_dtype), amp_dtype
    # The frozen base can live in the 16-bit compute dtype on a GPU (half the memory); fp32 elsewhere
    return (amp_dtype if device.type == "cuda" and amp_dtype is not None else torch.float32), amp_dtype


def autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    """bf16 / fp16 autocast when mixed precision is on, otherwise a context that does nothing."""
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def count_target_tokens(labels: torch.Tensor) -> int:
    """Positions that contribute to the loss (the model shifts labels left by one internally)."""
    return int((labels[:, 1:] != IGNORE_INDEX).sum())


def iterate_in_groups(loader: DataLoader, group_size: int):
    """Yield lists of `group_size` consecutive batches = one optimizer step (gradient accumulation).
    The last group of an epoch may be smaller; it still makes one correctly weighted step."""
    group = []
    for batch in loader:
        group.append(batch)
        if len(group) == group_size:
            yield group
            group = []
    if group:
        yield group


def build_optimizer(model: PeftModel, cfg: ScriptConfig) -> AdamW:
    # AdamW is standard choice of optimizer for transformer models. 
    # In example we have 3 epochs for simplicity and simple data. 
    # But for the real work scenario we need learning rate schedulers, validation sets and early stopping.
    # Setup optimizer - only optimize trainable parameters
    # NOTE: model.parameters() would also hand every frozen base weight to AdamW. Only the
    #       requires_grad=True tensors (the LoRA A/B matrices) are passed now. Weight decay is
    #       not applied to 1-D tensors (biases / norms, only trainable if bias != "none").
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    param_groups = [{"params": decay, "weight_decay": cfg.weight_decay}]
    if no_decay:
        param_groups.append({"params": no_decay, "weight_decay": 0.0})

    optimizer = AdamW(
        param_groups,
        lr=cfg.learning_rate,  # Learning rate - higher than full fine-tuning
    )
    return optimizer


@torch.no_grad()
def evaluate(model: PeftModel, eval_loader: DataLoader, device: torch.device,
             amp_dtype: Optional[torch.dtype]) -> dict:
    """Validation loss = mean cross-entropy over every target token; perplexity = exp(loss)."""
    model.eval()  # dropout (including lora_dropout) off
    loss_sum, n_targets = 0.0, 0
    for batch in eval_loader:
        batch_targets = count_target_tokens(batch["labels"])
        if batch_targets == 0:
            continue
        batch = {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}
        with autocast_context(device, amp_dtype):
            loss = model(**batch).loss
        loss_sum += loss.item() * batch_targets  # mean over the batch -> sum, weighted by its target tokens
        n_targets += batch_targets
    model.train()
    eval_loss = loss_sum / max(n_targets, 1)
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return {"eval_loss": eval_loss, "eval_perplexity": perplexity}


@dataclass
class EarlyStopping:
    """Tracks the best validation loss. should_stop = `patience` evaluations in a row that did not
    beat the best loss by more than `threshold`."""

    patience: int
    threshold: float = 0.0
    best: float = math.inf
    best_step: Optional[int] = None
    bad_evals: int = 0

    def update(self, eval_loss: float, step: int) -> bool:
        """Record one evaluation. Returns True if it is the best so far (=> save that adapter)."""
        self.bad_evals = 0 if eval_loss < self.best - self.threshold else self.bad_evals + 1
        if eval_loss < self.best:
            self.best, self.best_step = eval_loss, step
            return True
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad_evals >= self.patience


def train(model: PeftModel, train_loader: DataLoader, eval_loader: Optional[DataLoader], cfg: ScriptConfig,
          device: torch.device, amp_dtype: Optional[torch.dtype]) -> dict:
    optimizer = build_optimizer(model, cfg)

    # Training configuration, check if CUDA is available, better use mixed precision training and scheduler
    num_epochs = cfg.num_epochs
    streaming = isinstance(train_loader.dataset, IterableDataset)
    if streaming:
        num_epochs = sys.maxsize  # NOTE: a stream has no length; passes repeat until max_steps is reached
    steps_per_epoch = None if streaming else math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    if steps_per_epoch == 0:
        raise ValueError("The training DataLoader is empty. Check the data source and preprocessing filters.")
    total_steps = cfg.max_steps if cfg.max_steps > 0 else num_epochs * steps_per_epoch
    if cfg.max_steps > 0 and not streaming:
        num_epochs = math.ceil(total_steps / steps_per_epoch)  # NOTE: max_steps decides the number of passes

    # NOTE: the schedule length is computed from the data (optimizer steps, after gradient
    #       accumulation) instead of a fixed 1000, and warmup is a share of it
    #       (the original 100 warmup steps of 1000 = warmup_ratio 0.1).
    scheduler = get_scheduler(
        cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    # NOTE: mixed precision = matmuls run in bf16 / fp16 under autocast while the LoRA weights stay fp32.
    #       fp16 has a small range, so GradScaler multiplies the loss to stop gradients underflowing
    #       (and skips a step if they overflow); bf16 has the fp32 range and needs no scaler.
    use_grad_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=use_grad_scaler)

    if not cfg.use_qlora:
        model.to(device)  # NOTE: a 4-bit QLoRA model is already on the GPU (placed while loading)
    if streaming:
        print(f"Training for {total_steps} optimizer steps on {device} (streaming)")
    else:
        print(f"Training for {num_epochs} epochs on {device}")
        print(f"[train] {steps_per_epoch} optimizer steps per epoch, {total_steps} in total | effective batch "
              f"{cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps} sequences")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    clip_norm = cfg.max_grad_norm if cfg.max_grad_norm > 0 else float("inf")  # inf = measure only, no clipping
    best_dir = os.path.join(cfg.output_dir, "checkpoint-best")
    do_eval = eval_loader is not None and cfg.eval_strategy != "no"
    stopper = EarlyStopping(cfg.early_stopping_patience, cfg.early_stopping_threshold)
    history = []
    global_step, epoch, early_stopped = 0, 0, False

    def evaluate_and_checkpoint(current_epoch: int) -> bool:
        """Validation + best-adapter checkpoint + early-stopping bookkeeping. True = stop training."""
        metrics = evaluate(model, eval_loader, device, amp_dtype)
        is_best = stopper.update(metrics["eval_loss"], global_step)
        history.append({"step": global_step, "epoch": current_epoch + 1, **metrics})
        if is_best:
            model.save_pretrained(best_dir)  # adapter weights only (a few MB)
        status = "new best -> saved checkpoint-best" if is_best else "no improvement"
        if stopper.bad_evals and cfg.early_stopping_patience > 0:  # a new best can still count if gain <= threshold
            status += f" | early-stopping patience {stopper.bad_evals}/{cfg.early_stopping_patience}"
        print(f"[eval] step {global_step} | loss {metrics['eval_loss']:.4f} | "
              f"perplexity {metrics['eval_perplexity']:.2f} | {status}")
        return stopper.should_stop

    if do_eval:
        metrics = evaluate(model, eval_loader, device, amp_dtype)
        history.append({"step": 0, "epoch": 0, **metrics})
        print(f"[eval] before training | loss {metrics['eval_loss']:.4f} | perplexity "
              f"{metrics['eval_perplexity']:.2f} (LoRA B is still zero: this is the base model)")

    window_loss, window_steps, window_tokens, window_start = 0.0, 0, 0, time.perf_counter()

    # Training loop
    model.train()
    try:
        for epoch in range(num_epochs):
            if streaming:
                train_loader.dataset.set_epoch(epoch)  # NOTE: new shuffle-buffer order on every pass
            total_loss, epoch_steps = 0.0, 0

            # NOTE: one optimizer step = gradient_accumulation_steps micro-batches. Each micro-batch loss
            #       is weighted by its share of the step's target tokens, so the step optimizes the exact
            #       mean loss over all target tokens, whatever the sequence lengths and padding are.
            for micro_batches in iterate_in_groups(train_loader, cfg.gradient_accumulation_steps):
                n_targets = [count_target_tokens(batch["labels"]) for batch in micro_batches]
                step_targets = sum(n_targets)
                if step_targets == 0:
                    continue
                step_loss = 0.0
                for batch, batch_targets in zip(micro_batches, n_targets):
                    if batch_targets == 0:
                        continue
                    window_tokens += int(batch["attention_mask"].sum())  # counted on the CPU copy: no GPU sync
                    # Move batch to device
                    batch = {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}

                    # Forward pass
                    # NOTE: the batch is a dict: input_ids + attention_mask + labels (prompt / padding = -100);
                    #       the model shifts the labels by one and returns the mean next-token loss.
                    with autocast_context(device, amp_dtype):
                        outputs = model(**batch)
                    loss = outputs.loss * (batch_targets / step_targets)

                    # Backward pass
                    scaler.scale(loss).backward()
                    step_loss += loss.item()

                if not use_grad_scaler and not math.isfinite(step_loss):
                    raise RuntimeError(f"Loss became {step_loss} at step {global_step + 1}: lower --learning_rate")

                # Update weights
                scaler.unscale_(optimizer)  # NOTE: true gradient values again before clipping (no-op without fp16)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, clip_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)  # NOTE: with fp16 this skips the update if gradients overflowed
                scaler.update()
                if not use_grad_scaler or scaler.get_scale() >= scale_before:
                    scheduler.step()  # advance the LR schedule only when the optimizer really stepped
                optimizer.zero_grad(set_to_none=True)
                total_loss += step_loss
                epoch_steps += 1
                global_step += 1

                window_loss += step_loss
                window_steps += 1
                if global_step % cfg.logging_steps == 0 or global_step == total_steps:
                    elapsed = max(time.perf_counter() - window_start, 1e-9)
                    print(f"step {global_step:>5}/{total_steps} | epoch {epoch + 1}"
                          f" | loss {window_loss / window_steps:.4f}"
                          f" | lr {scheduler.get_last_lr()[0]:.2e} | grad_norm {float(grad_norm):.3f}"
                          f" | {window_tokens / elapsed:,.0f} tok/s")
                    window_loss, window_steps, window_tokens, window_start = 0.0, 0, 0, time.perf_counter()

                if do_eval and cfg.eval_strategy == "steps" and global_step % cfg.eval_steps == 0:
                    eval_start = time.perf_counter()
                    early_stopped = evaluate_and_checkpoint(epoch)
                    window_start += time.perf_counter() - eval_start  # keep eval time out of tok/s
                if early_stopped or global_step >= total_steps:
                    break

            if epoch_steps == 0:
                raise RuntimeError("No training batch with target tokens was produced in this pass. "
                                   "The dataset may be empty or all examples were filtered / masked. "
                                   "Check the data source and --max_seq_length.")

            # Print epoch statistics
            avg_loss = total_loss / max(epoch_steps, 1)
            print(f"Epoch {epoch+1}/{'-' if streaming else num_epochs}, Loss: {avg_loss:.4f}")
            if do_eval and cfg.eval_strategy == "epoch" and not early_stopped:
                eval_start = time.perf_counter()
                early_stopped = evaluate_and_checkpoint(epoch)
                window_start += time.perf_counter() - eval_start
            if early_stopped:
                print(f"[train] early stopping: {cfg.early_stopping_patience} evaluation(s) in a row did not beat the "
                      f"best loss by more than {cfg.early_stopping_threshold}")
            if early_stopped or global_step >= total_steps:
                break
        # The last steps may not have been evaluated yet (eval_strategy="steps")
        if do_eval and not early_stopped and history[-1]["step"] != global_step:
            evaluate_and_checkpoint(epoch)
    except KeyboardInterrupt:
        print("\n[train] interrupted with Ctrl+C - keeping the best (or the latest) adapter")

    # NOTE: the adapter with the lowest validation loss is restored before saving, so overfitting
    #       in the last epochs does not end up in the saved adapter.
    if stopper.best_step is not None:
        lora_device = trainable_params[0].device
        set_peft_model_state_dict(model, load_peft_weights(best_dir, device=str(lora_device)))
        print(f"[train] restored best adapter: step {stopper.best_step} | eval loss {stopper.best:.4f}")
    return {
        "optimizer_steps": global_step,
        "best_step": stopper.best_step,
        "best_eval_loss": stopper.best if stopper.best_step is not None else None,
        "history": history,
    }


#=====================================================================================
# Saving Model adapters
#=====================================================================================


def package_versions() -> dict:
    versions = {}
    for package in ("torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def save_adapter(model: PeftModel, tokenizer: PreTrainedTokenizerBase, cfg: ScriptConfig, summary: dict) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 5. Save adapter, LoRA adapter weights and configuration are saved
    # Automatically loads base model + adapter
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    # NOTE: adapter_config.json records base_model_name_or_path; that is how AutoPeftModelForCausalLM
    #       (next section) finds and loads the base model + adapter automatically. The tokenizer is
    #       saved too because training changed it (pad token, maybe a chat template).
    #       run_summary.json = layout, settings, package versions and the validation history.
    with open(os.path.join(cfg.output_dir, "run_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"[save] adapter + tokenizer + run_summary.json -> {os.path.abspath(cfg.output_dir)}")
    # NOTE: share it with model.push_to_hub("user/my-lora") and tokenizer.push_to_hub(...) (after `hf auth login`).


#=====================================================================================
# Loading Model
#=====================================================================================


def load_finetuned_model(cfg: ScriptConfig, model_dtype: torch.dtype) -> PeftModel:
    # NOTE: the path now matches the save folder (it was "./my_lora_adapter" vs "./lora_adapter"), and
    #       `torch_dtype=` is called `dtype=` in current transformers. model_dtype=auto uses fp32 on
    #       CPU (see resolve_precision). --use_qlora keeps inference in 4-bit;
    #       an explicit --save_merged_model reloads an unquantized base for export (more memory).
    #       Manual alternative: PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), adapter_dir)

    load_kwargs = {"device_map": "auto", "dtype": model_dtype, "trust_remote_code": cfg.trust_remote_code}
    if cfg.use_qlora and not cfg.save_merged_model:
        load_kwargs.update(qlora_load_kwargs(get_device(), model_dtype))
    elif cfg.use_qlora:
        print("[load] merged export requested: reloading an unquantized base (needs more memory)")
    model = AutoPeftModelForCausalLM.from_pretrained(cfg.output_dir, **load_kwargs)
    print("Model with adapter loaded automatically!")
    return model


#=====================================================================================
# Test Model
#=====================================================================================


def encode_prompt(prompt: str, fmt: str, tokenizer: PreTrainedTokenizerBase) -> dict[str, torch.Tensor]:
    """A raw user prompt -> model inputs, built exactly like the training prompts."""
    if fmt == "chat":
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    else:
        text = format_instruction_prompt(prompt) if fmt == "instruction" else prompt
        ids = bos_ids(tokenizer) + tokenizer(text, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([ids], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def test_model(model: PeftModel, cfg: ScriptConfig, fmt: str) -> None:
    #max_length: Maximum tokens to generate   
    #temperature: Randomness (0.7-1.0)   
    #top_p: Nucleus sampling (0.9-0.95)   
    #do_sample: Enable sampling
    # NOTE: max_length counts prompt + generated tokens; max_new_tokens (used below) counts only the
    #       generated ones. repetition_penalty > 1 discourages the loops small models like GPT-2 fall into.

    # Load tokenizer
    # NOTE: from the adapter folder instead of "gpt2": it holds the pad token / chat template used in training
    tokenizer = AutoTokenizer.from_pretrained(cfg.output_dir, trust_remote_code=cfg.trust_remote_code)

    # Create prompt
    prompt = cfg.test_prompt

    # Tokenize input
    # NOTE: wrapped in the same instruction / chat template the adapter was trained with
    inputs = {name: tensor.to(model.device) for name, tensor in encode_prompt(prompt, fmt, tokenizer).items()}
    generation_kwargs = {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample,
        "repetition_penalty": cfg.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if cfg.do_sample:  # temperature / top_p only exist for sampling (avoids a warning with greedy decoding)
        generation_kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p)

    # Generate text
    model.eval()
    set_seed(cfg.seed)  # NOTE: the same sampling seed for both runs -> a fair base vs LoRA comparison
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    # Decode and print
    prompt_length = inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    print(f"\n[test] layout: {fmt} | prompt: {prompt}")
    print(f"--- base model + LoRA adapter ---\n{generated_text.strip()}")

    if cfg.compare_with_base:
        # NOTE: disable_adapter() temporarily switches LoRA off -> the untouched base model, same weights in memory
        set_seed(cfg.seed)
        with torch.no_grad(), model.disable_adapter():
            base_outputs = model.generate(**inputs, **generation_kwargs)
        base_text = tokenizer.decode(base_outputs[0][prompt_length:], skip_special_tokens=True)
        print(f"--- base model only (adapter disabled) ---\n{base_text.strip()}")


#=====================================================================================
# NOTE: Optional - merge the adapter into the base model (deployment)
#=====================================================================================
# NOTE: merge_and_unload() adds B @ A * (alpha / r) into the frozen weights and removes the LoRA
#       layers: a plain transformers model, no PEFT needed and no extra latency at inference
#       (vLLM / TGI / llama.cpp GGUF conversion expect this). It costs a full model on disk.


def merge_and_save(model: PeftModel, cfg: ScriptConfig) -> None:
    merged_dir = os.path.normpath(cfg.output_dir) + "-merged"
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(cfg.output_dir, trust_remote_code=cfg.trust_remote_code).save_pretrained(merged_dir)
    print(f"[merge] standalone merged model -> {os.path.abspath(merged_dir)} "
          "(load it with AutoModelForCausalLM.from_pretrained)")


def read_trained_layout(cfg: ScriptConfig) -> str:
    """Inference-only runs: the layout the adapter was trained with decides how prompts are built."""
    summary_path = os.path.join(cfg.output_dir, "run_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as file:
            return json.load(file)["dataset_format"]
    return cfg.dataset_format if cfg.dataset_format != "auto" else "text"


def main() -> None:
    with contextlib.suppress(AttributeError, ValueError):
        # UTF-8: Windows consoles can print any generated text; line_buffering: logs appear immediately
        # even when the output is redirected to a file (python best-LORA-Example.py > train.log)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    cfg = parse_config()
    set_seed(cfg.seed)  # python, numpy and torch RNGs -> reproducible splits, LoRA init and dropout
    device = get_device()
    model_dtype, amp_dtype = resolve_precision(cfg, device)
    print(f"[setup] model: {cfg.model_name} | device: {device} | base weights: {model_dtype} | "
          f"mixed precision: {amp_dtype or 'off'} | QLoRA: {cfg.use_qlora}")

    if cfg.do_train:
        tokenizer = load_tokenizer(cfg)  # section 2 helper, needed first by section 1
        check_max_seq_length(cfg)
        train_ds, eval_ds, fmt, cols = load_and_prepare_data(cfg, tokenizer)  # section 1
        train_loader, eval_loader = build_dataloaders(train_ds, eval_ds, tokenizer, cfg, device)
        model = apply_lora(load_base_model(cfg, device, model_dtype), cfg)  # section 2
        results = train(model, train_loader, eval_loader, cfg, device, amp_dtype)  # section 3
        summary = {
            "base_model": cfg.model_name,
            "dataset_format": fmt,
            "columns": cols,
            "train_sequences": None if isinstance(train_ds, IterableDataset) else len(train_ds),
            "eval_sequences": 0 if eval_ds is None else len(eval_ds),
            "device": str(device),
            "base_weights_dtype": str(model_dtype),
            "mixed_precision": str(amp_dtype) if amp_dtype else "off",
            **results,
            "config": asdict(cfg),
            "versions": package_versions(),
        }
        save_adapter(model, tokenizer, cfg, summary)
        del model, train_loader, eval_loader  # free the training copy before loading the inference copy
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        if not os.path.isfile(os.path.join(cfg.output_dir, "adapter_config.json")):
            raise SystemExit(f"No adapter in '{cfg.output_dir}'. Train one first (--do_train true).")
        fmt = read_trained_layout(cfg)

    model = load_finetuned_model(cfg, model_dtype)
    test_model(model, cfg, fmt)
    if cfg.save_merged_model:
        merge_and_save(model, cfg)


if __name__ == "__main__":
    main()
