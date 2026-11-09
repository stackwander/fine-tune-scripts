"""
best-VLM-LoRA-Example.py - LoRA / QLoRA fine-tuning of open VISION-LANGUAGE models (VLMs)
with Hugging Face PEFT. Companion of best-LORA-Example.py (text-only LLMs) in this folder.

WHAT A VLM IS
  image -> vision encoder (ViT / SigLIP) -> projector -> "image tokens" -> language model -> text
  The processor turns every image into a fixed or dynamic number of image tokens that sit inside
  the prompt (SmolVLM: 64 per 512 px tile, Qwen2.5-VL: one per 28 x 28 px patch).

WHAT IS TRAINED
  LoRA adapters on the LANGUAGE MODEL's linear layers (attention + MLP). The vision encoder and the
  projector stay frozen - the standard recipe: the model learns your task / format / domain wording
  while keeping its pretrained eyesight. --lora_target_modules all-linear also adapts the vision side.
  The loss is computed on the ANSWER tokens only (prompt, image tokens and padding are masked -100).

PIPELINE (main() at the bottom runs it in this order)
  1. Data  : load (Hub / imagefolder / JSON-CSV-Parquet with image paths or embedded images /
             save_to_disk / streaming) -> detect layout -> resolve image paths -> clean (without
             decoding images) -> de-duplicate -> validation split -> per batch: images + chat template
             -> processor -> label masking (collator)
  2. Model : processor (+ pad token, image resolution cap) -> AutoModelForImageTextToText
             (fp32 CPU / bf16-fp16 GPU / 4-bit QLoRA) -> LoRA on the language model
  3. Train : AdamW on LoRA weights, warmup + schedule, mixed precision, token-weighted gradient
             accumulation, clipping, validation loss / perplexity, early stopping, best adapter restored
  Then     : save adapter + processor -> reload base + adapter -> compare LoRA vs base on validation
             images (reference answer shown) -> optional merge

DATA LAYOUTS (auto-detected from column names, or forced with --dataset_format)
  caption : image + text                     {"image": ..., "text": "..."}   prompt = --instruction
  vqa     : image + question + answer        {"image": ..., "question": "...", "answer": "..."}
  chat    : images + conversation turns      {"images": [...], "messages": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
                {"role": "assistant", "content": [{"type": "text", "text": "..."}]}]}
            also: string content with "<image>" placeholders, LLaVA / ShareGPT {"from", "value"}
            turns, the_cauldron {"user", "assistant"} turns. Images without placeholders are put in
            front of the first user message.
  An image cell can be an embedded image (Hub / Parquet / imagefolder), a file path (absolute, or
  relative to --image_root), an http(s) URL, a .npy array file, or a list of those for multi-image rows.

IMAGES WITH MORE THAN 8 BITS (electron / light microscopy, medical, astronomy, thermal or depth cameras)
  VLMs see 8-bit RGB. 8-bit images (JPEG, PNG, ...) are used as they are; transparent ones are put on
  white. 16-bit / 32-bit / float images (TIFF, PNG) and .npy arrays are mapped to 0-255 first:
    --image_scaling percentile   per-image contrast stretch between --image_percentiles (default 0.5 99.5,
                                 robust to hot pixels; 0 100 = plain min-max). The default.
    --image_scaling fixed --image_value_range 0 4095   one raw range for every image (here a 12-bit
                                 detector), so brightness stays comparable between images.
  Without this a 16-bit image is clipped at 255 and comes out almost completely white. NaN pixels become
  black. Multi-page TIFFs: the first page is used. Vendor files (.dm3 / .dm4 / .emd / .mrc): export them
  to TIFF or .npy first (e.g. with HyperSpy). Use the same image settings at inference (--do_train false
  warns when they differ from the ones the adapter was trained with).

RUN - THIS MACHINE (Windows, CPU): small model, small images
  python best-VLM-LoRA-Example.py                    # SmolVLM-256M on LaTeX_OCR: ~20 min, ~4 GB RAM
  python best-VLM-LoRA-Example.py --help             # every option with its default
  python best-VLM-LoRA-Example.py --dataset_name data/my_imagefolder            # images + metadata.jsonl
  python best-VLM-LoRA-Example.py --train_file data/vqa.jsonl --image_root data/images
  python best-VLM-LoRA-Example.py --do_train false --test_image photo.jpg --test_prompt "What is shown?"

RUN - VAST.AI (Linux, NVIDIA GPU): pip install -U "transformers>=4.56" peft datasets accelerate pillow bitsandbytes
  python best-VLM-LoRA-Example.py --model_name Qwen/Qwen2.5-VL-7B-Instruct --use_qlora \\
      --gradient_checkpointing --image_longest_edge 768 --max_seq_length 2048 --max_train_samples 0
  python best-VLM-LoRA-Example.py --model_name Qwen/Qwen2.5-VL-3B-Instruct \\
      --dataset_name HuggingFaceM4/ChartQA --max_train_samples 0          # plain LoRA, 24 GB GPU
  Use --use_qlora on inference-only runs too (--do_train false) to keep the base in 4-bit.
  --save_merged_model explicitly reloads an unquantized base for export and needs more memory.

MEMORY (the usual VLM failure is running out of it - on Windows that can look like a crash, exit code 139)
  A step holds the image tokens of every example plus the vocabulary logits of the positions that are
  trained (this script computes logits only there). If memory runs out, in this order: lower
  --per_device_train_batch_size (raise --gradient_accumulation_steps to keep the effective batch),
  lower --image_longest_edge, add --gradient_checkpointing, use --use_qlora (GPU). Long answers (chat
  data such as llava-instruct-mix) need smaller batches than short ones (captions, VQA, OCR).

HARMLESS MESSAGES
  transformers 5.12 prints "[ERROR] `loss` is part of Qwen2_5_VLCausalLMOutputWithPast.__init__'s
  signature, but not documented" when a Qwen-VL class is loaded (a library docstring check), and
  SmolVLM prints "pad_token_id must be None or an integer within the vocabulary". Training is unaffected.

Tested (Oct 2026) with torch 2.12.1, transformers 5.12.1, peft 0.19.1, datasets 5.0.0 on a CPU laptop:
default run (SmolVLM-256M-Instruct, 300 LaTeX_OCR images, 38 steps) took 19 minutes at 4.1 GB peak RAM,
validation loss 1.98 -> 1.14. Every data format above and tiny test versions of Qwen2-VL, Qwen2.5-VL,
Qwen3-VL, Idefics3 / SmolVLM, Gemma 3 and LLaVA were run end to end; Idefics2, SmolVLM 2, Gemma 4,
InternVL (-hf checkpoints) and LLaVA-NeXT passed the collator / loss / training-step checks.
16-bit / float TIFF, 16-bit PNG and .npy images (imagefolder and JSONL, both --image_scaling modes,
two-image rows, inference-only) were run end to end; 8-bit images give pixel-identical input as before.
NOT supported: Llama 3.2 Vision (its per-token cross_attention_mask is refused with a clear error) and
models without a chat template such as PaliGemma (refused). SmolVLM 2 needs `pip install num2words`.
The GPU-only paths (bf16 / fp16 autocast, QLoRA) were not run on a GPU.
"""

import contextlib
import gc
import hashlib
import inspect
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from io import BytesIO
from typing import Literal, Optional
from urllib.request import urlopen

import numpy as np
import torch
from datasets import Dataset, DatasetDict, IterableDataset, LargeList, List, Sequence, load_dataset, load_from_disk
from datasets import Image as ImageFeature
from jinja2 import TemplateError
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    load_peft_weights,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from PIL import Image, ImageOps
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    HfArgumentParser,
    get_scheduler,
    set_seed,
)

IGNORE_INDEX = -100  # label value skipped by the loss (prompt, image tokens, padding)
IMAGE_PLACEHOLDER = "<image>"  # placeholder used inside string content (LLaVA / ShareGPT style data)
MIN_IMAGE_SIDE = 32  # tiny images are upscaled to this (Qwen-VL rejects sides under 28 px)
NUMPY_MAGIC = b"\x93NUMPY"  # first bytes of a .npy file: embedded / downloaded arrays are recognized by it
# image settings that must match between training and inference (checked with --do_train false)
IMAGE_SETTINGS = ("image_longest_edge", "image_scaling", "image_percentiles", "image_value_range")
PER_TOKEN_KEYS = ("input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids")  # one value per token

SPLIT_ALIASES = ("validation", "valid", "val", "dev")  # "test" is never used for training decisions

COLUMN_ALIASES = {  # first match wins; explicit --*_column flags override these
    "messages": ("messages", "conversations", "conversation", "texts"),
    "images": ("images", "image", "img", "image_path", "image_paths", "image_url", "file_name", "path"),
    "question": ("question", "query", "prompt", "instruction", "input"),
    "answer": ("answer", "answers", "label", "labels", "response", "output", "completion"),
    "caption": ("caption", "captions", "text", "description", "latex", "formula", "sentences"),
}

ROLE_ALIASES = {
    "system": "system", "user": "user", "human": "user",
    "assistant": "assistant", "gpt": "assistant", "bot": "assistant", "model": "assistant",
}

FILE_BUILDERS = {  # file extension -> (packaged `datasets` builder, builder kwargs)
    ".csv": ("csv", {}), ".tsv": ("csv", {"sep": "\t"}),
    ".json": ("json", {}), ".jsonl": ("json", {}), ".ndjson": ("json", {}),
    ".parquet": ("parquet", {}), ".arrow": ("arrow", {}),
}
COMPRESSION_SUFFIXES = (".gz", ".bz2", ".xz", ".zst")

# Module-name fragments of the vision side (encoder, projector / connector, patch merger). LoRA skips
# them by default so only the language model is adapted.
VISION_MODULE_KEYWORDS = ("vision", "visual", "image", "multi_modal_projector", "mm_projector", "connector",
                          "merger", "patch_embed", "audio")


# =====================================================================================
# Configuration - edit the defaults here, or override on the command line (--name value)
# or with a .json / .yaml file. `python best-VLM-LoRA-Example.py --help` lists everything.
# =====================================================================================
def option(default, help_text: str):
    """A config field whose help text and default are shown by --help."""
    if isinstance(default, list):  # a dataclass needs a factory for a mutable default
        return field(default_factory=lambda: list(default), metadata={"help": help_text})
    return field(default=default, metadata={"help": help_text})


@dataclass
class ScriptConfig:
    # ---- Model ---------------------------------------------------------------------
    model_name: str = option("HuggingFaceTB/SmolVLM-256M-Instruct",
                             "Instruct VLM (needs a chat template), e.g. Qwen/Qwen2.5-VL-7B-Instruct.")
    model_revision: Optional[str] = option(None, "Pin a model commit hash / tag for reproducible runs.")
    trust_remote_code: bool = option(False, "Allow models that ship their own code; read it first.")
    attn_implementation: Optional[str] = option(
        None, "None = library default (sdpa); eager; flash_attention_2 (needs the flash-attn package).")
    model_dtype: Literal["auto", "float32", "bfloat16", "float16"] = option(
        "auto", "Dtype of the frozen base weights. auto = bf16 / fp16 on CUDA, fp32 on CPU / MPS.")
    image_longest_edge: int = option(
        512, "Cap on the image's longest side in pixels (fewer image tokens = faster). 768-1024 on a GPU.")
    image_scaling: Literal["percentile", "fixed"] = option(
        "percentile", "Images with MORE than 8 bits per value (16-bit / 32-bit / float TIFF or PNG, .npy) -> 0-255: "
        "percentile = per-image contrast stretch; fixed = --image_value_range for all. 8-bit images: unchanged.")
    image_percentiles: list[float] = option(
        [0.5, 99.5], "percentile scaling: LOW HIGH percentiles that become black / white (0 100 = min-max).")
    image_value_range: Optional[list[float]] = option(
        None, "fixed scaling: raw LOW HIGH values that become black / white, e.g. 0 4095 (12-bit) or 0 65535.")

    # ---- Data source: if several are set, load_from_disk_path > train_file > dataset_name ----
    dataset_name: Optional[str] = option("unsloth/LaTeX_OCR",
                                         "Hub dataset repo id, or a local folder (e.g. imagefolder).")
    dataset_config_name: Optional[str] = option(None, "Hub subset / config name, e.g. ai2d for the_cauldron.")
    dataset_revision: Optional[str] = option(None, "Pin a dataset commit / tag / branch.")
    train_file: Optional[str] = option(
        None, "Path, glob or URL: .json .jsonl .csv .tsv .parquet .arrow (may be compressed).")
    validation_file: Optional[str] = option(None, "Optional validation file in the same format as --train_file.")
    load_from_disk_path: Optional[str] = option(None, "Folder written by Dataset / DatasetDict.save_to_disk().")
    image_root: Optional[str] = option(
        None, "Base folder for RELATIVE image paths. None = folder of --train_file / local dataset, else cwd.")
    json_field: Optional[str] = option(None, "JSON only: key that holds the list of records, e.g. data.")
    train_split: str = option("train", "Name of the training split.")
    eval_split: Optional[str] = option(
        None, "Validation split. None = validation / valid / val / dev if present, else carved from train.")
    streaming: bool = option(False, "Read rows lazily (huge datasets). Needs --max_steps and --max_eval_samples.")
    cache_dir: Optional[str] = option(None, "Downloads + processed Arrow caches (default ~/.cache/huggingface).")

    # ---- Data layout ---------------------------------------------------------------
    dataset_format: Literal["auto", "caption", "vqa", "chat"] = option(
        "auto", "auto = detect from the column names (see COLUMN_ALIASES).")
    images_column: Optional[str] = option(None, "Column with the image(s) / path(s) / URL(s). None = auto-detect.")
    caption_column: Optional[str] = option(None, "Caption / target text column (caption layout). None = auto.")
    question_column: Optional[str] = option(None, "Question column (vqa layout). None = auto-detect.")
    answer_column: Optional[str] = option(None, "Answer column (vqa layout); lists use the first item. None = auto.")
    messages_column: Optional[str] = option(None, "Conversation column (chat layout). None = auto-detect.")
    instruction: str = option("Write the LaTeX representation for this image.",
                              "User prompt for the caption layout (change it to match your task).")
    system_prompt: Optional[str] = option(None, "Optional system message added to every example.")

    # ---- Preprocessing -------------------------------------------------------------
    max_seq_length: int = option(
        1024, "Max tokens per example INCLUDING image tokens. Only the answer side is ever truncated.")
    deduplicate: bool = option(True, "Drop exact duplicates (same text + same image bytes/path) before splitting.")
    validation_split_percentage: float = option(
        5.0, "Percent of train held out when the source has no validation split.")
    max_train_samples: int = option(300, "Seeded random subset for quick runs. 0 = use every example.")
    max_eval_samples: int = option(
        40, "Validation subset. 0 = every example. Streaming: rows taken from the stream for validation.")
    preprocessing_num_workers: Optional[int] = option(None, "Processes for .map / .filter on big datasets.")
    overwrite_cache: bool = option(False, "Ignore the cached processed datasets and recompute them.")
    shuffle_buffer_size: int = option(1000, "Streaming only: rows held in the shuffle buffer.")

    # ---- LoRA / QLoRA --------------------------------------------------------------
    lora_r: int = option(16, "LoRA rank r.")
    lora_alpha: int = option(32, "LoRA alpha (the update is scaled by alpha / r).")
    lora_dropout: float = option(
        0.0, "Dropout on the LoRA branch. 0.05-0.1 regularizes but costs memory + ~25 percent speed (input copies).")
    lora_target_modules: Optional[list[str]] = option(
        None, "None = every linear layer of the LANGUAGE model; all-linear = also the vision side; or names.")
    use_qlora: bool = option(
        False, "Load the base in 4-bit NF4 for training and inference (this template uses CUDA + bitsandbytes).")
    gradient_checkpointing: bool = option(
        False, "Recompute activations in the backward pass: much less memory, roughly 30 percent slower.")

    # ---- Training ------------------------------------------------------------------
    output_dir: str = option("./my_vlm_lora_adapter", "Where the adapter, processor and run_summary.json go.")
    do_train: bool = option(True, "false = skip training; only load and test the adapter in --output_dir.")
    num_epochs: int = option(1, "Passes over the training data.")
    max_steps: int = option(0, "> 0 overrides num_epochs (required with --streaming).")
    per_device_train_batch_size: int = option(
        2, "Examples per training micro-batch. Images + long answers are memory-heavy: 2 fits a 16 GB laptop.")
    per_device_eval_batch_size: int = option(4, "Examples per validation batch.")
    gradient_accumulation_steps: int = option(
        4, "Micro-batches per optimizer step: effective batch = batch size x accumulation steps.")
    learning_rate: float = option(2e-4, "Peak learning rate.")
    weight_decay: float = option(0.01, "AdamW weight decay (not applied to 1-D tensors).")
    lr_scheduler_type: str = option("linear", "linear, cosine, constant_with_warmup, ... (transformers schedulers).")
    warmup_ratio: float = option(0.1, "Share of all optimizer steps used to warm the LR up from 0.")
    max_grad_norm: float = option(1.0, "Gradient clipping max norm. 0 = off.")
    mixed_precision: Literal["auto", "no", "bf16", "fp16"] = option(
        "auto", "auto = bf16 (fp16 on older GPUs) on CUDA, off on CPU / MPS.")
    eval_strategy: Literal["epoch", "steps", "no"] = option(
        "steps", "When to compute the validation loss (no = never; nothing is held out then).")
    eval_steps: int = option(10, "Evaluate every N optimizer steps when --eval_strategy steps.")
    early_stopping_patience: int = option(3, "Stop after N evaluations in a row without improvement. 0 = off.")
    early_stopping_threshold: float = option(0.0, "Minimum eval-loss decrease that resets the patience counter.")
    logging_steps: int = option(5, "Print the training loss every N optimizer steps.")
    dataloader_num_workers: Optional[int] = option(
        None, "Image-processing workers. None = 0 on Windows, 4 on Linux / macOS (e.g. Vast.ai).")
    seed: int = option(42, "Seed for splits, shuffling, LoRA init, dropout and sampling.")

    # ---- Inference / test ----------------------------------------------------------
    num_test_samples: int = option(2, "Validation images to caption / answer after training (vs reference).")
    test_image: Optional[str] = option(None, "Also run on this image file or URL.")
    test_prompt: Optional[str] = option(None, "Prompt for --test_image (None = --instruction).")
    max_new_tokens: int = option(128, "Tokens to generate (the prompt is not counted).")
    do_sample: bool = option(False, "false = greedy decoding (best for OCR / extraction); true = sampling.")
    temperature: float = option(0.7, "Sampling temperature (only with --do_sample true).")
    top_p: float = option(0.9, "Nucleus sampling threshold (only with --do_sample true).")
    repetition_penalty: float = option(1.05, "> 1.0 discourages repeating tokens.")
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
    if cfg.gradient_accumulation_steps < 1 or cfg.logging_steps < 1 or cfg.max_seq_length < 16:
        raise ValueError("--gradient_accumulation_steps / --logging_steps must be >= 1, --max_seq_length >= 16")
    if cfg.image_longest_edge < MIN_IMAGE_SIDE:
        raise ValueError(f"--image_longest_edge must be >= {MIN_IMAGE_SIDE}")
    low_high = cfg.image_percentiles
    if len(low_high) != 2 or not 0 <= low_high[0] < low_high[1] <= 100:
        raise ValueError("--image_percentiles needs LOW HIGH with 0 <= LOW < HIGH <= 100, e.g. 0.5 99.5")
    if cfg.image_scaling == "fixed":
        low_high = cfg.image_value_range or []
        if len(low_high) != 2 or not all(map(math.isfinite, low_high)) or low_high[0] >= low_high[1]:
            raise ValueError("--image_scaling fixed needs --image_value_range LOW HIGH with LOW < HIGH, e.g. 0 4095")
    elif cfg.image_value_range is not None:
        raise ValueError("--image_value_range is only used with --image_scaling fixed")


#=====================================================================================
#1. Data loading and preprocessing (vision + text)
#=====================================================================================
# 1a load raw splits -> 1b detect layout + columns -> 1c turn image paths into images -> 1d clean
# (images NOT decoded, so it is fast) + de-duplicate -> 1e validation split + subsample.
# Images are decoded, resized and turned into image tokens per batch in VisionLanguageCollator (1f):
# pixel tensors are too large to cache on disk, and on-the-fly processing keeps memory flat.


def infer_file_builder(path: str) -> tuple[str, dict]:
    """Pick the packaged `datasets` builder ("json", "csv", ...) from a file name, glob or URL."""
    name = path.lower().split("?")[0]
    for suffix in COMPRESSION_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    extension = os.path.splitext(name)[1]
    if extension not in FILE_BUILDERS:
        raise ValueError(f"Unsupported file type '{extension}' in {path}. Supported: {sorted(FILE_BUILDERS)}. "
                         "For a folder of images + metadata.jsonl/csv use --dataset_name <folder> (imagefolder).")
    builder, builder_kwargs = FILE_BUILDERS[extension]
    return builder, dict(builder_kwargs)


def load_raw_splits(cfg: ScriptConfig):
    """1a. Load the raw train split (+ the validation split when the source has one)."""
    if cfg.load_from_disk_path:
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
        # A local folder of images + metadata.jsonl / metadata.csv is loaded by the "imagefolder" builder
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


def first_row(ds) -> dict:
    return next(iter(ds.take(1))) if isinstance(ds, IterableDataset) else ds[0]


def get_column_names(ds) -> list[str]:
    """Column names of a Dataset, or of a streamed IterableDataset (peeks at the first row if unknown)."""
    if ds.column_names is not None:
        return list(ds.column_names)
    return list(first_row(ds).keys())


def resolve_layout(cfg: ScriptConfig, column_names: list[str]) -> tuple[str, dict]:
    """1b. Decide caption / vqa / chat and which column holds what (explicit flags win)."""

    def pick(role: str) -> Optional[str]:
        explicit = getattr(cfg, f"{role}_column")
        if explicit:
            if explicit not in column_names:
                raise ValueError(f"--{role}_column '{explicit}' not found. Columns: {column_names}")
            return explicit
        return next((name for name in COLUMN_ALIASES[role] if name in column_names), None)

    found = {role: pick(role) for role in COLUMN_ALIASES}
    if not found["images"]:
        raise ValueError(f"No image column found in {column_names}. Set --images_column.")
    fmt = cfg.dataset_format
    if fmt == "auto":
        if found["messages"]:
            fmt = "chat"
        elif found["question"] and found["answer"]:
            fmt = "vqa"
        elif found["caption"]:
            fmt = "caption"
        else:
            raise ValueError(f"Cannot detect the data layout from columns {column_names}. "
                             "Set --dataset_format and the matching --*_column flags.")
    required = {"chat": ("images", "messages"), "vqa": ("images", "question", "answer"),
                "caption": ("images", "caption")}[fmt]
    missing = [role for role in required if not found[role]]
    if missing:
        raise ValueError(f"Layout '{fmt}' needs a column for {missing}; columns are {column_names}. "
                         f"Use --{missing[0]}_column.")
    cols = {role: found[role] for role in required}
    print(f"[data] layout: {fmt} | columns: {cols}" + (f" | prompt: {cfg.instruction!r}" if fmt == "caption" else ""))
    return fmt, cols


def default_image_root(cfg: ScriptConfig) -> str:
    """Folder that RELATIVE image paths are resolved against."""
    if cfg.image_root:
        return os.path.abspath(cfg.image_root)
    if cfg.train_file and "://" not in cfg.train_file:
        return os.path.dirname(os.path.abspath(cfg.train_file))
    if cfg.dataset_name and os.path.isdir(cfg.dataset_name):
        return os.path.abspath(cfg.dataset_name)
    return os.getcwd()


def keep_images_undecoded(ds, column: str):
    """1c. Image feature columns are switched to decode=False: cells stay {"bytes", "path"} dicts, so
    cleaning, hashing and splitting never decode a JPEG. Path / URL string columns stay strings.
    Every image is decoded exactly once, when its batch is built (load_image in the collator)."""
    feature = ds.features.get(column) if ds.features is not None else None
    if isinstance(feature, ImageFeature):
        return ds.cast_column(column, ImageFeature(decode=False))
    if isinstance(feature, (List, LargeList, Sequence)) and isinstance(feature.feature, ImageFeature):
        return ds.cast_column(column, List(ImageFeature(decode=False)))
    return ds


def image_source(cell, root: str):
    """Image cell -> ('bytes', data) | ('url', url) | ('path', absolute path) | (None, None)."""
    if isinstance(cell, dict):
        if cell.get("bytes"):
            return "bytes", cell["bytes"]
        cell = cell.get("path")
    if not isinstance(cell, str) or not cell.strip():
        return None, None
    if "://" in cell:
        return "url", cell
    return "path", cell if os.path.isabs(cell) else os.path.normpath(os.path.join(root, cell))


def image_cell_ok(cell, root: str) -> bool:
    """Cheap validity check without decoding: embedded bytes, a URL, or a local file that exists."""
    if isinstance(cell, (Image.Image, np.ndarray)):
        return True
    kind, value = image_source(cell, root)
    return kind in ("bytes", "url") or (kind == "path" and os.path.isfile(value))


def load_image(cell, root: str):
    """PIL image | numpy array | {"bytes", "path"} dict | local path (absolute or relative to root) | http(s)
    URL -> PIL image, or a numpy array for .npy data. to_rgb (1f) turns either into 8-bit RGB."""
    if isinstance(cell, (Image.Image, np.ndarray)):
        return cell
    kind, value = image_source(cell, root)
    if kind == "url":
        with urlopen(value, timeout=30) as response:  # downloaded every time: pre-download big URL datasets
            kind, value = "bytes", response.read()
    if kind == "bytes":
        if value[: len(NUMPY_MAGIC)] == NUMPY_MAGIC:
            return np.load(BytesIO(value), allow_pickle=False)  # allow_pickle=False: data only, never code
        return Image.open(BytesIO(value))
    if kind == "path":
        if value.lower().endswith(".npy"):
            return np.load(value, allow_pickle=False)
        return Image.open(value)
    raise ValueError(f"Image cell without bytes or path: {str(cell)[:100]}")


def as_text(value) -> str:
    """Cells can be None / NaN (empty CSV cells) or numbers: turn them into plain strings."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def first_text(value) -> str:
    """Answer / caption cells may hold a list (several captions, ChartQA labels): use the first one."""
    if isinstance(value, (list, tuple)):
        return next((as_text(v).strip() for v in value if as_text(v).strip()), "")
    return as_text(value).strip()


def content_to_parts(content) -> Optional[list[dict]]:
    """Message content -> [{"type": "image"} | {"type": "text", "text": ...}] (empty texts dropped)."""
    parts = []
    if isinstance(content, str):
        for index, piece in enumerate(content.split(IMAGE_PLACEHOLDER)):
            if index > 0:
                parts.append({"type": "image"})
            if piece.strip():
                parts.append({"type": "text", "text": piece.strip()})
        return parts
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                return None
            kind = part.get("type")
            if kind in ("image", "image_url"):
                parts.append({"type": "image"})
            elif kind == "text" and isinstance(part.get("text"), str):
                if part["text"].strip():
                    parts.append({"type": "text", "text": part["text"].strip()})
            else:
                return None
        return parts
    return None


def normalize_turns(raw_turns) -> Optional[list[dict]]:
    """Conversation cell -> [{"role", "content": parts}]. Accepts OpenAI style, LLaVA / ShareGPT
    {"from", "value"} and the_cauldron {"user", "assistant"} turns."""
    if not isinstance(raw_turns, list) or not raw_turns:
        return None
    messages = []
    for turn in raw_turns:
        if not isinstance(turn, dict):
            return None
        if turn.get("role") is None and turn.get("from") is None and "user" in turn:  # the_cauldron pair
            for role in ("user", "assistant"):
                parts = content_to_parts(as_text(turn.get(role)))
                if not parts:
                    return None
                messages.append({"role": role, "content": parts})
            continue
        role = ROLE_ALIASES.get(str(turn.get("role") or turn.get("from") or "").strip().lower())
        content = turn.get("content") if turn.get("content") is not None else turn.get("value")
        parts = content_to_parts(content)
        if role is None or not parts:
            return None
        messages.append({"role": role, "content": parts})
    return messages


def row_images(cell) -> list:
    if cell is None:
        return []
    return list(cell) if isinstance(cell, (list, tuple)) else [cell]


def row_to_conversation(row: dict, fmt: str, cols: dict, instruction: str,
                        system_prompt: Optional[str]) -> Optional[tuple[list[dict], list]]:
    """One raw row (any layout) -> (multimodal chat messages, images in placeholder order) or None."""
    images = row_images(row[cols["images"]])
    if not images or any(image is None for image in images):
        return None
    if fmt == "chat":
        messages = normalize_turns(row[cols["messages"]])
        if messages is None:
            return None
    else:
        question = first_text(row[cols["question"]]) if fmt == "vqa" else instruction
        answer = first_text(row[cols["answer"] if fmt == "vqa" else cols["caption"]])
        if not question or not answer:
            return None
        messages = [{"role": "user", "content": [{"type": "text", "text": question}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    if system_prompt and messages[0]["role"] != "system":
        messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}] + messages
    n_placeholders = sum(part["type"] == "image" for message in messages for part in message["content"])
    if n_placeholders == 0:  # images without placeholders go in front of the first user message
        first_user = next((m for m in messages if m["role"] == "user"), None)
        if first_user is None:
            return None
        first_user["content"] = [{"type": "image"} for _ in images] + first_user["content"]
    elif n_placeholders != len(images):
        return None
    if not any(m["role"] == "assistant" and any(p["type"] == "text" for p in m["content"]) for m in messages):
        return None
    return messages, images


def conversation_segments(messages: list[dict], processor) -> Optional[list[tuple[str, bool]]]:
    """Render with the processor's chat template and cut into (text, trainable) pieces.

    Piece 0 is the prompt up to the first trained answer and contains EVERY image placeholder (it goes
    through the processor, which expands placeholders into image tokens). The following pieces are
    plain text: assistant answers (trainable) and later user turns (context). Assistant turns that
    come before the last image stay context. Works with templates that render earlier turns unchanged
    when later turns are added (Qwen-VL, SmolVLM / Idefics3, Gemma 3, LLaVA, ...)."""
    last_image = max(i for i, m in enumerate(messages) if any(p["type"] == "image" for p in m["content"]))
    segments, rendered = [], ""
    try:
        for index, message in enumerate(messages):
            if message["role"] != "assistant" or index < last_image:
                continue
            prompt = processor.apply_chat_template(messages[:index], add_generation_prompt=True, tokenize=False)
            full = processor.apply_chat_template(messages[: index + 1], tokenize=False)
            if not (prompt.startswith(rendered) and full.startswith(prompt)):
                return None
            segments += [(prompt[len(rendered):], False), (full[len(prompt):], True)]
            rendered = full
    except TemplateError:  # e.g. the template forbids this role order or a system message
        return None
    return segments or None


def is_valid_batch(batch: dict, fmt: str, cols: dict, instruction: str, system_prompt: Optional[str],
                   processor, root: str) -> list[bool]:
    """1d. Keep rows whose images exist, that have an answer and that render with the chat template."""
    keep = []
    for i in range(len(batch[cols["images"]])):
        row = {key: batch[key][i] for key in batch}
        conversation = row_to_conversation(row, fmt, cols, instruction, system_prompt)
        keep.append(conversation is not None and all(image_cell_ok(cell, root) for cell in conversation[1])
                    and conversation_segments(conversation[0], processor) is not None)
    return keep


def hash_batch(batch: dict, cols: dict) -> dict:
    """Fingerprint = text columns + image bytes (or path). Image cells are undecoded here."""
    digests = []
    for i in range(len(batch[cols["images"]])):
        digest = hashlib.sha256()
        for role, column in sorted(cols.items()):
            if role == "images":
                for cell in row_images(batch[column][i]):
                    kind, value = image_source(cell, "")
                    digest.update(value if kind == "bytes" else str(value).encode("utf-8"))
            else:
                digest.update(json.dumps(batch[column][i], ensure_ascii=False, default=str).encode("utf-8"))
        digests.append(digest.hexdigest())
    return {"_hash": digests}


def deduplicate(ds: Dataset, cols: dict, cfg: ScriptConfig) -> Dataset:
    """1d. Drop exact duplicates (same texts + same image bytes / path); keeps the first copy."""
    hashes = ds.map(hash_batch, batched=True, fn_kwargs={"cols": cols}, remove_columns=ds.column_names,
                    num_proc=cfg.preprocessing_num_workers, load_from_cache_file=not cfg.overwrite_cache,
                    desc="Hashing rows for de-duplication")["_hash"]
    seen, keep = set(), []
    for index, digest in enumerate(hashes):
        if digest not in seen:
            seen.add(digest)
            keep.append(index)
    return ds.select(keep) if len(keep) < len(ds) else ds


def split_and_subsample(train_ds: Dataset, eval_ds: Optional[Dataset], cfg: ScriptConfig):
    """1e. Hold out a validation set when the source has none, then take seeded random subsets."""
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
    """1e (streaming). The validation rows are the FIRST max_eval_samples rows of the stream (a stream
    cannot be sampled at random); they are materialized in memory, training data stays lazy."""
    n_eval = max(cfg.max_eval_samples, 0) if cfg.eval_strategy != "no" else 0
    if eval_ds is None:
        eval_rows = list(train_ds.take(n_eval))
        train_ds = train_ds.skip(n_eval)  # skip BEFORE shuffling, so train and validation never overlap
    else:
        eval_rows = list(eval_ds.take(n_eval))
    if cfg.max_train_samples > 0:
        train_ds = train_ds.take(cfg.max_train_samples)
    train_ds = train_ds.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer_size)
    # a plain list (no Arrow table): stream rows may mix value types (e.g. an answer that is a string in one
    # row and a list in another), which Arrow would reject; a DataLoader takes a list as it is
    return train_ds, (eval_rows or None)


def load_and_prepare_data(cfg: ScriptConfig, processor):
    """Runs 1a-1e and returns (train_ds, eval_ds or None, layout, columns). Rows keep their raw
    columns; images are decoded later, batch by batch, in the collator."""
    train_raw, eval_raw = load_raw_splits(cfg)  # 1a
    if cfg.eval_strategy == "no":
        eval_raw = None
    fmt, cols = resolve_layout(cfg, get_column_names(train_raw))  # 1b
    streaming = isinstance(train_raw, IterableDataset)

    root = default_image_root(cfg)  # 1c
    train_raw = keep_images_undecoded(train_raw, cols["images"])
    if eval_raw is not None:
        eval_raw = keep_images_undecoded(eval_raw, cols["images"])
    print(f"[data] images: '{cols['images']}' column, decoded per batch; relative paths are resolved from {root}")
    print(f"[data] images with more than 8 bits per value (16-bit / float TIFF or PNG, .npy) -> 0-255 by "
          f"{ImageScaling.from_config(cfg)}")

    # 1d. clean (images exist, answer present, template renders) + de-duplicate
    clean_kwargs = {"batched": True, "fn_kwargs": {"fmt": fmt, "cols": cols, "instruction": cfg.instruction,
                                                   "system_prompt": cfg.system_prompt, "processor": processor,
                                                   "root": root}}
    if streaming:
        train_raw = train_raw.filter(is_valid_batch, **clean_kwargs)  # lazy, applied while streaming
    else:
        n_rows = len(train_raw)
        train_raw = train_raw.filter(is_valid_batch, num_proc=cfg.preprocessing_num_workers,
                                     load_from_cache_file=not cfg.overwrite_cache,
                                     desc="Removing rows without image / answer", **clean_kwargs)
        print(f"[data] cleaning kept {len(train_raw):,} of {n_rows:,} train rows "
              "(dropped: missing image or file, empty answer, image/placeholder mismatch)")
        if cfg.deduplicate:
            n_before = len(train_raw)
            train_raw = deduplicate(train_raw, cols, cfg)
            print(f"[data] de-duplication removed {n_before - len(train_raw):,} exact duplicate rows")
    if eval_raw is not None:
        if isinstance(eval_raw, IterableDataset):
            eval_raw = eval_raw.filter(is_valid_batch, **clean_kwargs)
        else:
            eval_raw = eval_raw.filter(is_valid_batch, load_from_cache_file=not cfg.overwrite_cache,
                                       desc="Removing rows without image / answer", **clean_kwargs)

    # 1e. validation split + subsampling
    if streaming:
        train_ds, eval_ds = split_and_subsample_streaming(train_raw, eval_raw, cfg)
    else:
        train_ds, eval_ds = split_and_subsample(train_raw, eval_raw, cfg)
        if len(train_ds) == 0:
            raise ValueError("No usable training rows - check the layout, the columns and the chat template.")
        print(f"[data] using {len(train_ds):,} train rows and {len(eval_ds) if eval_ds is not None else 0:,} "
              "validation rows")
    if eval_ds is not None and len(eval_ds) == 0:
        eval_ds = None
    return train_ds, eval_ds, fmt, cols


@dataclass(frozen=True)
class ImageScaling:
    """How images with more than 8 bits per value become 0-255 (--image_scaling / _percentiles / _value_range)."""
    mode: str = "percentile"
    percentiles: tuple = (0.5, 99.5)
    value_range: Optional[tuple] = None

    @classmethod
    def from_config(cls, cfg: ScriptConfig) -> "ImageScaling":
        return cls(cfg.image_scaling, tuple(cfg.image_percentiles),
                   tuple(cfg.image_value_range) if cfg.image_value_range else None)

    def __str__(self) -> str:
        if self.mode == "fixed":
            return f"the fixed raw range {self.value_range[0]:g}-{self.value_range[1]:g}"
        return f"per-image percentiles {self.percentiles[0]:g}-{self.percentiles[1]:g}"


def scale_to_uint8(values: np.ndarray, scaling: ImageScaling) -> np.ndarray:
    """Values of any dtype (uint16, int32, float, big-endian, ...) -> uint8. percentile: this image's LOW / HIGH
    percentiles become 0 / 255 (a few hot or dead pixels cannot wash out the contrast); fixed: the same raw
    range for every image. Values outside the range are clipped; NaN -> 0, +inf -> 255."""
    values = np.array(values, dtype=np.float32)  # own native-endian copy, changed in place below
    if scaling.mode == "fixed":
        low, high = scaling.value_range
    else:
        flat = values.reshape(-1)
        sample = flat[:: max(1, flat.size // 1_000_000)]  # <= ~1M values: fast, practically the same percentiles
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            return np.zeros(values.shape, dtype=np.uint8)
        low, high = (float(v) for v in np.percentile(sample, scaling.percentiles))
    with np.errstate(over="ignore", invalid="ignore"):
        values -= low
        values *= 255.0 / max(high - low, 1e-12)  # a flat image (high == low) becomes black
        np.nan_to_num(values, copy=False, nan=0.0, posinf=255.0, neginf=0.0)
        np.clip(values, 0.0, 255.0, out=values)
    return np.rint(values).astype(np.uint8)


def array_to_rgb(array: np.ndarray, scaling: ImageScaling) -> Image.Image:
    """numpy image (.npy) -> RGB. Shapes (H, W), (H, W, C) or (C, H, W) with C = 1, 3 or 4. uint8 is used as
    it is, bool masks become 0 / 255, every other dtype goes through scale_to_uint8 (all channels together,
    so the colour balance is kept)."""
    shape = array.shape
    if array.ndim == 3 and array.shape[-1] not in (1, 3, 4) and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)  # channels first -> channels last
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if not (array.ndim == 2 or (array.ndim == 3 and array.shape[-1] in (3, 4))):
        raise ValueError(f"Image array of shape {shape}: expected (H, W), (H, W, C) or (C, H, W) with C = 1, 3 "
                         "or 4. Store each frame of a stack as its own image.")
    if array.dtype == np.bool_:
        array = array.astype(np.uint8) * 255
    if array.dtype != np.uint8:
        if array.ndim == 3:
            array = array[..., :3]  # the alpha channel of a high-bit-depth array is ignored
        array = scale_to_uint8(array, scaling)
    return to_rgb(Image.fromarray(np.ascontiguousarray(array)), scaling)


def to_rgb(image, scaling: ImageScaling) -> Image.Image:
    """Decoded image (PIL or numpy array) -> 8-bit RGB, the input every VLM processor expects.
    8-bit modes (L, P, RGB, CMYK, ...) are converted as they are; transparent pixels are put on white
    (a plain .convert("RGB") shows whatever colour they store, often black). PIL modes I;16 / I / F
    (16-bit / 32-bit / float TIFF and PNG files) go through scale_to_uint8: a plain .convert("RGB")
    clips every value above 255, which turns a 16-bit microscope image almost completely white."""
    if isinstance(image, np.ndarray):
        return array_to_rgb(image, scaling)
    if image.mode == "F" or image.mode.startswith("I"):
        return Image.fromarray(scale_to_uint8(np.asarray(image), scaling)).convert("RGB")
    if image.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")
    return image.convert("RGB")


def prepare_image(image, longest_edge: int, scaling: ImageScaling = ImageScaling()) -> Image.Image:
    """EXIF rotation fix, 8-bit RGB (to_rgb), shrink to longest_edge (never enlarge), enlarge only tiny images."""
    if isinstance(image, Image.Image):
        image = ImageOps.exif_transpose(image)
    image = to_rgb(image, scaling)
    scale = min(longest_edge / max(image.size), 1.0)
    if min(image.size) * scale < MIN_IMAGE_SIDE:
        scale = MIN_IMAGE_SIDE / min(image.size)
    if scale != 1.0:
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(new_size, Image.Resampling.BICUBIC)
    return image


def template_lacks_end_marker(processor) -> bool:
    """True if the chat template ends an assistant answer without any end-of-turn marker (seen with
    some LLaVA templates). Then EOS is appended to every answer, so the model learns to stop."""
    reply = "Probe reply."
    probe = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Probe?"}]},
             {"role": "assistant", "content": [{"type": "text", "text": reply}]}]
    prompt = processor.apply_chat_template(probe[:1], add_generation_prompt=True, tokenize=False)
    completion = processor.apply_chat_template(probe, tokenize=False)[len(prompt):]
    return completion.replace(reply, "").strip() == ""


@dataclass
class VisionLanguageCollator:
    """1f. Raw rows -> one model batch: decode + resize images, chat template, processor (image tokens,
    pixel tensors), label masking, truncation of the answer side only, right padding.

    labels = input_ids on assistant answers, -100 on the prompt, the image tokens and the padding.
    Image tokens are never truncated: the model checks that the number of image tokens matches the
    image features, so a cut through them would crash."""

    processor: object
    fmt: str
    cols: dict
    instruction: str
    system_prompt: Optional[str]
    max_seq_length: int
    image_longest_edge: int
    append_eos: bool
    image_root: str = "."
    pad_to_multiple_of: Optional[int] = None
    image_scaling: ImageScaling = ImageScaling()

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        tokenizer = self.processor.tokenizer
        prompts, images, tails = [], [], []
        for row in rows:
            conversation = row_to_conversation(row, self.fmt, self.cols, self.instruction, self.system_prompt)
            segments = conversation_segments(conversation[0], self.processor) if conversation else None
            if not segments:  # cannot happen after the cleaning filter; a hard error beats silent garbage
                raise ValueError("A row could not be formatted - was the cleaning step skipped?")
            prompts.append(segments[0][0])
            images.append([prepare_image(load_image(cell, self.image_root), self.image_longest_edge, self.image_scaling)
                           for cell in conversation[1]])
            tails.append(segments[1:])
        # add_special_tokens=False: the chat template already wrote BOS etc. (same as at inference)
        encoded = self.processor(text=prompts, images=images, return_tensors="pt", padding=True,
                                 add_special_tokens=False)
        token_keys = [key for key in PER_TOKEN_KEYS if key in encoded]
        unknown = [key for key, value in encoded.items() if key not in token_keys and torch.is_tensor(value)
                   and value.dim() >= 2 and tuple(value.shape[:2]) == tuple(encoded["input_ids"].shape)]
        if unknown:  # e.g. Llama 3.2 Vision's cross_attention_mask: would need extending for the answer tokens
            raise NotImplementedError(
                f"This model's processor returns per-token inputs {unknown} that this script does not handle "
                "(e.g. Llama 3.2 Vision). Use Qwen-VL, SmolVLM, Gemma 3, LLaVA, InternVL, ...")

        sequences = []
        for row in range(len(prompts)):
            keep = encoded["attention_mask"][row].bool()
            values = {key: encoded[key][row][keep].tolist() for key in token_keys}
            prompt_length = len(values["input_ids"])
            labels = [IGNORE_INDEX] * prompt_length
            for text, trainable in tails[row]:
                piece = tokenizer(text, add_special_tokens=False)["input_ids"]
                if trainable and self.append_eos:
                    piece = piece + [tokenizer.eos_token_id]
                values["input_ids"] += piece
                labels += piece if trainable else [IGNORE_INDEX] * len(piece)
                for key in token_keys:
                    if key not in ("input_ids", "attention_mask"):
                        values[key] += [0] * len(piece)  # text tokens: type 0
            limit = max(self.max_seq_length, prompt_length)  # never cut into the prompt / image tokens
            values = {key: (v[:limit] if key != "attention_mask" else None) for key, v in values.items()}
            values["labels"] = labels[:limit]
            sequences.append(values)

        max_length = max(len(seq["input_ids"]) for seq in sequences)
        if self.pad_to_multiple_of:
            max_length = math.ceil(max_length / self.pad_to_multiple_of) * self.pad_to_multiple_of
        pad_id = tokenizer.pad_token_id
        batch = {
            "input_ids": torch.full((len(sequences), max_length), pad_id, dtype=torch.long),
            "attention_mask": torch.zeros((len(sequences), max_length), dtype=torch.long),
            "labels": torch.full((len(sequences), max_length), IGNORE_INDEX, dtype=torch.long),
        }
        for key in token_keys:
            if key not in batch:
                batch[key] = torch.zeros((len(sequences), max_length), dtype=encoded[key].dtype)
        for row, seq in enumerate(sequences):
            length = len(seq["input_ids"])
            batch["attention_mask"][row, :length] = 1
            batch["labels"][row, :length] = torch.tensor(seq["labels"], dtype=torch.long)
            for key in token_keys:
                if key != "attention_mask":
                    batch[key][row, :length] = torch.tensor(seq[key], dtype=batch[key].dtype)
        for key, value in encoded.items():  # pixel_values, image_grid_thw, pixel_attention_mask, ...
            if key not in token_keys:
                batch[key] = value
        return batch


def describe_data(name: str, ds, collator: VisionLanguageCollator, show_example: bool = False) -> None:
    """Sanity check on a few collated rows: sequence lengths, image tokens, share of trained tokens."""
    if isinstance(ds, IterableDataset):
        rows = list(ds.take(8))
        print(f"[data] {name}: streaming (processed on the fly while training)")
    else:
        rows = [ds[i] for i in range(min(len(ds), 8))]
    if not rows:
        return
    batch = collator(rows)
    lengths = batch["attention_mask"].sum(1).tolist()
    prompt_lengths = [(row != IGNORE_INDEX).nonzero()[0].item() if (row != IGNORE_INDEX).any() else len(row)
                      for row in batch["labels"]]
    trained = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())
    no_answer = sum(1 for row in batch["labels"] if not (row[1:] != IGNORE_INDEX).any())
    print(f"[data] {name}: {len(ds) if not isinstance(ds, IterableDataset) else '?'} rows | first {len(rows)}: "
          f"tokens per example {min(lengths)}-{max(lengths)} (prompt + image tokens {min(prompt_lengths)}-"
          f"{max(prompt_lengths)}) | trained tokens {100 * trained / max(sum(lengths), 1):.0f}%")
    if no_answer:
        needed = max(lengths) + 128  # longest sampled prompt (with its image tokens) + room for an answer
        message = (f"{no_answer} of {len(rows)} sampled {name} rows have NO answer tokens left: the prompt and its "
                   f"image tokens already fill --max_seq_length {collator.max_seq_length}. Use --max_seq_length "
                   f"{needed} (or more), or lower --image_longest_edge.")
        if no_answer == len(rows):
            raise ValueError(message)  # training would silently learn nothing
        print(f"[data] WARNING: {message}")
    if show_example:
        tokenizer = collator.processor.tokenizer
        ids, labels = batch["input_ids"][0].tolist(), batch["labels"][0].tolist()
        length = lengths[0]
        pieces, start = [], 0
        for end in range(1, length + 1):
            if end == length or (labels[end] != IGNORE_INDEX) != (labels[start] != IGNORE_INDEX):
                text = tokenizer.decode(ids[start:end])
                pieces.append(f"[[{text}]]" if labels[start] != IGNORE_INDEX else text)
                start = end
        rendered = re.sub(r"((<[^<>\s]{1,40}>)\2{4,})", lambda m: f"{m.group(2)} x{len(m.group(1)) // len(m.group(2))}",
                          "".join(pieces))
        print("[data] training example 0 - the loss is computed only on the [[...]] part:")
        print(rendered[:900] + (" ..." if len(rendered) > 900 else ""))


def build_dataloaders(train_ds, eval_ds, collator: VisionLanguageCollator, cfg: ScriptConfig):
    """1f. DataLoaders. Worker processes decode / resize images in parallel with training."""
    workers = cfg.dataloader_num_workers
    if workers is None:
        workers = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
    if workers > 0:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoid the fork warning / deadlock
    loader_kwargs = {"collate_fn": collator, "num_workers": workers, "pin_memory": torch.cuda.is_available(),
                     "persistent_workers": workers > 0}
    if isinstance(train_ds, IterableDataset):
        train_loader = DataLoader(train_ds, batch_size=cfg.per_device_train_batch_size, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.per_device_train_batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(cfg.seed), **loader_kwargs)
    eval_loader = None
    if eval_ds is not None:
        eval_loader = DataLoader(eval_ds, batch_size=cfg.per_device_eval_batch_size, shuffle=False, **loader_kwargs)
    return train_loader, eval_loader


#=====================================================================================
#2. Load processor + base VLM & setup the LoRA configuration
#=====================================================================================


def load_processor(source: str, cfg: ScriptConfig, revision: Optional[str] = None):
    """Processor = tokenizer + image processor + chat template, with the image resolution capped."""
    processor = AutoProcessor.from_pretrained(source, revision=revision, trust_remote_code=cfg.trust_remote_code)
    if getattr(processor, "chat_template", None) is None:
        raise ValueError(f"{source} has no chat template - use an Instruct / chat VLM "
                         "(e.g. HuggingFaceTB/SmolVLM-256M-Instruct, Qwen/Qwen2.5-VL-3B-Instruct).")
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # SmolVLM / Idefics3 resize every image UP to size["longest_edge"] (default 2048 = up to 16 tiles of
    # 512 px). Cap it at the largest multiple of the tile size <= image_longest_edge (documented knob).
    image_processor = getattr(processor, "image_processor", None)
    tile = getattr(image_processor, "max_image_size", None)
    if isinstance(tile, dict) and tile.get("longest_edge"):
        tile_edge = tile["longest_edge"]
        image_processor.size = {"longest_edge": max(tile_edge, cfg.image_longest_edge // tile_edge * tile_edge)}
    return processor


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
    load_kwargs = {"revision": cfg.model_revision, "trust_remote_code": cfg.trust_remote_code}
    if cfg.attn_implementation:
        load_kwargs["attn_implementation"] = cfg.attn_implementation
    if cfg.use_qlora:
        # QLoRA: the frozen base is stored in 4-bit NF4 (about 4x less memory); LoRA weights stay in
        # 16/32-bit and are the only thing trained. prepare_model_for_kbit_training freezes the base,
        # upcasts norms / embeddings to fp32 and (optionally) turns on gradient checkpointing.
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_name, dtype=model_dtype, **qlora_load_kwargs(device, model_dtype), **load_kwargs)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model = AutoModelForImageTextToText.from_pretrained(cfg.model_name, dtype=model_dtype, **load_kwargs)
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for config in (model.config, getattr(model.config, "text_config", None)):
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False  # the generation KV-cache is useless in training and costs memory
    return model


def language_model_target_regex(model) -> str:
    """Regex for PEFT that matches every linear layer of the LANGUAGE model (attention + MLP) and none of
    the vision encoder / projector / lm_head. Names differ per architecture (Qwen: model.language_model,
    SmolVLM: model.text_model, ...), so they are discovered from the loaded model."""
    leaves = sorted({name.rsplit(".", 1)[-1] for name, module in model.named_modules()
                     if isinstance(module, torch.nn.Linear) and "lm_head" not in name
                     and not any(keyword in name.lower() for keyword in VISION_MODULE_KEYWORDS)})
    if not leaves:
        raise ValueError("No language-model linear layers found - pass --lora_target_modules explicitly.")
    excluded = "|".join(VISION_MODULE_KEYWORDS)
    return rf"^(?!.*({excluded})).*\.({'|'.join(map(re.escape, leaves))})$"


def apply_lora(model, cfg: ScriptConfig) -> PeftModel:
    # r = rank (capacity), lora_alpha = scale (update x alpha / r), lora_dropout = regularization.
    # target modules: default = all linear layers of the language model (q/k/v/o + MLP) - the usual
    # VLM recipe. "all-linear" also adapts the vision encoder (helps for very new kinds of images,
    # e.g. medical scans, at the cost of more memory).
    if cfg.lora_target_modules is None:
        target_modules = language_model_target_regex(model)
    elif cfg.lora_target_modules == ["all-linear"]:
        target_modules = "all-linear"
    else:
        target_modules = cfg.lora_target_modules
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,  # VLMs generate text, so they use the causal-LM PEFT wrapper
    )
    model = get_peft_model(model, lora_config, revision=cfg.model_revision)
    model.print_trainable_parameters()
    adapted = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    vision = sum(any(k in name.lower() for k in VISION_MODULE_KEYWORDS) for name in adapted)
    print(f"[model] LoRA on {len(adapted)} layers ({len(adapted) - vision} language, {vision} vision) | "
          f"layer types: {sorted({name.rsplit('.', 1)[-1] for name in adapted})}")
    return model


#=====================================================================================
#3. Training Section (same engine as best-LORA-Example.py)
#=====================================================================================


def get_device() -> torch.device:
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
        print("[setup] WARNING: bf16 on a CPU is only fast with native bf16 units (AMX / AVX512-BF16). On laptop "
              "CPUs (AVX2) it is emulated and can be 100x slower - use --mixed_precision no.")
    amp_dtype = {"no": None, "bf16": torch.bfloat16, "fp16": torch.float16}[mode]
    if cfg.model_dtype != "auto":
        return getattr(torch, cfg.model_dtype), amp_dtype
    return (amp_dtype if device.type == "cuda" and amp_dtype is not None else torch.float32), amp_dtype


def autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def count_target_tokens(labels: torch.Tensor) -> int:
    """Positions that contribute to the loss (the model shifts labels left by one internally)."""
    return int((labels[:, 1:] != IGNORE_INDEX).sum())


def to_device(batch: dict, device: torch.device) -> dict:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def supports_logits_to_keep(model) -> bool:
    base = model.get_base_model() if isinstance(model, PeftModel) else model
    return "logits_to_keep" in inspect.signature(base.forward).parameters


def compute_loss(model, batch: dict, device: torch.device, amp_dtype: Optional[torch.dtype],
                 answer_logits_only: bool) -> torch.Tensor:
    """Mean next-token cross-entropy over the answer tokens of the batch.

    Memory: the vocabulary logits (batch x positions x vocab, e.g. 4 x 761 x 151,936 floats = 1.9 GB for
    Qwen) are the biggest tensor of a VLM step, yet most positions are prompt / image tokens that are never
    trained. With answer_logits_only the model computes logits ONLY at positions that predict an answer
    token (logits_to_keep) - same loss value, a fraction of the memory. Position p predicts token p + 1."""
    batch = to_device(batch, device)
    labels = batch.pop("labels")
    if not answer_logits_only:
        with autocast_context(device, amp_dtype):
            return model(**batch, labels=labels).loss
    targets = labels[:, 1:]
    positions = (targets != IGNORE_INDEX).any(dim=0).nonzero().squeeze(-1)
    with autocast_context(device, amp_dtype):
        logits = model(**batch, logits_to_keep=positions).logits  # (batch, kept positions, vocab)
    return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                             targets[:, positions].reshape(-1), ignore_index=IGNORE_INDEX)


def iterate_in_groups(loader: DataLoader, group_size: int):
    """Yield lists of `group_size` consecutive batches = one optimizer step (gradient accumulation)."""
    group = []
    for batch in loader:
        group.append(batch)
        if len(group) == group_size:
            yield group
            group = []
    if group:
        yield group


def build_optimizer(model: PeftModel, cfg: ScriptConfig) -> AdamW:
    """AdamW on the trainable (LoRA) tensors only; no weight decay on 1-D tensors."""
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    param_groups = [{"params": decay, "weight_decay": cfg.weight_decay}]
    if no_decay:
        param_groups.append({"params": no_decay, "weight_decay": 0.0})
    return AdamW(param_groups, lr=cfg.learning_rate)


@torch.no_grad()
def evaluate(model: PeftModel, eval_loader: DataLoader, device: torch.device,
             amp_dtype: Optional[torch.dtype], answer_logits_only: bool) -> dict:
    """Validation loss = mean cross-entropy over every answer token; perplexity = exp(loss)."""
    model.eval()
    loss_sum, n_targets = 0.0, 0
    for batch in eval_loader:
        batch_targets = count_target_tokens(batch["labels"])
        if batch_targets == 0:
            continue
        loss = compute_loss(model, batch, device, amp_dtype, answer_logits_only)
        loss_sum += loss.item() * batch_targets
        n_targets += batch_targets
    model.train()
    if n_targets == 0:
        raise ValueError("The validation set has no answer tokens (all cut by --max_seq_length?) - raise "
                         "--max_seq_length or lower --image_longest_edge.")
    eval_loss = loss_sum / n_targets
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
    answer_logits_only = supports_logits_to_keep(model)
    if not answer_logits_only:
        print("[train] this model has no logits_to_keep: full-sequence logits are computed (more memory)")
    num_epochs = cfg.num_epochs
    streaming = isinstance(train_loader.dataset, IterableDataset)
    if streaming:
        num_epochs = sys.maxsize  # a stream has no length; passes repeat until max_steps is reached
    steps_per_epoch = None if streaming else math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    if steps_per_epoch == 0:
        raise ValueError("The training DataLoader is empty. Check the data source and preprocessing filters.")
    total_steps = cfg.max_steps if cfg.max_steps > 0 else num_epochs * steps_per_epoch
    if cfg.max_steps > 0 and not streaming:
        num_epochs = math.ceil(total_steps / steps_per_epoch)
    scheduler = get_scheduler(cfg.lr_scheduler_type, optimizer=optimizer,
                              num_warmup_steps=math.ceil(cfg.warmup_ratio * total_steps),
                              num_training_steps=total_steps)
    use_grad_scaler = amp_dtype == torch.float16  # fp16 needs loss scaling; bf16 does not
    scaler = torch.amp.GradScaler(device.type, enabled=use_grad_scaler)

    if not cfg.use_qlora:
        model.to(device)  # a 4-bit QLoRA model is already on the GPU (placed while loading)
    if streaming:
        print(f"Training for {total_steps} optimizer steps on {device} (streaming)")
    else:
        print(f"Training for {num_epochs} epoch(s) on {device} | {steps_per_epoch} optimizer steps per epoch, "
              f"{total_steps} in total | effective batch "
              f"{cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps} examples")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    clip_norm = cfg.max_grad_norm if cfg.max_grad_norm > 0 else float("inf")
    best_dir = os.path.join(cfg.output_dir, "checkpoint-best")
    do_eval = eval_loader is not None and cfg.eval_strategy != "no"
    stopper = EarlyStopping(cfg.early_stopping_patience, cfg.early_stopping_threshold)
    history = []
    global_step, epoch, early_stopped, interrupted = 0, 0, False, False

    def evaluate_and_checkpoint(current_epoch: int) -> bool:
        metrics = evaluate(model, eval_loader, device, amp_dtype, answer_logits_only)
        is_best = stopper.update(metrics["eval_loss"], global_step)
        history.append({"step": global_step, "epoch": current_epoch + 1, **metrics})
        if is_best:
            model.save_pretrained(best_dir)  # adapter weights only
        status = "new best -> saved checkpoint-best" if is_best else "no improvement"
        if stopper.bad_evals and cfg.early_stopping_patience > 0:
            status += f" | early-stopping patience {stopper.bad_evals}/{cfg.early_stopping_patience}"
        print(f"[eval] step {global_step} | loss {metrics['eval_loss']:.4f} | "
              f"perplexity {metrics['eval_perplexity']:.2f} | {status}")
        return stopper.should_stop

    if do_eval:
        metrics = evaluate(model, eval_loader, device, amp_dtype, answer_logits_only)
        history.append({"step": 0, "epoch": 0, **metrics})
        print(f"[eval] before training | loss {metrics['eval_loss']:.4f} | perplexity "
              f"{metrics['eval_perplexity']:.2f} (LoRA B is still zero: this is the base model)")

    window_loss, window_steps, window_tokens, window_start = 0.0, 0, 0, time.perf_counter()
    model.train()
    try:
        for epoch in range(num_epochs):
            if streaming:
                train_loader.dataset.set_epoch(epoch)
            total_loss, epoch_steps = 0.0, 0
            # One optimizer step = gradient_accumulation_steps micro-batches; each micro-batch loss is
            # weighted by its share of the step's answer tokens -> exact token-level mean loss.
            for micro_batches in iterate_in_groups(train_loader, cfg.gradient_accumulation_steps):
                n_targets = [count_target_tokens(batch["labels"]) for batch in micro_batches]
                step_targets = sum(n_targets)
                if step_targets == 0:
                    continue
                step_loss = 0.0
                for batch, batch_targets in zip(micro_batches, n_targets):
                    if batch_targets == 0:
                        continue
                    window_tokens += int(batch["attention_mask"].sum())
                    loss = compute_loss(model, batch, device, amp_dtype, answer_logits_only)
                    loss = loss * (batch_targets / step_targets)
                    scaler.scale(loss).backward()
                    step_loss += loss.item()
                if not use_grad_scaler and not math.isfinite(step_loss):
                    raise RuntimeError(f"Loss became {step_loss} at step {global_step + 1}: lower --learning_rate")
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, clip_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if not use_grad_scaler or scaler.get_scale() >= scale_before:
                    scheduler.step()
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
                    window_start += time.perf_counter() - eval_start
                if early_stopped or global_step >= total_steps:
                    break
            if epoch_steps == 0:
                raise RuntimeError("No training batch with answer tokens was produced in this pass. "
                                   "The dataset may be empty or all examples were filtered / masked. "
                                   "Check the data source, --max_seq_length and --image_longest_edge.")
            print(f"Epoch {epoch + 1}/{'-' if streaming else num_epochs}, Loss: {total_loss / max(epoch_steps, 1):.4f}")
            if do_eval and cfg.eval_strategy == "epoch" and not early_stopped:
                eval_start = time.perf_counter()
                early_stopped = evaluate_and_checkpoint(epoch)
                window_start += time.perf_counter() - eval_start
            if early_stopped:
                print(f"[train] early stopping: {cfg.early_stopping_patience} evaluation(s) in a row did not beat "
                      f"the best loss by more than {cfg.early_stopping_threshold}")
            if early_stopped or global_step >= total_steps:
                break
        if do_eval and not early_stopped and history[-1]["step"] != global_step:
            evaluate_and_checkpoint(epoch)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[train] interrupted with Ctrl+C - keeping the best (or the latest) adapter")

    if global_step == 0 and not interrupted:
        raise RuntimeError("No optimizer step was taken: no training batch had answer tokens. Raise "
                           "--max_seq_length or lower --image_longest_edge.")
    if stopper.best_step is not None:  # restore the adapter with the lowest validation loss
        set_peft_model_state_dict(model, load_peft_weights(best_dir, device=str(trainable_params[0].device)))
        print(f"[train] restored best adapter: step {stopper.best_step} | eval loss {stopper.best:.4f}")
    return {"optimizer_steps": global_step, "best_step": stopper.best_step,
            "best_eval_loss": stopper.best if stopper.best_step is not None else None, "history": history}


#=====================================================================================
# Saving the adapter
#=====================================================================================


def package_versions() -> dict:
    versions = {}
    for package in ("torch", "transformers", "peft", "datasets", "accelerate", "pillow", "bitsandbytes"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def save_adapter(model: PeftModel, processor, cfg: ScriptConfig, summary: dict) -> None:
    """Adapter (a few MB) + processor (tokenizer, image settings, chat template) + run_summary.json.
    adapter_config.json records the base model id, used to rebuild base + adapter later."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    model.save_pretrained(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    with open(os.path.join(cfg.output_dir, "run_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"[save] adapter + processor + run_summary.json -> {os.path.abspath(cfg.output_dir)}")


#=====================================================================================
# Loading the model (base + adapter)
#=====================================================================================


def load_finetuned_model(cfg: ScriptConfig, model_dtype: torch.dtype):
    """AutoPeftModelForCausalLM only knows text models, so a VLM is rebuilt explicitly:
    base id from adapter_config.json -> AutoModelForImageTextToText -> PeftModel.from_pretrained."""
    peft_config = PeftConfig.from_pretrained(cfg.output_dir)
    load_kwargs = {"device_map": "auto", "dtype": model_dtype, "trust_remote_code": cfg.trust_remote_code}
    if cfg.use_qlora and not cfg.save_merged_model:
        load_kwargs.update(qlora_load_kwargs(get_device(), model_dtype))
    elif cfg.use_qlora:
        print("[load] merged export requested: reloading an unquantized base (needs more memory)")
    base = AutoModelForImageTextToText.from_pretrained(
        peft_config.base_model_name_or_path, revision=peft_config.revision, **load_kwargs)
    model = PeftModel.from_pretrained(base, cfg.output_dir)
    processor = AutoProcessor.from_pretrained(cfg.output_dir, trust_remote_code=cfg.trust_remote_code)
    print(f"[load] base {peft_config.base_model_name_or_path} + adapter {cfg.output_dir}")
    return model, processor


#=====================================================================================
# Test the model on images
#=====================================================================================


def generate_answer(model, processor, messages: list[dict], images: list, cfg: ScriptConfig) -> str:
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    scaling = ImageScaling.from_config(cfg)
    inputs = processor(text=[prompt], images=[[prepare_image(img, cfg.image_longest_edge, scaling) for img in images]],
                       return_tensors="pt", add_special_tokens=False)
    # pixel tensors come out as fp32; a bf16 / fp16 model (GPU) needs them in its own dtype
    inputs = {key: value.to(model.device, dtype=model.dtype) if torch.is_floating_point(value)
              else value.to(model.device) for key, value in inputs.items()}
    generation_kwargs = {"max_new_tokens": cfg.max_new_tokens, "do_sample": cfg.do_sample,
                         "repetition_penalty": cfg.repetition_penalty,
                         "pad_token_id": processor.tokenizer.pad_token_id}
    if cfg.do_sample:
        generation_kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p)
    set_seed(cfg.seed)
    with torch.no_grad():
        output = model.generate(**inputs, **generation_kwargs)
    return processor.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def test_model(model, processor, cfg: ScriptConfig, fmt: str, cols: dict, eval_rows: list[dict], root: str) -> None:
    """Answer = text generated after the prompt. For validation rows the reference answer is shown,
    so you can judge the adapter (and the base model with the adapter switched off) yourself."""
    model.eval()
    cases = []
    for row in eval_rows:
        conversation = row_to_conversation(row, fmt, cols, cfg.instruction, cfg.system_prompt)
        if conversation is None:
            continue
        messages, images = conversation[0], [load_image(cell, root) for cell in conversation[1]]
        # first answer that comes after the last image, so the prompt contains every image
        last_image = max(i for i, m in enumerate(messages) if any(p["type"] == "image" for p in m["content"]))
        answer = next(i for i, m in enumerate(messages) if m["role"] == "assistant" and i > last_image)
        reference = " ".join(p["text"] for p in messages[answer]["content"] if p["type"] == "text")
        cases.append((messages[:answer], images, reference))
    if cfg.test_image:
        question = cfg.test_prompt or cfg.instruction
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        if cfg.system_prompt:
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": cfg.system_prompt}]})
        cases.append((messages, [load_image(cfg.test_image, os.getcwd())], None))
    for number, (messages, images, reference) in enumerate(cases, start=1):
        question = " ".join(p["text"] for m in messages if m["role"] == "user" for p in m["content"]
                            if p["type"] == "text")
        sizes = [tuple(img.shape) if isinstance(img, np.ndarray) else img.size for img in images]
        print(f"\n[test {number}] {len(images)} image(s) {sizes} | prompt: {question[:200]}")
        if reference is not None:
            print(f"--- reference answer ---\n{reference[:600]}")
        print(f"--- base model + LoRA adapter ---\n{generate_answer(model, processor, messages, images, cfg)[:600]}")
        if cfg.compare_with_base:
            with model.disable_adapter():
                base_answer = generate_answer(model, processor, messages, images, cfg)
            print(f"--- base model only (adapter disabled) ---\n{base_answer[:600]}")


def merge_and_save(model: PeftModel, processor, cfg: ScriptConfig) -> None:
    """Fold B @ A * (alpha / r) into the base weights: a plain transformers VLM, no PEFT at inference."""
    merged_dir = os.path.normpath(cfg.output_dir) + "-merged"
    model.merge_and_unload().save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)
    print(f"[merge] standalone merged model -> {os.path.abspath(merged_dir)} "
          "(load with AutoModelForImageTextToText.from_pretrained)")


def main() -> None:
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    cfg = parse_config()
    set_seed(cfg.seed)
    device = get_device()
    model_dtype, amp_dtype = resolve_precision(cfg, device)
    print(f"[setup] model: {cfg.model_name} | device: {device} | base weights: {model_dtype} | "
          f"mixed precision: {amp_dtype or 'off'} | QLoRA: {cfg.use_qlora} | images <= {cfg.image_longest_edge} px")

    if cfg.do_train:
        processor = load_processor(cfg.model_name, cfg, revision=cfg.model_revision)
        train_ds, eval_ds, fmt, cols = load_and_prepare_data(cfg, processor)  # section 1
        collator = VisionLanguageCollator(
            processor=processor, fmt=fmt, cols=cols, instruction=cfg.instruction, system_prompt=cfg.system_prompt,
            max_seq_length=cfg.max_seq_length, image_longest_edge=cfg.image_longest_edge,
            append_eos=template_lacks_end_marker(processor), image_root=default_image_root(cfg),
            pad_to_multiple_of=8 if device.type == "cuda" else None, image_scaling=ImageScaling.from_config(cfg))
        if collator.append_eos:
            print("[data] this chat template ends answers without an end marker -> EOS appended to every answer")
        describe_data("train", train_ds, collator, show_example=True)
        if eval_ds is not None:
            describe_data("validation", eval_ds, collator)
        train_loader, eval_loader = build_dataloaders(train_ds, eval_ds, collator, cfg)
        model = apply_lora(load_base_model(cfg, device, model_dtype), cfg)  # section 2
        results = train(model, train_loader, eval_loader, cfg, device, amp_dtype)  # section 3
        summary = {
            "base_model": cfg.model_name, "dataset_format": fmt, "columns": cols,
            "train_rows": None if isinstance(train_ds, IterableDataset) else len(train_ds),
            "eval_rows": 0 if eval_ds is None else len(eval_ds), "device": str(device),
            "base_weights_dtype": str(model_dtype), "mixed_precision": str(amp_dtype) if amp_dtype else "off",
            **results, "config": asdict(cfg), "versions": package_versions(),
        }
        save_adapter(model, processor, cfg, summary)
        test_rows = [] if eval_ds is None else [eval_ds[i] for i in range(min(cfg.num_test_samples, len(eval_ds)))]
        del model, train_loader, eval_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        if not os.path.isfile(os.path.join(cfg.output_dir, "adapter_config.json")):
            raise SystemExit(f"No adapter in '{cfg.output_dir}'. Train one first (--do_train true).")
        with open(os.path.join(cfg.output_dir, "run_summary.json"), encoding="utf-8") as file:
            saved = json.load(file)
        fmt, cols = saved["dataset_format"], saved["columns"]
        trained = saved.get("config", {})
        changed = [f"--{key} {trained[key]} (now {getattr(cfg, key)})" for key in IMAGE_SETTINGS
                   if key in trained and trained[key] != getattr(cfg, key)]
        if changed:
            print("[load] WARNING: this adapter was trained with " + ", ".join(changed) + " - pass the same "
                  "image settings, so the model sees images the way it learned them")
        test_rows = []  # inference-only: use --test_image / --test_prompt
        if not cfg.test_image:
            raise SystemExit("--do_train false needs --test_image (an image file or URL) to test the adapter.")

    model, processor = load_finetuned_model(cfg, model_dtype)
    test_model(model, processor, cfg, fmt, cols, test_rows, default_image_root(cfg))
    if cfg.save_merged_model:
        merge_and_save(model, processor, cfg)


if __name__ == "__main__":
    main()
