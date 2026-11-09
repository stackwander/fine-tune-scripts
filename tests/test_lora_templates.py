"""Offline regression checks. Run: python -B -m unittest discover -s tests -v"""

import contextlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
from datasets import Dataset
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    CLIPImageProcessor,
    GPT2Config,
    GPT2LMHeadModel,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    PreTrainedTokenizerFast,
)


ROOT = Path(__file__).resolve().parents[1]


def load_template(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TEMPLATES = {
    "text": load_template("text_lora_template", "best-LORA-Example.py"),
    "vlm": load_template("vlm_lora_template", "best-VLM-LoRA-Example.py"),
}


def tiny_tokenizer():
    words = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "<image>", "user", "assistant",
             "red", "blue", "color", "What", "Name", "a", "animal", "cat", "dog"]
    words += [f"unused_{i}" for i in range(32 - len(words))]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", bos_token="[BOS]",
        eos_token="[EOS]", additional_special_tokens=["<image>"], model_max_length=64)


class TemplateRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.temp = tempfile.TemporaryDirectory(prefix="lora-regression-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        tokenizer = tiny_tokenizer()
        cls.text_base = cls.root / "text-base"
        GPT2LMHeadModel(GPT2Config(
            vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=64,
            bos_token_id=2, eos_token_id=3, pad_token_id=1,
            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).save_pretrained(cls.text_base)
        tokenizer.save_pretrained(cls.text_base)

        cls.vlm_base = cls.root / "vlm-base"
        LlavaForConditionalGeneration(LlavaConfig(
            text_config={"model_type": "llama", "vocab_size": 32, "hidden_size": 16,
                         "intermediate_size": 32, "num_hidden_layers": 1,
                         "num_attention_heads": 2, "num_key_value_heads": 2,
                         "max_position_embeddings": 64, "bos_token_id": 2,
                         "eos_token_id": 3, "pad_token_id": 1},
            vision_config={"model_type": "clip_vision_model", "hidden_size": 16,
                           "intermediate_size": 32, "num_hidden_layers": 1,
                           "num_attention_heads": 2, "image_size": 32, "patch_size": 16},
            image_token_index=4, image_seq_length=4, vision_feature_layer=-1,
            vision_feature_select_strategy="default")).save_pretrained(cls.vlm_base)
        chat_template = (
            "{% for message in messages %}{{ message['role'] + ': ' }}"
            "{% for part in message['content'] %}"
            "{% if part['type'] == 'image' %}{{ '<image> ' }}"
            "{% else %}{{ part['text'] + ' ' }}{% endif %}{% endfor %}"
            "{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}")
        LlavaProcessor(
            image_processor=CLIPImageProcessor(
                size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}),
            tokenizer=tokenizer, patch_size=16, num_additional_image_tokens=1,
            vision_feature_select_strategy="default", chat_template=chat_template,
        ).save_pretrained(cls.vlm_base)

    def config(self, name, **kwargs):
        defaults = dict(
            output_dir=str(self.root / name), max_steps=2, gradient_accumulation_steps=1,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            warmup_ratio=0, logging_steps=1, lora_r=2, lora_alpha=4, lora_dropout=0,
            dataloader_num_workers=0, eval_strategy="no", max_seq_length=64)
        defaults.update(kwargs)
        return TEMPLATES[name].ScriptConfig(**defaults)

    def test_empty_finite_loader_fails_before_step_count_division(self):
        for name, module in TEMPLATES.items():
            with self.subTest(template=name), self.assertRaisesRegex(ValueError, "DataLoader is empty"):
                module.train(torch.nn.Linear(2, 2), DataLoader([], batch_size=1), None,
                             self.config(name, max_steps=1), torch.device("cpu"), None)

    def test_empty_and_fully_masked_streams_fail_after_one_pass(self):
        for name, module in TEMPLATES.items():
            for masked in (False, True):
                with self.subTest(template=name, masked=masked):
                    stream = Dataset.from_list([
                        {"input_ids": [5, 7, 3], "labels": [-100, -100, -100]},
                    ]).to_iterable_dataset()
                    if not masked:
                        stream = stream.filter(lambda _row: False)
                    original_set_epoch = stream.set_epoch

                    def first_pass_only(epoch):
                        self.assertEqual(epoch, 0, "An unusable stream was retried instead of rejected")
                        original_set_epoch(epoch)

                    loader = DataLoader(stream, batch_size=1, collate_fn=TEMPLATES["text"].CausalLMCollator(1))
                    with patch.object(stream, "set_epoch", side_effect=first_pass_only) as set_epoch:
                        with self.assertRaisesRegex(RuntimeError, "No training batch"):
                            module.train(torch.nn.Linear(2, 2), loader, None,
                                         self.config(name, streaming=True, max_steps=1), torch.device("cpu"), None)
                        set_epoch.assert_called_once_with(0)

    def test_nonempty_stream_repeats_until_max_steps(self):
        for name, module in TEMPLATES.items():
            with self.subTest(template=name):
                stream = Dataset.from_list([{"input_ids": [5, 7, 3], "labels": [-100, 7, 3]}]).to_iterable_dataset()
                loader = DataLoader(stream, batch_size=1, collate_fn=TEMPLATES["text"].CausalLMCollator(1))
                model = GPT2LMHeadModel(GPT2Config(
                    vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=64,
                    bos_token_id=2, eos_token_id=3, pad_token_id=1))
                with patch.object(stream, "set_epoch", wraps=stream.set_epoch) as set_epoch:
                    result = module.train(model, loader, None, self.config(name, streaming=True, max_steps=3),
                                          torch.device("cpu"), None)
                    self.assertEqual(result["optimizer_steps"], 3)
                    self.assertEqual([call.args[0] for call in set_epoch.call_args_list], [0, 1, 2])

    def test_interrupt_before_first_step_preserves_partial_run(self):
        def interrupt(_rows):
            raise KeyboardInterrupt

        for name, module in TEMPLATES.items():
            with self.subTest(template=name):
                loader = DataLoader([{}], batch_size=1, collate_fn=interrupt)
                result = module.train(torch.nn.Linear(2, 2), loader, None, self.config(name, max_steps=1),
                                      torch.device("cpu"), None)
                self.assertEqual(result["optimizer_steps"], 0)

    def test_inference_respects_qlora_and_explicit_merged_export(self):
        for name, module in TEMPLATES.items():
            for use_qlora, merge in ((False, False), (True, False), (True, True)):
                with self.subTest(template=name, qlora=use_qlora, merge=merge), contextlib.ExitStack() as stack:
                    cfg = self.config(name, use_qlora=use_qlora, save_merged_model=merge)
                    quantization = object()
                    qloader = stack.enter_context(patch.object(module, "qlora_load_kwargs", return_value={
                        "quantization_config": quantization, "device_map": {"": 2}}))
                    stack.enter_context(patch.object(module, "get_device", return_value=torch.device("cuda:2")))
                    if name == "text":
                        loader = stack.enter_context(patch.object(module.AutoPeftModelForCausalLM, "from_pretrained"))
                    else:
                        stack.enter_context(patch.object(module.PeftConfig, "from_pretrained", return_value=
                            SimpleNamespace(base_model_name_or_path="saved-base", revision="pinned-revision")))
                        loader = stack.enter_context(patch.object(module.AutoModelForImageTextToText, "from_pretrained"))
                        stack.enter_context(patch.object(module.PeftModel, "from_pretrained"))
                        stack.enter_context(patch.object(module.AutoProcessor, "from_pretrained"))
                    module.load_finetuned_model(cfg, torch.bfloat16)
                    if use_qlora and not merge:
                        qloader.assert_called_once_with(torch.device("cuda:2"), torch.bfloat16)
                        self.assertIs(loader.call_args.kwargs["quantization_config"], quantization)
                        self.assertEqual(loader.call_args.kwargs["device_map"], {"": 2})
                    else:
                        qloader.assert_not_called()
                        self.assertNotIn("quantization_config", loader.call_args.kwargs)
                        self.assertEqual(loader.call_args.kwargs["device_map"], "auto")
                    self.assertEqual(loader.call_args.kwargs["dtype"], torch.bfloat16)
                    if name == "vlm":
                        self.assertEqual(loader.call_args.args[0], "saved-base")
                        self.assertEqual(loader.call_args.kwargs["revision"], "pinned-revision")

    @unittest.skipUnless(importlib.util.find_spec("bitsandbytes"), "bitsandbytes is optional for LoRA")
    def test_training_and_inference_share_nf4_configuration(self):
        for name, module in TEMPLATES.items():
            with self.subTest(template=name), patch.object(torch.cuda, "current_device", return_value=2):
                options = module.qlora_load_kwargs(torch.device("cuda"), torch.bfloat16)
                quantization = options["quantization_config"]
                self.assertTrue(quantization.load_in_4bit)
                self.assertEqual(quantization.bnb_4bit_quant_type, "nf4")
                self.assertTrue(quantization.bnb_4bit_use_double_quant)
                self.assertEqual(quantization.bnb_4bit_compute_dtype, torch.bfloat16)
                self.assertEqual(options["device_map"], {"": 2})
                cfg = self.config(name, use_qlora=True, gradient_checkpointing=True)
                base = SimpleNamespace(config=SimpleNamespace(use_cache=True))
                auto = module.AutoModelForCausalLM if name == "text" else module.AutoModelForImageTextToText
                with patch.object(auto, "from_pretrained", return_value=base) as loader:
                    with patch.object(module, "prepare_model_for_kbit_training", return_value=base) as prepare:
                        module.load_base_model(cfg, torch.device("cuda"), torch.bfloat16)
                        self.assertEqual(loader.call_args.kwargs["quantization_config"].to_dict(),
                                         quantization.to_dict())
                        self.assertEqual(loader.call_args.kwargs["device_map"], options["device_map"])
                        self.assertTrue(prepare.call_args.kwargs["use_gradient_checkpointing"])
                        self.assertFalse(base.config.use_cache)

    def test_qlora_rejects_unsupported_device_and_missing_package(self):
        for name, module in TEMPLATES.items():
            with self.subTest(template=name, device="cpu"), self.assertRaisesRegex(SystemExit, "CUDA"):
                module.qlora_load_kwargs(torch.device("cpu"), torch.float32)
            with self.subTest(template=name, missing="bitsandbytes"), patch.object(module, "find_spec", return_value=None):
                with self.assertRaisesRegex(SystemExit, "pip install bitsandbytes"):
                    module.qlora_load_kwargs(torch.device("cuda"), torch.bfloat16)

    def model_loss(self, name, model, batch):
        model.eval()
        with torch.no_grad():
            if name == "vlm":
                return TEMPLATES[name].compute_loss(model, batch, model.device, None, True).cpu()
            return model(**{key: value.to(model.device) for key, value in batch.items()}).loss.cpu()

    def test_lora_training_reload_and_unquantized_merged_export(self):
        for name, module in TEMPLATES.items():
            for checkpointing in (False, True):
                with self.subTest(template=name, checkpointing=checkpointing):
                    cfg = self.config(name, model_name=str(self.text_base if name == "text" else self.vlm_base),
                                      output_dir=str(self.root / f"{name}-{checkpointing}"),
                                      gradient_checkpointing=checkpointing, eval_strategy="epoch")
                    if name == "text":
                        processor = module.load_tokenizer(cfg)
                        data = module.preprocess(Dataset.from_list([
                            {"instruction": "Name a color", "output": "red"},
                            {"instruction": "Name a animal", "output": "cat"},
                        ]), "instruction", {"instruction": "instruction", "response": "output", "input": None},
                            processor, cfg, "regression")
                        train_loader, eval_loader = module.build_dataloaders(data, data, processor, cfg, torch.device("cpu"))
                    else:
                        processor = module.load_processor(cfg.model_name, cfg)
                        data = [{"image": Image.new("RGB", (32, 32), color),
                                 "question": "What color", "answer": color} for color in ("red", "blue")]
                        collator = module.VisionLanguageCollator(
                            processor, "vqa", {"images": "image", "question": "question", "answer": "answer"},
                            "What color", None, 64, 32, module.template_lacks_end_marker(processor))
                        train_loader, eval_loader = module.build_dataloaders(data, data, collator, cfg)
                    model = module.apply_lora(module.load_base_model(cfg, torch.device("cpu"), torch.float32), cfg)
                    before = {key: value.detach().clone() for key, value in model.named_parameters()}
                    report = module.train(model, train_loader, eval_loader, cfg, torch.device("cpu"), None)
                    self.assertEqual(report["optimizer_steps"], 2)
                    changed = [key for key, value in model.named_parameters() if not torch.equal(value, before[key])]
                    self.assertTrue(changed)
                    self.assertTrue(all("lora_" in key for key in changed), changed)
                    module.save_adapter(model, processor, cfg, {})
                    batch = next(iter(eval_loader))
                    expected = self.model_loss(name, model, batch)
                    loaded = module.load_finetuned_model(cfg, torch.float32)
                    reloaded = loaded if name == "text" else loaded[0]
                    torch.testing.assert_close(self.model_loss(name, reloaded, batch), expected)

                    # An explicit merged export must load a dense base even when --use_qlora is selected.
                    cfg.use_qlora, cfg.save_merged_model = True, True
                    loaded = module.load_finetuned_model(cfg, torch.float32)
                    export = loaded if name == "text" else loaded[0]
                    self.assertFalse(getattr(export, "is_loaded_in_4bit", False))
                    if name == "text":
                        module.merge_and_save(export, cfg)
                    else:
                        module.merge_and_save(export, loaded[1], cfg)
                    auto = AutoModelForCausalLM if name == "text" else AutoModelForImageTextToText
                    merged = auto.from_pretrained(cfg.output_dir + "-merged", device_map="cpu", dtype=torch.float32)
                    self.assertFalse(getattr(merged, "is_loaded_in_4bit", False))
                    torch.testing.assert_close(self.model_loss(name, merged, batch), expected)

    def test_run_summary_records_bitsandbytes_version(self):
        for name, module in TEMPLATES.items():
            with self.subTest(template=name):
                self.assertIn("bitsandbytes", module.package_versions())


if __name__ == "__main__":
    unittest.main()
