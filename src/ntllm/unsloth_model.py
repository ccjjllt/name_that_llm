"""Experimental Unsloth backend for the unchanged q8 feature/classification task.

Import this module before importing Transformers/PEFT model modules. Benchmark
it in its own process: Unsloth patches their classes globally. No quantization,
chat template, LM loss, or dropout removal is introduced here.

The implementation follows the inspected Unsloth source revision recorded in
SOURCE_REVISION. It still needs a real Kaggle comparison; source compatibility
is not a measured speedup or proof of equivalent floating-point outputs.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path

# Unsloth must install its patches before the imports in lora_model.
from unsloth import FastLanguageModel

import torch
from torch import nn
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_layers import GradientCheckpointingLayer

from .lora_model import Q8Classifier, TARGET_MODULES


SOURCE_REVISION = "3fc6b699e65e8c6c375c29ea4f7fa80185f1e6a0"


def _package_evidence(name):
    distribution = importlib.metadata.distribution(name)
    direct = distribution.read_text("direct_url.json")
    return {"version": distribution.version, "direct_url": json.loads(direct) if direct else None}


def _callable_name(value):
    value = getattr(value, "__func__", value)
    value = getattr(value, "func", value)
    return getattr(value, "__name__", type(value).__name__)


class UnslothQ8Classifier(Q8Classifier):
    """Same public interface as Q8Classifier, with a separately saved backend."""

    def __init__(self, base: Path, config: dict, labels: list[str],
                 device: torch.device, dtype: torch.dtype,
                 checkpoint: Path | None = None):
        nn.Module.__init__(self)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("This Unsloth q8 experiment runs on a Kaggle CUDA GPU.")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.compute_dtype = dtype
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Use the same FP16/BF16 compute precision as the reference experiment.")
        self.labels = list(labels)
        self.layers = list(config["layers"])
        self.max_len = int(config.get("max_len", 384))
        self._checkpointing = bool(config.get("gradient_checkpointing", True))
        self.base_config_sha256 = hashlib.sha256((base / "config.json").read_bytes()).hexdigest()
        self.model_config = {
            "keep": int(config["keep"]), "layers": self.layers,
            "max_len": self.max_len, "r": int(config.get("r", 16)),
            "alpha": int(config.get("alpha", 32)),
            "lora_dropout": float(config.get("lora_dropout", 0.05)),
            "attention_implementation": config.get("attention_implementation", "sdpa"),
        }
        if checkpoint is not None:
            saved = json.loads((checkpoint / "model_config.json").read_text())
            if saved.get("backend") != "unsloth":
                raise ValueError("Unsloth checkpoints have a different PEFT wrapper; do not silently mix backends.")
            if saved["labels"] != self.labels or saved["base_config_sha256"] != self.base_config_sha256:
                raise ValueError("Checkpoint labels or base configuration differ.")
            for key in ("keep", "layers", "r", "alpha", "lora_dropout"):
                if saved["model_config"][key] != self.model_config[key]:
                    raise ValueError(f"Checkpoint architecture differs for {key}.")

        model_config = AutoConfig.from_pretrained(str(base), local_files_only=True)
        keep = self.model_config["keep"]
        if (model_config.model_type != "qwen3" or not self.layers
                or self.layers != sorted(set(self.layers))
                or not 0 < self.layers[0] <= self.layers[-1] <= keep <= model_config.num_hidden_layers
                or self.max_len < 2):
            raise ValueError("Expected Qwen3 and valid retained/pooling layers and token length.")
        model_config.num_hidden_layers = keep
        model_config.use_cache = False
        model_config.output_hidden_states = True
        if getattr(model_config, "layer_types", None) is not None:
            model_config.layer_types = model_config.layer_types[:keep]
        self.hidden_size = model_config.hidden_size
        if checkpoint is not None and saved["hidden_size"] != self.hidden_size:
            raise ValueError("Checkpoint hidden width differs.")

        # num_labels selects the small SequenceClassification container rather
        # than allocating the unused 151k-vocabulary LM head on a 16GB T4.
        # The original checkpoint is truncated by config BEFORE weight loading.
        container, _ = FastLanguageModel.from_pretrained(
            model_name=str(base), config=model_config, num_labels=len(labels),
            max_seq_length=self.max_len, dtype=dtype,
            load_in_4bit=False, load_in_8bit=False, load_in_16bit=True,
            full_finetuning=False, fast_inference=False,
            use_gradient_checkpointing=False, offload_embedding=False,
            device_map={"": self.device.index}, local_files_only=True,
            fix_tokenizer=False, use_exact_model_name=True,
            trust_remote_code=False, random_state=int(config.get("seed", 42)),
        )
        if len(container.model.layers) != keep:
            raise ValueError("The Unsloth loader did not retain the requested layer count.")
        container.requires_grad_(False)
        if checkpoint is None:
            self.backbone = FastLanguageModel.get_peft_model(
                container, r=self.model_config["r"], lora_alpha=self.model_config["alpha"],
                lora_dropout=self.model_config["lora_dropout"], bias="none",
                target_modules=TARGET_MODULES, use_gradient_checkpointing=False,
                random_state=int(config.get("seed", 42)), use_rslora=False,
            )
        else:
            self.backbone = PeftModel.from_pretrained(
                container, str(checkpoint / "adapter"), is_trainable=True,
                autocast_adapter_dtype=True, local_files_only=True,
            )
            self.backbone = FastLanguageModel.patch_peft_model(
                self.backbone, use_gradient_checkpointing=False)

        # The container's score head is not our classifier. Only LoRA is trained
        # inside the backbone; the existing eight LayerNorms/two heads follow.
        for name, parameter in self.backbone.named_parameters():
            is_lora = ".lora_A." in name or ".lora_B." in name
            parameter.requires_grad_(is_lora)
            if is_lora:
                parameter.data = parameter.data.float()
        self.tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("The original tokenizer needs padding or EOS.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        width = self.hidden_size * len(self.layers)
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size, device=self.device, dtype=torch.float32)
            for _ in self.layers
        ])
        self.plain = nn.Linear(width, len(labels), device=self.device, dtype=torch.float32)
        self.centered = nn.Linear(width, len(labels), device=self.device, dtype=torch.float32)
        if checkpoint is not None:
            heads = load_file(str(checkpoint / "heads.safetensors"))
            for name in ("norms", "plain", "centered"):
                getattr(self, name).load_state_dict({key.removeprefix(name + "."): value
                    for key, value in heads.items() if key.startswith(name + ".")})

        self.backend_evidence = {
            "inspected_unsloth_revision": SOURCE_REVISION,
            "unsloth": _package_evidence("unsloth"),
            "unsloth_zoo": _package_evidence("unsloth_zoo"),
            "quantization": "none", "unused_sequence_score_head": "frozen and bypassed",
            "gradient_checkpointing": "HF non-reentrant with RNG preservation",
            "fused_lora_dropout_requested": self.model_config["lora_dropout"],
            "patched_layer_counts": {
                "qkv": sum(_callable_name(getattr(layer.self_attn, "apply_qkv", None)) == "apply_lora_qkv"
                           for layer in self.backbone.get_base_model().model.layers),
                "o": sum(_callable_name(getattr(layer.self_attn, "apply_o", None)) == "apply_lora_o"
                         for layer in self.backbone.get_base_model().model.layers),
                "mlp": sum(any(_callable_name(getattr(layer.mlp, slot, None)).startswith("apply_lora_mlp_")
                               for slot in ("forward", "_unsloth_forward", "_old_forward"))
                           for layer in self.backbone.get_base_model().model.layers),
            },
            "runtime_validation": "Experimental backend; actual agreement and throughput require a real GPU comparison.",
        }
        self._configure_checkpointing()

    def _configure_checkpointing(self):
        container = self.backbone.get_base_model()
        trunk = container.model
        # Legacy Unsloth's loop checkpoints with preserve_rng_state=False.
        # Modern HF layer wrappers let us keep the exact nonzero-dropout rule.
        if self._checkpointing and not all(isinstance(layer, GradientCheckpointingLayer)
                                           for layer in trunk.layers):
            raise ValueError("RNG-preserving q8 checkpointing needs HF GradientCheckpointingLayer support.")
        self.backbone.gradient_checkpointing_disable()
        if hasattr(container, "_require_grads_hook"):
            container.disable_input_require_grads()
        if self._checkpointing:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True})
            self.backbone.enable_input_require_grads()
        self.backbone._unsloth_gradient_checkpointing = self._checkpointing

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode and hasattr(self, "backbone"):
            self._configure_checkpointing()
        return self

    def representation(self, tokens):
        batch = self.tokenizer.pad({"input_ids": tokens}, return_tensors="pt").to(self.device)
        # Bypass the unused classifier score head but retain patched Qwen blocks.
        trunk = self.backbone.get_base_model().model
        with torch.autocast("cuda", dtype=self.compute_dtype):
            hidden = trunk(**batch, output_hidden_states=True, use_cache=False,
                           return_dict=True).hidden_states
        mask = batch["attention_mask"].unsqueeze(-1).float()
        mask[:, 0] = 0
        count = mask.sum(dim=1).clamp(min=1)
        return torch.cat([norm((hidden[layer].float() * mask).sum(dim=1) / count)
                          for layer, norm in zip(self.layers, self.norms)], dim=1)

    def save(self, directory):
        super().save(directory)
        path = Path(directory) / "model_config.json"
        metadata = json.loads(path.read_text())
        metadata.update(backend="unsloth", backend_evidence=self.backend_evidence)
        path.write_text(json.dumps(metadata, indent=2) + "\n")
