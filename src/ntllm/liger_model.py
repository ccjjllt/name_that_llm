"""Experimental Liger transformer kernels with the existing q8 model/adapter layout.

Run in a separate process from the plain HF and Unsloth backends: the Liger
Qwen3 helper patches classes globally. The 18-class objective is unchanged.
"""

import importlib.metadata
import json
from pathlib import Path

import torch

from .lora_model import Q8Classifier


# v0.8.4 is the last released tag retaining Transformers 4.x support;
# v0.8.5 and newer main reject the pinned 4.57.6 at import time.
SOURCE_REVISION = "43293d84d760e402e4fd974b12410a1d47c31761"


class LigerQ8Classifier(Q8Classifier):
    def __init__(self, base: Path, config: dict, labels: list[str],
                 device: torch.device, dtype: torch.dtype,
                 checkpoint: Path | None = None):
        if torch.device(device).type != "cuda":
            raise ValueError("This Liger q8 experiment runs on a Kaggle CUDA GPU.")
        source_backend = None
        if checkpoint is not None:
            saved = json.loads((checkpoint / "model_config.json").read_text())
            source_backend = saved.get("backend", "hf")
            if source_backend not in ("hf", "liger"):
                raise ValueError("Liger keeps the HF PEFT layout; other wrapper checkpoints require explicit conversion.")

        from liger_kernel.transformers import apply_liger_kernel_to_qwen3

        apply_liger_kernel_to_qwen3(
            rope=True, rms_norm=True, swiglu=True,
            cross_entropy=False, fused_linear_cross_entropy=False,
        )
        super().__init__(base, config, labels, device, dtype, checkpoint)
        distribution = importlib.metadata.distribution("liger_kernel")
        direct = distribution.read_text("direct_url.json")
        self.backend_evidence = {
            "inspected_liger_revision": SOURCE_REVISION,
            "liger_kernel_version": distribution.version,
            "direct_url": json.loads(direct) if direct else None,
            "kernels": {"rope": True, "rms_norm": True, "swiglu": True,
                        "cross_entropy": False, "fused_linear_cross_entropy": False},
            "quantization": "none", "checkpoint_layout": "same as HF Q8Classifier",
            "initialization_backend": source_backend,
            "module_counts": {
                name: sum(type(module).__name__ == name for module in self.backbone.modules())
                for name in ("LigerRMSNorm", "LigerSwiGLUMLP")
            },
            "runtime_validation": "Experimental backend; throughput and numerical agreement need a real GPU comparison.",
        }

    def save(self, directory):
        super().save(directory)
        path = Path(directory) / "model_config.json"
        metadata = json.loads(path.read_text())
        metadata.update(backend="liger", backend_evidence=self.backend_evidence)
        path.write_text(json.dumps(metadata, indent=2) + "\n")
