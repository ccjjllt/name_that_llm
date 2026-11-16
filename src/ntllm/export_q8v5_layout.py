"""Export a local Q8 checkpoint dir to submission q8v5 layout (CPU-only).

Local (Q8Classifier.save):
  run_dir/adapter/adapter_model.safetensors
  run_dir/adapter/adapter_config.json
  run_dir/heads.safetensors
  run_dir/model_config.json

Submission (q8v5_f*/q8v5):
  out_dir/adapter_config.json      (inference_mode=true)
  out_dir/adapter_model.safetensors
  out_dir/heads.pt                 (torch.save dict, same keys)

The submission loader (core_worker.py) is dual-compatible, so this export
is for byte-level alignment, not for readability.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def export_q8v5_layout(run_dir: Path, out_dir: Path) -> dict:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    adapter_dir = run_dir / "adapter"
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"missing {adapter_dir / 'adapter_model.safetensors'}")
    if not (run_dir / "heads.safetensors").is_file():
        raise FileNotFoundError(f"missing {run_dir / 'heads.safetensors'}")
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"missing {adapter_dir / 'adapter_config.json'}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. adapter weights top-level (bytes preserved via tensor round-trip).
    sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
    save_file(sd, str(out_dir / "adapter_model.safetensors"))

    # 2. adapter config with inference_mode=true for submission parity.
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    cfg["inference_mode"] = True
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

    # 3. heads.safetensors -> torch.save heads.pt (keys unchanged).
    heads = dict(load_file(str(run_dir / "heads.safetensors")))
    torch.save(heads, str(out_dir / "heads.pt"))

    return {
        "adapter_tensors": len(sd),
        "head_keys": sorted(heads.keys()),
        "adapter_config_inference_mode": cfg["inference_mode"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    info = export_q8v5_layout(a.run_dir, a.out_dir)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
