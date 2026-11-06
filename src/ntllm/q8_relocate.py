"""Move an unexecuted external plan without resampling or changing its inputs.

Only machine paths are rebound. The data manifest and all numerical/text
artifacts remain byte-identical. A pending initialization remains pending;
moving a plan is neither a model initialization nor proof of execution.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .q8_data import file_sha
from .q8_exposure import load_exposure_manifest
from .q8_initialization import initialization_identity
from .q8_supervision import align_anchor_rows
from .q8_tokens import tokenizer_identity


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _check_file(path, expected, description):
    if not path.is_file() or file_sha(path) != expected:
        raise ValueError(f"{description} differs from the frozen content: {path}")


def _inside(directory, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Frozen artifact names must stay within their directory.")
    return directory / relative


def _read_plan(prepared, metadata, row_weights, centered_weights):
    with np.load(prepared / "plan.npz", allow_pickle=False) as handle:
        arrays = {key: handle[key].copy() for key in ("rows", "sizes", "same_prompt", "epochs")}
    if not (len(arrays["sizes"]) == len(arrays["same_prompt"]) == len(arrays["epochs"])):
        raise ValueError("Frozen plan arrays have inconsistent batch counts.")
    if int(arrays["sizes"].sum()) != len(arrays["rows"]):
        raise ValueError("Frozen plan sizes do not cover its rows.")
    digest = hashlib.sha256()
    for key, values in {**arrays, "plain_weights": row_weights,
                        "centered_weights": centered_weights}.items():
        digest.update(json.dumps([key, str(values.dtype), list(values.shape)]).encode())
        digest.update(values.tobytes())
    if digest.hexdigest() != metadata["plan_sha256"]:
        raise ValueError("Frozen plan or loss weights changed.")


def _initial_reference(metadata, exposure, target):
    proof = metadata.get("initialization_proof") or {}
    frozen_proof = exposure.get("references", {}).get("initialization") or {}
    expected = proof.get("initialization_sha256") or frozen_proof.get("initialization_sha256")
    for reference in (proof, frozen_proof):
        if reference.get("initialization_sha256") not in (None, expected):
            raise ValueError("The two frozen initialization references disagree.")
    if expected is not None:
        current = initialization_identity(target)
        if current.get("initialization_sha256") != expected:
            raise ValueError("A materialized initial state can only move with the same contents.")
        context = current.get("context", {})
        if (current.get("assets", {}).get("content", {}).get("kind") != "transformer_base"
                or context.get("labels") != metadata["labels"]
                or context.get("backend") != metadata["settings"].get("backend", "hf")):
            raise ValueError("The initial state is not the frozen Q8 label/backend/base context.")
        for key, value in context.get("architecture", {}).items():
            if key in metadata["settings"] and metadata["settings"][key] != value:
                raise ValueError(f"The initial architecture differs from the frozen setting: {key}")
        manifest = target / "initialization.json"
        return dict(status="available", directory=str(target), model_kind="transformer_base",
                    manifest_path=str(manifest), manifest_sha256=file_sha(manifest),
                    initialization_sha256=expected)
    if proof.get("status") != "pending" or frozen_proof.get("status") != "pending":
        raise ValueError("Only an explicitly pending common start may be rebound without a parameter SHA.")
    old_path = metadata.get("shared_initialization_path")
    if not old_path or any(Path(ref.get("directory", "")).resolve() != Path(old_path).resolve()
                           for ref in (proof, frozen_proof)):
        raise ValueError("The pending initialization path is inconsistent.")
    if Path(old_path).exists() or target.exists():
        raise ValueError("A materialized directory cannot be silently substituted for a pending start; bind its explicit identity first.")
    return dict(status="pending", directory=str(target), model_kind="transformer_base")


def relocate_prepared(prepared, data_dir, tokenizer_dir, initial_state, out):
    """Copy a reviewed external plan and explicitly bind new machine paths.

    `data_dir` must already contain the exact original data-manifest bytes and
    referenced files, with its relative evidence-ledger path intact. This
    helper does not rewrite that manifest or rebuild training data.
    """
    prepared, data_dir, tokenizer_dir, initial_state, out = (
        Path(value).resolve() for value in (prepared, data_dir, tokenizer_dir, initial_state, out))
    if out.exists() or out.is_relative_to(prepared):
        raise FileExistsError("Use a new output directory outside the source plan.")
    metadata = json.loads((prepared / "config.json").read_text())
    if metadata.get("settings", {}).get("initialization_mode") == "canonical_backend":
        raise ValueError("Canonical backend plans cannot be relocated by this helper. Relocate the original strict plan first, then derive both backend plans with the explicit canonical initialization path.")
    exposure = json.loads((prepared / "exposure_manifest.json").read_text())
    if (metadata.get("phase") != "external" or metadata.get("status") != "planned"
            or metadata.get("prepared_only") is not True or metadata.get("resume_from")
            or metadata.get("initialization_run") or metadata.get("external_initialization_config")):
        raise ValueError("Relocation supports only unexecuted fresh external plans; official lineage needs its own explicit evidence.")
    executed = ("executed_initialization.json", "metrics.json", "train_log.jsonl", "checkpoints",
                "actual_row_indices.npy", "adapter", "heads.safetensors")
    if any((prepared / name).exists() for name in executed):
        raise ValueError("Execution artifacts are present; do not rebind an already executed run.")
    _check_file(prepared / "exposure_manifest.json", metadata["exposure_manifest_sha256"], "Exposure manifest")
    for name, digest in exposure["files"].items():
        _check_file(_inside(prepared, name), digest, "Exposure artifact")
    for key, filename in (("actual_exposure_sha256", "actual_exposure.parquet"),
                          ("update_schedule_sha256", "update_schedule.parquet"),
                          ("token_rows_sha256", "token_rows.parquet"),
                          ("token_profile_sha256", "token_profile.json")):
        if key not in metadata:
            raise ValueError(f"The prepared plan lacks a required content binding: {key}")
        _check_file(prepared / filename, metadata[key], "Prepared artifact")

    _check_file(data_dir / "manifest.json", metadata["data_manifest_sha256"], "Prepared data manifest")
    data_manifest = json.loads((data_dir / "manifest.json").read_text())
    for name, digest in data_manifest.get("files_sha256", {}).items():
        _check_file(_inside(data_dir, name), digest, "Relocated data file")
    data_path = data_dir / Path(metadata["data_path"]).name
    _check_file(data_path, metadata["data_sha256"], "Training data")
    ledger = Path(data_manifest["evidence_ledger"])
    if not ledger.is_absolute():
        ledger = data_dir / ledger
    ledger = ledger.resolve()
    _check_file(ledger, data_manifest["evidence_ledger_sha256"], "Evidence ledger")
    expected_tokenizer = metadata.get("tokenizer_identity") or {}
    new_tokenizer = tokenizer_identity(tokenizer_dir)
    if (new_tokenizer["files_identity_sha256"] != expected_tokenizer.get("files_identity_sha256")
            or new_tokenizer["files_sha256"] != expected_tokenizer.get("files_sha256")):
        raise ValueError("Tokenizer files differ from the frozen tokenizer.")
    new_initial = _initial_reference(metadata, exposure, initial_state)

    frame = pd.read_parquet(data_path)
    training_rows = pd.read_parquet(prepared / "training_rows.parquet")
    required_columns = {"id", "model", "word_bin", "prompt_id"}
    if "training_slot_id" in frame:
        required_columns.add("training_slot_id")
    if not required_columns.issubset(training_rows.columns):
        raise ValueError("The saved training-row identity is incomplete.")
    for column in sorted(set(training_rows.columns) & set(frame.columns)):
        if not frame[column].equals(training_rows[column]):
            raise ValueError(f"Full training-row identity differs: {column}")
    row_weights = training_rows.loss_weight.to_numpy(dtype=np.float32)
    centered_weights = np.load(prepared / "centered_loss_weights.npy", allow_pickle=False)
    _read_plan(prepared, metadata, row_weights, centered_weights)
    relation_frame, relation_path = None, None
    if metadata.get("relation_data_path"):
        relation_path = data_dir / Path(metadata["relation_data_path"]).name
        _check_file(relation_path, metadata["relation_data_sha256"], "Relation data")
        relation_frame = align_anchor_rows(frame, pd.read_parquet(relation_path))
    _, source_positions, source_report = load_exposure_manifest(
        prepared, frame, row_weights, centered_weights, relation_frame,
        metadata["settings"]["centered_weight"], expected_config=metadata["settings"])

    refs = copy.deepcopy(exposure.get("references", {}))
    for filename in ("official.parquet", "prompt_folds.json", "labels.json"):
        original = refs.get("official_files", {}).get(filename)
        expected = original.get("sha256") if isinstance(original, dict) else original
        if not expected:
            raise ValueError(f"Missing official-file identity: {filename}")
        _check_file(data_dir / filename, expected, "Official split/labels")
        refs["official_files"][filename] = dict(path=str(data_dir / filename), sha256=expected)
    refs["initialization"] = new_initial
    refs["model_kind"] = "transformer_base"
    moved = copy.deepcopy(metadata)
    moved.update(data_path=str(data_path), tokenizer_path=str(tokenizer_dir),
                 shared_initialization_path=str(initial_state), initialization_proof=new_initial,
                 prepared_plan_source=None, base=None)
    moved["tokenizer_identity"]["path"] = str(tokenizer_dir)
    if relation_path is not None:
        moved["relation_data_path"] = str(relation_path)
    for name, sha_name in (("anchor_path", "anchor_sha256"), ("peer_path", "peer_sha256")):
        if moved.get("relations", {}).get(name):
            target = data_dir / Path(moved["relations"][name]).name
            _check_file(target, moved["relations"][sha_name], "Relation/peer data")
            moved["relations"][name] = str(target)
    # plan_source is historical provenance, not an active input of --prepared-plan.
    if moved.get("evidence_reference"):
        moved["evidence_reference"]["evidence_ledger"] = str(ledger)
    source_files = {path.name: file_sha(path) for path in prepared.iterdir() if path.is_file()}
    if any(not path.is_file() for path in prepared.iterdir()):
        raise ValueError("Unexpected subdirectories in a prepared-only run.")
    out.mkdir(parents=True)
    for name in source_files:
        shutil.copy2(prepared / name, out / name)
    new_exposure = copy.deepcopy(exposure)
    new_exposure["references"] = refs
    _write(out / "exposure_manifest.json", new_exposure)
    moved["exposure_manifest_sha256"] = file_sha(out / "exposure_manifest.json")
    moved["relocated_from"] = dict(directory=str(prepared), config_sha256=source_files["config.json"],
                                    exposure_manifest_sha256=source_files["exposure_manifest.json"])
    _write(out / "config.json", moved)
    unchanged = {name: file_sha(out / name) == digest for name, digest in source_files.items()
                 if name not in ("config.json", "exposure_manifest.json", "relocation.json")}
    if not all(unchanged.values()):
        raise ValueError("Relocation changed a frozen artifact.")
    _, destination_positions, destination_report = load_exposure_manifest(
        out, frame, row_weights, centered_weights, relation_frame,
        moved["settings"]["centered_weight"], expected_config=moved["settings"])
    if source_positions != destination_positions or source_report["settings_sha256"] != destination_report["settings_sha256"]:
        raise ValueError("Relocation changed actual update contents or settings.")
    report = dict(status="relocated_without_execution", source=str(prepared), destination=str(out),
                  source_files_sha256=source_files, unchanged_artifacts=unchanged,
                  actual_updates=len(destination_positions), actual_rows=destination_report["actual_rows"],
                  relation_rows=destination_report["relation_rows"], actual_windows_identical=True,
                  scientific_settings_unchanged=True, plan_sha256=moved["plan_sha256"],
                  data_sha256=moved["data_sha256"], data_manifest_sha256=moved["data_manifest_sha256"],
                  initialization_status=new_initial["status"], shared_initialization_path=str(initial_state),
                  pending_binding_origin=metadata.get("shared_initialization_path"),
                  active_paths=dict(data_dir=str(data_dir), tokenizer_dir=str(tokenizer_dir), evidence_ledger=str(ledger)),
                  historical_paths_retained=[name for name in ("parent_manifest_path",)
                                             if name in moved.get("evidence_reference", {})],
                  scope="Machine-path rebinding only. No tokenizer/model execution, new initialization, resampling or upload.")
    _write(out / "relocation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    report = relocate_prepared(**vars(parser.parse_args()))
    print(json.dumps({key: report[key] for key in ("status", "actual_updates", "actual_rows", "relation_rows", "initialization_status")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
