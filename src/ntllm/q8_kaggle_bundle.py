"""Package the current 40 Q8 training stages and their inputs for Kaggle."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


PROJECT = Path(__file__).resolve().parents[2]
SOURCE_DATA = Path("data/q8_source_full/historical_anchor_vs_protocol_unknown")
PROTOCOL_DATA = Path("data/q8_protocol")
REFERENCE_FIELDS = (
    "settings", "labels", "plan_sha256", "data_sha256", "data_manifest_sha256",
    "relation_data_sha256", "planned_token_identity", "planned_relation_token_identity",
)


def _read(path):
    return json.loads(Path(path).read_text())


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def _atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_catalog(root=PROJECT):
    """Record existing plan identities; do not regenerate or modify any plan."""
    from .q8_config import load_q8_config
    from .q8_stages import resolve_stage_supervision

    root = Path(root).resolve()
    centered = _read(root / "runs/centered_reference_20261107/preparation/summary.json")
    scale = _read(root / "runs/scale_preparation_20261107/summary.json")
    jobs, tokenizers = {}, set()

    def add(job_id, profile, plan_path, initial_group, parent, updates):
        plan = Path(plan_path)
        metadata = _read(plan / "config.json")
        if metadata.get("status") != "planned" or not metadata.get("prepared_only"):
            raise ValueError(f"Expected an unexecuted prepared plan: {plan}")
        data_dir = Path(metadata["data_path"]).resolve().parent.relative_to(root)
        tokenizers.add(Path(metadata["tokenizer_path"]).resolve().relative_to(root).as_posix())
        jobs[job_id] = dict(
            profile=profile, phase=metadata["phase"], variant=metadata["variant"],
            fold=metadata["fold"], max_updates=metadata["max_updates"], updates=updates,
            data_dir=data_dir.as_posix(), initial_group=initial_group, parent=parent,
            reference={key: metadata[key] for key in REFERENCE_FIELDS},
            historical_prepared_plan=str(plan),
        )

    external_ids = {
        "off_off": "centered_off", "off_on": "centered_off",
        "on_off": "centered_class", "on_on": "centered_class", "global_on": "centered_global",
    }
    for arm in ("on_on", "global_on", "off_off"):
        add(external_ids[arm], arm, centered["arms"][arm]["external"], "auxiliary", None,
            centered["external_updates"])
    for arm, external_id in external_ids.items():
        if centered["arms"][arm]["external"] != jobs[external_id]["historical_prepared_plan"]:
            raise ValueError(f"Stage {arm} no longer shares its expected external model.")
        for fold in range(5):
            add(f"{arm}_fold{fold}", arm, centered["arms"][arm]["official"][str(fold)],
                "auxiliary", external_id, centered["official_updates_per_fold"])
    for item in scale["plans"]:
        if item["experiment"] != "source":
            continue
        external_id = f"source_{item['variant']}"
        external = item["phase"] == "external"
        job_id = external_id if external else f"{external_id}_fold{item['fold']}"
        add(job_id, "source", item["prepared_plan"], "source", None if external else external_id,
            item["updates"])
    if len(jobs) != 40 or len(tokenizers) != 1:
        raise ValueError(f"Expected 40 jobs and one tokenizer; got {len(jobs)} / {len(tokenizers)}.")
    first, second = centered["planned_budget"], scale["source"]
    budget = dict(
        training_stages=len(jobs), updates=first["optimizer_updates"] + second["updates"],
        exposures=first["row_exposures"] + second["exposures"],
        input_tokens=first["input_tokens"] + second["input_tokens"],
        padded_tokens=first["padded_tokens"] + second["padded_tokens"],
    )
    if sum(job["updates"] for job in jobs.values()) != budget["updates"]:
        raise ValueError("Catalog updates differ from the existing preparation summaries.")
    for job_id, job in jobs.items():
        settings = load_q8_config(root / "configs/q8.toml", job["profile"])
        settings.setdefault("loraplus_ratio", 16.0)
        parent = jobs[job["parent"]]["reference"] if job["parent"] else None
        resolve_stage_supervision(settings, job["phase"], parent)
        if settings != job["reference"]["settings"]:
            raise ValueError(f"{job_id}: current profile differs from the frozen scientific settings.")
    return dict(
        format="ntllm_kaggle_jobs_v1", jobs=jobs, tokenizer_dir=tokenizers.pop(), budget=budget,
        token_scope="Successful training forwards, including repeated exposure; excludes evaluation, diagnostics and retries.",
        reference_scope="Historical paths are informational; execution uses bundled relative paths and recorded identities.",
    )


def payload_files(root, catalog):
    """Explicit runtime inputs; no model weights, raw source downloads or old runs."""
    root = Path(root).resolve()
    files = {Path("pyproject.toml"), Path("README.md")}
    if (root / "uv.lock").is_file():
        files.add(Path("uv.lock"))
    for directory, suffixes in (("src", {".py"}), ("configs", {".toml", ".txt"})):
        files.update(p.relative_to(root) for p in (root / directory).rglob("*")
                     if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts)
    files.update(PROTOCOL_DATA / name for name in (
        "external_expanded.parquet", "official.parquet", "labels.json", "manifest.json",
        "protocol_evidence.parquet", "prompt_folds.json",
    ))
    files.update(PROTOCOL_DATA / "governance" / name for name in (
        "external_expanded.governance.parquet", "manifest.json", "summary.json", "source_history_registry.json",
    ))
    files.update(SOURCE_DATA / name for name in (
        "external_anchor.parquet", "external_baseline.parquet", "external_mix20.parquet",
        "official.parquet", "labels.json", "manifest.json", "prompt_folds.json", "replacement_slots.parquet",
    ))
    for directory in (SOURCE_DATA / "governance", SOURCE_DATA / "intervention_scope",
                      Path(catalog["tokenizer_dir"])):
        files.update(p.relative_to(root) for p in (root / directory).rglob("*") if p.is_file())
    for relative in files:
        path = root / relative
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing, linked or out-of-project payload input: {relative}")
    return sorted(files)


def build_bundle(output_dir, *, root=PROJECT, catalog_only=False):
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog(root)
    catalog_bytes = _json_bytes(catalog)
    _atomic_bytes(output_dir / "kaggle_jobs.json", catalog_bytes)
    if catalog_only:
        return dict(catalog=str(output_dir / "kaggle_jobs.json"), budget=catalog["budget"])
    descriptor, temporary = tempfile.mkstemp(prefix=".ntllm-kaggle.", suffix=".zip", dir=output_dir)
    os.close(descriptor)
    manifest = dict(format="ntllm_kaggle_bundle_v1", files={},
                    scope="Every payload member except this manifest; paths are relative to the ntllm/ archive root.")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1,
                             allowZip64=True) as archive:
            for relative in payload_files(root, catalog):
                digest, size = hashlib.sha256(), 0
                with (root / relative).open("rb") as source, archive.open(f"ntllm/{relative.as_posix()}", "w", force_zip64=True) as target:
                    while block := source.read(8 * 1024 * 1024):
                        target.write(block)
                        digest.update(block)
                        size += len(block)
                manifest["files"][relative.as_posix()] = dict(sha256=digest.hexdigest(), bytes=size)
            archive.writestr("ntllm/kaggle_jobs.json", catalog_bytes)
            manifest["files"]["kaggle_jobs.json"] = dict(sha256=hashlib.sha256(catalog_bytes).hexdigest(), bytes=len(catalog_bytes))
            for job_id, job in catalog["jobs"].items():
                data_name = "official.parquet" if job["phase"] == "official" else f"external_{job['variant']}.parquet"
                for name, key in ((data_name, "data_sha256"), ("manifest.json", "data_manifest_sha256")):
                    relative = f"{job['data_dir']}/{name}"
                    if manifest["files"][relative]["sha256"] != job["reference"][key]:
                        raise ValueError(f"{job_id}: packaged {name} differs from the frozen reference.")
            manifest["totals"] = dict(files=len(manifest["files"]), bytes=sum(item["bytes"] for item in manifest["files"].values()))
            archive.writestr("ntllm/bundle_manifest.json", _json_bytes(manifest))
        destination = output_dir / "ntllm-kaggle.zip"
        os.replace(temporary, destination)
        _atomic_bytes(output_dir / "bundle_manifest.json", _json_bytes(manifest))
        return dict(archive=str(destination), archive_bytes=destination.stat().st_size,
                    payload=manifest["totals"], budget=catalog["budget"])
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "runs/kaggle_preparation")
    parser.add_argument("--catalog-only", action="store_true", help="Write the current job catalog without creating the zip.")
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.output_dir, catalog_only=args.catalog_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
