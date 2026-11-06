"""Compare complete two-stage data × centered-supervision OOF experiments."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .compare_lora import KEYS, combine_predictions, metrics, outcomes, read_prediction
from .q8_data import file_sha


ARMS = ("baseline_off", "mix20_off", "baseline_on", "mix20_on")
PAIRS = {
    "data_off": ("baseline_off", "mix20_off"),
    "data_on": ("baseline_on", "mix20_on"),
    "aux_baseline": ("baseline_off", "baseline_on"),
    "aux_mix20": ("mix20_off", "mix20_on"),
}


def read_config(path, phase, variant, enabled):
    config = json.loads((path / "config.json").read_text())
    if (config.get("phase") != phase or config.get("variant") != variant
            or config.get("status") != "complete" or config.get("prepared_only", True)):
        raise ValueError(f"{path}: expected a completed {phase}/{variant} training run.")
    settings = config["settings"]
    weight = settings["centered_weight"]
    if not np.isfinite(weight) or (weight <= 0 if enabled else weight != 0):
        raise ValueError(f"{path}: centered_weight does not match this off/on arm.")
    if (settings.get("centered_plan") not in ("shared_triplets", "shared_stratified_triplets")
            or settings.get("centered_balance") != "class_mass"):
        raise ValueError(f"{path}: shared or shared-stratified triplets with class_mass supervision are required.")
    if not config.get("data_manifest_sha256") or not config.get("base"):
        raise ValueError(f"{path}: data manifest and base model identity are required.")
    return config


def validate_fold_predictions(prediction, official, fold, source):
    """Check the exact in-memory prediction snapshot selected for scoring."""
    expected = official.set_index('id').sort_index()
    expected = expected.loc[expected.fold.eq(fold), ['prompt_id', 'fold', 'model', 'n_words']]
    expected = expected.rename(columns={'model': 'true_model'})
    selected = prediction.set_index('id').sort_index()
    if (not selected.index.is_unique or not selected.index.equals(expected.index)
            or not np.array_equal(selected[KEYS].to_numpy(), expected[KEYS].to_numpy())):
        raise ValueError(f'{source}: predictions do not cover exactly the recorded held-out fold.')


def verify_design(paths, return_predictions=False):
    """Verify both stages and the actual official rows underlying all five folds."""
    records, manifest_hashes, base_paths, on_weights = {}, set(), set(), set()
    settings_by_phase, official_hashes = {}, set()
    contexts, data_hashes, budgets = {}, {}, {}
    code_by_phase, dtype_by_phase, devices_by_phase = {}, {}, {}
    requested_budgets = {}
    snapshots = {arm: [] for arm in ARMS}
    for arm in ARMS:
        variant, state = arm.rsplit("_", 1)
        enabled = state == "on"
        records[arm] = {}
        for path in paths[arm]:
            config = read_config(path, "official", variant, enabled)
            fold = config.get("fold")
            if type(fold) is not int or fold not in range(5) or fold in records[arm]:
                raise ValueError(f"{path}: each arm needs exactly one run for each fold 0..4.")
            if config.get("initialization_partial") or not config.get("initialization_run"):
                raise ValueError(f"{path}: a complete external initialization is required.")
            parent_path = Path(config["initialization_run"])
            parent = read_config(parent_path, "external", variant, enabled)
            if parent.get("fold") != "shared" or parent.get("conditioning_fold") != "shared":
                raise ValueError(f"{parent_path}: external data arms require a shared starting checkpoint.")
            if config.get("conditioning_fold") != "shared":
                raise ValueError(f"{path}: unexpected target-conditioned external checkpoint.")
            if parent["settings"]["centered_weight"] != config["settings"]["centered_weight"]:
                raise ValueError(f"{path}: off/on supervision must match across both training stages.")
            if parent["labels"] != config["labels"]:
                raise ValueError(f"{path}: label order differs from the external checkpoint.")
            for stage in (config, parent):
                manifest_hashes.add(stage["data_manifest_sha256"])
                base_paths.add(str(Path(stage["base"]).resolve()))
                if enabled:
                    on_weights.add(stage["settings"]["centered_weight"])
                comparable = {key: value for key, value in stage["settings"].items()
                              if key != "centered_weight"}
                phase = stage["phase"]
                code, dtype = stage.get("training_code_sha256"), stage.get("dtype")
                if not code or not dtype:
                    raise ValueError(f"{path}: {phase} training-code and actual precision records are required.")
                if phase in code_by_phase and code != code_by_phase[phase]:
                    raise ValueError(f"{path}: {phase} training code differs across arms.")
                if phase in dtype_by_phase and dtype != dtype_by_phase[phase]:
                    raise ValueError(f"{path}: {phase} actual training precision differs across arms.")
                code_by_phase[phase], dtype_by_phase[phase] = code, dtype
                devices_by_phase.setdefault(phase, set()).add(stage.get("device", "unrecorded"))
                if phase in settings_by_phase and comparable != settings_by_phase[phase]:
                    raise ValueError(f"{path}: {phase} settings differ beyond centered_weight.")
                settings_by_phase[phase] = comparable
                budget = stage["supervision"]["active_updates"]
                if phase in budgets and budgets[phase] != budget:
                    raise ValueError(f"{path}: {phase} update budgets differ across arms.")
                budgets[phase] = budget
                requested_budgets.setdefault(phase, set()).add(stage.get("max_updates"))
                data_key = (phase, variant)
                if data_key in data_hashes and data_hashes[data_key] != stage["data_sha256"]:
                    raise ValueError(f"{path}: {phase}/{variant} off/on data hashes differ.")
                data_hashes[data_key] = stage["data_sha256"]
                epoch_signature = [
                    (epoch.get("centered_context_sha256"), epoch["optimizer_updates"])
                    for epoch in stage["epochs"]
                ]
                if not epoch_signature or any(not digest for digest, _ in epoch_signature):
                    raise ValueError(f"{path}: {phase} relation-context fingerprints are missing.")
                context_key = (phase, stage["fold"])
                if context_key in contexts and contexts[context_key] != epoch_signature:
                    raise ValueError(f"{path}: {phase} fold {stage['fold']} contexts or update counts differ.")
                contexts[context_key] = epoch_signature
            official_path = Path(config["data_path"])
            if file_sha(official_path) != config["data_sha256"]:
                raise ValueError(f"{path}: official data no longer match the recorded input hash.")
            official_hashes.add(config["data_sha256"])
            official = pd.read_parquet(official_path)
            prediction, source = read_prediction(path, view='plain', cohort='full')
            validate_fold_predictions(prediction, official, fold, source['path'])
            snapshots[arm].append(prediction)
            records[arm][fold] = dict(path=str(path.resolve()), config=config,
                                      initialization_run=str(parent_path.resolve()),
                                      prediction_source=source)
        if set(records[arm]) != set(range(5)):
            raise ValueError(f"{arm}: all five complete official folds are required.")
        initializations = {value["initialization_run"] for value in records[arm].values()}
        if len(initializations) != 1:
            raise ValueError(f"{arm}: official folds must share the same external-stage checkpoint.")
    if len(manifest_hashes) != 1 or len(official_hashes) != 1:
        raise ValueError("All stages and arms must use the same prepared data manifest and official table.")
    if len(base_paths) != 1 or len(on_weights) != 1:
        raise ValueError("All arms need the same base path and the same positive on weight.")
    design = dict(data_manifest_sha256=manifest_hashes.pop(), official_data_sha256=official_hashes.pop(),
                centered_on_weight=on_weights.pop(), base_path=base_paths.pop(),
                base_identity_scope="Recorded local path and settings; this does not independently fingerprint base weights or tokenizer.",
                completed_stages_only=True, settings_by_phase=settings_by_phase,
                active_updates_by_phase=budgets,
                requested_max_updates_by_phase={phase: sorted(values, key=str)
                                                for phase, values in requested_budgets.items()},
                training_code_by_phase=code_by_phase, dtype_by_phase=dtype_by_phase,
                devices_by_phase={phase: sorted(values) for phase, values in devices_by_phase.items()},
                relation_contexts={f"{phase}/{fold}": value for (phase, fold), value in contexts.items()},
                runs={arm: {str(fold): {key: value for key, value in record.items() if key != "config"}
                            for fold, record in folds.items()} for arm, folds in records.items()})
    if return_predictions:
        return design, {arm: combine_predictions(snapshots[arm]) for arm in ARMS}
    return design


def factorial_outcomes(rows, predictions):
    if not len(rows):
        return dict(rows=0)
    paired = {name: outcomes(rows, predictions[first], predictions[second])
              for name, (first, second) in PAIRS.items()}
    scores = {
        "baseline_off": paired["data_off"]["baseline_ba"],
        "mix20_off": paired["data_off"]["candidate_ba"],
        "baseline_on": paired["data_on"]["baseline_ba"],
        "mix20_on": paired["data_on"]["candidate_ba"],
    }
    differences = {f"delta_{name}": result["delta"] for name, result in paired.items()}
    return dict(rows=len(rows), prompts=int(rows.prompt_id.nunique()), balanced_accuracy=scores,
                **differences, interaction=differences["delta_data_on"] - differences["delta_data_off"],
                paired=paired)


def interaction_interval(rows, predictions, samples=2000, seed=42):
    group_ids, groups = pd.factorize(rows.prompt_id)
    class_ids, labels = pd.factorize(rows.true_model)
    truth = rows.true_model.to_numpy()
    correct = {name: (values == truth).astype(np.int8) for name, values in predictions.items()}
    interaction = correct["mix20_on"] - correct["baseline_on"] - correct["mix20_off"] + correct["baseline_off"]
    count = np.zeros((len(groups), len(labels)), dtype=np.int64)
    change = np.zeros_like(count)
    np.add.at(count, (group_ids, class_ids), 1)
    np.add.at(change, (group_ids, class_ids), interaction)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=samples)
    denominator = draws @ count
    rates = (draws @ change) / np.where(denominator, denominator, np.nan)
    return dict(unit="prompt_id", samples=samples, seed=seed,
                statistic="(mix20_on - baseline_on) - (mix20_off - baseline_off) balanced accuracy",
                descriptive_percentile_95pct=np.quantile(np.nanmean(rates, axis=1), [.025, .975]).tolist())


def compare(paths):
    design, loaded = verify_design(paths, return_predictions=True)
    rows, labels, _ = loaded["baseline_off"]
    for arm, (other, other_labels, _) in loaded.items():
        if other_labels != labels or not rows[KEYS].equals(other[KEYS]):
            raise ValueError(f"{arm}: IDs, labels, folds, prompts or lengths differ across arms.")
    if rows.groupby("prompt_id").fold.nunique().max() != 1:
        raise ValueError("A validation prompt spans multiple folds.")
    predictions = {arm: np.asarray(labels)[probability.argmax(axis=1)]
                   for arm, (_, _, probability) in loaded.items()}
    slices = {f"class/{label}": rows.true_model.eq(label).to_numpy() for label in labels}
    length = pd.cut(rows.n_words, [-1, 59, 99, 159, np.inf], labels=["<60", "60-99", "100-159", ">=160"])
    slices.update({f"words/{name}": length.eq(name).to_numpy() for name in length.cat.categories})
    slices.update({f"fold/{fold}": rows.fold.eq(fold).to_numpy() for fold in sorted(rows.fold.unique())})
    report = dict(
        scope="Four-arm, complete two-stage plain-head OOF comparison with fixed shared relation contexts. These effects are conditional on the selected data and recipe, not universal text-quality or causality claims.",
        design=design, labels=labels,
        arm_metrics={arm: metrics(rows, labels, probability) for arm, (_, _, probability) in loaded.items()},
        factorial=factorial_outcomes(rows, predictions),
        interaction_bootstrap=interaction_interval(rows, predictions),
        slices={name: factorial_outcomes(rows.loc[mask], {arm: pred[mask] for arm, pred in predictions.items()})
                for name, mask in slices.items()},
    )
    paired_rows = rows[KEYS].copy()
    for arm, prediction in predictions.items():
        paired_rows[arm] = prediction
    return report, paired_rows.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument("--" + arm.replace("_", "-"), nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    report, paired_rows = compare({arm: getattr(args, arm) for arm in ARMS})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"{stamp}_centered_factorial_{uuid4().hex[:6]}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    paired_rows.to_parquet(output / "paired_predictions.parquet", index=False)
    print(json.dumps(report["factorial"], ensure_ascii=False), flush=True)
    print(output.resolve(), flush=True)


if __name__ == "__main__":
    main()
