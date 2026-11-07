"""Freeze relation examples and compare source/protocol strata in plain slots.

These are source/protocol combination experiments, not causal estimates of a
rewrite operation. Only explicit pairs from one original response support that
separate question. No validation outcomes are used to choose replacement rows.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .q8_data import file_sha
from .q8_sampling import WORD_BINS
from .q8_supervision import common_relation_rows, shared_triplet_batches
from .train_lora import update_windows


FOCUS = ("kimi-k3", "glm-5.3", "glm-5.3-flash", "claude-fable-5")
CONDITIONS = ("protocol_documented", "protocol_unknown", "transformed", "normalization_reported")
COMPARISONS = (("historical_anchor", "protocol_unknown"),
               ("protocol_request_logged", "protocol_unknown"),
               ("protocol_source_documented", "protocol_unknown"),
               ("protocol_request_logged", "transformed"),
               ("protocol_unknown", "transformed"),
               ("protocol_unknown", "normalization_reported"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def anchor_slots(anchor, config, max_updates):
    """Keep even unexecuted epoch relations unchanged, without shrinking support."""
    common, _ = common_relation_rows(anchor, anchor)
    relation = set()
    actual = []
    remaining = max_updates
    for epoch in range(config["external_epochs"]):
        batches = shared_triplet_batches(anchor, common, config["seed"] + epoch, config["batch_size"])
        relation.update(row for rows, same in batches if same for row in rows)
        windows = update_windows(batches, config["batch_size"] * config["accumulation"])
        active = min(len(windows), remaining)
        for update, chunk in enumerate(windows[:active]):
            for microbatch, (rows, same) in enumerate(chunk):
                for row in rows:
                    actual.append(dict(position=row, epoch=epoch, update=update,
                                       microbatch=microbatch, centered=same))
        remaining -= active
    if remaining:
        raise ValueError("Requested updates exceed the anchor's full training plan.")
    exposure = pd.DataFrame(actual)
    safe = sorted(set(exposure.position) if config["centered_plan"] == "anchored_separate"
                  else set(exposure.position) - relation)
    return safe, exposure, relation


def condition_inventory(pool, anchor_ids):
    candidates = pool.loc[~pool.id.isin(anchor_ids)].copy()
    if not candidates.id.is_unique:
        raise ValueError("Candidate pool must have unique training IDs; retain variant evidence separately.")
    if not candidates.research_condition.isin(CONDITIONS).all():
        raise ValueError("Every candidate needs an explicit protocol research condition.")
    identity_conflict = candidates.evidence_conflicts.map(lambda value: any(
        name in json.loads(value) for name in
        ("returned_model_differs_from_label", "requested_model_differs_from_label")))
    unresolved = candidates.loc[identity_conflict].copy()
    # Conflicting identity is a distinct question from an unknown generation
    # protocol. Preserve it for investigation; do not silently use it as a
    # valid-label diversity intervention.
    return candidates.loc[~identity_conflict].copy(), unresolved


def condition_mask(frame, condition):
    if condition == "protocol_request_logged":
        return frame.research_condition.eq("protocol_documented") & frame.protocol_evidence_level.eq("request_logged")
    if condition == "protocol_source_documented":
        return frame.research_condition.eq("protocol_documented") & frame.protocol_evidence_level.eq("source_documented")
    return frame.research_condition.eq(condition)


def replace_slots(anchor, replacements, positions, condition):
    """Original positions are the experimental slots, not model inputs."""
    arm = anchor.drop(index=positions).copy()
    if positions:
        replacement = replacements.copy()
        replacement.index = positions
        replacement["group_eligible"] = False
        replacement["relation_use_reason"] = "plain_intervention_frozen_anchor"
        replacement["intervention_condition"] = condition
        arm = pd.concat([arm, replacement], axis=0).sort_index()
    arm = arm.reset_index(drop=True)
    arm["training_slot_id"] = anchor.id.to_numpy()
    arm["intervention_condition"] = arm.intervention_condition.fillna("unchanged_background")
    if not arm.id.is_unique or len(arm) != len(anchor):
        raise ValueError("An intervention duplicated or dropped a training row.")
    if not arm[["model", "word_bin"]].equals(anchor[["model", "word_bin"]]):
        raise ValueError("Replacement altered a slot's class or word-count bin.")
    return arm


def build_pair(anchor, candidates, safe, left, right, seed, focus):
    """Use the intersection of available supply, retaining every shortfall."""
    positions, selected = [], {left: [], right: []}
    strata = []
    safe_rows = anchor.loc[safe]
    for model in sorted(focus):
        for bin_index, word_bin in enumerate(WORD_BINS):
            # Pandas may expose read-only index storage; NumPy's permutation
            # path for very small strata can require a writable array.
            slots = safe_rows.index[safe_rows.model.eq(model) & safe_rows.word_bin.eq(word_bin)].to_numpy(copy=True)
            pools = {}
            for condition in (left, right):
                if condition == "historical_anchor":
                    pools[condition] = anchor.loc[slots]
                else:
                    pools[condition] = candidates.loc[candidates.model.eq(model)
                                                     & candidates.word_bin.eq(word_bin)
                                                     & condition_mask(candidates, condition)].sort_values("id")
            count = min(len(slots), *(len(frame) for frame in pools.values()))
            strata.append(dict(model=model, word_bin=word_bin, eligible_slots=len(slots),
                               supply={key: len(value) for key, value in pools.items()},
                               replaced_slots=count, unfilled_slots=len(slots) - count))
            # Separate deterministic random streams avoid dependence on other strata's supply.
            salt = int(hashlib.sha256(f"{model}:{bin_index}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed + salt)
            chosen_slots = rng.permutation(slots)[:count].tolist()
            positions.extend(chosen_slots)
            for condition, frame in pools.items():
                chosen = (anchor.loc[chosen_slots] if condition == "historical_anchor" else
                          frame.iloc[rng.permutation(len(frame))[:count]])
                selected[condition].append(chosen)
    replacements = {key: pd.concat(value, ignore_index=True) for key, value in selected.items()}
    arms = {key: replace_slots(anchor, replacements[key], positions, key) for key in selected}
    return arms, positions, strata


def build(args):
    if args.out.exists():
        raise FileExistsError(f"Use a new output directory to preserve research history: {args.out}")
    from .q8_config import load_q8_config
    from .q8_stages import resolve_stage_supervision
    config = load_q8_config(args.config, getattr(args, 'experiment', None), default_experiment='source')
    resolve_stage_supervision(config, 'external')
    if config["centered_plan"] not in ("anchored_triplets", "anchored_separate"):
        raise ValueError("Protocol controls require a frozen anchor plan.")
    selected = getattr(args, 'comparison', None)
    names = {left + '_vs_' + right for left, right in COMPARISONS}
    if selected and not set(selected) <= names:
        raise ValueError('Unknown source comparison; choose an explicitly named condition pair.')
    comparisons = [(left, right) for left, right in COMPARISONS
                   if not selected or left + '_vs_' + right in selected]
    separate = config["centered_plan"] == "anchored_separate"
    anchor = pd.read_parquet(args.data_dir / "external_baseline.parquet")
    anchor["training_slot_id"] = anchor.id
    anchor["intervention_condition"] = "unchanged_background"
    pool = pd.read_parquet(args.data_dir / "external_expanded.parquet")
    candidates, unresolved_identity = condition_inventory(pool, set(anchor.id))
    safe, exposure, relations = anchor_slots(anchor, config, args.max_updates)
    args.out.mkdir(parents=True)
    unresolved_identity.to_parquet(args.out / "identity_conflicts_pending.parquet", index=False)
    evidence = (args.data_dir / "protocol_evidence.parquet").resolve()
    if not getattr(args, 'reference_evidence', False):
        shutil.copy2(evidence, args.out / "protocol_evidence.parquet")
        evidence = (args.out / "protocol_evidence.parquet").resolve()
    exposure["training_slot_id"] = anchor.iloc[exposure.position.to_numpy()].id.to_numpy()
    exposure.to_parquet(args.out / "anchor_actual_slots.parquet", index=False)
    report = dict(
        method="Frozen baseline relation plan; same class and length at every corresponding plain slot",
        centered_plan=config["centered_plan"],
        relation_execution="Independent frozen-anchor forwards in both arms" if separate else "Shared forward for unchanged relation rows",
        interpretation="Changed text and source imply a source/protocol combination effect; not an isolated transformation effect.",
        input_data_dir=str(args.data_dir.resolve()), input_manifest_sha256=file_sha(args.data_dir / "manifest.json"),
        config_sha256=file_sha(args.config), settings=config, seed=config["seed"], external_epochs=config["external_epochs"],
        selected_comparisons=[left + '_vs_' + right for left, right in comparisons],
        evidence_ledger=os.path.relpath(evidence, args.out.resolve()),
        evidence_ledger_sha256=file_sha(evidence),
        max_updates=args.max_updates, focus_labels=list(args.focus), pool_rows=len(pool),
        anchor_rows=len(anchor), actual_exposures=len(exposure),
        actual_centered_exposures=int(exposure.centered.sum()),
        actual_centered_unique=int(exposure.loc[exposure.centered, "position"].nunique()),
        replaceable_actual_plain_slots=len(safe),
        safe_actual_plain_slots=int(len(set(exposure.position) - relations)),
        full_plan_centered_union=len(relations),
        candidate_counts=candidates.research_condition.value_counts().to_dict(),
        documented_reference_levels=candidates.loc[candidates.research_condition.eq("protocol_documented"),
                                                    "protocol_evidence_level"].value_counts().to_dict(),
        unresolved_identity_candidates=len(unresolved_identity),
        identity_policy="Unknown model evidence remains explicit; confirmed request/return label conflicts are held separately, not used as protocol-unknown interventions.",
        comparisons=[])
    for left, right in comparisons:
        arms, positions, strata = build_pair(anchor, candidates, safe, left, right, config["seed"], args.focus)
        name = left + "_vs_" + right
        detail = dict(name=name, left=left, right=right, replacement_slots=len(positions), strata=strata,
                      status="prepared" if positions else "unavailable_no_matched_supply",
                      interpretation=report["interpretation"])
        if positions:
            path = args.out / name
            path.mkdir()
            anchor.to_parquet(path / "external_anchor.parquet", index=False)
            for filename in ("official.parquet", "labels.json", "prompt_folds.json"):
                shutil.copy2(args.data_dir / filename, path / filename)
            for variant, condition in (("baseline", left), ("mix20", right)):
                arms[condition].to_parquet(path / f"external_{variant}.parquet", index=False)
            relation_positions = sorted(relations)
            columns = ["id", "text", "model", "prompt_id"]
            if not separate and any(not arms[c].loc[relation_positions, columns].equals(anchor.loc[relation_positions, columns])
                                    for c in arms):
                raise ValueError("An intervention changed an anchor relation.")
            paired = anchor.loc[positions, ["id", "model", "word_bin"]].rename(columns={"id": "training_slot_id"})
            for condition in (left, right):
                paired[condition + "_id"] = arms[condition].loc[positions, "id"].to_numpy()
                paired[condition + "_source"] = arms[condition].loc[positions, "source_dataset"].to_numpy()
            paired.to_parquet(path / "replacement_slots.parquet", index=False)
            manifest = dict(detail, policy="q8_protocol_fixed_relation_v1",
                            parent_manifest_path=str((args.data_dir / "manifest.json").resolve()),
                            parent_manifest_sha256=report["input_manifest_sha256"],
                            anchor_plan=dict(seed=config["seed"], epochs=config["external_epochs"],
                                             batch_size=config["batch_size"], accumulation=config["accumulation"],
                                             max_updates=args.max_updates, mode=config["centered_plan"]),
                            labels=json.loads((path / "labels.json").read_text()),
                            files_sha256={p.name: file_sha(p) for p in sorted(path.iterdir()) if p.is_file()},
                            evidence_ledger=os.path.relpath(evidence, path.resolve()),
                            evidence_ledger_sha256=report['evidence_ledger_sha256'])
            write_json(path / "manifest.json", manifest)
            # Scope is a separately bound annotation. Adding it cannot change
            # this manifest's data identity or the original experimental slots.
            from .q8_source_scope import build_source_scope
            scope = build_source_scope(path, exposure=exposure)
            detail['display_name'] = scope['display_name']
            detail['changed_text_rows'] = scope['summary']['changed_text_rows']
            detail['changed_fraction'] = scope['summary']['changed_fraction']
            detail['directly_replaced_labels'] = scope['summary']['directly_replaced_labels']
            detail['intervention_scope'] = {key: scope[key] for key in
                                           ('status', 'display_name', 'manifest_path', 'manifest_sha256')}
            detail["data_dir"] = str(path.resolve())
        report["comparisons"].append(detail)
    write_json(args.out / "controls.json", report)
    print(json.dumps({key: report[key] for key in ("actual_centered_unique", "safe_actual_plain_slots", "candidate_counts")}, ensure_ascii=False))
    print(json.dumps([{k: row[k] for k in ("name", "display_name", "replacement_slots", "changed_text_rows", "changed_fraction", "status")
                       if k in row} for row in report["comparisons"]], ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/q8_protocol"))
    parser.add_argument("--config", type=Path, default=Path("configs/q8.toml"))
    parser.add_argument("--experiment", help="Named Q8 experiment; defaults to source for data preparation")
    parser.add_argument("--out", type=Path, default=Path("data/q8_protocol_controls"))
    parser.add_argument("--max-updates", type=int, default=2000)
    parser.add_argument("--comparison", action="append", choices=[a + '_vs_' + b for a, b in COMPARISONS],
                        help="Prepare only this condition pair; repeat for more pairs. Default: all pairs.")
    parser.add_argument("--reference-evidence", action="store_true",
                        help="Reference the immutable input ledger by relative path instead of copying it.")
    parser.add_argument("--focus", nargs="+", default=list(FOCUS))
    build(parser.parse_args())


if __name__ == "__main__":
    main()
