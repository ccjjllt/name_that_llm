"""Prepare a controlled q8-style baseline and a class/length-matched data update."""

import argparse
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold

from .q8_isolation import screen_pool_risks


STRONG_EVIDENCE = ("own_bank", "own_text", "other_text")
WORD_BINS = ("<60", "60-99", "100-159", ">=160")
MISSING_METADATA = {"", "none", "nan", "unknown", "unspecified", "n/a", "unverified"}


def text_key(text):
    """Normalize only the duplicate key; training always receives original text."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def add_text_metadata(frame):
    frame = frame.copy()
    frame["_key"] = frame.text.fillna("").map(text_key)
    frame["n_words"] = frame.text.fillna("").str.split().str.len()
    frame["word_bin"] = pd.cut(
        frame.n_words, [-1, 59, 99, 159, np.inf], labels=WORD_BINS,
    ).astype(str)
    return frame


def prepare_official(train_path):
    frame = pd.read_csv(train_path, dtype=str, keep_default_na=False)
    frame = add_text_metadata(frame[["id", "prompt_id", "text", "model"]])
    if not frame.id.is_unique or frame.text.str.strip().eq("").any():
        raise ValueError("Official IDs must be unique and official text must be nonempty.")
    labels = sorted(frame.model.unique())
    if len(labels) != 18:
        raise ValueError("Expected the competition's 18 model labels.")
    frame["fold"] = np.int8(-1)
    for fold, (_, valid) in enumerate(GroupKFold(n_splits=5).split(frame, groups=frame.prompt_id)):
        frame.loc[valid, "fold"] = fold
    frame["group_is_strong"] = True
    frame["group_eligible"] = True
    frame["group_evidence"] = "official"
    frame["source_dataset"] = "official"
    frame["origin"] = "official"
    return frame, labels


def filter_new_pool(frame, isolated, labels):
    """Apply new-pool quality requirements after joint raw-pool isolation."""
    required = ["prompt_canon", "prompt_evidence", "source_dataset", "source_license"]
    if any(column not in frame for column in required):
        raise ValueError(f"New pool must include {required}.")
    documented = frame[["source_dataset", "source_license"]].apply(
        lambda col: ~col.fillna("").str.strip().str.casefold().isin(MISSING_METADATA)
    ).all(axis=1)
    valid = (
        frame.model.isin(labels) & frame.text.fillna("").str.strip().ne("")
        & documented & ~isolated
    )
    result = frame.loc[valid].copy()
    result["group_is_strong"] = (
        result.prompt_evidence.isin(STRONG_EVIDENCE)
        & result.prompt_canon.fillna("").str.strip().ne("")
    )
    result = result.sort_values("group_is_strong", ascending=False, kind="stable").drop_duplicates("_key")
    return result.reset_index(drop=True)


def clean_new_pool(path, official, labels):
    """Inspect the new pool alone; prepare_pools uses both raw pools jointly."""
    frame = add_text_metadata(pd.read_parquet(path))
    masks, _, summary = screen_pool_risks({"new": frame}, official)
    result = filter_new_pool(frame, masks["new"], labels)
    return result, {**summary["pools"]["new"], "retained_rows": len(result)}


def assign_baseline_groups(baseline, newer):
    """Share trusted canonical IDs without guessing through ambiguous old keys."""
    strong = newer.loc[newer.group_is_strong, ["_key", "model", "prompt_canon"]]
    shared = baseline[["_key", "model", "prompt_key"]].merge(strong, on=["_key", "model"])
    nonempty = shared.prompt_key.str.strip().ne("") & ~shared.prompt_key.str.endswith(":")
    key_counts = shared.loc[nonempty].groupby("prompt_key").prompt_canon.nunique()
    unambiguous = key_counts.index[key_counts.eq(1)]
    mapping = (shared.loc[shared.prompt_key.isin(unambiguous)]
               .drop_duplicates("prompt_key").set_index("prompt_key").prompt_canon)
    direct = strong.set_index("_key").prompt_canon
    baseline["prompt_canon"] = baseline._key.map(direct).fillna("")
    exact = baseline.prompt_canon.ne("")
    baseline["group_evidence"] = np.where(exact, "canonical_direct", "row")
    inherited = baseline.prompt_key.map(mapping).fillna("")
    propagated = ~exact & inherited.ne("")
    baseline.loc[propagated, "prompt_canon"] = inherited[propagated]
    baseline.loc[propagated, "group_evidence"] = "canonical_mapped"
    canonical = baseline.prompt_canon.ne("")
    ambiguous = baseline.prompt_key.isin(key_counts.index[key_counts.gt(1)])
    legacy = (
        ~canonical & ~ambiguous & baseline.prompt_key.str.strip().ne("")
        & ~baseline.prompt_key.str.endswith(":")
    )
    baseline["prompt_id"] = "row:" + baseline.id
    baseline.loc[legacy, "prompt_id"] = "legacy:" + baseline.loc[legacy, "prompt_key"]
    baseline.loc[legacy, "group_evidence"] = "legacy"
    baseline.loc[canonical, "prompt_id"] = "canonical:" + baseline.loc[canonical, "prompt_canon"]
    baseline["group_is_strong"] = canonical
    baseline["group_eligible"] = canonical | legacy
    return {
        "unambiguous_old_keys": len(unambiguous), "ambiguous_old_keys": int(key_counts.gt(1).sum()),
        "direct_canonical_rows": int(exact.sum()), "propagated_canonical_rows": int(propagated.sum()),
        "legacy_rows": int(legacy.sum()), "independent_rows": int((~canonical & ~legacy).sum()),
    }


def select_grouped(frame, quotas, rng, prefer_groups=()):
    """Select exact per-stratum quotas using a shared randomized group order."""
    groups = frame.prompt_id.drop_duplicates().to_numpy()
    ranks = pd.Series(rng.permutation(len(groups)), index=groups)
    ranked = frame.assign(_rank=frame.prompt_id.map(ranks), _tie=rng.permutation(len(frame)))
    ranked.loc[ranked.prompt_id.isin(prefer_groups), "_rank"] -= len(groups)
    ranked = ranked.sort_values(["_rank", "_tie"], kind="stable")
    stratum = pd.MultiIndex.from_frame(ranked[["model", "word_bin"]])
    available = quotas.reindex(stratum, fill_value=0).to_numpy()
    ordinal = ranked.groupby(["model", "word_bin"]).cumcount().to_numpy()
    return ranked.loc[ordinal < available].drop(columns=["_rank", "_tie"])


def corpus_stats(frame):
    eligible = frame.loc[frame.group_eligible]
    labels_per_group = eligible.groupby("prompt_id").model.nunique()
    result = {
        "rows": len(frame), "unique_ids": int(frame.id.nunique()),
        "rows_per_model": frame.groupby("model").size().to_dict(),
        "rows_per_word_bin": frame.groupby("word_bin").size().to_dict(),
        "rows_per_source": frame.groupby("source_dataset").size().to_dict(),
        "rows_per_origin": frame.groupby("origin").size().to_dict(),
        "group_evidence_rows": frame.groupby("group_evidence").size().to_dict(),
        "strong_group_rows": int(frame.group_is_strong.sum()),
        "eligible_group_rows": len(eligible),
        "same_label_extra_generations": int(eligible.duplicated(["prompt_id", "model"]).sum()),
    }
    for minimum in (2, 3):
        groups = labels_per_group.index[labels_per_group.ge(minimum)]
        result[f"groups_with_{minimum}plus_labels"] = len(groups)
        result[f"rows_in_{minimum}plus_label_groups"] = int(frame.prompt_id.isin(groups).sum())
    return result


def prepare_pools(pool_path, new_pool_path, official, labels, seed=42, mix_fraction=0.2,
                  audit_dir=None):
    if not 0 <= mix_fraction <= 1:
        raise ValueError("mix_fraction must be between 0 and 1.")
    baseline = add_text_metadata(pd.read_parquet(pool_path))
    if not baseline.model.isin(labels).all() or baseline.text.fillna("").str.strip().eq("").any():
        raise ValueError("Original pool contains an unknown label or empty text.")
    if baseline._key.duplicated().any():
        raise ValueError("Original pool must have unique text.")
    original_keys = set(baseline._key)
    baseline["id"] = baseline._key.map(lambda value: "text:" + hashlib.sha256(value.encode()).hexdigest())
    baseline["prompt_key"] = baseline.prompt_key.fillna("").astype(str).str.strip()
    baseline["source_dataset"] = baseline.src
    baseline["origin"] = "q8v5_original"
    raw_newer = add_text_metadata(pd.read_parquet(new_pool_path))
    masks, ledger, isolation = screen_pool_risks(
        {"original": baseline, "new": raw_newer}, official)
    baseline = baseline.loc[~masks["original"]].copy()
    newer = filter_new_pool(raw_newer, masks["new"], labels)
    if set(baseline.model) != set(labels):
        raise ValueError("Prompt isolation must retain every competition class.")
    if audit_dir is not None:
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(audit_dir / "isolated_rows.csv", index=False)
        (audit_dir / "isolation.json").write_text(
            json.dumps(isolation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    screening = {**isolation["pools"]["new"], "retained_rows": len(newer)}
    grouping = assign_baseline_groups(baseline, newer)
    # Excluded originals must never return as purportedly new examples.
    new_only = newer.loc[~newer._key.isin(original_keys) & newer.group_is_strong].copy()
    new_only["id"] = new_only._key.map(lambda value: "text:" + hashlib.sha256(value.encode()).hexdigest())
    new_only["prompt_key"] = new_only.prompt_canon
    new_only["prompt_id"] = "canonical:" + new_only.prompt_canon
    new_only["group_eligible"] = True
    new_only["group_evidence"] = "canonical_direct"
    new_only["origin"] = "open_pool_new"
    new_only["w_host"] = np.float32(1)
    columns = ["id", "text", "model", "prompt_id", "group_is_strong", "group_eligible",
               "group_evidence", "prompt_key", "prompt_canon", "source_dataset", "n_words",
               "word_bin", "origin", "w_host"]
    baseline = baseline[columns].sort_values("id").reset_index(drop=True)
    new_only = new_only[columns].sort_values("id").reset_index(drop=True)
    counts = baseline.groupby(["model", "word_bin"]).size()
    quotas = np.floor(counts * mix_fraction).astype(int)
    supply = new_only.groupby(["model", "word_bin"]).size().reindex(counts.index, fill_value=0)
    quotas = pd.Series(np.minimum(quotas, supply), index=counts.index)
    rng = np.random.RandomState(seed)
    added = select_grouped(new_only, quotas, rng)
    retained = select_grouped(baseline, counts - quotas, rng, prefer_groups=set(added.prompt_id))
    mixed = pd.concat([retained, added], ignore_index=True).sort_values("id").reset_index(drop=True)
    actual = mixed.groupby(["model", "word_bin"]).size().reindex(counts.index, fill_value=0)
    if not actual.equals(counts) or not mixed.id.is_unique:
        raise ValueError("Mix must retain exact class/length quotas and unique text IDs.")
    strata = pd.DataFrame({"baseline_rows": counts, "new_available": supply, "replacement_rows": quotas})
    strata["replacement_fraction"] = strata.replacement_rows / strata.baseline_rows
    stats = {
        "prompt_isolation": isolation, "new_pool_screening": screening, "group_mapping": grouping,
        "baseline": corpus_stats(baseline), "mix20": corpus_stats(mixed),
        "new_eligible_unique_rows": len(new_only), "new_selected_rows": len(added),
        "actual_new_fraction": len(added) / len(baseline), "seed": seed,
        "maximum_stratum_replacement_fraction": mix_fraction,
        "strata": strata.reset_index().to_dict(orient="records"),
        "selection": "Quotas are recomputed from the isolated baseline. New means absent from the raw original pool. Shared random canonical-group priority, then exact model/word-bin quotas. Retained originals prioritize groups receiving new examples. Group completeness is measured rather than enforced by duplicating rows.",
        "grouping": "Canonical IDs come from strong evidence and unambiguous old-key mappings. Valid remaining legacy keys are training-eligible but are not marked canonical-strong. Empty, empty-suffix and ambiguous unmapped keys use independent row groups.",
        "weighting": "w_host is retained as provenance only. Its original training use is unknown. Neither w_host nor published match_weight contributes to selection or loss weights.",
        "evaluation_scope": "Public upstream pools used official-informed cleaning. These prepared data support controlled development comparisons, not proof of fully independent external validation.",
    }
    return baseline, mixed, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--new-pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    official, labels = prepare_official(args.train)
    baseline, mixed, stats = prepare_pools(
        args.pool, args.new_pool, official, labels, audit_dir=args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    official.drop(columns="_key").to_parquet(args.out / "official.parquet", index=False)
    baseline.to_parquet(args.out / "external_baseline.parquet", index=False)
    mixed.to_parquet(args.out / "external_mix20.parquet", index=False)
    fold_map = official.groupby("prompt_id").fold.first().astype(int).to_dict()
    manifest = {
        "labels": labels, "official_rows": len(official), "official_prompts": len(fold_map),
        "official_rows_per_fold": official.groupby("fold").size().to_dict(),
        "fold_method": "Own GroupKFold(n_splits=5), shuffle=False. This is not the author's q8v5 adapter mapping.",
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
        "input_sha256": {"train": file_sha(args.train), "pool": file_sha(args.pool), "new_pool": file_sha(args.new_pool)},
        "input_paths": {"train": str(args.train.resolve()), "pool": str(args.pool.resolve()), "new_pool": str(args.new_pool.resolve())},
        "preparation_version": "q8_joint_prompt_isolation_v1",
        "preparation_code_sha256": {name: file_sha(Path(__file__).with_name(name))
                                    for name in ("q8_data.py", "q8_isolation.py")},
        "external": stats,
    }
    for filename, value in (("labels.json", labels), ("prompt_folds.json", fold_map), ("manifest.json", manifest)):
        (args.out / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"official": len(official), "baseline": len(baseline), "mix20": len(mixed),
                      "new_rows": stats["new_selected_rows"], "new_fraction": stats["actual_new_fraction"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
