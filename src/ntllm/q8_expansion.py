"""Build a larger, source-audited Q8 pool without changing the frozen reference.

Raw source evidence is joined before filtering. Known official regenerations can
therefore remove copies hidden behind opaque keys in older aggregate datasets.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from .q8_data import (STRONG_EVIDENCE, add_text_metadata, assign_baseline_groups,
                     corpus_stats, file_sha, select_grouped, text_key)
from .q8_isolation import screen_pool_risks


LABEL_ALIASES = {"gpt-5.5-2026-04-23": "gpt-5.5", "qwen3.8-max-0902": "qwen3.8-max"}
POLICY = "q8_expansion_raw_provenance_v2"
TRAIN_COLUMNS = ["id", "text", "model", "prompt_id", "group_is_strong", "group_eligible",
                 "group_evidence", "prompt_key", "prompt_canon", "source_dataset", "n_words",
                 "word_bin", "origin", "w_host", "source_pool", "source_version", "source_row",
                 "source_record_id", "source_license", "license_basis", "provider_route",
                 "served_provider", "served_model_id", "source_split", "provenance_tier"]


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def served_label(value):
    value = str(value).rsplit("/", 1)[-1].removesuffix("@default")
    return LABEL_ALIASES.get(value, value)


def public_license(value):
    value = str(value).lower().replace("_", "-")
    if any(token in value for token in ("noncommercial", "non-commercial", "-nc", "-nd", "no derivatives")):
        return False
    return any(token in value for token in ("cc by 4.0", "cc-by-4.0", "cc by-sa 4.0", "cc-by-sa-4.0", "cc0"))


def base_annotations(frame, pool, version):
    frame = add_text_metadata(frame.reset_index(drop=True))
    frame["source_row"] = np.arange(len(frame), dtype=np.int32)
    frame["source_record_id"] = frame["id"].astype(str) if "id" in frame else frame.source_row.astype(str)
    frame["id"] = frame._key.map(lambda key: "text:" + hashlib.sha256(key.encode()).hexdigest())
    frame["source_pool"], frame["source_version"] = pool, version
    frame["admission_reasons"] = ""
    for column in ("prompt_key", "prompt_group", "prompt_canon", "prompt_evidence", "source_license",
                   "license_basis", "provider_route", "served_provider", "served_model_id", "source_split"):
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    for flag in ("flag_train_prompt_regen", "flag_near_train_prompt", "flag_gemini37_routing_risk"):
        if flag not in frame:
            frame[flag] = False
    if "w_host" not in frame:
        frame["w_host"] = 1.0
    return frame


def reject(frame, mask, reason):
    frame.loc[np.asarray(mask, dtype=bool), "admission_reasons"] += reason + "|"


def propagate_row_evidence(pools):
    """A re-export cannot erase adverse evidence for the same text and label.

    Route uncertainty is not a prompt-level risk: other models' answers to the
    same prompt remain eligible. No current source has independent, stronger
    row-level routing evidence that would override these exclusions.
    """
    policies = {
        "shared_route_uncertainty": ("gemini37_unverified_route",),
        "shared_served_identity_mismatch": ("served_identity_mismatch",),
        "shared_incomplete_generation": ("incomplete_generation", "source_is_truncated"),
    }
    report = {}
    for shared_reason, seed_reasons in policies.items():
        evidence = []
        for name, frame in pools.items():
            selected = frame.admission_reasons.str.split("|").map(
                lambda reasons: bool(set(seed_reasons).intersection(reasons)))
            part = frame.loc[selected, ["_key", "model", "source_row", "id"]].copy()
            part["source_pool"] = name
            evidence.append(part)
        seeds = pd.concat(evidence, ignore_index=True)
        identities = pd.MultiIndex.from_frame(seeds[["_key", "model"]].drop_duplicates())
        affected = {}
        for name, frame in pools.items():
            match = pd.MultiIndex.from_frame(frame[["_key", "model"]]).isin(identities)
            reject(frame, match, shared_reason)
            affected[name] = int(match.sum())
        report[shared_reason] = {
            "seed_records": len(seeds), "unique_text_label_identities": len(identities),
            "affected_rows_by_pool": affected,
            "additional_source_copies": sum(affected.values()) - len(seeds),
            "seed_rows": seeds.drop(columns="_key").to_dict(orient="records"),
        }
    return report


def strong_map(frame):
    return frame.loc[frame.prompt_evidence.isin(STRONG_EVIDENCE), ["_key", "model", "prompt_canon"]]


def adapt_extended(frame, prompts, open_pool):
    frame = frame.rename(columns={"split": "source_split"}).copy()
    frame["source_dataset"] = "prvsiyan/prvsiyans-extended-name-that-llm-dataset"
    frame = base_annotations(frame, "extended_v8", 8)
    body_by_id = prompts.set_index("prompt_id").prompt_body
    if not body_by_id.index.is_unique:
        raise ValueError("Extended prompt IDs need unique raw prompt bodies.")
    bodies = frame.prompt_id.map(body_by_id).fillna("")
    frame["prompt_key"] = "prvsiyan:" + frame.prompt_id.astype(str)
    frame["prompt_group"] = frame.prompt_key
    frame["prompt_canon"] = bodies.map(
        lambda value: "prvs-body:" + hashlib.sha256(text_key(value).encode()).hexdigest() if value else "")
    # Identical answers with the same label can link a source prompt to an
    # existing canonical. Multiple canonical associations remain unmerged.
    linked = frame[["_key", "model", "prompt_key"]].merge(strong_map(open_pool), on=["_key", "model"])
    counts = linked.groupby("prompt_key").prompt_canon.nunique()
    mapping = linked[linked.prompt_key.isin(counts[counts.eq(1)].index)].drop_duplicates("prompt_key").set_index("prompt_key").prompt_canon
    inherited = frame.prompt_key.map(mapping)
    frame.loc[inherited.notna(), "prompt_canon"] = inherited.dropna()
    frame["prompt_evidence"] = np.where(bodies.ne(""), "own_text", "none")
    frame["flag_train_prompt_regen"] = frame.source_split.eq("train_regen")
    frame["source_license"] = "CC BY 4.0"
    frame["license_basis"] = "source dataset v8; prompt-level licence retained in raw prompts.csv"
    frame["origin"] = "extended_v8"
    frame["provenance_tier"] = "api_logged_source"
    reject(frame, bodies.eq(""), "missing_prompt_body")
    reject(frame, ~frame.source_split.isin(["new", "new2", "proxy_v1", "new_kimi_nothink", "train_regen", "pilot_range"]), "unknown_source_split")
    reject(frame, frame.source_split.eq("pilot_range"), "shared_length_range_protocol")
    reject(frame, ~frame.served_model_id.map(served_label).eq(frame.model), "served_identity_mismatch")
    for flag in ("is_empty", "is_refusal", "is_truncated"):
        if not pd.api.types.is_bool_dtype(frame[flag]):
            raise ValueError(f"Expected boolean source quality flag {flag}.")
        reject(frame, frame[flag], "source_" + flag)
    reject(frame, frame.finish_reason_norm.ne("stop"), "incomplete_generation")
    vertex = frame.provider_route.eq("google") & frame.served_provider.str.contains("Vertex", case=False)
    reject(frame, frame.model.eq("gemini-3.7-flash") & ~vertex, "gemini37_unverified_route")
    return frame


def adapt_exact_source(frame, known_canons):
    frame = frame.rename(columns={"route": "provider_route", "provider": "served_provider",
                                  "model_seen": "served_model_id"}).copy()
    frame["source_dataset"] = "zeminglii/ntl-own-exact-route-generations"
    frame = base_annotations(frame, "exact_route_v2", 2)
    frame["source_split"] = frame.batch
    frame["prompt_key"] = frame.prompt_canon
    frame["prompt_group"] = "zeminglii:" + frame.prompt_canon
    frame["prompt_evidence"] = np.where(frame.prompt_canon.isin(known_canons), "own_bank", "id_only")
    frame["source_license"] = "CC BY 4.0"
    frame["license_basis"] = "source dataset v2; upstream Open Pool prompt terms also apply"
    frame["origin"] = "exact_route_v2"
    frame["provenance_tier"] = "served_id_recorded"
    reject(frame, ~frame.served_model_id.map(served_label).eq(frame.model), "served_identity_mismatch")
    reject(frame, frame.finish_reason.ne("stop"), "incomplete_generation")
    # The card says Vertex, but these rows only record provider='Google'. Do not
    # infer a verified Vertex route for new 3.7 texts from that ambiguous field.
    reject(frame, frame.model.eq("gemini-3.7-flash") & ~frame.served_provider.str.contains("Vertex", case=False), "gemini37_unverified_route")
    return frame


def assign_new_groups(frame):
    frame = frame.copy()
    strong = frame.prompt_evidence.isin(STRONG_EVIDENCE) & frame.prompt_canon.ne("")
    frame["group_is_strong"] = strong
    frame["group_eligible"] = strong
    frame["group_evidence"] = np.where(strong, "canonical_source_evidence", "row")
    frame["prompt_id"] = np.where(strong, "canonical:" + frame.prompt_canon, "row:" + frame.id)
    return frame


def punctuation_key(text):
    return " ".join(re.findall(r"\w+", text_key(text)))


def enrich_protocol_evidence(args, output):
    """Add evidence without reselecting, rewriting or regrouping frozen texts."""
    from .q8_provenance import build_evidence, attach_evidence, confirmed_transformation_pairs

    frozen_manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in frozen_manifest.get("input_sha256", {}).items():
        if hasattr(args, name) and file_sha(getattr(args, name)) != expected:
            raise ValueError(f"Frozen {name} input differs; source-row admission cannot be transferred across versions.")
    sources = {
        "original_v3": {"path": str(args.old_pool), "version": 3},
        "open_pool_v15": {"path": str(args.new_pool), "version": 15},
        "extended_v8": {"path": str(args.extended), "version": 8},
        "exact_route_v2": {"path": str(args.exact_source), "version": 2},
    }
    supplement_path = getattr(args, "evidence_sources", None)
    supplement = json.loads(supplement_path.read_text()) if supplement_path else {}
    sources.update(supplement.get("sources", {}))
    evidence, report = build_evidence(sources, prompt_bank=args.prompt_bank,
                                      extended_prompts=args.extended_prompts,
                                      source_cards=supplement.get("source_cards", {}))
    # The original admission ledger remains authoritative for dataset selection.
    # Supplementary source logs explain observations; they do not admit new text.
    admission_path = output / "provenance_rows.parquet"
    if admission_path.exists():
        admission = pd.read_parquet(admission_path)
        fields = ["source_pool", "source_row", "model", "id", "admission_reasons", "admitted"]
        evidence = evidence.merge(admission[fields], on=fields[:4], how="left", validate="many_to_one")
    evidence.to_parquet(output / "protocol_evidence.parquet", index=False)
    pairs, pairing_report = confirmed_transformation_pairs(evidence)
    pairs.to_parquet(output / "transformation_pairs.parquet", index=False)
    report["transformation_pairs"] = pairing_report
    report["frozen_membership"] = {}
    for variant in ("baseline", "mix20", "expanded"):
        path = output / f"external_{variant}.parquet"
        before = pd.read_parquet(path)
        after = attach_evidence(before, evidence)
        if not after[before.columns].equals(before):
            raise ValueError("Evidence enrichment changed a pre-existing training field.")
        after.to_parquet(path, index=False)
        report["frozen_membership"][variant] = {
            "rows": len(after), "preexisting_columns_unchanged": True,
            "research_conditions": after.research_condition.value_counts().to_dict(),
            "focus_conditions": {str(label): group.research_condition.value_counts().to_dict()
                                 for label, group in after.loc[after.model.isin(
                                     ["kimi-k3", "glm-5.3", "glm-5.3-flash", "claude-fable-5"])].groupby("model")},
            "missing_exact_evidence": int(after.evidence_record_ids.eq("[]").sum()),
        }
    json_write(output / "protocol_evidence_report.json", report)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evidence_ledger"] = "protocol_evidence.parquet"
    manifest["evidence_ledger_sha256"] = file_sha(output / "protocol_evidence.parquet")
    manifest["protocol_evidence_report"] = report
    manifest["protocol_enrichment_code_sha256"] = {
        name: file_sha(Path(__file__).with_name(name)) for name in ("q8_expansion.py", "q8_provenance.py")}
    json_write(manifest_path, manifest)
    return report


def enrich_frozen(args):
    if args.out.exists():
        raise FileExistsError("Evidence enrichment needs a new output directory; retain frozen historical data.")
    source = args.enrich_prepared
    manifest = json.loads((source / "manifest.json").read_text())
    shutil.copytree(source, args.out)
    manifest["frozen_parent_path"] = str(source.resolve())
    manifest["frozen_parent_manifest_sha256"] = file_sha(source / "manifest.json")
    json_write(args.out / "manifest.json", manifest)
    report = enrich_protocol_evidence(args, args.out)
    for filename in ("official.parquet", "labels.json", "prompt_folds.json"):
        if file_sha(source / filename) != file_sha(args.out / filename):
            raise ValueError("Protocol enrichment changed official data or folds.")
    print(json.dumps(report["frozen_membership"], ensure_ascii=False))
    return report


def build(args):
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    labels = json.loads((args.reference_dir / "labels.json").read_text())
    official = add_text_metadata(pd.read_parquet(args.reference_dir / "official.parquet"))
    original = pd.read_parquet(args.old_pool)
    original["source_dataset"] = original.src
    original = base_annotations(original, "original_v3", 3)
    original["origin"] = "q8v5_original"
    original["provenance_tier"] = "legacy_reference"
    catalog = pd.read_csv(args.source_catalog, keep_default_na=False).set_index("source_dataset").source_license
    original["source_license"] = original.source_dataset.map(catalog).fillna("upstream terms; legacy attribution unresolved")
    original["license_basis"] = "Open Pool v15 source catalog where available; original aggregate states upstream terms apply"
    newer = base_annotations(pd.read_parquet(args.new_pool), "open_pool_v15", 15)
    newer["origin"] = "open_pool_v15"
    newer["provenance_tier"] = "curated_public_pool"
    newer["license_basis"] = "row source_license; aggregate CC BY-SA 4.0"
    extended = adapt_extended(pd.read_csv(args.extended, keep_default_na=False),
                              pd.read_csv(args.extended_prompts, keep_default_na=False), newer)
    prompt_bank = pd.read_parquet(args.prompt_bank)
    exact = adapt_exact_source(pd.read_csv(args.exact_source, keep_default_na=False), set(prompt_bank.prompt_canon))
    pools = {"original_v3": original, "open_pool_v15": newer, "extended_v8": extended, "exact_route_v2": exact}
    row_evidence = propagate_row_evidence(pools)

    # Exact word sequences differing only in punctuation are high-specificity
    # copy candidates. Keep their raw text; record, do not delete generic fuzzy matches.
    official_words = set(official.text.map(punctuation_key))
    punctuation_audit = []
    hard_bad_keys = set(extended.loc[extended.admission_reasons.str.contains(
        "source_is_|shared_length_range_protocol|incomplete_generation", regex=True), "_key"]) - {""}
    for name, frame in pools.items():
        reject(frame, ~frame.model.isin(labels), "unknown_label")
        reject(frame, frame.text.fillna("").str.strip().eq(""), "empty_text")
        reject(frame, frame._key.isin(hard_bad_keys), "shared_source_quality_or_protocol_exclusion")
        if name != "original_v3":
            reject(frame, ~frame.source_license.map(public_license), "unverified_public_license")
        word_keys = frame.text.fillna("").map(punctuation_key)
        near_official = word_keys.isin(official_words) & frame.n_words.ge(25) & ~frame._key.isin(official._key)
        frame.loc[near_official, "flag_near_train_prompt"] = True
        punctuation_audit.append({"pool": name, "punctuation_only_official_matches": int(near_official.sum())})
    # Blank/refusal outputs should not act as cross-label or cross-prompt bridges.
    screening = {}
    for name, frame in pools.items():
        screened = frame.copy()
        meaningless = screened._key.eq("") | screened.admission_reasons.str.contains("source_is_refusal", regex=False)
        screened.loc[meaningless, "_key"] = "nonbridge:" + name + ":" + screened.loc[meaningless, "source_row"].astype(str)
        screening[name] = screened
    masks, isolation_ledger, isolation = screen_pool_risks(screening, official)
    for name, frame in pools.items():
        selected = isolation_ledger.pool.eq(name)
        isolation_ledger.loc[selected, "text_id"] = frame.iloc[
            isolation_ledger.loc[selected, "source_row"].to_numpy(dtype=int)].id.to_numpy()
    isolation_ledger.to_csv(output / "isolated_rows.csv", index=False)
    json_write(output / "isolation.json", isolation)

    historical = pd.read_csv(args.reference_dir / "isolated_rows.csv", keep_default_na=False)
    previous_exclusions = set(historical.text_id)
    admitted, provenance = {}, []
    for name, frame in pools.items():
        reject(frame, masks[name], "joint_prompt_or_label_isolation")
        reject(frame, frame.id.isin(previous_exclusions), "previously_isolated_text")
        frame["admission_reasons"] = frame.admission_reasons.str.rstrip("|")
        frame["admitted"] = frame.admission_reasons.eq("")
        admitted[name] = frame.loc[frame.admitted].copy()
        provenance.append(frame[["id", "model", "source_dataset", "source_pool", "source_version",
                                 "source_row", "source_record_id", "source_license", "license_basis",
                                 "origin", "provenance_tier", "prompt_key", "prompt_group", "prompt_canon",
                                 "prompt_evidence", "provider_route", "served_provider", "served_model_id",
                                 "source_split", "admission_reasons", "admitted"]].copy())

    evidence = pd.concat([assign_new_groups(admitted[name]) for name in
                          ("open_pool_v15", "extended_v8", "exact_route_v2")], ignore_index=True)
    evidence = evidence.sort_values("group_is_strong", ascending=False, kind="stable").drop_duplicates("_key")
    baseline = admitted["original_v3"].copy()
    group_mapping = assign_baseline_groups(baseline, evidence)
    baseline = baseline[TRAIN_COLUMNS].sort_values("id").reset_index(drop=True)

    # The existing mix20 remains a same-size class/length control. New means not
    # present in the RAW original, including any now-excluded original text.
    raw_original_ids = set(original.id)
    added_candidates = evidence.loc[~evidence.id.isin(raw_original_ids) & evidence.group_is_strong].copy()
    counts = baseline.groupby(["model", "word_bin"]).size()
    supply = added_candidates.groupby(["model", "word_bin"]).size().reindex(counts.index, fill_value=0)
    quotas = pd.Series(np.minimum(np.floor(counts * .2).astype(int), supply), index=counts.index)
    rng = np.random.RandomState(args.seed)
    added = select_grouped(added_candidates, quotas, rng)
    kept = select_grouped(baseline, counts - quotas, rng, prefer_groups=set(added.prompt_id))
    mixed = pd.concat([kept, added[TRAIN_COLUMNS]], ignore_index=True).sort_values("id").reset_index(drop=True)
    # Preserve reference text/group metadata for shared rows. New weak-prompt
    # rows remain usable for plain CE and never masquerade as centered evidence.
    expanded = pd.concat([baseline, evidence[TRAIN_COLUMNS]], ignore_index=True)
    expanded = expanded.drop_duplicates("id", keep="first").sort_values("id").reset_index(drop=True)
    if set(expanded.model) != set(labels) or not expanded.id.is_unique:
        raise ValueError("Expanded pool must retain all 18 labels with unique normalized text.")
    if not mixed.groupby(["model", "word_bin"]).size().reindex(counts.index).equals(counts):
        raise ValueError("The governed mix20 must match every baseline class/length quota.")
    for name, frame in (("baseline", baseline), ("mix20", mixed), ("expanded", expanded)):
        if frame.id.isin(previous_exclusions).any():
            raise ValueError(f"Historical exclusions reappeared in {name}.")
        frame.to_parquet(output / f"external_{name}.parquet", index=False)
    for filename in ("official.parquet", "labels.json", "prompt_folds.json"):
        shutil.copy2(args.reference_dir / filename, output / filename)

    ledger = pd.concat(provenance, ignore_index=True)
    for name, frame in (("baseline", baseline), ("mix20", mixed), ("expanded", expanded)):
        ledger["in_" + name] = ledger.id.isin(frame.id)
    chosen = set(zip(expanded.source_pool, expanded.source_row))
    ledger["selected_representative"] = [key in chosen for key in zip(ledger.source_pool, ledger.source_row)]
    ledger["deduplicated_copy"] = ledger.admitted & ~ledger.selected_representative
    ledger.to_parquet(output / "provenance_rows.parquet", index=False)
    paths = {name: getattr(args, name) for name in
             ("old_pool", "new_pool", "extended", "extended_prompts", "exact_source", "source_catalog", "prompt_bank")}
    before = pd.read_parquet(args.reference_dir / "external_baseline.parquet")
    removed_reference = before.loc[~before.id.isin(baseline.id)]
    removed_reference.drop(columns="text").to_parquet(output / "removed_reference_rows.parquet", index=False)
    report = {
        "policy": POLICY, "seed": args.seed, "labels": labels,
        "input_sha256": {name: file_sha(path) for name, path in paths.items()},
        "input_paths": {name: str(path.resolve()) for name, path in paths.items()},
        "reference_manifest_sha256": file_sha(args.reference_dir / "manifest.json"),
        "preparation_code_sha256": {name: file_sha(Path(__file__).with_name(name)) for name in
                                    ("q8_expansion.py", "q8_data.py", "q8_isolation.py")},
        "official_rows": len(official), "official_prompts": int(official.prompt_id.nunique()),
        "official_fold_definition": "Byte-identical copy of reference official.parquet and prompt_folds.json.",
        "sources": {name: {"raw_rows": len(frame), "admitted_rows": int(frame.admitted.sum()),
                            "reasons": frame.loc[~frame.admitted, "admission_reasons"].value_counts().to_dict()}
                    for name, frame in pools.items()},
        "group_mapping": group_mapping, "prompt_isolation": isolation,
        "row_evidence_propagation": row_evidence,
        "punctuation_copy_audit": punctuation_audit,
        "baseline": corpus_stats(baseline), "mix20": corpus_stats(mixed), "expanded": corpus_stats(expanded),
        "expanded_new_vs_baseline": int((~expanded.id.isin(baseline.id)).sum()),
        "mix20_new_vs_baseline": int((~mixed.id.isin(baseline.id)).sum()),
        "removed_prior_reference_rows": len(removed_reference),
        "retained_legacy_unresolved_license_rows": int(baseline.source_license.str.contains("unresolved").sum()),
        "new_weak_prompt_plain_only_rows": int((~expanded.group_eligible & ~expanded.id.isin(baseline.id)).sum()),
        "governance_notes": [
            "Raw punctuation, case, spacing and paragraph layout are preserved; normalized keys only govern identity.",
            "New source records carry version, row, licence, route and served-model evidence. Legacy reference rows retain an explicit unresolved-attribution tier where original upstream licence is missing.",
            "Source-confirmed train_regen is excluded jointly, including exact copies behind opaque legacy keys.",
            "Shared-length-range pilot rows and hard source failures are excluded by text identity across copies; markdown/length/style funnel flags are not used as admission gates.",
            "Unverified Gemini 3.7 routes, served identity mismatches and incomplete generations propagate to every normalized-text+model copy, including aggregate re-exports. These exclusions do not spread through prompt groups. No current source provides stronger independent routing evidence overriding them.",
            "No official-informed match_weight, style_score or passes_funnel becomes a training weight or selection score.",
            "Only strong new prompt evidence enables centered supervision. Weak id/centroid evidence remains plain-only.",
            "Normalized exact duplicate copies are coalesced; punctuation-only official copies are quarantined. General semantic near-duplicate detection and upstream-label correctness are not proven.",
            "Public pools have a history of official-informed filtering; the resulting official CV remains a development comparison.",
        ],
    }
    json_write(output / "manifest.json", report)
    enrich_protocol_evidence(args, output)
    print(json.dumps({key: report[key] for key in ("expanded_new_vs_baseline", "mix20_new_vs_baseline", "removed_prior_reference_rows")}
                     | {name: len(frame) for name, frame in (("baseline", baseline), ("mix20", mixed), ("expanded", expanded))}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=Path("data/q8"))
    parser.add_argument("--old-pool", type=Path, default=Path("data/external/nlm-matched-clean-v3/nlm_pool_v5.parquet"))
    parser.add_argument("--new-pool", type=Path, default=Path("data/external/open-pool-v15/ntl_open_pool.parquet"))
    parser.add_argument("--source-catalog", type=Path, default=Path("data/external/open-pool-v15/sources.csv"))
    parser.add_argument("--prompt-bank", type=Path, default=Path("data/external/open-pool-v15/prompts.parquet"))
    parser.add_argument("--extended", type=Path, default=Path("data/external/prvsiyan-extended-v8/passages.csv"))
    parser.add_argument("--extended-prompts", type=Path, default=Path("data/external/prvsiyan-extended-v8/prompts.csv"))
    parser.add_argument("--exact-source", type=Path, default=Path("data/external/zeminglii-exact-v2/passages.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/q8_scale"))
    parser.add_argument("--enrich-prepared", type=Path,
                        help="Copy frozen prepared data to a new --out and add protocol evidence only")
    parser.add_argument("--evidence-sources", type=Path,
                        help="JSON with additional frozen sources and source_cards; no new text is admitted")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.enrich_prepared:
        enrich_frozen(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
