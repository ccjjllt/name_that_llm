"""Conservative prompt-risk isolation on raw pools, before sampling or grouping.

An official-looking ID is a quarantine signal, not proof of an identical prompt.
Text similarity scores are deliberately not used to decide exclusions.
"""

import hashlib
import re

import numpy as np
import pandas as pd


RISK_FLAGS = ("flag_train_prompt_regen", "flag_near_train_prompt", "flag_gemini37_routing_risk")
MISSING_KEYS = {"", "none", "nan", "null", "unknown", "unspecified", "n/a", "unverified"}
PROMPT_COLUMNS = ("prompt_key", "prompt_group", "prompt_canon")


def valid_group_key(values):
    return ~values.str.casefold().isin(MISSING_KEYS) & ~values.str.endswith(":")


def screen_pool_risks(pools, official):
    """Return per-pool exclusion masks, a row ledger, and joint screening counts.

    Pools and official must already have normalized ``_key`` values. Group keys
    retain their full existing namespace and are scoped by pool/column; canonical
    IDs and exact normalized text bridge pools. Ambiguous keys cannot spread risk, whereas a
    direct official-ID match remains a seed regardless of key ambiguity.
    """
    parts = []
    for name, frame in pools.items():
        part = frame[["_key", "model"]].reset_index(drop=True).copy()
        part["pool"] = name
        part["source_row"] = np.arange(len(frame))
        source = "source_dataset" if "source_dataset" in frame else "src"
        part["source_dataset"] = frame[source].fillna("").astype(str).to_numpy()
        for column in (*PROMPT_COLUMNS, "prompt_evidence"):
            part[column] = (frame[column].fillna("").astype(str).str.strip().to_numpy()
                            if column in frame else "")
        for flag in RISK_FLAGS:
            if flag in frame:
                if not pd.api.types.is_bool_dtype(frame[flag]):
                    raise ValueError(f"Expected boolean {flag} in {name}.")
                part[flag] = frame[flag].fillna(True).to_numpy(dtype=bool)
            else:
                part[flag] = False
        parts.append(part)
    rows = pd.concat(parts, ignore_index=True)
    tokens = "|".join(re.escape(value.casefold()) for value in official.prompt_id.unique())
    pattern = rf"(?<![a-z0-9])(?:{tokens})(?![a-z0-9])"
    matches = rows[list(PROMPT_COLUMNS)].apply(
        lambda col: col.str.casefold().str.findall(pattern))
    rows["official_prompt_markers"] = [
        ",".join(sorted(set(sum(values, [])))) for values in matches.itertuples(index=False, name=None)
    ]
    direct = pd.DataFrame({
        "official_text": rows._key.isin(official._key),
        "conflicting_text_label": rows.groupby("_key").model.transform("nunique").gt(1),
        "official_prompt_marker": rows.official_prompt_markers.ne(""),
        **{flag: rows[flag] for flag in RISK_FLAGS},
    })
    rows["seed_reasons"] = ""
    for reason in direct:
        rows.loc[direct[reason], "seed_reasons"] += reason + "|"
    rows["seed_reasons"] = rows.seed_reasons.str.rstrip("|")
    seeds = direct.any(axis=1)
    blocked = seeds.copy()
    rows["isolation_reason"] = rows.seed_reasons
    rows["propagation_round"] = np.where(blocked, 0, -1)

    canonical = rows.prompt_canon.where(valid_group_key(rows.prompt_canon), "")
    # Use RAW canonical associations, including the rows about to be excluded.
    text_canons = pd.DataFrame({"_key": rows._key, "canon": canonical})
    text_canons = text_canons.loc[text_canons.canon.ne("")].drop_duplicates()
    links = {"shared_text": rows._key, "canonical": canonical}
    key_summary = {}
    for column in ("prompt_key", "prompt_group"):
        valid = valid_group_key(rows[column])
        # The original pool reuses fully namespaced keys across source copies.
        # Adding source_dataset here would incorrectly sever those prompt links.
        scoped = rows.pool + "\x1f" + rows[column]
        associations = pd.DataFrame({"key": scoped[valid], "_key": rows.loc[valid, "_key"]})
        counts = associations.merge(text_canons, on="_key").groupby("key").canon.nunique()
        ambiguous = scoped.isin(counts.index[counts.gt(1)])
        links[column] = scoped.where(valid & ~ambiguous, "")
        key_summary[column] = {
            "ambiguous_keys_not_propagated": int(counts.gt(1).sum()),
            "ambiguous_rows_not_propagated": int((valid & ambiguous).sum()),
            "invalid_rows_not_propagated": int((~valid).sum()),
        }

    rounds = []
    while True:
        additions = pd.DataFrame({
            reason: values.ne("") & values.isin(set(values[blocked]) - {""})
            for reason, values in links.items()
        })
        added = ~blocked & additions.any(axis=1)
        if not added.any():
            break
        rows.loc[added, "isolation_reason"] = additions.loc[added].apply(
            lambda row: "|".join(row.index[row]), axis=1)
        rows.loc[added, "propagation_round"] = len(rounds) + 1
        rounds.append(rows.loc[added].groupby("pool").size().to_dict())
        blocked |= added

    masks = {name: blocked[rows.pool.eq(name)].to_numpy() for name in pools}
    summary = {
        "policy": "joint_raw_pools_v1",
        "interpretation": "Conservative prompt-risk quarantine; official-looking IDs and canonical links do not prove identical generation prompts. No similarity threshold is used.",
        "propagation": "Fixed point over normalized text, valid canonical IDs, and complete namespaced prompt keys/groups with at most one raw linked canonical. Pool/column namespaces remain separate. Empty/placeholder/empty-suffix and ambiguous group keys do not propagate.",
        "group_keys": key_summary,
        "propagation_rounds": rounds,
        "pools": {},
    }
    for name in pools:
        selected = rows.pool.eq(name)
        summary["pools"][name] = {
            "input_rows": int(selected.sum()),
            "seed_rows": int((selected & seeds).sum()),
            "isolated_rows": int((selected & blocked).sum()),
            "propagated_rows": int((selected & blocked & ~seeds).sum()),
            "official_text_overlaps": int(direct.loc[selected, "official_text"].sum()),
            "conflicting_text_rows": int(direct.loc[selected, "conflicting_text_label"].sum()),
            "explicit_official_prompt_rows": int(direct.loc[selected, "official_prompt_marker"].sum()),
            "blocked_canons": int(canonical[selected & blocked].replace("", pd.NA).nunique()),
        }
    ledger = rows.loc[blocked].copy()
    ledger.insert(2, "text_id", ledger._key.map(
        lambda value: "text:" + hashlib.sha256(value.encode()).hexdigest()))
    # Text is recoverable from source_row; avoid copying full passages to CSV.
    ledger = ledger.drop(columns=["_key", *RISK_FLAGS])
    return masks, ledger, summary
