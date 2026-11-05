"""Shared relation batches and explicit target-class auxiliary loss budgets."""

import hashlib

import numpy as np
import pandas as pd

from .q8_sampling import WORD_BINS, prompt_batches


def common_relation_rows(frame, peer):
    """Find identical, eligible relation texts in two class/length-matched arms."""
    labels = sorted(frame.model.unique())
    if len(labels) != 18 or set(peer.model) != set(labels):
        raise ValueError("Both relation arms must contain the same 18 model labels.")
    strata = pd.MultiIndex.from_product([labels, WORD_BINS], names=["model", "word_bin"])
    counts = []
    for arm in (frame, peer):
        if not arm.id.is_unique:
            raise ValueError("Relation arms require unique text IDs.")
        if not arm.word_bin.isin(WORD_BINS).all():
            raise ValueError("Relation arms contain an unknown word-count bin.")
        counts.append(arm.groupby(["model", "word_bin"]).size().reindex(strata, fill_value=0))
    if not counts[0].equals(counts[1]):
        raise ValueError("Shared relation arms must have identical class/length quotas.")
    first, second = frame.set_index("id", drop=False), peer.set_index("id", drop=False)
    ids = sorted(set(first.index) & set(second.index))
    left, right = first.loc[ids], second.loc[ids]
    if not left[["text", "model", "prompt_id"]].equals(right[["text", "model", "prompt_id"]]):
        raise ValueError("A shared text ID has inconsistent text, label or prompt grouping.")
    eligible = np.ones(len(ids), dtype=bool)
    for arm in (left, right):
        column = "group_eligible" if "group_eligible" in arm else "group_is_strong"
        eligible &= arm[column].fillna(False).to_numpy(dtype=bool)
        eligible &= arm.prompt_id.fillna("").astype(str).str.strip().ne("").to_numpy()
    common = left.iloc[np.flatnonzero(eligible)].reset_index(drop=True)
    identity = "\n".join(f"{row.id}\t{row.model}\t{row.prompt_id}" for row in common.itertuples())
    report = {
        "selection": "identical shared text/label/prompt, eligible in both arms",
        "arm_rows": len(frame), "peer_rows": len(peer), "shared_text_rows": len(ids),
        "common_eligible_rows": len(common), "common_prompts": int(common.prompt_id.nunique()),
        "common_rows_per_class": common.groupby("model").size().reindex(labels, fill_value=0).to_dict(),
        "common_relation_sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "class_length_quotas_equal": True,
    }
    return common, report


def shared_triplet_batches(frame, common, seed, batch_size=3):
    """Use identical shared triplets, then cover every remaining row with plain CE.

    The batch-size sequence and relation positions depend only on the common
    rows, seed and total row count, so matched arms have identical window shapes.
    """
    if batch_size != 3:
        raise ValueError("shared_triplets requires batch_size=3.")
    if not frame.id.is_unique or not common.id.is_unique:
        raise ValueError("Shared triplet plans require unique text IDs.")
    common = common.sort_values("id").reset_index(drop=True)
    positions = pd.Series(np.arange(len(frame), dtype=int), index=frame.id)
    if not common.id.isin(positions.index).all():
        raise ValueError("A shared relation text is absent from the training arm.")
    batches = []
    used = np.zeros(len(frame), dtype=bool)
    for rows, same in prompt_batches(common, seed, batch_size):
        if same and len(rows) == 3:
            mapped = positions.loc[common.iloc[rows].id].to_numpy(dtype=int).tolist()
            batches.append((mapped, True))
            used[mapped] = True
    rng = np.random.default_rng(seed)
    remaining = rng.permutation(np.flatnonzero(~used)).tolist()
    batches.extend((remaining[start:start + batch_size], False)
                   for start in range(0, len(remaining), batch_size))
    return [batches[index] for index in rng.permutation(len(batches))]


def _exposure(frame, targets, plain_weights, windows, classes):
    denominator = np.zeros(len(frame), dtype=np.float64)
    group_size = np.zeros(len(frame), dtype=np.int16)
    prompts = frame.prompt_id.to_numpy()
    for chunk in windows:
        count = sum(len(rows) for rows, _ in chunk)
        if count == 0:
            raise ValueError("An optimizer window must contain training rows.")
        for rows, same in chunk:
            index = np.asarray(rows, dtype=np.int64)
            if not len(index) or ((index < 0) | (index >= len(frame))).any():
                raise ValueError("Invalid row positions in the auxiliary plan.")
            if len(np.unique(index)) != len(index) or (denominator[index] > 0).any():
                raise ValueError("Each row may occur only once within an epoch.")
            denominator[index] = count
            if same:
                if len(index) < 2 or len(np.unique(targets[index])) != len(index):
                    raise ValueError("A centered batch needs distinct labels and at least two rows.")
                if not np.all(prompts[index] == prompts[index[0]]):
                    raise ValueError("A centered batch must use one prompt.")
                group_size[index] = len(index)
    active = denominator > 0
    auxiliary = group_size > 0
    coefficient = np.divide(plain_weights, denominator, out=np.zeros(len(frame)), where=active)
    return {
        "denominator": denominator, "group_size": group_size, "active": active,
        "auxiliary": auxiliary, "coefficient": coefficient,
        "plain_mass": np.bincount(targets, weights=coefficient, minlength=classes),
        "centered_mass": np.bincount(targets[auxiliary], weights=coefficient[auxiliary], minlength=classes),
    }


def centered_loss_weights(frame, labels, plain_weights, windows_by_epoch,
                          balance="class_mass", max_updates=None, enabled=True):
    """Calibrate centered weights on each epoch's actually requested prefix.

    class_mass assigns U/K total coefficients per target class in an active
    epoch, including the real optimizer-window denominators. This balances loss
    budgets, not unique-text coverage or the gradients received by each class.
    global_mass normalizes only the total mass to U, retaining class shares.
    none keeps the former per-row plain weights on active relation rows.
    """
    if balance not in ("class_mass", "global_mass", "none"):
        raise ValueError("centered balance must be class_mass, global_mass or none.")
    if max_updates is not None and (not isinstance(max_updates, (int, np.integer)) or max_updates < 1):
        raise ValueError("max_updates must be a positive integer.")
    if len(labels) != 18 or len(set(labels)) != len(labels) or not frame.model.isin(labels).all():
        raise ValueError("Auxiliary budgets require the competition's 18-label order.")
    plain_weights = np.asarray(plain_weights, dtype=np.float64)
    if plain_weights.shape != (len(frame),) or not np.isfinite(plain_weights).all() or (plain_weights < 0).any():
        raise ValueError("Plain weights must be finite, nonnegative and aligned with training rows.")
    targets = frame.model.map({label: i for i, label in enumerate(labels)}).to_numpy(dtype=int)
    classes = len(labels)
    result = np.zeros((len(windows_by_epoch), len(frame)), dtype=np.float32)
    planned_updates = sum(len(windows) for windows in windows_by_epoch)
    remaining = planned_updates if max_updates is None else min(planned_updates, max_updates)
    report = {
        "balance": balance, "enabled": bool(enabled), "labels": list(labels),
        "scope": "Each active epoch prefix, after applying the global update budget.",
        "target_definition": {
            "class_mass": "active epoch updates / 18 per target class",
            "global_mass": "active epoch updates in total, retaining unbalanced class shares",
            "none": "unmodified plain row weights on centered rows",
        }[balance],
        "planned_updates": planned_updates, "requested_max_updates": max_updates,
        "active_updates": remaining, "partial": remaining < planned_updates, "epochs": [],
        "interpretation": "Equal target-class loss coefficients do not imply equal text coverage or gradients. global_mass controls total strength without balancing classes. Changing a partial budget recalibrates its normalized weights.",
    }
    for epoch, windows in enumerate(windows_by_epoch):
        full = _exposure(frame, targets, plain_weights, windows, classes)
        if not full["active"].all():
            raise ValueError("The complete epoch plan must expose every training row once.")
        active_updates = min(len(windows), remaining)
        remaining -= active_updates
        actual = full if active_updates == len(windows) else _exposure(
            frame, targets, plain_weights, windows[:active_updates], classes)
        mass = actual["centered_mass"]
        target = active_updates / classes
        missing = [label for label, value in zip(labels, mass) if value <= 0] if active_updates else []
        if balance == "class_mass" and enabled and missing:
            raise ValueError(
                f"Epoch {epoch + 1}'s {active_updates}-update prefix has no centered support for "
                f"{missing}. Increase the update budget; 18-class auxiliary balancing is undefined.")
        if balance == "class_mass":
            factors = np.divide(target, mass, out=np.zeros(classes), where=mass > 0)
        elif balance == "global_mass":
            total_mass = float(mass.sum())
            if active_updates and enabled and total_mass <= 0:
                raise ValueError(
                    f"Epoch {epoch + 1}'s {active_updates}-update prefix has no centered support. "
                    "Increase the update budget; global auxiliary normalization is undefined.")
            factors = np.full(classes, active_updates / total_mass if total_mass > 0 else 0.0)
        else:
            factors = np.ones(classes)
        if not np.isfinite(factors).all():
            raise ValueError("Nonfinite centered class factors; inspect the active support and plain weights.")
        aux = actual["auxiliary"]
        result[epoch, aux] = (plain_weights[aux] * factors[targets[aux]]).astype(np.float32)
        if not np.isfinite(result[epoch]).all():
            raise ValueError("Centered row weights cannot be represented as finite float32 values.")
        candidate = np.divide(result[epoch].astype(np.float64), actual["denominator"],
                              out=np.zeros(len(frame)), where=actual["active"])
        rows = []
        for label_id, label in enumerate(labels):
            class_mask = targets == label_id
            selected = class_mask & aux
            coefficients = candidate[selected]
            total = float(coefficients.sum())
            square = float(np.square(coefficients).sum())
            sizes = []
            for size in sorted(np.unique(actual["group_size"][selected])):
                part = selected & (actual["group_size"] == size)
                sizes.append({"group_size": int(size), "rows": int(part.sum()),
                              "base_mass": float(actual["coefficient"][part].sum()),
                              "candidate_mass": float(candidate[part].sum()),
                              "effective_mass": float(candidate[part].sum()) if enabled else 0.0})
            active_rows = int((class_mask & actual["active"]).sum())
            rows.append({
                "model": label, "full_plain_rows": int(class_mask.sum()),
                "full_centered_rows": int((class_mask & full["auxiliary"]).sum()),
                "full_plain_mass": float(full["plain_mass"][label_id]),
                "full_base_centered_mass": float(full["centered_mass"][label_id]),
                "active_plain_rows": active_rows, "active_centered_rows": int(selected.sum()),
                "active_centered_coverage": float(selected.sum() / active_rows) if active_rows else 0.0,
                "active_plain_mass": float(actual["plain_mass"][label_id]),
                "base_centered_mass": float(mass[label_id]), "factor": float(factors[label_id]),
                "target_centered_mass": target if balance == "class_mass" else None,
                "candidate_centered_mass": total, "effective_centered_mass": total if enabled else 0.0,
                "coefficient_ess": total * total / square if square else 0.0,
                "maximum_row_weight": float(result[epoch, selected].max()) if selected.any() else 0.0,
                "by_group_size": sizes,
            })
        report["epochs"].append({
            "epoch": epoch + 1, "planned_updates": len(windows), "active_updates": active_updates,
            "full_centered_rows": int(full["auxiliary"].sum()),
            "active_plain_rows": int(actual["active"].sum()), "active_centered_rows": int(aux.sum()),
            "missing_centered_classes": missing,
            "complete_class_support": not missing, "max_factor": float(factors.max()),
            "target_total_mass": active_updates if balance != "none" else None,
            "candidate_total_mass": float(candidate.sum()),
            "effective_total_mass": float(candidate.sum()) if enabled else 0.0,
            "per_class": rows,
        })
    return result, report
