"""Shared relation batches and explicit target-class auxiliary loss budgets."""

import hashlib
import json

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
    if not left[["text", "model", "prompt_id", "word_bin"]].equals(right[["text", "model", "prompt_id", "word_bin"]]):
        raise ValueError("A shared text ID has inconsistent text, label, prompt grouping or length bin.")
    eligible = np.ones(len(ids), dtype=bool)
    for arm in (left, right):
        column = "group_eligible" if "group_eligible" in arm else "group_is_strong"
        eligible &= arm[column].fillna(False).to_numpy(dtype=bool)
        eligible &= arm.prompt_id.fillna("").astype(str).str.strip().ne("").to_numpy()
    common = left.iloc[np.flatnonzero(eligible)].reset_index(drop=True)
    identity = "\n".join(f"{ident}\t{label}\t{prompt}" for ident, label, prompt in
                         zip(common.id, common.model, common.prompt_id))
    report = {
        "selection": "identical shared text/label/prompt, eligible in both arms",
        "arm_rows": len(frame), "peer_rows": len(peer), "shared_text_rows": len(ids),
        "common_eligible_rows": len(common), "common_prompts": int(common.prompt_id.nunique()),
        "common_rows_per_class": common.groupby("model").size().reindex(labels, fill_value=0).to_dict(),
        "common_relation_sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "class_length_quotas_equal": True,
    }
    return common, report


def shared_triplet_batches(frame, common, seed, batch_size=3, stratify_plain=False):
    """Use identical shared triplets, then cover every remaining row with plain CE.

    The batch-size sequence and relation positions depend only on the common
    rows, seed and total row count, so matched arms have identical window shapes.
    stratify_plain couples each remaining slot's model/length stratum between
    equal-quota arms, including any finite prefix. It does not enlarge the
    shared relation set or make different protocols a text-transformation pair.
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
    common_positions = positions.loc[common.id].to_numpy(dtype=int)
    for rows, same in prompt_batches(common, seed, batch_size):
        if same and len(rows) == 3:
            mapped = common_positions[rows].tolist()
            batches.append((mapped, True))
            used[mapped] = True
    rng = np.random.default_rng(seed)
    remaining = np.flatnonzero(~used)
    if stratify_plain:
        if not frame.word_bin.isin(WORD_BINS).all():
            raise ValueError("Stratified plain slots require known word-count bins.")
        # Keep within-stratum random draws separate from the historical global
        # permutation and batch-placement RNG, preserving relation positions.
        row_rng = np.random.default_rng(seed + 100000)
        groups = frame.iloc[remaining].groupby(
            ["model", "word_bin"], observed=True, sort=False).indices
        ordered = []
        for label in sorted(frame.model.unique()):
            for word_bin in WORD_BINS:
                local = groups.get((label, word_bin), np.array([], dtype=int))
                ordered.extend(row_rng.permutation(remaining[local]).tolist())
        remaining = np.asarray(ordered, dtype=int)
    remaining = rng.permutation(remaining).tolist()
    batches.extend((remaining[start:start + batch_size], False)
                   for start in range(0, len(remaining), batch_size))
    return [batches[index] for index in rng.permutation(len(batches))]


def align_anchor_rows(frame, anchor):
    """Align immutable relation examples to a plain arm's experimental slots."""
    if not anchor.id.is_unique or not frame.id.is_unique:
        raise ValueError("Anchored plans require unique text IDs in each arm.")
    if "training_slot_id" not in frame or not frame.training_slot_id.is_unique:
        raise ValueError("Anchored arms need unique training_slot_id values.")
    if set(frame.training_slot_id) != set(anchor.id):
        raise ValueError("Anchored arms must cover exactly the anchor's slots.")
    aligned = anchor.set_index("id", drop=False).loc[frame.training_slot_id].reset_index(drop=True)
    if not frame.reset_index(drop=True)[["model", "word_bin"]].equals(aligned[["model", "word_bin"]]):
        raise ValueError("An anchored replacement changes its slot's class or length bin.")
    aligned["training_slot_id"] = frame.training_slot_id.to_numpy()
    return aligned


def anchored_triplet_batches(frame, anchor, seed, batch_size=3, permit_plain_replacement_all=False):
    """Replay an anchor plan while allowing only plain-slot text replacements.

    Class and length are fixed for every slot. By default the same forward is
    used by both heads, so centered slots cannot change text. The explicit
    separate-forward mode allows any plain replacement while the caller must
    compute centered loss from aligned anchor texts, never from the changed arm.
    """
    align_anchor_rows(frame, anchor)
    position = pd.Series(np.arange(len(frame), dtype=int), index=frame.training_slot_id)
    mapped = position.loc[anchor.id].to_numpy(dtype=int)
    aligned = frame.iloc[mapped].reset_index(drop=True)
    anchor = anchor.reset_index(drop=True)
    common, _ = common_relation_rows(anchor, anchor)
    plan = shared_triplet_batches(anchor, common, seed, batch_size)
    relation_positions = np.asarray([i for rows, same in plan if same for i in rows], dtype=int)
    columns = ["id", "text", "model", "prompt_id"]
    if not permit_plain_replacement_all and not aligned.iloc[relation_positions][columns].equals(anchor.iloc[relation_positions][columns]):
        raise ValueError("An anchored replacement changes a centered text, label or prompt.")
    return [(mapped[rows].tolist(), same) for rows, same in plan]


def plan_evidence(frame, windows_by_epoch, plain_weights, centered_weights,
                  max_updates=None, centered_strength=1.0, relation_frame=None):
    """Describe actual exposure and coefficients without loading a model.

    Per-slot stratum equality is stronger than equal whole-pool quotas. The
    centered fingerprint includes window position, partners and effective
    per-row coefficients, not just aggregate target-class loss mass.
    """
    labels = sorted(frame.model.unique())
    ids = frame.id.to_numpy()
    models = frame.model.to_numpy()
    bins = frame.word_bin.to_numpy()
    slots = frame.get("training_slot_id", frame.id).to_numpy()
    relations = frame if relation_frame is None else relation_frame
    if len(relations) != len(frame) or not relations.model.reset_index(drop=True).equals(frame.model.reset_index(drop=True)):
        raise ValueError("Relation rows and plain targets must be aligned by experimental slot.")
    if relation_frame is not None and not np.array_equal(relations.training_slot_id.to_numpy(), slots):
        raise ValueError("Relation data do not match the plain arm's slot order.")
    relation_ids, relation_prompts = relations.id.to_numpy(), relations.prompt_id.to_numpy()
    remaining = sum(map(len, windows_by_epoch)) if max_updates is None else max_updates
    reports = []

    def describe(windows, epoch):
        active, relation_rows = [], []
        stratum_hash, slot_hash, text_hash, context_hash = [hashlib.sha256() for _ in range(4)]
        coefficient_mass = {label: 0.0 for label in labels}
        for update, chunk in enumerate(windows):
            denominator = sum(len(rows) for rows, _ in chunk)
            for batch, (rows, same) in enumerate(chunk):
                positions = np.asarray(rows, dtype=int)
                active.extend(rows)
                for slot, i in enumerate(rows):
                    location = [update, batch, slot, denominator, bool(same)]
                    for digest, identity in ((stratum_hash, [models[i], bins[i]]),
                                             (slot_hash, [slots[i]]), (text_hash, [ids[i]])):
                        digest.update((json.dumps(location + identity, ensure_ascii=False,
                                                  separators=(",", ":")) + "\n").encode())
                if same:
                    relation_rows.extend(rows)
                    event = [update, batch, denominator, relation_ids[positions].tolist(),
                             models[positions].tolist(), relation_prompts[positions].tolist(),
                             (np.asarray(plain_weights)[positions] / denominator).tolist(),
                             (centered_weights[epoch, positions] * centered_strength / denominator).tolist()]
                    context_hash.update((json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
                    for i in rows:
                        coefficient_mass[models[i]] += float(centered_weights[epoch, i]) * centered_strength / denominator
        active = np.asarray(active, dtype=int)
        relation_rows = np.asarray(relation_rows, dtype=int)
        counts = frame.iloc[active].groupby(["model", "word_bin"], observed=True).size()
        per_class = []
        for label in labels:
            per_class.append(dict(model=label, rows=int((models[active] == label).sum()),
                                  centered_rows=int((models[relation_rows] == label).sum()),
                                  centered_coefficient_mass=coefficient_mass[label]))
        return dict(updates=len(windows), rows=len(active), unique_rows=len(np.unique(active)),
                    centered_rows=len(relation_rows), centered_unique_rows=len(np.unique(relation_rows)),
                    plain_forward_rows=len(active),
                    additional_relation_forward_rows=(len(relation_rows) if relation_frame is not None and centered_strength > 0 else 0),
                    total_forward_rows=(len(active) + (len(relation_rows) if relation_frame is not None and centered_strength > 0 else 0)),
                    stratum_slot_sha256=stratum_hash.hexdigest(), anchor_slot_sha256=slot_hash.hexdigest(),
                    text_slot_sha256=text_hash.hexdigest(),
                    centered_context_coefficient_sha256=context_hash.hexdigest(),
                    class_length_counts=[dict(model=label, word_bin=length, rows=int(counts.get((label, length), 0)))
                                         for label in labels for length in WORD_BINS], per_class=per_class)

    for epoch, windows in enumerate(windows_by_epoch):
        actual = min(len(windows), max(0, remaining))
        remaining -= actual
        full = describe(windows, epoch)
        selected = full if actual == len(windows) else describe(windows[:actual], epoch)
        reports.append(dict(epoch=epoch + 1, full=full, actual=selected,
                            full_epoch_once=full['rows'] == full['unique_rows'] == len(frame)))
    return dict(scope="Actual requested update prefixes, with full-epoch coverage retained.",
                relation_input="separate_anchor" if relation_frame is not None else "shared_plain_forward",
                coefficient_note="Weights outside the requested budget are zero; full records describe layout, not an additional training budget.",
                epochs=reports)


def relation_coverage(frame, windows_by_epoch, max_updates=None, labels=None):
    """Measure physical relation support, independent of auxiliary loss weights.

    A group contributes each unordered label pair once. Repeated encounters of
    the same two text IDs count toward encounters but not unique_text_pairs.
    The per-class view also counts distinct partner texts and target texts;
    equal loss mass cannot change any of these quantities.
    """
    labels = sorted(frame.model.unique()) if labels is None else list(labels)
    if len(set(labels)) != len(labels) or not frame.model.isin(labels).all():
        raise ValueError("Coverage labels must uniquely include every relation label.")
    if max_updates is not None and (not isinstance(max_updates, (int, np.integer)) or max_updates < 1):
        raise ValueError("Coverage update budget must be a positive integer.")
    if not frame.id.is_unique:
        raise ValueError("Physical relation coverage requires unique text IDs per frame.")
    ids, models, prompts = (frame[column].to_numpy() for column in ("id", "model", "prompt_id"))
    label_order = {label: index for index, label in enumerate(labels)}
    pairs = [(left, right) for index, left in enumerate(labels) for right in labels[index + 1:]]

    def new_state():
        return dict(updates=0, groups=0, rows=0, sizes={}, fingerprint=hashlib.sha256(),
                    classes={label: dict(exposures=0, texts=set(), prompts=set(), partner_texts=set(),
                                         partner_labels=set(), partner_encounters=0) for label in labels},
                    pairs={pair: dict(encounters=0, text_pairs=set(), left=set(), right=set(), prompts=set())
                           for pair in pairs})

    def add_group(state, rows, epoch, update, denominator):
        rows = np.asarray(rows, dtype=int)
        if (rows < 0).any() or (rows >= len(frame)).any() or len(rows) < 2:
            raise ValueError("A relation group needs valid positions and at least two rows.")
        if len(set(models[rows])) != len(rows) or len(set(prompts[rows])) != 1:
            raise ValueError("Relation coverage requires one prompt and distinct labels per group.")
        state['groups'] += 1
        state['rows'] += len(rows)
        state['sizes'][len(rows)] = state['sizes'].get(len(rows), 0) + 1
        event = [epoch, update, denominator, ids[rows].tolist(), models[rows].tolist(), str(prompts[rows[0]])]
        state['fingerprint'].update((json.dumps(event, ensure_ascii=False, separators=(',', ':')) + '\n').encode())
        for row in rows:
            cls = state['classes'][models[row]]
            cls['exposures'] += 1
            cls['texts'].add(ids[row])
            cls['prompts'].add(prompts[row])
            partners = rows[rows != row]
            cls['partner_texts'].update(ids[partners])
            cls['partner_labels'].update(models[partners])
            cls['partner_encounters'] += len(partners)
        for index, row in enumerate(rows):
            for other in rows[index + 1:]:
                left, right = (row, other) if label_order[models[row]] < label_order[models[other]] else (other, row)
                pair = state['pairs'][(models[left], models[right])]
                pair['encounters'] += 1
                pair['text_pairs'].add((ids[left], ids[right]))
                pair['left'].add(ids[left])
                pair['right'].add(ids[right])
                pair['prompts'].add(prompts[left])

    def summary(state):
        pair_rows = [dict(left_label=left, right_label=right, encounters=values['encounters'],
                          unique_text_pairs=len(values['text_pairs']), unique_left_texts=len(values['left']),
                          unique_right_texts=len(values['right']), unique_prompts=len(values['prompts']))
                     for (left, right), values in state['pairs'].items()]
        return dict(active_updates=state['updates'], centered_groups=state['groups'],
                    centered_row_exposures=state['rows'],
                    centered_unique_texts=sum(len(value['texts']) for value in state['classes'].values()),
                    group_size_counts={str(key): value for key, value in state['sizes'].items()},
                    relation_context_sha256=state['fingerprint'].hexdigest(),
                    possible_label_pairs=len(pairs), observed_label_pairs=sum(row['encounters'] > 0 for row in pair_rows),
                    pair_encounters=sum(row['encounters'] for row in pair_rows),
                    unique_text_pairs=sum(row['unique_text_pairs'] for row in pair_rows),
                    per_class=[dict(model=label, centered_row_exposures=value['exposures'],
                                    centered_unique_texts=len(value['texts']), unique_prompts=len(value['prompts']),
                                    unique_partner_texts=len(value['partner_texts']),
                                    unique_partner_labels=len(value['partner_labels']),
                                    partner_encounters=value['partner_encounters'])
                               for label, value in state['classes'].items()], per_label_pair=pair_rows)

    total, epochs = new_state(), []
    remaining = sum(map(len, windows_by_epoch)) if max_updates is None else max_updates
    for epoch, windows in enumerate(windows_by_epoch):
        count = min(len(windows), max(0, remaining))
        remaining -= count
        current = new_state()
        current['updates'] = count
        total['updates'] += count
        for update, chunk in enumerate(windows[:count]):
            denominator = sum(len(rows) for rows, _ in chunk)
            for rows, same in chunk:
                if same:
                    add_group(current, rows, epoch, update, denominator)
                    add_group(total, rows, epoch, update, denominator)
        epochs.append(dict(epoch=epoch + 1, **summary(current)))
    return dict(scope="Actual update prefix; physical text and partner support, without loss weighting.",
                pair_direction="Unordered distinct-label pairs; one encounter per text pair within a group.",
                repeated_text_note="Unique text and pair counts deduplicate across epochs; exposures retain repetitions.",
                requested_max_updates=max_updates, total=summary(total), epochs=epochs)


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
