"""Freeze the actual optimizer-update contents, then read those contents back.

This is a research artifact: rows, relation partners, loss coefficients and the
learning-rate schedule remain inspectable. Matching a random seed or whole-pool
quota does not establish matching actual exposure or initialization.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .q8_compute import packed_forwards, learning_rate_schedule
from .q8_data import file_sha
from .q8_sampling import WORD_BINS


POLICY = "actual_update_exposure_v1"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _frame_hashes(frame):
    return frame.text.map(_sha).to_numpy()


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_exposure_manifest(frame, labels, weights, centered_weights, windows_by_epoch,
                            max_updates, config, out, relation_frame=None, *,
                            phase="external", references=None):
    """Write only the actual prefix; retain the distinct complete-plan budget.

    relation_frame is already aligned to frame positions when the explicitly
    separate-forward experiment is used. Shared relation slots always contain
    the exact text used for both losses. File creation never overwrites an
    existing frozen artifact.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "exposure_manifest.json").exists():
        raise FileExistsError("An exposure manifest is frozen; use a new output directory.")
    frame = frame.reset_index(drop=True)
    relations = frame if relation_frame is None else relation_frame.reset_index(drop=True)
    if len(frame) != len(relations) or not frame.model.equals(relations.model):
        raise ValueError("Relation targets must align with the plain slots.")
    if not frame.word_bin.isin(WORD_BINS).all() or set(frame.model) != set(labels):
        raise ValueError("Unexpected classes or length bins in exposure data.")
    plain_weights = np.asarray(weights, dtype=np.float32)
    centered_weights = np.asarray(centered_weights, dtype=np.float32)
    if plain_weights.shape != (len(frame),) or centered_weights.shape != (len(windows_by_epoch), len(frame)):
        raise ValueError("Loss weights do not cover the requested training plan.")
    if not np.isfinite(plain_weights).all() or not np.isfinite(centered_weights).all():
        raise ValueError("Exposure weights must be finite.")
    total = sum(len(epoch) for epoch in windows_by_epoch)
    active = total if max_updates is None else min(total, int(max_updates))
    if active < 1:
        raise ValueError("An actual exposure plan needs a positive update budget.")
    schedule = learning_rate_schedule(config, phase, active)
    strength = float(config.get("centered_weight", 1.0))
    maximum = int(config.get("forward_batch_rows", config["batch_size"]))
    body_hash = _frame_hashes(frame)
    relation_hash = body_hash if relation_frame is None else _frame_hashes(relations)
    rows_by_index = frame.to_dict(orient="records")
    relation_by_index = rows_by_index if relation_frame is None else relations.to_dict(orient="records")
    records, updates = [], []
    global_update = 0
    for epoch_index, windows in enumerate(windows_by_epoch):
        for window_index, chunk in enumerate(windows):
            if global_update >= active:
                break
            global_update += 1
            denominator = sum(len(rows) for rows, _ in chunk)
            if denominator < 1:
                raise ValueError("Empty optimizer window.")
            microbatch, window_slot, relation_count = 0, 0, 0
            for physical_pack, groups in enumerate(packed_forwards(chunk, maximum)):
                pack_rows = sum(len(rows) for rows, _ in groups)
                pack_relation_rows = sum(len(rows) for rows, same in groups if same)
                physical_slot = 0
                for positions, same in groups:
                    group = [int(position) for position in positions]
                    if not group or min(group) < 0 or max(group) >= len(frame):
                        raise ValueError("Invalid exposure row positions.")
                    partners = [relation_by_index[i] for i in group] if same else []
                    if same:
                        if len(group) < 2 or len({p["model"] for p in partners}) != len(group):
                            raise ValueError("Relations require distinct target classes.")
                        if len({p["prompt_id"] for p in partners}) != 1:
                            raise ValueError("Relation partners belong to different prompts.")
                        if any(not p.get("group_eligible", p.get("group_is_strong", True)) for p in partners):
                            raise ValueError("An ineligible weak-prompt row entered a relation slot.")
                    partner_ids = [str(p["id"]) for p in partners]
                    partner_hashes = [str(relation_hash[i]) for i in group] if same else []
                    group_id = _sha(_json([partner_ids, partner_hashes])) if same else ""
                    for group_slot, position in enumerate(group):
                        row, relation = rows_by_index[position], relation_by_index[position]
                        coefficient = float(plain_weights[position]) / denominator
                        auxiliary = float(centered_weights[epoch_index, position]) if same else 0.0
                        record = dict(update=global_update, epoch_index=epoch_index,
                                      window_index=window_index, microbatch=microbatch,
                                      slot_in_update=window_slot, slot_in_microbatch=group_slot,
                                      physical_pack=physical_pack, slot_in_physical_pack=physical_slot,
                                      physical_pack_rows=pack_rows,
                                      additional_relation_pack_rows=pack_relation_rows if relation_frame is not None else 0,
                                      row_position=position, training_slot_id=str(row.get("training_slot_id", row["id"])),
                                      text_id=str(row["id"]), text=row["text"], exact_text_sha256=str(body_hash[position]),
                                      model=str(row["model"]), label_index=labels.index(row["model"]),
                                      word_bin=str(row["word_bin"]), n_words=int(row.get("n_words", len(row["text"].split()))),
                                      prompt_id=str(row["prompt_id"]),
                                      role="relation" if same else "plain_only",
                                      shared_relation_forward=bool(same and relation_frame is None),
                                      relation_group_id=group_id, relation_group_size=len(group) if same else 0,
                                      relation_text_id=str(relation["id"]) if same else "",
                                      relation_exact_text_sha256=str(relation_hash[position]) if same else "",
                                      relation_prompt_id=str(relation["prompt_id"]) if same else "",
                                      relation_partner_ids=_json(partner_ids), relation_partner_text_sha256=_json(partner_hashes),
                                      window_denominator=denominator, plain_loss_weight=float(plain_weights[position]),
                                      plain_loss_coefficient=coefficient, centered_loss_weight=auxiliary,
                                      centered_strength=strength, centered_loss_coefficient=auxiliary * strength / denominator,
                                      plain_group_eligible=bool(row.get("group_eligible", row.get("group_is_strong", True))))
                        for column in ("source_dataset", "source_pool", "source_version", "source_row", "source_record_id",
                                       "source_file", "origin_hint", "source_files", "origin_hints", "research_condition",
                                       "intervention_condition", "protocol_evidence_level", "evidence_record_ids",
                                       "model_identity_status", "prompt_identity_status", "generation_status", "transformation_status"):
                            if column in row:
                                record[column] = row[column]
                        records.append(record)
                        window_slot += 1
                        physical_slot += 1
                    relation_count += len(group) if same else 0
                    microbatch += 1
            rates = schedule[global_update - 1]
            updates.append(dict(update=global_update, epoch_index=epoch_index, window_index=window_index,
                                plain_rows=denominator, relation_rows=relation_count,
                                additional_relation_forward_rows=relation_count if relation_frame is not None else 0,
                                microbatches=microbatch, physical_packs=physical_pack + 1,
                                **{key: value for key, value in rates.items() if key not in ("update", "group_lrs")},
                                group_lrs_json=_json(rates["group_lrs"])))
    ledger = pd.DataFrame(records)
    update_table = pd.DataFrame(updates)
    ledger.to_parquet(out / "actual_exposure.parquet", index=False)
    update_table.to_parquet(out / "update_schedule.parquet", index=False)
    counts = ledger.groupby(["model", "word_bin"]).size()
    report = dict(policy=POLICY, phase=phase, labels=list(labels), training_rows=len(frame),
                  full_planned_updates=total, active_updates=active, requested_max_updates=max_updates,
                  actual_rows=len(ledger), unique_text_ids=int(ledger.text_id.nunique()),
                  unique_exact_texts=int(ledger.exact_text_sha256.nunique()),
                  relation_rows=int(ledger.role.eq("relation").sum()),
                  unique_relation_texts=int(ledger.loc[ledger.role.eq("relation"), "relation_exact_text_sha256"].nunique()),
                  forward_mode="separate_anchor" if relation_frame is not None else "shared_relation_forward",
                  settings=config, settings_sha256=_sha(_json(config)), references=references or {},
                  class_length_counts=[dict(model=label, word_bin=word_bin, rows=int(counts.get((label, word_bin), 0)))
                                       for label in labels for word_bin in WORD_BINS],
                  files={name: file_sha(out / name) for name in ("actual_exposure.parquet", "update_schedule.parquet")},
                  interpretation="The frozen actual prefix is consumed as optimizer windows. Initialization equality needs actual saved parameter evidence, not just the same seed.")
    _write(out / "exposure_manifest.json", report)
    return ledger, report


def load_exposure_manifest(out, frame, weights=None, centered_weights=None, relation_frame=None,
                           centered_strength=None, expected_config=None):
    """Read authoritative actual windows and check their rows against current data."""
    out = Path(out)
    report = json.loads((out / "exposure_manifest.json").read_text())
    if report["policy"] != POLICY:
        raise ValueError("Unsupported exposure manifest policy.")
    for name, digest in report["files"].items():
        if file_sha(out / name) != digest:
            raise ValueError(f"Frozen exposure artifact changed: {name}")
    if expected_config is not None and _sha(_json(expected_config)) != report["settings_sha256"]:
        raise ValueError("Training settings differ from the frozen update manifest.")
    if len(frame) != report["training_rows"]:
        raise ValueError("Training row count changed after exposure freezing.")
    ledger = pd.read_parquet(out / "actual_exposure.parquet")
    if len(ledger) != report["actual_rows"] or ledger["update"].nunique() != report["active_updates"]:
        raise ValueError("Frozen exposure row/update counts disagree.")
    positions = ledger.row_position.to_numpy(dtype=int)
    selected = frame.iloc[positions].reset_index(drop=True)
    for saved, current in (("text_id", "id"), ("text", "text"), ("model", "model"), ("word_bin", "word_bin"), ("prompt_id", "prompt_id")):
        if not np.array_equal(ledger[saved].to_numpy(), selected[current].to_numpy()):
            raise ValueError(f"Frozen exposure {saved} changed.")
    if not np.array_equal(ledger.exact_text_sha256.to_numpy(), _frame_hashes(selected)):
        raise ValueError("Frozen exact text hashes disagree.")
    label_ids = {label: index for index, label in enumerate(report["labels"])}
    if not np.array_equal(ledger.label_index.to_numpy(), selected.model.map(label_ids).to_numpy()):
        raise ValueError("Frozen label indices disagree with the label order.")
    if not ledger.role.isin(["relation", "plain_only"]).all():
        raise ValueError("Unknown frozen slot role.")
    slots = selected.get("training_slot_id", selected.id).astype(str).to_numpy()
    if not np.array_equal(ledger.training_slot_id.to_numpy(), slots):
        raise ValueError("Frozen experimental slots changed.")
    if weights is not None and not np.array_equal(ledger.plain_loss_weight.to_numpy(), np.asarray(weights, dtype=np.float32)[positions].astype(float)):
        raise ValueError("Plain loss weights differ from the frozen plan.")
    active = ledger.role.eq("relation").to_numpy()
    relations = frame if relation_frame is None else relation_frame
    if len(relations) != len(frame) or not np.array_equal(relations.model.to_numpy(), frame.model.to_numpy()):
        raise ValueError("Frozen relation targets do not align with the plain slots.")
    if report["forward_mode"] == "separate_anchor" and relation_frame is None:
        raise ValueError("This plan requires separately aligned anchor texts.")
    if report["forward_mode"] != "separate_anchor" and relation_frame is not None:
        raise ValueError("A shared-input plan cannot switch to a separate anchor stream.")
    # Materialize only the relation identity columns once. Repeated DataFrame
    # iloc on three rows can concatenate every Arrow string chunk on large pools.
    relation_ids = relations.id.astype(str).to_numpy()
    relation_models = relations.model.to_numpy()
    relation_prompts = relations.prompt_id.to_numpy()
    relation_model_present = pd.notna(relation_models)
    relation_prompt_present = pd.notna(relation_prompts)
    relation_hashes = _frame_hashes(relations)
    if not np.array_equal(ledger.shared_relation_forward.to_numpy(), active & (relation_frame is None)):
        raise ValueError("Frozen shared-forward flags disagree with their roles.")
    if not ledger.loc[~active, "centered_loss_weight"].eq(0).all():
        raise ValueError("Plain-only slots cannot carry centered loss.")
    rel = relations.iloc[positions[active]].reset_index(drop=True)
    if not np.array_equal(ledger.loc[active, "relation_text_id"].to_numpy(), rel.id.to_numpy()) or not np.array_equal(ledger.loc[active, "relation_exact_text_sha256"].to_numpy(), _frame_hashes(rel)):
        raise ValueError("Frozen relation texts changed.")
    if not np.array_equal(ledger.loc[active, "relation_prompt_id"].to_numpy(), rel.prompt_id.to_numpy()):
        raise ValueError("Frozen relation prompt identities changed.")
    if centered_weights is not None:
        values = np.asarray(centered_weights, dtype=np.float32)[ledger.epoch_index.to_numpy(dtype=int), positions].astype(float)
        values[~active] = 0
        if not np.array_equal(ledger.centered_loss_weight.to_numpy(), values):
            raise ValueError("Auxiliary weights differ from the frozen plan.")
    if centered_strength is not None and not ledger.centered_strength.eq(centered_strength).all():
        raise ValueError("Auxiliary strength differs from the frozen plan.")
    if not np.array_equal(ledger.plain_loss_coefficient.to_numpy(), ledger.plain_loss_weight.to_numpy() / ledger.window_denominator.to_numpy()):
        raise ValueError("Plain loss coefficients disagree with the frozen weights and denominators.")
    if not np.array_equal(ledger.centered_loss_coefficient.to_numpy(), ledger.centered_loss_weight.to_numpy() * ledger.centered_strength.to_numpy() / ledger.window_denominator.to_numpy()):
        raise ValueError("Auxiliary coefficients disagree with the frozen weights and denominators.")
    flat_positions = []
    for update, chunk in ledger.groupby("update", sort=False):
        if int(update) != len(flat_positions) + 1:
            raise ValueError("Frozen update order is not contiguous.")
        if chunk.epoch_index.nunique() != 1 or chunk.window_index.nunique() != 1:
            raise ValueError("A frozen update spans multiple epoch/window identities.")
        if not np.array_equal(chunk.slot_in_update.to_numpy(), np.arange(len(chunk))):
            raise ValueError("Frozen optimizer-slot order is not contiguous.")
        batches = []
        for batch, rows in chunk.groupby("microbatch", sort=False):
            if int(batch) != len(batches) or rows.role.nunique() != 1:
                raise ValueError("Frozen microbatch order or role is inconsistent.")
            if not np.array_equal(rows.slot_in_microbatch.to_numpy(), np.arange(len(rows))):
                raise ValueError("Frozen within-microbatch order changed.")
            same = rows.role.iloc[0] == "relation"
            index = rows.row_position.astype(int).tolist()
            if same:
                if (len(set(relation_prompts[index][relation_prompt_present[index]])) != 1
                        or len(set(relation_models[index][relation_model_present[index]])) != len(index)):
                    raise ValueError("Frozen relation group lost its prompt or distinct labels.")
                group_ids = relation_ids[index].tolist()
                if rows.relation_partner_ids.nunique() != 1 or json.loads(rows.relation_partner_ids.iloc[0]) != group_ids:
                    raise ValueError("Frozen relation partner order changed.")
                hashes = relation_hashes[index].tolist()
                if rows.relation_partner_text_sha256.nunique() != 1 or json.loads(rows.relation_partner_text_sha256.iloc[0]) != hashes:
                    raise ValueError("Frozen relation partner text hashes changed.")
                if not rows.relation_group_size.eq(len(rows)).all() or not rows.relation_group_id.eq(_sha(_json([group_ids, hashes]))).all():
                    raise ValueError("Frozen relation-group identity changed.")
                if relation_frame is None and not rows.plain_group_eligible.all():
                    raise ValueError("An ineligible weak-prompt row entered shared relation supervision.")
            batches.append((index, same))
        if not chunk.window_denominator.eq(len(chunk)).all():
            raise ValueError("Frozen loss denominator disagrees with its window.")
        physical = []
        maximum = int(report["settings"].get("forward_batch_rows", report["settings"]["batch_size"]))
        for pack_index, groups in enumerate(packed_forwards(batches, maximum)):
            count = sum(len(rows) for rows, _ in groups)
            extra = sum(len(rows) for rows, same in groups if same) if relation_frame is not None else 0
            physical.extend((pack_index, offset, count, extra) for offset in range(count))
        columns = ["physical_pack", "slot_in_physical_pack", "physical_pack_rows", "additional_relation_pack_rows"]
        if not np.array_equal(chunk[columns].to_numpy(), np.asarray(physical)):
            raise ValueError("Frozen physical-forward layout differs from the consumed batches.")
        flat_positions.append((int(chunk.epoch_index.iloc[0]), int(chunk.window_index.iloc[0]), batches))
    saved_schedule = pd.read_parquet(out / "update_schedule.parquet")
    expected_schedule = learning_rate_schedule(report["settings"], report["phase"], report["active_updates"])
    if len(saved_schedule) != len(flat_positions):
        raise ValueError("Frozen learning-rate schedule has the wrong update count.")
    for row, rates, (epoch, window, batches) in zip(saved_schedule.to_dict(orient="records"), expected_schedule, flat_positions):
        for key, value in rates.items():
            observed = json.loads(row["group_lrs_json"]) if key == "group_lrs" else row[key]
            if observed != value:
                raise ValueError(f"Frozen schedule differs from the actual optimizer policy: {key}")
        if row["epoch_index"] != epoch or row["window_index"] != window or row["plain_rows"] != sum(len(rows) for rows, _ in batches):
            raise ValueError("Frozen schedule describes a different optimizer window.")
    return ledger, flat_positions, report


def compare_pair(left, right, *, output=None):
    """Compare explicit controls; absent initialization evidence stays pending."""
    paths = [Path(left), Path(right)]
    reports = [json.loads((path / "exposure_manifest.json").read_text()) for path in paths]
    for path, report in zip(paths, reports):
        for name, digest in report["files"].items():
            if file_sha(path / name) != digest:
                raise ValueError(f"Changed frozen artifact: {path / name}")
    frames = [pd.read_parquet(path / "actual_exposure.parquet") for path in paths]
    schedules = [pd.read_parquet(path / "update_schedule.parquet") for path in paths]
    columns = ["update", "epoch_index", "window_index", "microbatch", "slot_in_update", "slot_in_microbatch",
               "physical_pack", "physical_pack_rows", "additional_relation_pack_rows", "shared_relation_forward",
               "role", "model", "label_index", "word_bin", "window_denominator",
               "plain_loss_weight", "plain_loss_coefficient", "centered_loss_weight", "centered_strength", "centered_loss_coefficient"]
    equal_length = len(frames[0]) == len(frames[1])
    differences = {column: int((frames[0][column].to_numpy() != frames[1][column].to_numpy()).sum()) if equal_length else None
                   for column in columns}
    relation_columns = ["relation_group_id", "relation_group_size", "relation_text_id", "relation_exact_text_sha256",
                        "relation_prompt_id", "relation_partner_ids", "relation_partner_text_sha256"]
    relations = [frame.loc[frame.role.eq("relation"), relation_columns].reset_index(drop=True) for frame in frames]
    refs = [report.get("references", {}) for report in reports]
    official = []
    official_file_checks = []
    for ref in refs:
        declared = ref.get("official_files") or {}
        canonical, verified = {}, bool(declared)
        for name, value in declared.items():
            if isinstance(value, dict) and value.get("path") and value.get("sha256"):
                if file_sha(Path(value["path"])) != value["sha256"]:
                    raise ValueError(f"Official input changed: {value['path']}")
                canonical[name] = value["sha256"]
            else:
                canonical[name] = value
                verified = False
        official.append(canonical)
        official_file_checks.append(verified)
    initialization, initialization_reference_sources = [], []
    for path, ref in zip(paths, refs):
        receipt = path / "executed_initialization.json"
        if receipt.is_file():
            initialization.append(json.loads(receipt.read_text()))
            initialization_reference_sources.append("executed_initialization.json")
        else:
            initialization.append(ref.get("initialization"))
            initialization_reference_sources.append("frozen_manifest_reference")
    # Verify actual saved parameter/RNG files, not a seed or a metadata Boolean.
    from .q8_initialization import initialization_identity, asset_identity
    initialization_objects = []
    for arm_index, value in enumerate(initialization):
        identity = None
        if isinstance(value, dict) and (value.get("directory") or value.get("manifest_path")):
            directory = Path(value["directory"]) if value.get("directory") else Path(value["manifest_path"]).parent
            if value.get("manifest_path") and value.get("manifest_sha256"):
                if not Path(value["manifest_path"]).is_file():
                    initialization_objects.append(None)
                    continue
                if file_sha(Path(value["manifest_path"])) != value["manifest_sha256"]:
                    raise ValueError("Common initialization manifest reference changed.")
            if (directory / "initialization.json").is_file():
                manifest_format = json.loads((directory / "initialization.json").read_text()).get("format")
                from .q8_backend_compare import FORMAT as CANONICAL_FORMAT, backend_initialization_identity
                candidate = (backend_initialization_identity(directory) if manifest_format == CANONICAL_FORMAT
                             else initialization_identity(directory))
            else:
                candidate = {"status": "pending"}
            if candidate["status"] == "available":
                expected = value.get("initialization_sha256")
                if expected is not None and candidate["initialization_sha256"] != expected:
                    raise ValueError("Common initialization identity changed.")
                assets = candidate["assets"]
                if assets["content"].get("kind") == "transformer_base":
                    locations = value.get("asset_locations", assets)
                    if locations.get("identity_sha256") != candidate["context"]["asset_identity_sha256"]:
                        raise ValueError("Relocated initialization assets have a different content identity.")
                    try:
                        actual = asset_identity(locations["base_directory"], locations["tokenizer_directory"])
                    except FileNotFoundError:
                        # A downloaded run may retain the recorded asset SHA
                        # without the multi-GB base. Do not invent a local
                        # re-verification or discard the other control results.
                        initialization_objects.append(None)
                        continue
                    if actual["identity_sha256"] != candidate["context"]["asset_identity_sha256"]:
                        raise ValueError("Common base weights or tokenizer changed.")
                expected_kind = refs[arm_index].get("model_kind", value.get("model_kind"))
                if refs[arm_index].get("model_kind") and value.get("model_kind") and refs[arm_index]["model_kind"] != value["model_kind"]:
                    raise ValueError("Conflicting expected initialization model domains.")
                context = candidate["context"]
                domain_matches = expected_kind is not None and assets["content"].get("kind") == expected_kind
                context_matches = context.get("labels") == reports[arm_index]["labels"]
                for reference_key, context_key in (("expected_asset_identity", "asset_identity_sha256"),
                                                   ("expected_backend", "backend"),
                                                   ("expected_architecture", "architecture"),
                                                   ("expected_compute_dtype", "compute_dtype")):
                    if reference_key in refs[arm_index]:
                        if context_key == "backend" and candidate.get("format") == CANONICAL_FORMAT:
                            context_matches &= (refs[arm_index][reference_key] in candidate["allowed_backends"]
                                                and refs[arm_index][reference_key] == reports[arm_index]["settings"].get("backend"))
                        else:
                            context_matches &= refs[arm_index][reference_key] == context.get(context_key)
                bound = bool(value.get("initialization_sha256") or value.get("manifest_sha256"))
                if domain_matches and context_matches and bound:
                    identity = candidate
        initialization_objects.append(identity)
    init_equal = all(initialization_objects) and initialization_objects[0]["initialization_sha256"] == initialization_objects[1]["initialization_sha256"]
    label_order_equal = reports[0]["labels"] == reports[1]["labels"]
    phase_equal = reports[0]["phase"] == reports[1]["phase"]
    controls = equal_length and label_order_equal and phase_equal and all(value == 0 for value in differences.values()) and relations[0].equals(relations[1]) and schedules[0].equals(schedules[1])
    official_equal = all(official_file_checks) and official[0] == official[1]
    result = dict(actual_control_fields_equal=bool(controls), field_difference_counts=differences,
                  relation_text_partners_positions_equal=relations[0].equals(relations[1]),
                  learning_rate_schedule_equal=schedules[0].equals(schedules[1]),
                  label_order_equal=label_order_equal,
                  phase_equal=phase_equal,
                  official_split_equal=bool(official_equal),
                  official_files_verified=official_file_checks,
                  declared_official_hashes_equal=bool(official[0]) and official[0] == official[1],
                  initialization_status="verified_same_saved_parameters" if init_equal else "not_verified",
                  initialization_reference_source=initialization_reference_sources,
                  initialization_asset_kinds=[value["assets"]["content"].get("kind") if value else None for value in initialization_objects],
                  settings_equal=reports[0]["settings_sha256"] == reports[1]["settings_sha256"],
                  changed_plain_texts=int(frames[0].exact_text_sha256.ne(frames[1].exact_text_sha256).sum()) if equal_length else None,
                  status="fully_verified" if controls and official_equal and init_equal and reports[0]["settings_sha256"] == reports[1]["settings_sha256"] else
                         "controls_verified_initialization_pending" if controls and official_equal and reports[0]["settings_sha256"] == reports[1]["settings_sha256"] else "differences_or_missing_evidence",
                  interpretation="Control equality is not a new accuracy result. Both arms require fresh training from the same saved initialization.")
    if output is not None:
        _write(Path(output), result)
    return result
