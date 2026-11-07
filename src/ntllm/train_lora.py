"""Prepare or train a controlled q8-style external-to-official experiment."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
from uuid import uuid4

import numpy as np
import pandas as pd

from .q8_data import file_sha
from .q8_config import load_q8_config
from .q8_sampling import matching_weights, plan_summary, prompt_batches
from .q8_supervision import (common_relation_rows, shared_triplet_batches, centered_loss_weights,
                             anchored_triplet_batches, plan_evidence, align_anchor_rows, relation_coverage)
from .q8_compute import packed_forwards, make_optimizer, lr_multiplier, optimizer_group_rates


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def synchronize_device(device):
    """Measure completed device work rather than asynchronous launch time."""
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def read_saved_plan(directory, epochs):
    """Use the reviewed physical row order instead of silently resampling it."""
    with np.load(Path(directory) / 'plan.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in ('rows', 'sizes', 'same_prompt', 'epochs')}
    if not (len(arrays['sizes']) == len(arrays['same_prompt']) == len(arrays['epochs'])):
        raise ValueError('Frozen batch arrays have different lengths.')
    plan, offset = [[] for _ in range(epochs)], 0
    for size, same, epoch in zip(arrays['sizes'], arrays['same_prompt'], arrays['epochs']):
        if size < 1 or epoch < 0 or epoch >= epochs:
            raise ValueError('Frozen batch size or epoch is invalid.')
        rows = arrays['rows'][offset:offset + size].astype(int).tolist()
        if len(rows) != size:
            raise ValueError('Frozen batch rows are incomplete.')
        plan[int(epoch)].append((rows, bool(same)))
        offset += int(size)
    if offset != len(arrays['rows']) or any(not epoch for epoch in plan):
        raise ValueError('Frozen plan does not cover the declared complete epochs.')
    return plan


def read_initialization_identity(directory):
    from .q8_initialization import initialization_identity
    marker = Path(directory) / 'initialization.json'
    if marker.is_file() and json.loads(marker.read_text()).get('format') == 'ntllm-canonical-q8-backend-initialization-v1':
        from .q8_backend_compare import backend_initialization_identity
        return backend_initialization_identity(directory)
    return initialization_identity(directory)


def initialization_reference(directory):
    if directory is None:
        return {'status': 'not_specified', 'model_kind': 'transformer_base'}
    identity = read_initialization_identity(directory)
    reference = dict(status=identity['status'], directory=str(Path(directory).resolve()),
                     model_kind='transformer_base')
    if identity['status'] == 'available':
        path = Path(directory) / 'initialization.json'
        reference.update(manifest_path=str(path.resolve()), manifest_sha256=file_sha(path),
                         initialization_sha256=identity['initialization_sha256'],
                         model_kind=identity['assets']['content']['kind'])
    return reference


def resumed_initialization_reference(saved, directory):
    """Keep a verified moved locator, while preserving the original root SHA."""
    previous = dict(saved.get('initialization_proof') or {})
    expected = (saved.get('shared_initialization') or previous).get('initialization_sha256')
    if directory is not None:
        current = initialization_reference(directory)
        if current['status'] == 'available':
            if expected is None or current['initialization_sha256'] != expected:
                raise ValueError('Resumed common initialization does not match its recorded root.')
            current['proof_scope'] = 'Same immutable initial root; actual parameters resume from the trained checkpoint.'
            return current
    if expected:
        previous.update(status='recorded_unavailable', initialization_sha256=expected,
                        proof_scope='Recorded initial root; its parameter snapshot is not available at the current locator.')
    return previous


def bind_initial_state(saved, requested):
    """A reviewed start cannot be replaced by a different same-shape model."""
    original = saved.get('shared_initialization_path')
    proof = saved.get('shared_initialization') or saved.get('initialization_proof') or {}
    expected = proof.get('initialization_sha256')
    if requested is None:
        return Path(original) if original else None
    if expected:
        current = read_initialization_identity(requested)
        if current.get('initialization_sha256') != expected:
            raise ValueError('The requested initial parameters differ from the frozen experiment.')
    elif original is None or Path(requested).resolve() != Path(original).resolve():
        raise ValueError('A pending common start has a frozen shared path; prepare a new paired plan to bind another start.')
    return Path(requested)


def token_rows_identity(stats):
    """Content signature independent of Parquet's environment-specific encoding."""
    columns = ['raw_text_sha256', 'input_ids_sha256', 'raw_tokens', 'input_tokens', 'removed_tokens', 'truncated']
    digest = hashlib.sha256()
    for raw, ids, before, after, removed, truncated in stats[columns].itertuples(index=False, name=None):
        digest.update(json.dumps([raw, ids, int(before), int(after), int(removed), bool(truncated)],
                                 separators=(',', ':')).encode())
        digest.update(b'\n')
    return {'rows': len(stats), 'sha256': digest.hexdigest()}


def measure_token_inputs(tokenizer, tokenizer_directory, frame, relation_frame, windows_by_epoch,
                         max_updates, config, run_dir, *, plan_only=False, prepared=None,
                         reference_identity=None, frozen_metadata=None, resume_metadata=None):
    """Encode each input stream once, verify its inputs, and profile once.

    Planning retains statistics only. Training returns the actual model
    tokenizer's IDs and checks them against frozen measurements, never against
    a second newly encoded copy of the entire pool.
    """
    from .q8_tokens import encode_with_statistics, tokenizer_identity, profile_plan
    identity = tokenizer_identity(tokenizer_directory, tokenizer)
    frozen_metadata = frozen_metadata or {}
    reference_identity = reference_identity or {}
    if reference_identity and identity['files_identity_sha256'] != reference_identity['files_identity_sha256']:
        raise ValueError('The actual tokenizer files differ from the requested or frozen tokenizer.')
    columns = ['raw_text_sha256', 'input_ids_sha256', 'raw_tokens', 'input_tokens', 'removed_tokens', 'truncated']
    frozen_rows = Path(prepared) / 'token_rows.parquet' if prepared is not None else None
    has_frozen_rows = frozen_rows is not None and frozen_rows.is_file()
    if not has_frozen_rows and not frozen_metadata.get('planned_token_identity') and reference_identity.get('runtime'):
        # Without frozen row evidence, only reuse a reference whose effective
        # implementation matches. Loading its identity does not encode a pool.
        for key in ('runtime', 'backend_tokenizer_sha256'):
            if identity.get(key) != reference_identity.get(key):
                raise ValueError(f'Tokenizer reference differs in {key}; prepare measured inputs before using another implementation.')
    tokens, stats = encode_with_statistics(tokenizer, frame.text.tolist(), config['max_len'],
                                           retain_tokens=not plan_only)
    relation_tokens, relation_stats = (encode_with_statistics(
        tokenizer, relation_frame.text.tolist(), config['max_len'], retain_tokens=not plan_only)
        if relation_frame is not None else (None, None))
    plain_identity = token_rows_identity(stats)
    relation_identity = token_rows_identity(relation_stats) if relation_stats is not None else None
    if has_frozen_rows:
        if not pd.read_parquet(frozen_rows, columns=columns).equals(stats[columns]):
            raise ValueError('Tokenizer inputs or truncation differ from the reviewed plan.')
    if prepared is not None and relation_stats is not None:
        frozen_relations = Path(prepared) / 'relation_token_rows.parquet'
        if has_frozen_rows and not frozen_relations.is_file():
            raise ValueError('The reviewed plan is missing its anchor token rows.')
        if frozen_relations.is_file() and not pd.read_parquet(frozen_relations, columns=columns).equals(relation_stats[columns]):
            raise ValueError('Anchor token inputs differ from the reviewed plan.')
    for key, value in (('planned_token_identity', plain_identity),
                       ('planned_relation_token_identity', relation_identity)):
        if key in frozen_metadata and frozen_metadata[key] != value:
            raise ValueError(f'Frozen token input identity differs: {key}.')
    if resume_metadata:
        for key, value in (('executed_token_identity', plain_identity),
                           ('executed_relation_token_identity', relation_identity)):
            if resume_metadata.get(key) != value:
                raise ValueError(f'Resume actual tokenizer inputs differ: {key}.')
        previous = resume_metadata.get('executed_tokenizer_identity') or {}
        for key in ('files_identity_sha256', 'runtime', 'backend_tokenizer_sha256'):
            if identity.get(key) != previous.get(key):
                raise ValueError(f'Resume tokenizer implementation differs: {key}.')
    profile_plan(frame, stats, windows_by_epoch, max_updates, config, run_dir,
                 relation_stats=relation_stats, relation_frame=relation_frame)
    measured = dict(tokenizer_identity=identity,
                    token_rows_sha256=file_sha(run_dir / 'token_rows.parquet'),
                    token_profile_sha256=file_sha(run_dir / 'token_profile.json'),
                    planned_token_identity=plain_identity, planned_relation_token_identity=relation_identity)
    if not plan_only:
        # Preserve both historical artifact locations with identical files,
        # not another profile_plan pass or another in-memory statistics table.
        execution_dir = run_dir / 'execution_tokens'
        execution_dir.mkdir(exist_ok=True)
        for name in ('token_rows.parquet', 'relation_token_rows.parquet', 'token_profile.json',
                     'step_token_metrics.parquet', 'physical_token_batches.parquet'):
            if (run_dir / name).is_file():
                shutil.copy2(run_dir / name, execution_dir / name)
        measured.update(executed_token_identity=plain_identity, executed_relation_token_identity=relation_identity,
                        executed_tokenizer_identity=identity)
    return tokens, relation_tokens, stats, relation_stats, measured


def update_windows(batches, target_rows):
    """Keep complete microbatches and use the same update count for equal row budgets."""
    cumulative = np.cumsum([len(rows) for rows, _ in batches])
    count = math.ceil(int(cumulative[-1]) / target_rows)
    targets = np.arange(1, count) * (cumulative[-1] / count)
    boundaries = [0]
    for number, target in enumerate(targets, start=1):
        proposed = int(np.searchsorted(cumulative, target, side='left')) + 1
        latest = len(batches) - (count - number)
        boundaries.append(max(boundaries[-1] + 1, min(proposed, latest)))
    boundaries.append(len(batches))
    return [batches[left:right] for left, right in zip(boundaries[:-1], boundaries[1:])]


def prepare(args, config, resume_metadata=None):
    labels = json.loads((args.data_dir / 'labels.json').read_text())
    data_manifest = json.loads((args.data_dir / 'manifest.json').read_text())
    evidence_reference = {key: data_manifest[key] for key in (
        'evidence_ledger', 'evidence_ledger_sha256', 'parent_manifest_path', 'parent_manifest_sha256')
        if key in data_manifest}
    if 'evidence_ledger' in evidence_reference:
        ledger_path = Path(evidence_reference['evidence_ledger'])
        if not ledger_path.is_absolute() and (args.data_dir / ledger_path).exists():
            ledger_path = args.data_dir / ledger_path
        actual_sha = file_sha(ledger_path)
        if actual_sha != evidence_reference.get('evidence_ledger_sha256'):
            raise ValueError('The protocol evidence ledger differs from its data manifest.')
        evidence_reference.update(evidence_ledger=str(ledger_path.resolve()), evidence_ledger_sha256=actual_sha)
    official = pd.read_parquet(args.data_dir / 'official.parquet')
    inner_split = None
    inner_fit = None
    if config.get('research_inner_validation', False):
        from .q8_inner import fixed_inner_split
        outer_fold = args.fold if isinstance(args.fold, int) else config.get('inner_outer_fold', 0)
        inner_fit, _, _, inner_split = fixed_inner_split(
            official, outer_fold, config.get('inner_seed', 42), config.get('inner_dev_fraction', .2))
        if args.phase == 'official' and not isinstance(args.fold, int):
            raise ValueError('Inner research requires an explicit outer fold; final refitting is a separate experiment.')
    valid = official.iloc[:0].copy()
    matching = None
    parent = (resume_metadata or {}).get('external_initialization_config')
    if args.init is not None:
        parent = json.loads((args.init / 'config.json').read_text())
    if parent is not None:
        backend = parent['settings'].get('backend', 'hf')
        selected = getattr(args, 'backend', None) or config.get('backend', backend)
        if getattr(args, 'backend', None) is None and selected == 'hf':
            selected = backend
        if selected != backend and not {selected, backend}.issubset({'hf', 'liger'}):
            raise ValueError('Unsloth checkpoint layout cannot be silently mixed with another backend.')
        config['backend'] = selected
    from .q8_stages import KEYS, resolve_stage_supervision
    stage_supervision = resolve_stage_supervision(
        config, args.phase, parent, {key: getattr(args, key, None) for key in KEYS},
        frozen=bool(resume_metadata or getattr(args, 'prepared_plan', None)))
    shared_policies = ('shared_triplets', 'shared_stratified_triplets')
    anchored_policies = ('anchored_triplets', 'anchored_separate')
    triplet_policies = (*shared_policies, 'grouped_triplets', *anchored_policies)
    if config['centered_plan'] not in ('native', *triplet_policies):
        raise ValueError('Unknown centered_plan; use native, shared_triplets, shared_stratified_triplets, grouped_triplets, anchored_triplets or anchored_separate.')
    if config['centered_balance'] not in ('none', 'global_mass', 'class_mass'):
        raise ValueError('centered_balance must be none, global_mass or class_mass.')
    if not np.isfinite(config['centered_weight']) or config['centered_weight'] < 0:
        raise ValueError('centered_weight must be finite and nonnegative.')
    variant = args.variant or (parent['variant'] if parent else 'baseline')
    conditioning_fold = 'shared'
    if args.phase == 'external':
        if args.init is not None:
            raise ValueError('External experiments start with a fresh adapter on the base model.')
        pool_name = variant if variant in ('mix20', 'expanded') else 'baseline'
        path = args.data_dir / f'external_{pool_name}.parquet'
        frame = pd.read_parquet(path)
        soft_weights = np.ones(len(frame), dtype=np.float32)
        if variant == 'matched':
            if args.fold == 'shared':
                raise ValueError('Matched external training needs a target fold, or final for all official rows.')
            target = inner_fit if inner_fit is not None else (official if args.fold == 'final' else official.loc[official.fold != args.fold])
            soft_weights, matching = matching_weights(frame, target, config['matching_strength'])
            conditioning_fold = args.fold
        elif args.fold != 'shared':
            raise ValueError('Baseline and mix20 external training use --fold shared.')
    else:
        if args.fold == 'shared':
            raise ValueError('Official training needs --fold 0..4 or final.')
        path = args.data_dir / 'official.parquet'
        frame = official
        if args.fold != 'final':
            frame = inner_fit if inner_fit is not None else official.loc[official.fold != args.fold]
            valid = official.loc[official.fold == args.fold].copy()
        frame = frame.reset_index(drop=True)
        soft_weights = np.ones(len(frame), dtype=np.float32)
        if variant == 'matched':
            conditioning_fold = args.fold
        if parent is not None:
            preview_parent = (args.plan_only and parent['status'] == 'planned'
                              and parent.get('prepared_only', False))
            if parent['phase'] != 'external' or (parent['status'] not in ('complete', 'partial', 'research_snapshot')
                                                and not preview_parent):
                raise ValueError('Official training requires a saved external-stage checkpoint.')
            if parent['variant'] != variant or parent['labels'] != labels:
                raise ValueError('Checkpoint variant or label order differs from this run.')
            conditioning_fold = parent['conditioning_fold']
            if args.fold != 'final' and conditioning_fold not in ('shared', args.fold):
                raise ValueError('The external checkpoint used target statistics from a different fold.')
            if parent['data_manifest_sha256'] != file_sha(args.data_dir / 'manifest.json'):
                raise ValueError('The prepared data differ from the external-stage experiment.')
        elif not args.plan_only:
            raise ValueError('Official training requires --init pointing to its external checkpoint.')
    frame = frame.reset_index(drop=True)
    if not len(frame) or (args.phase == 'official' and args.fold != 'final' and not len(valid)):
        raise ValueError('Empty training data or official validation fold.')
    if set(frame.prompt_id) & set(valid.prompt_id):
        raise ValueError('Training and validation prompts overlap.')
    label_ids = {label: index for index, label in enumerate(labels)}
    targets = frame.model.map(label_ids).to_numpy(dtype=np.int64)
    counts = np.bincount(targets, minlength=len(labels))
    if (counts == 0).any():
        raise ValueError('Every experiment must retain all competition classes.')
    # A fixed per-row objective preserves class balancing even in small batches.
    class_weights = len(frame) / (len(labels) * counts) if config['class_balance'] else np.ones(len(labels))
    weights = (soft_weights * class_weights[targets]).astype(np.float32)
    epochs = config['external_epochs'] if args.phase == 'external' else config['official_epochs']
    seed = config['seed'] + (args.fold if isinstance(args.fold, int) and args.phase == 'official' else 0)
    relations = dict(policy=config['centered_plan'], policy_source=stage_supervision['source'])
    relation_frame = None
    relation_data_path = relation_data_sha256 = None
    if args.phase == 'external' and config['centered_plan'] in anchored_policies:
        expected_plan = data_manifest.get('anchor_plan')
        requested_plan = dict(seed=seed, epochs=epochs, batch_size=config['batch_size'],
                              accumulation=config['accumulation'], max_updates=args.max_updates)
        if expected_plan is None or any(expected_plan.get(key) != value for key, value in requested_plan.items()):
            raise ValueError('Anchored controls require their frozen seed, epochs, batch size, accumulation and update budget. Rebuild the controls for another plan.')
        if expected_plan.get('mode', 'anchored_triplets') != config['centered_plan']:
            raise ValueError('The anchored forward mode differs from the frozen data plan; rebuild controls for a different objective.')
        anchor_path = args.data_dir / 'external_anchor.parquet'
        anchor = pd.read_parquet(anchor_path)
        separate = config['centered_plan'] == 'anchored_separate'
        if separate:
            relation_frame = align_anchor_rows(frame, anchor)
            relation_data_path, relation_data_sha256 = str(anchor_path.resolve()), file_sha(anchor_path)
        relations.update(anchor_path=str(anchor_path.resolve()), anchor_sha256=file_sha(anchor_path),
                         frozen_plan=requested_plan,
                         permit_plain_replacement_all=separate,
                         constraint='Identical class/length slot sequence; every centered text and partner preserved from anchor.',
                         forward_mode='separate_plain_and_anchor' if separate else 'shared_plain_forward')
        plan = [anchored_triplet_batches(frame, anchor, seed + epoch, config['batch_size'],
                                         permit_plain_replacement_all=separate)
                for epoch in range(epochs)]
    elif config['centered_plan'] in triplet_policies:
        if args.phase == 'external' and config['centered_plan'] in shared_policies:
            if variant == 'expanded':
                raise ValueError('Expanded data use grouped_triplets; shared_triplets needs equal-size paired pools.')
            peer_name = 'baseline' if variant == 'mix20' else 'mix20'
            peer_path = args.data_dir / f'external_{peer_name}.parquet'
            common, report = common_relation_rows(frame, pd.read_parquet(peer_path))
            relations.update(peer_path=str(peer_path.resolve()), peer_sha256=file_sha(peer_path))
        else:
            # Construct relations only after selecting this fold's training side.
            common, report = common_relation_rows(frame, frame)
        relations.update(report)
        plan = [shared_triplet_batches(frame, common, seed + epoch, config['batch_size'],
                                       stratify_plain=config['centered_plan'] == 'shared_stratified_triplets')
                for epoch in range(epochs)]
    else:
        plan = [prompt_batches(frame, seed + epoch, config['batch_size']) for epoch in range(epochs)]
    frozen_plan = getattr(args, 'prepared_plan', None)
    if frozen_plan is not None:
        plan = read_saved_plan(frozen_plan, epochs)
        relations['plan_source'] = str(Path(frozen_plan).resolve())
    windows_by_epoch = [update_windows(batches, config['batch_size'] * config['accumulation'])
                        for batches in plan]
    centered_weights, supervision = centered_loss_weights(
        frame if relation_frame is None else relation_frame, labels, weights, windows_by_epoch, balance=config['centered_balance'],
        max_updates=args.max_updates, enabled=config['centered_weight'] > 0)
    if frozen_plan is not None:
        frozen_weights = np.load(Path(frozen_plan) / 'centered_loss_weights.npy', allow_pickle=False)
        if not np.array_equal(centered_weights, frozen_weights):
            raise ValueError('Auxiliary weights differ from the reviewed actual update plan.')
    supervision['strength'] = config['centered_weight']
    for epoch_report in supervision['epochs']:
        epoch_report['effective_total_mass'] *= config['centered_weight']
        for item in epoch_report['per_class']:
            item['effective_centered_mass'] *= config['centered_weight']
            for size in item['by_group_size']:
                size['effective_mass'] *= config['centered_weight']
    summaries = [plan_summary(frame, epoch) for epoch in plan]
    row_ids = (frame if relation_frame is None else relation_frame).id.to_numpy()
    if relation_frame is not None:
        for summary in summaries:
            summary['centered_row_semantics'] = 'Immutable aligned anchor rows; plain row counts describe the intervention arm.'
    for windows, summary in zip(windows_by_epoch, summaries):
        sizes = [sum(len(rows) for rows, _ in chunk) for chunk in windows]
        summary.update(optimizer_updates=len(windows), minimum_update_rows=min(sizes), maximum_update_rows=max(sizes))
        events = [(index, sizes[index], row_ids[rows].tolist())
                  for index, chunk in enumerate(windows) for rows, same in chunk if same]
        summary['centered_context_sha256'] = hashlib.sha256(
            json.dumps(events, separators=(',', ':')).encode()).hexdigest()
    return frame, valid, labels, targets, weights, centered_weights, plan, dict(
        inner_split=inner_split,
        stage_supervision=stage_supervision,
        variant=variant, conditioning_fold=conditioning_fold, matching=matching,
        class_weights=class_weights.tolist(), epochs=summaries, relations=relations,
        relation_coverage=relation_coverage(frame if relation_frame is None else relation_frame,
                                            windows_by_epoch, args.max_updates, labels),
        supervision=supervision,
        evidence_reference=evidence_reference,
        sampling_evidence=plan_evidence(frame, windows_by_epoch, weights, centered_weights,
                                       args.max_updates, config['centered_weight'], relation_frame=relation_frame),
        relation_data_path=relation_data_path, relation_data_sha256=relation_data_sha256,
        data_path=str(path.resolve()),
        data_sha256=file_sha(path), data_manifest_sha256=file_sha(args.data_dir / 'manifest.json'),
        initialization_run=(str(args.init.resolve()) if args.init else
                            (resume_metadata or {}).get('initialization_run')),
        initialization_partial=bool(parent and parent['status'] == 'partial'),
        initialization_prepared_only=bool(parent and parent.get('prepared_only', False)),
        external_initialization_config=parent,
    )


def runtime(device_name, dtype_name):
    import torch
    if device_name == 'auto':
        device_name = 'cpu'
        if torch.cuda.is_available():
            device_name = 'cuda'
        elif torch.backends.mps.is_available():
            device_name = 'mps'
    device = torch.device(device_name)
    if device.type == 'cuda':
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
    if device.type not in ('cpu', 'mps', 'cuda'):
        raise ValueError('Supported devices are cpu, mps and cuda.')
    if dtype_name == 'auto':
        bf16 = device.type == 'mps' or (
            device.type == 'cuda' and torch.cuda.get_device_capability(device)[0] >= 8
            and torch.cuda.is_bf16_supported())
        dtype_name = 'bfloat16' if bf16 else ('float16' if device.type == 'cuda' else 'float32')
    if device.type == 'cpu' and dtype_name != 'float32':
        raise ValueError('Use float32 for CPU training.')
    if dtype_name == 'float16' and device.type != 'cuda':
        raise ValueError('Float16 training requires CUDA gradient scaling. Use bfloat16 or float32 on MPS.')
    return device, getattr(torch, dtype_name)


def branch_random_state(key, device):
    """A deterministic branch RNG scope, restored even after a failed update.

    The key is (training seed, epoch, successful-update cursor, physical pack,
    branch). It is recomputed on retry/resume; no hidden stream is advanced by
    a different plain token length. Checkpoint recomputation stays in the scope.
    """
    from contextlib import contextmanager
    import torch

    @contextmanager
    def scope():
        seed = int.from_bytes(hashlib.sha256(json.dumps(list(key), separators=(',', ':')).encode()).digest()[:8], 'big') % (2 ** 63 - 1)
        cpu_state = torch.get_rng_state()
        accelerator_state = (torch.cuda.get_rng_state(device) if device.type == 'cuda' else
                             torch.mps.get_rng_state() if device.type == 'mps' else None)
        try:
            # torch.manual_seed also changes other devices; only seed the
            # generators whose states are saved in this branch scope.
            torch.random.default_generator.manual_seed(seed)
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            elif device.type == 'mps':
                torch.mps.manual_seed(seed)
            yield
        finally:
            torch.set_rng_state(cpu_state)
            if device.type == 'cuda':
                torch.cuda.set_rng_state(accelerator_state, device)
            elif device.type == 'mps':
                torch.mps.set_rng_state(accelerator_state)
    return scope()


def backward_window(model, tokens, targets, weights, centered_weights, chunk, dropout, scaler, centered_weight,
                    forward_batch_rows=3, relation_tokens=None, branch_seed_key=None):
    import torch
    from torch.nn import functional as F

    row_count = sum(len(rows) for rows, _ in chunk)
    plain_total = centered_total = 0.0
    if relation_tokens is not None and branch_seed_key is None:
        raise ValueError('Separate anchor inputs require a deterministic (seed, epoch, update) RNG key.')
    for packed_index, groups in enumerate(packed_forwards(chunk, forward_batch_rows)):
        combined = [index for rows, _ in groups for index in rows]
        if relation_tokens is not None:
            # Always use two independent forwards, even if plain text equals
            # anchor text. The anchor is trainable, not cached or detached.
            with branch_random_state((*branch_seed_key, packed_index, 'plain'), model.device):
                features = model.representation([tokens[index] for index in combined])
                parts, offset = [], 0
                for rows, _ in groups:
                    z = features[offset:offset + len(rows)]
                    offset += len(rows)
                    y = torch.as_tensor(targets[rows], device=model.device)
                    w = torch.as_tensor(weights[rows], device=model.device)
                    parts.append((F.cross_entropy(model.plain(dropout(z)), y, reduction='none') * w).sum())
                plain = torch.stack(parts).sum()
                plain_value = float(plain.detach())
                if not math.isfinite(plain_value):
                    raise FloatingPointError(f'Nonfinite plain loss for training rows {combined}.')
                scaler.scale(plain / row_count).backward()
                plain_total += plain_value
            relation_groups = [(rows, same) for rows, same in groups if same]
            if relation_groups:
                relation_rows = [index for rows, _ in relation_groups for index in rows]
                with branch_random_state((*branch_seed_key, packed_index, 'anchor'), model.device):
                    relation_features = model.representation([relation_tokens[index] for index in relation_rows])
                    parts, offset = [], 0
                    for rows, _ in relation_groups:
                        z = relation_features[offset:offset + len(rows)]
                        offset += len(rows)
                        centered_z = z - z.mean(dim=0, keepdim=True)
                        y = torch.as_tensor(targets[rows], device=model.device)
                        auxiliary = torch.as_tensor(centered_weights[rows], device=model.device)
                        parts.append((F.cross_entropy(model.centered(dropout(centered_z)), y,
                                                     reduction='none') * auxiliary).sum())
                    centered = torch.stack(parts).sum()
                    centered_value = float(centered.detach())
                    if not math.isfinite(centered_value):
                        raise FloatingPointError(f'Nonfinite anchor loss for relation slots {relation_rows}.')
                    # Additional anchor rows never enlarge the plain window's
                    # denominator, nor do they receive a duplicate plain CE.
                    scaler.scale(centered_weight * centered / row_count).backward()
                    centered_total += centered_value
            continue
        # Historical shared-input behavior remains unchanged.
        features = model.representation([tokens[index] for index in combined])
        plain_parts, centered_parts, offset = [], [], 0
        for rows, same_prompt in groups:
            z = features[offset:offset + len(rows)]
            offset += len(rows)
            y = torch.as_tensor(targets[rows], device=model.device)
            w = torch.as_tensor(weights[rows], device=model.device)
            plain_parts.append((F.cross_entropy(model.plain(dropout(z)), y, reduction='none') * w).sum())
            if same_prompt:
                centered_z = z - z.mean(dim=0, keepdim=True)
                auxiliary = torch.as_tensor(centered_weights[rows], device=model.device)
                centered_parts.append((F.cross_entropy(model.centered(dropout(centered_z)), y,
                                                       reduction='none') * auxiliary).sum())
        plain = torch.stack(plain_parts).sum()
        centered = torch.stack(centered_parts).sum() if centered_parts else features.sum() * 0
        plain_value, centered_value = torch.stack([plain.detach(), centered.detach()]).tolist()
        if not math.isfinite(plain_value) or not math.isfinite(centered_value):
            raise FloatingPointError(f'Nonfinite forward loss for training rows {combined}.')
        scaler.scale((plain + centered_weight * centered) / row_count).backward()
        plain_total += plain_value
        centered_total += centered_value
    return row_count, plain_total, centered_total


def train(model, tokens, targets, weights, centered_weights, plan, config, phase, run_dir, max_updates=None,
          resume=None, checkpoint_every=None, stop_after_updates=None, segment_seconds=None, metadata=None,
          relation_tokens=None, frozen_positions=None, frozen_full_updates=None,
          token_stats=None, relation_token_stats=None, development_evaluator=None,
          checkpoint_seconds=None, segment_started_at=None):
    import torch
    from .q8_checkpoint import save_training_checkpoint, load_training_state, execution_policy

    optimizer, parameters = make_optimizer(model, config, phase)
    windows = [update_windows(epoch, config['batch_size'] * config['accumulation']) for epoch in plan]
    all_positions = [(epoch, window, chunk) for epoch, chunks in enumerate(windows)
                     for window, chunk in enumerate(chunks)]
    planned = len(all_positions)
    target_updates = min(planned, max_updates) if max_updates else planned
    from .q8_observation import observation_plan, save_learning_snapshot
    observations = observation_plan(windows, config, phase, target_updates)
    write_json(run_dir / 'observation_plan.json', observations)
    snapshot_events = observations['snapshot_events']
    diagnostic_updates = set(observations['diagnostic_updates'])
    positions = list(frozen_positions) if frozen_positions is not None else all_positions[:target_updates]
    if frozen_positions is not None:
        if frozen_full_updates != planned or len(positions) != target_updates:
            raise ValueError('Frozen actual/full update budgets differ from the training plan.')
        if positions != all_positions[:target_updates]:
            raise ValueError('The model must consume the reviewed update windows exactly.')
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, target_updates, config['warmup_fraction']))
    base_rates = optimizer_group_rates(config, phase)
    scale = model.device.type == 'cuda' and model.compute_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda', init_scale=256.0, enabled=scale)
    dropout = torch.nn.Dropout(config['head_dropout'])
    step, exposed, relation_exposed, previous_elapsed = 0, [], [], 0.0
    saved_policy = None
    if relation_tokens is not None and len(relation_tokens) != len(tokens):
        raise ValueError('Plain and anchor token arrays must cover the same aligned slots.')
    if resume is not None:
        restored = load_training_state(resume, optimizer, scheduler, scaler, model.device)
        saved_policy = restored['metadata'].get('execution_policy')
        progress = restored['progress']
        step = int(progress['step'])
        exposed = list(progress['exposed_rows'])
        previous_elapsed = float(progress['elapsed_seconds'])
        if progress['target_updates'] != target_updates or progress['planned_updates'] != planned:
            raise ValueError('Resume must retain the original research update budget and schedule.')
        if not 0 <= step <= target_updates:
            raise ValueError('Checkpoint update is outside the planned research budget.')
        expected_rows = [row for _, _, chunk in positions[:step] for rows, _ in chunk for row in rows]
        if exposed != expected_rows:
            raise ValueError('Checkpoint exposure does not match the reconstructed training plan.')
        if relation_tokens is not None:
            relation_exposed = list(progress.get('relation_exposed_rows', []))
            expected_relations = [row for _, _, chunk in positions[:step]
                                  for rows, same in chunk if same for row in rows]
            if relation_exposed != expected_relations:
                raise ValueError('Checkpoint anchor exposure does not match the frozen relation plan.')
            if progress.get('relation_data_sha256') != (metadata or {}).get('relation_data_sha256'):
                raise ValueError('Checkpoint anchor input differs from the current relation data.')
        elif progress.get('relation_exposed_rows'):
            raise ValueError('A separate-relation checkpoint cannot resume with a shared input stream.')
        next_position = all_positions[step][:2] if step < planned else (len(windows), 0)
        if (progress['next_epoch'], progress['next_window']) != next_position:
            raise ValueError('Checkpoint cursor differs from the reconstructed training plan.')
    initial_step = step
    policy = execution_policy(saved=(metadata or {}).get('execution_policy', saved_policy),
                              checkpoint_every=checkpoint_every, checkpoint_seconds=checkpoint_seconds,
                              stop_after_updates=stop_after_updates, segment_seconds=segment_seconds)
    metadata = dict(metadata or {}, execution_policy=policy)
    metadata.setdefault('settings', config)
    source_intervention = metadata.get('source_intervention') or {}
    source_changes = None
    if phase == 'external' and source_intervention.get('status') == 'available':
        if file_sha(Path(source_intervention['changed_rows_path'])) != source_intervention['changed_rows_sha256']:
            raise ValueError('Source replacement rows changed after their scope was recorded.')
        source_changes = pd.read_parquet(source_intervention['changed_rows_path'])

    def source_exposure():
        if source_changes is None:
            return metadata.get('actual_source_intervention_exposure')
        from .q8_source_scope import source_exposure_summary
        return source_exposure_summary(source_intervention, exposed, changed_rows=source_changes)
    write_json(run_dir / 'execution_policy.json', policy)
    checkpoint_every, checkpoint_seconds = policy['checkpoint_every'], policy['checkpoint_seconds']
    segment_seconds, stop_after_updates = policy['segment_seconds'], policy['stop_after_updates']
    stop_at = min(target_updates, step + stop_after_updates) if stop_after_updates else target_updates
    start = time.perf_counter()
    segment_started_at = start if segment_started_at is None else segment_started_at
    last_save_time = start
    last_checkpoint = None
    last_saved_step = None

    def segment_expired():
        return bool(segment_seconds and time.perf_counter() - segment_started_at >= segment_seconds)

    def completed_centered_mass(count):
        return sum(float(centered_weights[epoch, rows].astype(np.float64).sum())
                   * config['centered_weight'] / sum(len(part) for part, _ in chunk)
                   for epoch, _, chunk in positions[:count] for rows, same in chunk if same)

    actual_centered_mass = completed_centered_mass(step)

    def checkpoint(reason):
        nonlocal last_checkpoint, last_saved_step, last_save_time
        if last_saved_step == step:
            return last_checkpoint
        epoch, window = all_positions[step][:2] if step < planned else (len(windows), 0)
        progress = dict(step=step, next_epoch=epoch, next_window=window, exposed_rows=exposed,
                        elapsed_seconds=previous_elapsed + time.perf_counter() - start,
                        planned_updates=planned, target_updates=target_updates, save_reason=reason)
        if relation_tokens is not None:
            progress.update(relation_exposed_rows=relation_exposed,
                            relation_data_sha256=(metadata or {}).get('relation_data_sha256'),
                            relation_input_mode='separate_frozen_anchor')
        last_checkpoint = save_training_checkpoint(
            run_dir / 'checkpoints' / f'step_{step:08d}', model, optimizer, scheduler, scaler,
            progress, dict(metadata, actual_centered_coefficient_mass=actual_centered_mass,
                           actual_source_intervention_exposure=source_exposure()),
            keep=int(config.get('checkpoint_keep', 2)))
        last_saved_step = step
        last_save_time = time.perf_counter()
        print(f'Checkpoint ({reason}): {last_checkpoint}', flush=True)
        return last_checkpoint

    pending_observation = False

    def observe():
        nonlocal pending_observation
        # Also replay this event on resume: a process may have stopped after
        # saving recovery state but before completing its held-out observation.
        checkpoint('before_observation')
        next_epoch, next_window = all_positions[step][:2] if step < planned else (len(windows), 0)
        progress = dict(step=step, next_epoch=next_epoch, next_window=next_window, exposed_rows=exposed,
                        elapsed_seconds=previous_elapsed + time.perf_counter() - start,
                        planned_updates=planned, target_updates=target_updates,
                        actual_centered_coefficient_mass=actual_centered_mass)
        if relation_tokens is not None:
            progress.update(relation_exposed_rows=relation_exposed,
                            relation_data_sha256=metadata.get('relation_data_sha256'))
        note = save_learning_snapshot(run_dir / 'snapshots' / f'step_{step:08d}', model,
            optimizer, scheduler, scaler, progress,
            dict(metadata, actual_source_intervention_exposure=source_exposure()),
            snapshot_events[str(step)], len(snapshot_events), development_evaluator,
            evaluation_deadline=segment_started_at + segment_seconds if segment_seconds else None)
        pending_observation = note.get('development_status') == 'pending'

    model.train()
    if resume is not None and str(step) in snapshot_events:
        observe()
    with (run_dir / 'train_log.jsonl').open('w') as logfile:
        while step < stop_at:
            # The wall clock includes preparation when called through run().
            # Finish an in-flight optimizer window; never shorten its schedule.
            if segment_expired():
                break
            epoch_index, window_index, chunk = positions[step]
            chunk_rows = [row for rows, _ in chunk for row in rows]
            diagnostic = before_parameters = None
            diagnostic_seconds = 0.
            if step + 1 in diagnostic_updates:
                diagnostic_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                from .q8_diagnostics import diagnose_window, capture_parameters
                observer_scaler = torch.amp.GradScaler('cuda', init_scale=scaler.get_scale(), enabled=scale)
                def diagnostic_backward(strength):
                    return backward_window(
                        model, tokens, targets, weights, centered_weights[epoch_index], chunk, dropout,
                        observer_scaler, strength, config.get('forward_batch_rows', config['batch_size']),
                        relation_tokens=relation_tokens,
                        branch_seed_key=(config['seed'], epoch_index, step) if relation_tokens is not None else None)
                diagnostic = diagnose_window(
                    model, diagnostic_backward, centered_weight=config['centered_weight'],
                    gradient_scale=observer_scaler.get_scale(), clip_norm=config['gradient_clip'],
                    context=dict(update=step + 1, epoch_index=epoch_index, window=window_index,
                                 row_positions=chunk_rows, plan_sha256=(metadata or {}).get('plan_sha256'),
                                 actual_exposure_sha256=(metadata or {}).get('actual_exposure_sha256'),
                                 target_updates=target_updates,
                                 selection='Predeclared successful-update numbers, not selected from observed loss.'))
                before_parameters = capture_parameters(model)
                diagnostic_seconds = time.perf_counter() - diagnostic_start
            synchronize_device(model.device)
            compute_start = time.perf_counter()
            if scale:
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state(model.device)
            for retry in range(9):
                optimizer.zero_grad(set_to_none=True)
                row_count, plain_total, centered_total = backward_window(
                    model, tokens, targets, weights, centered_weights[epoch_index],
                    chunk, dropout, scaler, config['centered_weight'],
                    config.get('forward_batch_rows', config['batch_size']), relation_tokens=relation_tokens,
                    branch_seed_key=(config['seed'], epoch_index, step) if relation_tokens is not None else None)
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip']))
                if math.isfinite(grad_norm):
                    break
                if not scale or retry == 8:
                    write_json(run_dir / 'nonfinite_batch.json',
                               dict(epoch=epoch_index + 1, next_update=step + 1, rows=chunk_rows))
                    raise FloatingPointError('Nonfinite gradients after scaling. No invalid optimizer update was applied.')
                scaler = torch.amp.GradScaler('cuda', init_scale=scaler.get_scale() / 2, enabled=True)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng, model.device)
            rates = [group['lr'] for group in optimizer.param_groups]
            expected_rates = {name: value * lr_multiplier(step, target_updates, config['warmup_fraction'])
                              for name, value in base_rates.items()}
            if any(not math.isclose(group['lr'], expected_rates[group['name']], rel_tol=1e-12, abs_tol=0)
                   for group in optimizer.param_groups):
                raise ValueError('Optimizer learning rates diverged from the frozen per-update schedule.')
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            synchronize_device(model.device)
            update_compute_seconds = time.perf_counter() - compute_start
            if diagnostic is not None:
                diagnostic_start = time.perf_counter()
                from .q8_diagnostics import measure_parameter_updates
                diagnostic['actual_parameter_updates'] = measure_parameter_updates(model, before_parameters)
                diagnostic['actual_preclip_global_norm'] = grad_norm
                diagnostic['actual_clip_multiplier'] = min(1., config['gradient_clip'] / (grad_norm + 1e-6))
                diagnostic['actual_optimizer_group_lrs'] = {group['name']: rate for group,rate in zip(optimizer.param_groups,rates)}
                diagnostic['finite_gradient_retries'] = retry
                diagnostic_dir = run_dir / 'gradient_diagnostics'
                diagnostic_dir.mkdir(exist_ok=True)
                write_json(diagnostic_dir / f'update_{step:08d}.json', diagnostic)
                diagnostic_seconds += time.perf_counter() - diagnostic_start
            exposed.extend(chunk_rows)
            for rows, same in chunk:
                if same:
                    actual_centered_mass += (float(centered_weights[epoch_index, rows].astype(np.float64).sum())
                                             * config['centered_weight'] / row_count)
            relation_rows = [row for rows, same in chunk if same for row in rows]
            if relation_tokens is not None:
                relation_exposed.extend(relation_rows)
            record = dict(update=step, epoch=epoch_index + 1, window=window_index, rows=row_count,
                          plain_ce=plain_total / row_count, centered_ce=centered_total / row_count,
                          weighted_centered_ce=config['centered_weight'] * centered_total / row_count,
                          adapter_lr=rates[0], head_lr=rates[-1],
                          optimizer_group_lrs={group['name']: rate for group, rate in zip(optimizer.param_groups, rates)},
                          grad_norm=grad_norm, gradient_scale_retries=retry,
                          update_compute_seconds=update_compute_seconds, gradient_diagnostic_seconds=diagnostic_seconds,
                          elapsed_seconds=round(previous_elapsed + time.perf_counter() - start, 3),
                          segment_elapsed_seconds=round(time.perf_counter() - start, 3))
            if relation_tokens is not None:
                record.update(plain_forward_rows=row_count, relation_forward_rows=len(relation_rows),
                              relation_input_mode='separate_frozen_anchor')
            if token_stats is not None:
                from .q8_tokens import summarize_exposure, physical_token_counts
                record['plain_token_exposure'] = summarize_exposure(token_stats, chunk_rows)
                record['centered_token_exposure'] = summarize_exposure(
                    relation_token_stats if relation_token_stats is not None else token_stats, relation_rows)
                physical = physical_token_counts(tokens, chunk, config.get('forward_batch_rows', config['batch_size']), relation_tokens)
                record['physical_tokens'] = {name: sum(item[name] for item in physical)
                                             for name in ('rows', 'input_tokens', 'padded_tokens', 'padding_tokens')}
                record['physical_tokens_including_retries'] = {
                    name: value * (retry + 1) for name, value in record['physical_tokens'].items()}
            logfile.write(json.dumps(record) + '\n')
            logfile.flush()
            if step == initial_step + 1 or step % 25 == 0 or step == target_updates:
                print(f'update {step}/{target_updates} plain={record["plain_ce"]:.4f} centered={record["centered_ce"]:.4f}', flush=True)
            if checkpoint_every and step % checkpoint_every == 0:
                checkpoint('update_interval')
            elif checkpoint_seconds and time.perf_counter() - last_save_time >= checkpoint_seconds:
                checkpoint('time_interval')
            if str(step) in snapshot_events:
                observe()
            if segment_expired():
                break
    stop_reason = ('segment_seconds' if pending_observation else
                   'target_reached' if step >= target_updates else
                   'stop_after_updates' if step >= stop_at else 'segment_seconds')
    checkpoint(stop_reason)
    np.save(run_dir / 'actual_row_indices.npy', np.asarray(exposed, dtype=np.int32))
    if relation_tokens is not None:
        np.save(run_dir / 'actual_relation_row_indices.npy', np.asarray(relation_exposed, dtype=np.int32))
    elapsed = time.perf_counter() - start
    token_metrics = {}
    if token_stats is not None:
        from .q8_tokens import summarize_exposure
        centered_exposed = [row for _, _, chunk in positions[:step]
                            for rows, same in chunk if same for row in rows]
        token_metrics = dict(actual_plain_tokens=summarize_exposure(token_stats, exposed),
                             actual_centered_tokens=summarize_exposure(
                                 relation_token_stats if relation_token_stats is not None else token_stats,
                                 centered_exposed))
    return dict(successful_updates=step, segment_updates=step - initial_step,
                actual_source_intervention_exposure=source_exposure(),
                actual_centered_coefficient_mass=actual_centered_mass,
                planned_updates=planned, target_updates=target_updates,
                row_exposures=len(exposed), unique_training_rows=len(set(exposed)),
                separate_relation_row_exposures=len(relation_exposed),
                unique_separate_relation_rows=len(set(relation_exposed)),
                partial=step < planned, interrupted=step < target_updates or pending_observation,
                pending_observation_updates=[step] if pending_observation else [],
                train_seconds=round(previous_elapsed + elapsed, 3), segment_train_seconds=round(elapsed, 3),
                segment_wall_seconds=round(time.perf_counter() - segment_started_at, 3),
                stop_reason=stop_reason, execution_policy=policy,
                last_checkpoint=str(last_checkpoint), resume_from=str(resume) if resume else None, **token_metrics)


def evaluate(model, fit_frame, frame, labels, fold, config, run_dir):
    """Evaluate same-checkpoint heads and train-fitted Gaussian on unseen prompts."""
    from .q8_evaluation import evaluate_cohorts

    if set(fit_frame.prompt_id) & set(frame.prompt_id):
        raise ValueError('Evaluation fitting and query prompts must be disjoint.')
    use_gaussian = config.get('eval_gaussian', True)
    batch_size = config['eval_batch_size']
    query = model.predict_outputs(model.encode(frame.text.tolist()), batch_size,
                                  keep_features=use_gaussian)
    fitted = (model.predict_outputs(model.encode(fit_frame.text.tolist()), batch_size,
                                   keep_features=True)['features'] if use_gaussian else None)
    metadata = json.loads((run_dir / 'config.json').read_text())
    history = [metadata]
    if metadata.get('external_initialization_config'):
        history.append(metadata['external_initialization_config'])
    elif metadata.get('initialization_run'):
        history.append(json.loads((Path(metadata['initialization_run']) / 'config.json').read_text()))
    from .q8_stages import centered_training_history
    centered_evidence = centered_training_history(metadata)
    # A positive lambda alone is insufficient (e.g. native single-row batches).
    # A head trained in the external stage remains available after plain-only
    # adaptation; absent legacy evidence is not guessed to mean trained.
    centered_trained = any(stage['effective_mass'] > 0 for stage in centered_evidence)
    condition = dict(settings={key: value for key, value in config.items() if not key.startswith('eval_')},
                     variant=metadata['variant'],
                     data_manifest_sha256=metadata['data_manifest_sha256'],
                     base=metadata['base'], base_config_sha256=model.base_config_sha256,
                     training_code_sha256=metadata['training_code_sha256'],
                     active_updates=metadata.get('supervision', {}).get('active_updates'),
                     actual_dtype=metadata.get('dtype'),
                     external_settings=({key: value for key, value in history[1]['settings'].items()
                                         if not key.startswith('eval_')} if len(history) > 1 else None),
                     external_active_updates=(history[1].get('supervision', {}).get('active_updates')
                                              if len(history) > 1 else None),
                     external_actual_dtype=history[1].get('dtype') if len(history) > 1 else None)
    provenance = dict(model_kind='q8_lora_fold', checkpoint=str(run_dir.resolve()), fold=int(fold),
                      data_governance=metadata.get('data_governance'),
                      validation_scope=metadata.get('validation_scope'),
                      source_intervention=metadata.get('source_intervention'),
                      actual_source_intervention_exposure=metadata.get('actual_source_intervention_exposure'),
                      data_manifest_sha256=metadata['data_manifest_sha256'],
                      base_config_sha256=model.base_config_sha256,
                      model_config=model.model_config,
                      fit_ids=fit_frame.id.astype(str).tolist(),
                      fit_prompts=sorted(fit_frame.prompt_id.astype(str).unique().tolist()),
                      official_prompt_holdout=True,
                      centered_training_evidence=centered_evidence,
                      condition_signature=hashlib.sha256(json.dumps(
                          condition, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                      evaluation_scope='Official prompt holdout with fixed public external-data history; not a full-system independent OOF claim.')
    if metadata.get('inner_split'):
        from .q8_inner import checkpoint_identity
        artifacts = {str(path.relative_to(run_dir)): path for path in run_dir.rglob('*.safetensors')
                     if 'checkpoints' not in path.relative_to(run_dir).parts and 'snapshots' not in path.relative_to(run_dir).parts}
        for name in ('model_config.json', 'base_identity.json'):
            if (run_dir / name).exists(): artifacts[name] = run_dir / name
        identity = checkpoint_identity(artifacts)
        provenance.update(role='outer_test', split_sha256=metadata['inner_split']['split_sha256'],
                          model_identity_sha256=identity['model_identity_sha256'],
                          model_scope='q8_inner_fit')
    evaluation_config = dict(grouping_protocol=config.get('eval_grouping_protocol', 'train_fold'),
                             gaussian_noninjective=config.get('eval_gaussian_noninjective', True),
                             cluster_threshold=config.get('eval_cluster_threshold', .15),
                             seeds=config.get('eval_cohort_seeds', [42, 43]))
    if config.get('eval_imbalance_suite') is not None:
        evaluation_config['imbalance_suite'] = config['eval_imbalance_suite']
    report = evaluate_cohorts(
        fit_frame, frame.assign(fold=fold), labels, query['plain_logits'],
        query['raw_centered_logits'], query['centered_bias'], run_dir / 'evaluation',
        query_features=query['features'], fit_features=fitted,
        centered_trained=centered_trained, provenance=provenance, config=evaluation_config)
    # Preserve the previous plain-only readers while the richer branch outputs
    # remain explicitly named under evaluation/<cohort>/.
    shutil.copy2(run_dir / 'evaluation/full/plain_predictions.parquet',
                 run_dir / 'valid_predictions.parquet')
    np.save(run_dir / 'valid_logits.npy', query['plain_logits'])
    return dict(**report['cohorts']['full']['views']['plain'], evaluation=report,
                prediction_scope='Named head and train-fitted Gaussian views, recomputed per text-derived cohort; legacy root predictions remain plain.')


def run(args):
    segment_started_at = time.perf_counter()
    if getattr(args, 'backend_initial_state', None) is not None:
        if getattr(args, 'initial_state', None) is not None and args.initial_state != args.backend_initial_state:
            raise ValueError('Choose one common initialization path, not conflicting backend/strict paths.')
        args.initial_state = args.backend_initial_state
    resume = getattr(args, 'resume', None)
    prepared = getattr(args, 'prepared_plan', None)
    if (resume is not None or prepared is not None) and getattr(args, 'experiment', None) is not None:
        raise ValueError('Prepared plans and resumed runs use their saved settings; do not select a new experiment.')
    if resume is not None and prepared is not None:
        raise ValueError('A resume already carries its frozen plan; do not also supply --prepared-plan.')
    prepared_metadata = None
    if prepared is not None:
        if not (prepared / 'exposure_manifest.json').is_file():
            raise ValueError('--prepared-plan requires a frozen actual-update manifest, not a historical score.')
        prepared_metadata = json.loads((prepared / 'config.json').read_text())
    resume_metadata = None
    if resume is not None:
        if args.init is not None:
            raise ValueError('--resume restores a training cursor; do not also supply --init.')
        resume_metadata = json.loads((resume / 'run_config.json').read_text())
        args.initial_state = bind_initial_state(resume_metadata, getattr(args, 'initial_state', None))
        config = dict(resume_metadata['settings'])
        for name in ('phase', 'fold', 'variant', 'max_updates'):
            requested, saved = getattr(args, name, None), resume_metadata.get(name)
            if requested is not None and requested != saved:
                raise ValueError(f'Resume must retain {name}={saved!r}; a different budget is a new experiment.')
            setattr(args, name, saved)
        if args.base is None and resume_metadata.get('base'):
            args.base = Path(resume_metadata['base'])
        if args.data_dir == Path('data/q8'):
            args.data_dir = Path(resume_metadata['data_path']).parent
    elif prepared_metadata is not None:
        config = dict(prepared_metadata['settings'])
        for name in ('phase', 'fold', 'variant', 'max_updates'):
            requested, saved = getattr(args, name, None), prepared_metadata.get(name)
            if requested is not None and requested != saved:
                raise ValueError(f'Frozen plan requires {name}={saved!r}; prepare a new experiment to change it.')
            setattr(args, name, saved)
        if args.data_dir == Path('data/q8'):
            args.data_dir = Path(prepared_metadata['data_path']).parent
        if args.init is None and prepared_metadata.get('initialization_run'):
            args.init = Path(prepared_metadata['initialization_run'])
        args.initial_state = bind_initial_state(prepared_metadata, getattr(args, 'initial_state', None))
        if getattr(args, 'tokenizer', None) is None and prepared_metadata.get('tokenizer_path'):
            args.tokenizer = Path(prepared_metadata['tokenizer_path'])
        if args.base is None and prepared_metadata.get('base'):
            args.base = Path(prepared_metadata['base'])
        for name in ('centered_plan', 'centered_balance', 'centered_weight'):
            requested = getattr(args, name, None)
            if requested is not None and requested != config[name]:
                raise ValueError(f'Frozen plan cannot change {name}.')
    else:
        config = load_q8_config(args.config, getattr(args, 'experiment', None))
        if args.phase is None:
            raise ValueError('Choose --phase external/official, or supply --resume.')
        args.fold = args.fold if args.fold is not None else 'shared'
    from .q8_checkpoint import execution_policy, EXECUTION_DEFAULTS
    saved_execution = (resume_metadata or prepared_metadata or {}).get('execution_policy')
    # Old frozen plans need no rewriting. Operational overrides never enter
    # settings, its identity, loss normalization or the total LR horizon.
    policy = execution_policy(config_path=args.config if args.config.is_file() else None,
                              saved=saved_execution,
                              **{name: getattr(args, name, None) for name in EXECUTION_DEFAULTS})
    defaults = dict(backend='hf', optimizer='adamw', loraplus_ratio=16.0, fused_adamw=False,
                    forward_batch_rows=config['batch_size'])
    for name, default in defaults.items():
        config.setdefault(name, default)
        value = getattr(args, name, None)
        if value is not None:
            if (resume_metadata or prepared_metadata) and value != config[name]:
                raise ValueError(f'A frozen experiment must retain the original {name}.')
            config[name] = value
    if config['backend'] not in ('hf', 'unsloth', 'liger'):
        raise ValueError('backend must be hf, unsloth or liger.')
    if config.get('fusion_protocol', 'inner_prompt_fusion_v1') not in ('inner_prompt_fusion_v1', 'calibrated_inner_v2', 'crossfit_calibrated_inner_v3'):
        raise ValueError('Choose inner_prompt_fusion_v1, calibrated_inner_v2 or crossfit_calibrated_inner_v3 explicitly.')
    if config.get('fusion_protocol') in ('calibrated_inner_v2', 'crossfit_calibrated_inner_v3') and (
            not config.get('research_inner_validation') or not config.get('eval_gaussian', True)
            or not config.get('eval_gaussian_noninjective', True)
            or config.get('eval_grouping_protocol', 'train_fold') != 'train_fold'):
        raise ValueError('Calibrated fusion requires fixed inner validation and all Gaussian views with train-fold grouping.')
    if getattr(args, 'backend_initial_state', None) is not None and config.get('initialization_mode') != 'canonical_backend':
        raise ValueError('--backend-initial-state requires an explicitly derived canonical_backend experiment.')
    backend_classifier = None
    if config['backend'] == 'unsloth' and not args.plan_only:
        import sys
        if 'transformers' in sys.modules and 'ntllm.unsloth_model' not in sys.modules:
            raise RuntimeError('Unsloth training requires a fresh process before Transformers/tokenizer imports.')
        from .unsloth_model import UnslothQ8Classifier
        backend_classifier = UnslothQ8Classifier
    for key in ('batch_size', 'accumulation', 'external_epochs', 'official_epochs', 'eval_batch_size', 'forward_batch_rows'):
        if config[key] < 1:
            raise ValueError(f'{key} must be positive.')
    if config['forward_batch_rows'] < config['batch_size']:
        raise ValueError('forward_batch_rows cannot split a logical training microbatch.')
    frame, valid, labels, y, weights, centered_weights, plan, details = prepare(args, config, resume_metadata)
    # Governance annotates source evidence and interpretation; it never changes
    # the frozen rows, labels, sampling weights, or optimizer schedule.
    from .q8_governance import load_governance
    pool_name = details['variant'] if details['variant'] in ('mix20', 'expanded') else 'baseline'
    external_parent = details.get('external_initialization_config') or {}
    recorded_governance = ((resume_metadata or {}).get('data_governance') or
                           (external_parent.get('data_governance') if args.phase == 'official' else None))
    if recorded_governance and recorded_governance.get('status') == 'available':
        directory = (getattr(args, 'governance_dir', None)
                     or Path(recorded_governance['manifest_path']).parent)
        governance = load_governance(args.data_dir, pool_name, directory=directory)
        if (governance.get('status') != 'available' or
                governance['manifest_sha256'] != recorded_governance['manifest_sha256']):
            raise ValueError('Resume and official adaptation must retain the recorded governance version.')
    else:
        governance = load_governance(args.data_dir, pool_name, directory=getattr(args, 'governance_dir', None))
    governed_data_sha = details['data_sha256'] if args.phase == 'external' else external_parent.get('data_sha256')
    if (governance.get('status') == 'available' and governed_data_sha is not None
            and governance['data_sha256'] != governed_data_sha):
        raise ValueError('Data governance belongs to a different external training pool.')
    details['data_governance'] = governance
    details['validation_scope'] = dict(
        kind='fixed_public_pool_development',
        official_prompt_holdout=bool(len(valid) or details.get('inner_split')),
        upstream_official_dependence_removed=False,
        independent_model_version_verification=False,
        interpretation='Prompt-disjoint folds support development comparisons conditional on these public sources. They do not undo upstream official-informed prompt design, length targets or selection, nor independently verify source model versions.')
    from .q8_source_scope import load_source_scope
    recorded_source_scope = ((resume_metadata or {}).get('source_intervention') or
                             (external_parent.get('source_intervention') if args.phase == 'official' else None))
    scope_directory = getattr(args, 'source_scope_dir', None)
    if recorded_source_scope and recorded_source_scope.get('status') == 'available':
        scope_directory = scope_directory or Path(recorded_source_scope['manifest_path']).parent
    source_scope = load_source_scope(args.data_dir, directory=scope_directory)
    if recorded_source_scope and recorded_source_scope.get('status') == 'available' and (
            source_scope.get('status') != 'available' or source_scope['manifest_sha256'] != recorded_source_scope['manifest_sha256']):
        raise ValueError('Resume and official adaptation must retain the recorded source-intervention scope.')
    if source_scope.get('status') == 'available':
        if governed_data_sha is not None and source_scope['data_sha256'].get(pool_name) != governed_data_sha:
            raise ValueError('Source-intervention scope belongs to a different external training arm.')
        source_scope = dict(source_scope, arm_variant=pool_name)
    details['source_intervention'] = source_scope
    if args.phase == 'official':
        details['actual_source_intervention_exposure'] = external_parent.get('actual_source_intervention_exposure')
    relation_frame = None
    if details.get('relation_data_path'):
        from .q8_supervision import align_anchor_rows
        relation_path = Path(details['relation_data_path'])
        if file_sha(relation_path) != details['relation_data_sha256']:
            raise ValueError('Frozen relation data changed after the training plan was prepared.')
        relation_frame = align_anchor_rows(frame, pd.read_parquet(relation_path))
        details['relation_forward_policy'] = 'Independent frozen-anchor forward for every centered group; plain and relation exposures are recorded separately.'
        details['branch_rng_scheme'] = 'sha256-json(seed,epoch,successful_update,packed_index,plain_or_anchor); first64bits modulo 2**63-1; CPU and active device state restored after each forward/backward branch.'
    batches = [(epoch, rows, same) for epoch, items in enumerate(plan) for rows, same in items]
    plan_arrays = dict(rows=np.asarray([row for _, rows, _ in batches for row in rows], dtype=np.int32),
                       sizes=np.asarray([len(rows) for _, rows, _ in batches], dtype=np.int32),
                       same_prompt=np.asarray([same for _, _, same in batches], dtype=bool),
                       epochs=np.asarray([epoch for epoch, _, _ in batches], dtype=np.int16))
    fingerprint = hashlib.sha256()
    for name, values in {**plan_arrays, 'plain_weights': weights, 'centered_weights': centered_weights}.items():
        fingerprint.update(json.dumps([name, str(values.dtype), list(values.shape)]).encode())
        fingerprint.update(values.tobytes())
    details['plan_sha256'] = fingerprint.hexdigest()
    if prepared_metadata:
        for name in ('data_sha256', 'data_manifest_sha256', 'plan_sha256', 'relation_data_sha256'):
            if details.get(name) != prepared_metadata.get(name):
                raise ValueError(f'Current {name} differs from the reviewed experiment.')
        if labels != prepared_metadata['labels'] or config != prepared_metadata['settings']:
            raise ValueError('Frozen labels or training settings changed.')
        if args.phase == 'official':
            expected_parent = prepared_metadata.get('external_initialization_config')
            current_parent = details.get('external_initialization_config')
            if expected_parent and current_parent:
                for key in ('phase', 'variant', 'max_updates', 'data_sha256', 'data_manifest_sha256', 'plan_sha256', 'labels', 'settings'):
                    if expected_parent.get(key) != current_parent.get(key):
                        raise ValueError(f'Official adaptation switched its external experiment: {key}.')
                expected_root = (expected_parent.get('shared_initialization') or expected_parent.get('initialization_proof') or {}).get('initialization_sha256')
                current_root = (current_parent.get('shared_initialization') or current_parent.get('initialization_proof') or {}).get('initialization_sha256')
                if expected_root and expected_root != current_root:
                    raise ValueError('Official adaptation switched the common external initial parameters.')
                if not expected_root and expected_parent.get('shared_initialization_path') != current_parent.get('shared_initialization_path'):
                    raise ValueError('Official adaptation switched the pending common external start binding.')
    if resume_metadata:
        for name in ('data_sha256', 'data_manifest_sha256', 'plan_sha256'):
            if details[name] != resume_metadata[name]:
                raise ValueError(f'Resume {name} differs from the saved experiment.')
        if details.get('relation_data_sha256') != resume_metadata.get('relation_data_sha256'):
            raise ValueError('Resume must retain the same separate relation input, including its absence.')
        if labels != resume_metadata['labels'] or config != resume_metadata['settings']:
            raise ValueError('Resume label order or experiment settings changed.')
        if getattr(args, 'dtype', None) is None:
            args.dtype = resume_metadata['dtype'].removeprefix('torch.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    arm = 'off' if config['centered_weight'] == 0 else 'on'
    run_dir = args.output_dir / f'{stamp}_{args.phase}_{details["variant"]}_{arm}_f{args.fold}_{uuid4().hex[:6]}'
    run_dir.mkdir(parents=True, exist_ok=False)
    source_files = ('train_lora.py', 'q8_config.py', 'q8_sampling.py', 'q8_supervision.py', 'q8_compute.py',
                    'q8_observation.py', 'q8_diagnostics.py', 'q8_inner.py', 'q8_fusion.py', 'q8_crossfit_fusion.py',
                    'q8_fusion_stability.py', 'q8_evaluation.py', 'q8_cohort_suite.py', 'q8_group_diagnostics.py',
                    'q8_backend_compare.py', 'q8_stages.py',
                    'q8_exposure.py', 'q8_tokens.py', 'q8_initialization.py',
                    'q8_checkpoint.py', 'q8_governance.py', 'q8_source_scope.py', 'lora_model.py', 'unsloth_model.py', 'liger_model.py')
    metadata = dict(details, phase=args.phase, fold=args.fold, labels=labels, settings=config,
                    execution_policy=policy,
                    status='planned', prepared_only=args.plan_only, training_rows=len(frame), validation_rows=len(valid),
                    base=str(args.base.resolve()) if args.base else None, max_updates=args.max_updates,
                    prepared_plan_source=str(prepared.resolve()) if prepared else None,
                    shared_initialization_path=(str(args.initial_state.resolve())
                                                if getattr(args, 'initial_state', None) else None),
                    tokenizer_path=(str(args.tokenizer.resolve()) if getattr(args, 'tokenizer', None) else None),
                    resume_from=str(resume.resolve()) if resume else None,
                    training_code_sha256={name: file_sha(Path(__file__).with_name(name)) for name in source_files
                                          if Path(__file__).with_name(name).exists()},
                    recipe='Controlled reconstruction using q8v5 architecture; unpublished author training settings are explicit project choices.')
    if details.get('inner_split'):
        from .q8_inner import write_split
        write_split(run_dir / 'inner_split.json', details['inner_split'])
    if resume_metadata:
        metadata['resume_previous_code_sha256'] = resume_metadata['training_code_sha256']
    write_json(run_dir / 'config.json', metadata)
    write_json(run_dir / 'execution_policy.json', policy)
    # Keep per-row evidence references and independent protocol states in the
    # training artifact; features still come exclusively from frame.text.
    # Large response/prompt bodies remain in the source evidence ledger.
    body_columns = {'text', 'final_text', 'raw_response', 'raw_response_text', 'original_response',
                    'prompt_text', 'raw_prompt', 'system_prompt', 'raw_metadata_json'}
    row_manifest = frame[[name for name in frame.columns
                          if name not in body_columns and not name.endswith('_evidence_json')]].copy()
    row_manifest['loss_weight'] = weights
    row_manifest.to_parquet(run_dir / 'training_rows.parquet', index=False)
    if relation_frame is not None:
        relation_manifest = relation_frame[[name for name in relation_frame.columns
                                           if name not in body_columns and not name.endswith('_evidence_json')]].copy()
        relation_manifest['plain_slot_loss_weight'] = weights
        relation_manifest.to_parquet(run_dir / 'relation_training_rows.parquet', index=False)
    np.save(run_dir / 'centered_loss_weights.npy', centered_weights)
    np.savez_compressed(run_dir / 'plan.npz', **plan_arrays)
    from .q8_exposure import write_exposure_manifest, load_exposure_manifest
    windows_by_epoch = [update_windows(items, config['batch_size'] * config['accumulation']) for items in plan]
    from .q8_observation import observation_plan
    observation_spec = observation_plan(windows_by_epoch, config, args.phase,
        min(sum(map(len, windows_by_epoch)), args.max_updates) if args.max_updates else sum(map(len, windows_by_epoch)))
    write_json(run_dir / 'observation_plan.json', observation_spec)
    metadata['observation_plan'] = observation_spec
    references = {
        'model_kind': 'transformer_base',
        'official_files': {name: {'path': str((args.data_dir / name).resolve()),
                                  'sha256': file_sha(args.data_dir / name)}
                           for name in ('official.parquet', 'prompt_folds.json', 'labels.json')},
        'initialization': initialization_reference(getattr(args, 'initial_state', None)),
    }
    if resume_metadata:
        references['initialization'] = resumed_initialization_reference(
            resume_metadata, getattr(args, 'initial_state', None))
    if args.phase == 'official':
        references['initialization'] = dict(
            status='arm_specific_external_checkpoint', model_kind='transformer_base',
            source=details.get('initialization_run'),
            shared_root=(details.get('external_initialization_config') or {}).get('initialization_proof'))
    metadata['initialization_proof'] = references['initialization']
    if prepared:
        load_exposure_manifest(prepared, frame, weights, centered_weights, relation_frame,
                               config['centered_weight'], expected_config=config)
    exposure_ledger, exposure_report = write_exposure_manifest(
        frame, labels, weights, centered_weights, windows_by_epoch, args.max_updates,
        config, run_dir, relation_frame, phase=args.phase, references=references)
    del exposure_ledger
    frozen_ledger, frozen_positions, frozen_report = load_exposure_manifest(
        prepared or run_dir, frame, weights, centered_weights, relation_frame,
        config['centered_weight'], expected_config=config)
    # These full text-bearing ledgers are already on disk. Do not retain one
    # in an ignored `_` binding throughout the source experiment's training.
    del frozen_ledger
    metadata.update(exposure_manifest_sha256=file_sha(run_dir / 'exposure_manifest.json'),
                    actual_exposure_sha256=exposure_report['files']['actual_exposure.parquet'],
                    update_schedule_sha256=exposure_report['files']['update_schedule.parquet'])
    from .q8_tokens import tokenizer_identity
    tokenizer_reference = ((prepared_metadata or {}).get('tokenizer_identity')
                           or (resume_metadata or {}).get('executed_tokenizer_identity'))
    if getattr(args, 'tokenizer', None) is not None:
        requested = tokenizer_identity(args.tokenizer)
        if tokenizer_reference and requested['files_identity_sha256'] != tokenizer_reference['files_identity_sha256']:
            raise ValueError('Requested tokenizer files differ from the frozen tokenizer.')
        if args.plan_only or tokenizer_reference is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
            tokenizer.padding_side = 'right'
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            if args.plan_only:
                measurement = measure_token_inputs(
                    tokenizer, args.tokenizer, frame, relation_frame, windows_by_epoch,
                    args.max_updates, config, run_dir, plan_only=True, prepared=prepared,
                    reference_identity=tokenizer_reference, frozen_metadata=prepared_metadata)
                metadata.update(measurement[-1])
                del measurement
            else:
                # Fresh --tokenizer requests bind implementation identity only;
                # all training IDs will come from model.tokenizer after loading.
                tokenizer_reference = tokenizer_identity(args.tokenizer, tokenizer)
            del tokenizer
    if tokenizer_reference:
        metadata['requested_tokenizer_identity'] = tokenizer_reference
    write_json(run_dir / 'config.json', metadata)
    print(f'Run: {run_dir.resolve()}', flush=True)
    if args.plan_only:
        print(json.dumps(dict(training_rows=len(frame), validation_rows=len(valid), epochs=details['epochs']), ensure_ascii=False), flush=True)
        return run_dir
    if args.base is None:
        raise ValueError('Training needs --base pointing to an existing local model. Use --plan-only for data analysis.')
    if not args.base.is_dir():
        raise FileNotFoundError(f'The local base model directory does not exist: {args.base}')
    if args.phase == 'external' and config.get('require_shared_initialization', False) and getattr(args, 'initial_state', None) is None and resume is None:
        raise ValueError('This paired experiment needs --initial-state for the common materialized start; a seed alone is insufficient.')
    import torch
    device, dtype = runtime(args.device or config['device'], args.dtype or config['dtype'])
    if resume_metadata and (str(dtype) != resume_metadata['dtype']
                            or device.type != resume_metadata['device'].split(':')[0]):
        raise ValueError('Resume requires the original compute dtype and device type.')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    metadata.update(status='running', device=str(device), dtype=str(dtype))
    write_json(run_dir / 'config.json', metadata)
    try:
        model_load_start = time.perf_counter()
        # Unsloth must patch Transformers/PEFT before importing the HF model module.
        if config['backend'] == 'unsloth':
            Classifier = backend_classifier
        elif config['backend'] == 'liger':
            from .liger_model import LigerQ8Classifier as Classifier
        else:
            from .lora_model import Q8Classifier as Classifier
        model = Classifier(args.base, config, labels, device, dtype, resume or args.init)
        from .q8_initialization import asset_identity, save_initialization, load_initialization
        if config.get('initialization_mode') == 'canonical_backend':
            from .q8_backend_compare import save_backend_initialization as save_initialization
            from .q8_backend_compare import load_backend_initialization as load_initialization
        assets = asset_identity(args.base)
        metadata['model_assets'] = assets
        write_json(run_dir / 'base_identity.json', assets['content'])
        if resume_metadata and resume_metadata.get('model_assets', {}).get('identity_sha256') != assets['identity_sha256']:
            raise ValueError('Resume base weights or tokenizer differ from the saved content identity.')
        parent_assets = details.get('external_initialization_config', {}).get('model_assets') if details.get('external_initialization_config') else None
        if args.phase == 'official' and config.get('require_shared_initialization', False):
            parent = details.get('external_initialization_config') or {}
            if not parent_assets or not parent.get('shared_initialization', {}).get('values_verified'):
                raise ValueError('Paired official adaptation requires its external stage base identity and verified common initialization.')
        if parent_assets and parent_assets['identity_sha256'] != assets['identity_sha256']:
            raise ValueError('Official adaptation must use the external checkpoint base weights and tokenizer.')
        initial_state = getattr(args, 'initial_state', None)
        if args.phase == 'external' and initial_state is not None and resume is None:
            initial_context = dict(assets=assets, labels=labels, architecture=model.model_config, backend=config['backend'])
            if not initial_state.exists():
                if config.get('initialization_mode') == 'canonical_backend' and config['backend'] != 'hf':
                    raise ValueError('Materialize the canonical initial parameters with HF before the Unsloth arm.')
                save_initialization(model, initial_state, **initial_context)
            shared = load_initialization(model, initial_state, **initial_context)
            metadata['shared_initialization'] = shared
            metadata['initialization_proof'] = initialization_reference(initial_state)
            metadata['initialization_proof']['asset_locations'] = dict(
                base_directory=assets['base_directory'], tokenizer_directory=assets['tokenizer_directory'],
                identity_sha256=assets['identity_sha256'])
            write_json(run_dir / 'executed_initialization.json', metadata['initialization_proof'])
        elif resume_metadata:
            metadata['shared_initialization'] = resume_metadata.get('shared_initialization')
            if args.phase == 'external':
                receipt = dict(metadata['initialization_proof'],
                               resumed_checkpoint=str(resume.resolve()),
                               resumed_run_config_sha256=file_sha(resume / 'run_config.json'))
                receipt['asset_locations'] = dict(base_directory=assets['base_directory'],
                                                 tokenizer_directory=assets['tokenizer_directory'],
                                                 identity_sha256=assets['identity_sha256'])
                metadata['initialization_proof'] = receipt
                write_json(run_dir / 'executed_initialization.json', receipt)
        elif details.get('external_initialization_config'):
            metadata['shared_initialization'] = details['external_initialization_config'].get('shared_initialization')
        synchronize_device(model.device)
        model_load_seconds = time.perf_counter() - model_load_start
        tokens, relation_tokens, actual_stats, actual_relation_stats, measurement = measure_token_inputs(
            model.tokenizer, args.base, frame, relation_frame, windows_by_epoch,
            args.max_updates, config, run_dir, prepared=prepared,
            reference_identity=tokenizer_reference, frozen_metadata=prepared_metadata,
            resume_metadata=resume_metadata)
        metadata.update(measurement)
        write_json(run_dir / 'config.json', metadata)
        development_evaluator = None
        if details.get('inner_split'):
            from .q8_inner import fixed_inner_split
            from .q8_observation import evaluate_inner_development
            inner_fit, inner_dev, _, split = fixed_inner_split(
                pd.read_parquet(args.data_dir / 'official.parquet'), details['inner_split']['outer_fold'],
                config.get('inner_seed', 42), config.get('inner_dev_fraction', .2))
            if split != details['inner_split']:
                raise ValueError('The fixed inner holdout changed during preparation.')
            def development_evaluator(snapshot_dir, step):
                from .q8_stages import centered_training_history
                recorded = json.loads((snapshot_dir / 'config.json').read_text())
                centered_evidence = centered_training_history(recorded)
                return evaluate_inner_development(
                    model, inner_fit, inner_dev, labels, split, config, snapshot_dir / 'inner_evaluation',
                    phase=args.phase, training_rows_path=run_dir / 'training_rows.parquet', step=step,
                    centered_trained=any(item['effective_mass'] > 0 for item in centered_evidence),
                    centered_evidence=centered_evidence,
                    data_governance=recorded.get('data_governance'),
                    validation_scope=recorded.get('validation_scope'),
                    source_intervention=recorded.get('source_intervention'),
                    actual_source_intervention_exposure=recorded.get('actual_source_intervention_exposure'))
        metrics = train(model, tokens, y, weights, centered_weights,
                        plan, config, args.phase, run_dir, args.max_updates, resume=resume,
                        **policy, segment_started_at=segment_started_at, metadata=metadata,
                        relation_tokens=relation_tokens, frozen_positions=frozen_positions,
                        frozen_full_updates=frozen_report['full_planned_updates'],
                        token_stats=actual_stats, relation_token_stats=actual_relation_stats,
                        development_evaluator=development_evaluator)
        metadata['actual_centered_coefficient_mass'] = metrics['actual_centered_coefficient_mass']
        if metrics.get('actual_source_intervention_exposure') is not None:
            metadata['actual_source_intervention_exposure'] = metrics['actual_source_intervention_exposure']
        write_json(run_dir / 'config.json', metadata)
        model.save(run_dir)
        evaluation_start = time.perf_counter()
        final_evaluation_needed = development_evaluator is not None or len(valid) > 0
        if (not metrics['interrupted'] and final_evaluation_needed and policy['segment_seconds']
                and evaluation_start - segment_started_at >= policy['segment_seconds']):
            # The final update is already recoverable. Resume at the same
            # target cursor to do evaluation without any extra training.
            metrics.update(interrupted=True, stop_reason='segment_seconds')
        metrics['final_evaluation_pending'] = bool(
            final_evaluation_needed and metrics['interrupted']
            and metrics['successful_updates'] == metrics['target_updates'])
        locked_recipes = None
        if development_evaluator is not None and not metrics['interrupted']:
            metrics['inner_development'] = development_evaluator(run_dir, metrics['successful_updates'])
            if args.phase == 'official':
                from .q8_fusion import select_recipe, select_calibrated_recipe
                evidence = json.loads((run_dir / 'inner_evaluation/training_evidence.json').read_text())
                if config.get('fusion_protocol') == 'crossfit_calibrated_inner_v3':
                    from .q8_crossfit_fusion import select_crossfit_calibrated_recipe
                    selector = select_crossfit_calibrated_recipe
                else:
                    selector = (select_calibrated_recipe if config.get('fusion_protocol') == 'calibrated_inner_v2'
                                else select_recipe)
                primary = selector(run_dir / 'inner_evaluation', split, evidence, run_dir / 'fusion_primary')
                calibration = {'calibration': primary['calibration']} if 'calibration' in primary else {}
                locked_recipes = dict(primary=primary, exclusion_sensitivity=selector(
                    run_dir / 'inner_evaluation', split, evidence, run_dir / 'fusion_exclusion_sensitivity',
                    allow_exclusion=True, **calibration))
        if len(valid) and not metrics['interrupted']:
            metrics.update(evaluate(model, frame, valid, labels, args.fold, config, run_dir))
            if locked_recipes:
                from .q8_fusion import apply_recipe
                metrics['locked_fusion'] = {name: apply_recipe(recipe, run_dir / 'evaluation', split,
                    run_dir / f'outer_{name}') for name,recipe in locked_recipes.items()}
        synchronize_device(model.device)
        metrics.update(model_load_seconds=model_load_seconds,
                       segment_wall_seconds=round(time.perf_counter() - segment_started_at, 3),
                       evaluation_seconds=time.perf_counter() - evaluation_start,
                       timing_scope='Model loading includes asset hashing and common initialization; final evaluation includes inner fusion selection and outer reporting. Intermediate observations are included in training wall time and separately itemized.')
        write_json(run_dir / 'metrics.json', metrics)
        metadata['status'] = ('interrupted' if metrics['interrupted'] else
                              'partial' if metrics['partial'] else 'complete')
        metadata['last_checkpoint'] = metrics['last_checkpoint']
        write_json(run_dir / 'config.json', metadata)
    except Exception as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(run_dir / 'config.json', metadata)
        raise
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/q8.toml'))
    parser.add_argument('--experiment', help='Named experiment in the Q8 config; defaults to auxiliary for new runs')
    parser.add_argument('--data-dir', type=Path, default=Path('data/q8'))
    parser.add_argument('--governance-dir', type=Path, help='Immutable governance version; defaults to data-dir/governance, inherited on resume/official adaptation')
    parser.add_argument('--source-scope-dir', type=Path, help='Immutable measured replacement scope; defaults to data-dir/intervention_scope and is inherited on resume/official adaptation')
    parser.add_argument('--phase', choices=['external', 'official'])
    parser.add_argument('--variant', choices=['baseline', 'mix20', 'matched', 'expanded'])
    parser.add_argument('--centered-plan', choices=['native', 'shared_triplets', 'shared_stratified_triplets',
                                                   'grouped_triplets', 'anchored_triplets', 'anchored_separate'])
    parser.add_argument('--centered-balance', choices=['none', 'global_mass', 'class_mass'])
    parser.add_argument('--centered-weight', type=float, help='Auxiliary strength; 0/1 gives paired off/on arms')
    parser.add_argument('--fold', default=None)
    parser.add_argument('--base', type=Path, help='Existing local Transformers model directory')
    parser.add_argument('--init', type=Path, help='Completed external-stage run for official adaptation')
    parser.add_argument('--initial-state', type=Path, help='Common external-stage trainable tensors and RNG; first arm materializes it, both arms reload it')
    parser.add_argument('--backend-initial-state', type=Path, help='Explicit canonical HF/Unsloth common tensors; config must use initialization_mode=canonical_backend')
    parser.add_argument('--tokenizer', type=Path, help='Frozen local tokenizer for pre-training token/truncation measurement; never downloads model weights')
    parser.add_argument('--prepared-plan', type=Path, help='Previously reviewed run directory with actual-update manifest; consume its frozen settings and row order')
    parser.add_argument('--resume', type=Path, help='Own checkpoint directory with optimizer, schedule, cursor and RNG')
    parser.add_argument('--output-dir', type=Path, default=Path('runs'))
    parser.add_argument('--plan-only', action='store_true', help='Analyze real data without loading model weights')
    parser.add_argument('--max-updates', type=int, help='Research budget; remains fixed when resuming a segment')
    parser.add_argument('--checkpoint-every', type=int, help='Recovery cadence in successful updates (default 100); 0 disables this cadence')
    parser.add_argument('--checkpoint-seconds', type=float, help='Recovery cadence in elapsed seconds (default 900); checked at update boundaries; 0 disables')
    parser.add_argument('--stop-after-updates', type=int, help='Stop this segment after N additional updates; inherited on resume; 0 clears the cap')
    parser.add_argument('--segment-seconds', type=float, help='Soft wall-time limit from run entry (default 10800); finish active work then save; 0 disables')
    parser.add_argument('--backend', choices=['hf', 'unsloth', 'liger'])
    parser.add_argument('--forward-batch-rows', type=int)
    parser.add_argument('--optimizer', choices=['adamw', 'loraplus'])
    parser.add_argument('--loraplus-ratio', type=float)
    parser.add_argument('--fused-adamw', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--dtype', choices=['auto', 'float32', 'float16', 'bfloat16'])
    args = parser.parse_args()
    if args.phase is None and args.resume is None and args.prepared_plan is None:
        parser.error('Choose --phase, --prepared-plan or --resume.')
    if args.fold not in (None, 'shared', 'final'):
        try:
            args.fold = int(args.fold)
        except ValueError:
            parser.error('--fold must be shared, final or 0..4')
        if args.fold not in range(5):
            parser.error('--fold must be shared, final or 0..4')
    for name in ('max_updates', 'forward_batch_rows'):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    for name in ('checkpoint_every', 'stop_after_updates', 'checkpoint_seconds', 'segment_seconds'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f'--{name.replace("_", "-")} must be nonnegative and finite')
    run(args)


if __name__ == '__main__':
    main()
