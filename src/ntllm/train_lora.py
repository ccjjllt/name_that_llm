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
import tomllib
from uuid import uuid4

import numpy as np
import pandas as pd

from .q8_data import file_sha
from .q8_sampling import matching_weights, plan_summary, prompt_batches
from .q8_supervision import common_relation_rows, shared_triplet_batches, centered_loss_weights


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


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


def prepare(args, config):
    labels = json.loads((args.data_dir / 'labels.json').read_text())
    official = pd.read_parquet(args.data_dir / 'official.parquet')
    valid = official.iloc[:0].copy()
    matching = None
    parent = None
    if args.init is not None:
        parent = json.loads((args.init / 'config.json').read_text())
    # A two-stage ablation must not silently switch its auxiliary objective.
    legacy = dict(centered_plan='native', centered_balance='none', centered_weight=1.0)
    for key in legacy:
        inherited = parent['settings'].get(key, legacy[key]) if parent else None
        override = getattr(args, key, None)
        value = override if override is not None else (
            inherited if parent else config.get(key, legacy[key]))
        if parent and value != inherited:
            raise ValueError(f'{key} differs from the external checkpoint; keep both stages in the same ablation arm.')
        config[key] = value
    if config['centered_plan'] not in ('native', 'shared_triplets'):
        raise ValueError('centered_plan must be native or shared_triplets.')
    if config['centered_balance'] not in ('none', 'global_mass', 'class_mass'):
        raise ValueError('centered_balance must be none, global_mass or class_mass.')
    if not np.isfinite(config['centered_weight']) or config['centered_weight'] < 0:
        raise ValueError('centered_weight must be finite and nonnegative.')
    variant = args.variant or (parent['variant'] if parent else 'baseline')
    conditioning_fold = 'shared'
    if args.phase == 'external':
        if args.init is not None:
            raise ValueError('External experiments start with a fresh adapter on the base model.')
        pool_name = 'mix20' if variant == 'mix20' else 'baseline'
        path = args.data_dir / f'external_{pool_name}.parquet'
        frame = pd.read_parquet(path)
        soft_weights = np.ones(len(frame), dtype=np.float32)
        if variant == 'matched':
            if args.fold == 'shared':
                raise ValueError('Matched external training needs a target fold, or final for all official rows.')
            target = official if args.fold == 'final' else official.loc[official.fold != args.fold]
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
            frame = official.loc[official.fold != args.fold]
            valid = official.loc[official.fold == args.fold].copy()
        frame = frame.reset_index(drop=True)
        soft_weights = np.ones(len(frame), dtype=np.float32)
        if variant == 'matched':
            conditioning_fold = args.fold
        if parent is not None:
            preview_parent = (args.plan_only and parent['status'] == 'planned'
                              and parent.get('prepared_only', False))
            if parent['phase'] != 'external' or (parent['status'] not in ('complete', 'partial')
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
    relations = dict(policy=config['centered_plan'], policy_source='external_checkpoint' if parent else 'config_or_cli')
    if config['centered_plan'] == 'shared_triplets':
        if args.phase == 'external':
            peer_name = 'baseline' if variant == 'mix20' else 'mix20'
            peer_path = args.data_dir / f'external_{peer_name}.parquet'
            common, report = common_relation_rows(frame, pd.read_parquet(peer_path))
            relations.update(peer_path=str(peer_path.resolve()), peer_sha256=file_sha(peer_path))
        else:
            # Construct relations only after selecting this fold's training side.
            common, report = common_relation_rows(frame, frame)
        relations.update(report)
        plan = [shared_triplet_batches(frame, common, seed + epoch, config['batch_size'])
                for epoch in range(epochs)]
    else:
        plan = [prompt_batches(frame, seed + epoch, config['batch_size']) for epoch in range(epochs)]
    windows_by_epoch = [update_windows(batches, config['batch_size'] * config['accumulation'])
                        for batches in plan]
    centered_weights, supervision = centered_loss_weights(
        frame, labels, weights, windows_by_epoch, balance=config['centered_balance'],
        max_updates=args.max_updates, enabled=config['centered_weight'] > 0)
    supervision['strength'] = config['centered_weight']
    for epoch_report in supervision['epochs']:
        epoch_report['effective_total_mass'] *= config['centered_weight']
        for item in epoch_report['per_class']:
            item['effective_centered_mass'] *= config['centered_weight']
            for size in item['by_group_size']:
                size['effective_mass'] *= config['centered_weight']
    summaries = [plan_summary(frame, epoch) for epoch in plan]
    for windows, summary in zip(windows_by_epoch, summaries):
        sizes = [sum(len(rows) for rows, _ in chunk) for chunk in windows]
        summary.update(optimizer_updates=len(windows), minimum_update_rows=min(sizes), maximum_update_rows=max(sizes))
        events = [(index, sizes[index], frame.iloc[rows].id.tolist())
                  for index, chunk in enumerate(windows) for rows, same in chunk if same]
        summary['centered_context_sha256'] = hashlib.sha256(
            json.dumps(events, separators=(',', ':')).encode()).hexdigest()
    return frame, valid, labels, targets, weights, centered_weights, plan, dict(
        variant=variant, conditioning_fold=conditioning_fold, matching=matching,
        class_weights=class_weights.tolist(), epochs=summaries, relations=relations,
        supervision=supervision, data_path=str(path.resolve()),
        data_sha256=file_sha(path), data_manifest_sha256=file_sha(args.data_dir / 'manifest.json'),
        initialization_run=str(args.init.resolve()) if args.init else None,
        initialization_partial=bool(parent and parent['status'] == 'partial'),
        initialization_prepared_only=bool(parent and parent.get('prepared_only', False)),
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


def backward_window(model, tokens, targets, weights, centered_weights, chunk, dropout, scaler, centered_weight):
    import torch
    from torch.nn import functional as F

    row_count = sum(len(rows) for rows, _ in chunk)
    plain_total = centered_total = 0.0
    for rows, same_prompt in chunk:
        z = model.representation([tokens[index] for index in rows])
        y = torch.as_tensor(targets[rows], device=model.device)
        w = torch.as_tensor(weights[rows], device=model.device)
        plain = (F.cross_entropy(model.plain(dropout(z)), y, reduction='none') * w).sum()
        centered = z.sum() * 0
        if same_prompt:
            # Keep the same head-dropout draws in the paired off/on arms.
            # The outer strength disables this loss, not the batch layout.
            centered_z = z - z.mean(dim=0, keepdim=True)
            auxiliary = torch.as_tensor(centered_weights[rows], device=model.device)
            centered = (F.cross_entropy(model.centered(dropout(centered_z)), y, reduction='none') * auxiliary).sum()
        plain_value, centered_value = float(plain.detach()), float(centered.detach())
        if not math.isfinite(plain_value) or not math.isfinite(centered_value):
            raise FloatingPointError(f'Nonfinite forward loss for training rows {rows}.')
        scaler.scale((plain + centered_weight * centered) / row_count).backward()
        plain_total += plain_value
        centered_total += centered_value
    return row_count, plain_total, centered_total


def train(model, tokens, targets, weights, centered_weights, plan, config, phase, run_dir, max_updates=None):
    import torch

    adapter = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    heads = list(model.norms.parameters()) + list(model.plain.parameters()) + list(model.centered.parameters())
    parameters = adapter + heads
    learning_rate = config[f'{phase}_lr']
    head_rate = config[f'{phase}_head_lr']
    optimizer = torch.optim.AdamW([
        {'params': adapter, 'lr': learning_rate},
        {'params': heads, 'lr': head_rate},
    ], weight_decay=config['weight_decay'])
    windows = [update_windows(epoch, config['batch_size'] * config['accumulation']) for epoch in plan]
    planned = sum(len(epoch) for epoch in windows)
    target_updates = min(planned, max_updates) if max_updates else planned
    warmup = max(1, int(target_updates * config['warmup_fraction']))

    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        fraction = (step - warmup) / max(1, target_updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(fraction, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    scale = model.device.type == 'cuda' and model.compute_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda', init_scale=256.0, enabled=scale)
    dropout = torch.nn.Dropout(config['head_dropout'])
    step = 0
    exposed = []
    start = time.perf_counter()
    model.train()
    with (run_dir / 'train_log.jsonl').open('w') as logfile:
        for epoch_number, epoch in enumerate(windows, start=1):
            for chunk in epoch:
                if step == target_updates:
                    break
                chunk_rows = [row for rows, _ in chunk for row in rows]
                if scale:
                    cpu_rng = torch.get_rng_state()
                    cuda_rng = torch.cuda.get_rng_state(model.device)
                for retry in range(9):
                    optimizer.zero_grad(set_to_none=True)
                    row_count, plain_total, centered_total = backward_window(
                        model, tokens, targets, weights, centered_weights[epoch_number - 1],
                        chunk, dropout, scaler, config['centered_weight'])
                    scaler.unscale_(optimizer)
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip']))
                    if math.isfinite(grad_norm):
                        break
                    if not scale or retry == 8:
                        write_json(run_dir / 'nonfinite_batch.json', dict(epoch=epoch_number, next_update=step + 1, rows=chunk_rows))
                        raise FloatingPointError('Nonfinite gradients after scaling. No invalid optimizer update was applied.')
                    # Replay the same rows and dropout draws at a smaller scale.
                    scaler = torch.amp.GradScaler('cuda', init_scale=scaler.get_scale() / 2, enabled=True)
                    torch.set_rng_state(cpu_rng)
                    torch.cuda.set_rng_state(cuda_rng, model.device)
                rates = [group['lr'] for group in optimizer.param_groups]
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1
                exposed.extend(chunk_rows)
                record = dict(update=step, epoch=epoch_number, rows=row_count,
                              plain_ce=plain_total / row_count, centered_ce=centered_total / row_count,
                              weighted_centered_ce=config['centered_weight'] * centered_total / row_count,
                              adapter_lr=rates[0], head_lr=rates[1], grad_norm=grad_norm,
                              gradient_scale_retries=retry,
                              elapsed_seconds=round(time.perf_counter() - start, 3))
                logfile.write(json.dumps(record) + '\n')
                logfile.flush()
                if step == 1 or step % 25 == 0 or step == target_updates:
                    print(f'update {step}/{target_updates} plain={record["plain_ce"]:.4f} centered={record["centered_ce"]:.4f}', flush=True)
            if step == target_updates:
                break
    np.save(run_dir / 'actual_row_indices.npy', np.asarray(exposed, dtype=np.int32))
    return dict(successful_updates=step, planned_updates=planned, target_updates=target_updates,
                row_exposures=len(exposed), unique_training_rows=len(set(exposed)),
                partial=step < planned, train_seconds=round(time.perf_counter() - start, 3))


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
    if metadata.get('initialization_run'):
        history.append(json.loads((Path(metadata['initialization_run']) / 'config.json').read_text()))
    centered_evidence = [dict(
        phase=stage['phase'],
        effective_mass=sum(epoch.get('effective_total_mass', 0.)
                           for epoch in stage.get('supervision', {}).get('epochs', [])),
        recorded=bool(stage.get('supervision')),
    ) for stage in history]
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
    evaluation_config = dict(grouping_protocol=config.get('eval_grouping_protocol', 'train_fold'),
                             cluster_threshold=config.get('eval_cluster_threshold', .15),
                             seeds=config.get('eval_cohort_seeds', [42, 43]))
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
    with args.config.open('rb') as handle:
        config = tomllib.load(handle)
    for key in ('batch_size', 'accumulation', 'external_epochs', 'official_epochs', 'eval_batch_size'):
        if config[key] < 1:
            raise ValueError(f'{key} must be positive.')
    frame, valid, labels, y, weights, centered_weights, plan, details = prepare(args, config)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    arm = 'off' if config['centered_weight'] == 0 else 'on'
    run_dir = args.output_dir / f'{stamp}_{args.phase}_{details["variant"]}_{arm}_f{args.fold}_{uuid4().hex[:6]}'
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(details, phase=args.phase, fold=args.fold, labels=labels, settings=config,
                    status='planned', prepared_only=args.plan_only, training_rows=len(frame), validation_rows=len(valid),
                    base=str(args.base.resolve()) if args.base else None,
                    max_updates=args.max_updates,
                    training_code_sha256={name: file_sha(Path(__file__).with_name(name)) for name in
                                          ('train_lora.py', 'q8_sampling.py', 'q8_supervision.py')},
                    recipe='Controlled reconstruction using q8v5 architecture; unpublished author training settings are explicit project choices.')
    write_json(run_dir / 'config.json', metadata)
    row_manifest = frame[['id', 'prompt_id', 'model', 'source_dataset', 'word_bin', 'origin']].copy()
    row_manifest['loss_weight'] = weights
    row_manifest.to_parquet(run_dir / 'training_rows.parquet', index=False)
    np.save(run_dir / 'centered_loss_weights.npy', centered_weights)
    batches = [(epoch, rows, same) for epoch, items in enumerate(plan) for rows, same in items]
    np.savez_compressed(run_dir / 'plan.npz',
                        rows=np.asarray([row for _, rows, _ in batches for row in rows], dtype=np.int32),
                        sizes=np.asarray([len(rows) for _, rows, _ in batches], dtype=np.int8),
                        same_prompt=np.asarray([same for _, _, same in batches], dtype=bool),
                        epochs=np.asarray([epoch for epoch, _, _ in batches], dtype=np.int16))
    print(f'Run: {run_dir.resolve()}', flush=True)
    if args.plan_only:
        print(json.dumps(dict(training_rows=len(frame), validation_rows=len(valid), epochs=details['epochs']), ensure_ascii=False), flush=True)
        return run_dir
    if args.base is None:
        raise ValueError('Training needs --base pointing to an existing local model. Use --plan-only for data analysis.')
    if not args.base.is_dir():
        raise FileNotFoundError(f'The local base model directory does not exist: {args.base}')
    import torch
    from .lora_model import Q8Classifier
    device, dtype = runtime(args.device or config['device'], args.dtype or config['dtype'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    metadata.update(status='running', device=str(device), dtype=str(dtype), max_updates=args.max_updates)
    write_json(run_dir / 'config.json', metadata)
    try:
        model = Q8Classifier(args.base, config, labels, device, dtype, args.init)
        metrics = train(model, model.encode(frame.text.tolist()), y, weights, centered_weights,
                        plan, config, args.phase, run_dir, args.max_updates)
        model.save(run_dir)
        if len(valid):
            metrics.update(evaluate(model, frame, valid, labels, args.fold, config, run_dir))
        write_json(run_dir / 'metrics.json', metrics)
        metadata['status'] = 'partial' if metrics['partial'] else 'complete'
        write_json(run_dir / 'config.json', metadata)
    except Exception as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(run_dir / 'config.json', metadata)
        raise
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/q8.toml'))
    parser.add_argument('--data-dir', type=Path, default=Path('data/q8'))
    parser.add_argument('--phase', choices=['external', 'official'], required=True)
    parser.add_argument('--variant', choices=['baseline', 'mix20', 'matched'])
    parser.add_argument('--centered-plan', choices=['native', 'shared_triplets'])
    parser.add_argument('--centered-balance', choices=['none', 'global_mass', 'class_mass'])
    parser.add_argument('--centered-weight', type=float, help='Auxiliary strength; 0/1 gives paired off/on arms')
    parser.add_argument('--fold', default='shared')
    parser.add_argument('--base', type=Path, help='Existing local Transformers model directory')
    parser.add_argument('--init', type=Path, help='Completed external-stage run for official adaptation')
    parser.add_argument('--output-dir', type=Path, default=Path('runs'))
    parser.add_argument('--plan-only', action='store_true', help='Analyze real data without loading model weights')
    parser.add_argument('--max-updates', type=int, help='Explicit partial run with a schedule fitted to this update budget')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--dtype', choices=['auto', 'float32', 'float16', 'bfloat16'])
    args = parser.parse_args()
    if args.fold not in ('shared', 'final'):
        try:
            args.fold = int(args.fold)
        except ValueError:
            parser.error('--fold must be shared, final or 0..4')
        if args.fold not in range(5):
            parser.error('--fold must be shared, final or 0..4')
    if args.max_updates is not None and args.max_updates < 1:
        parser.error('--max-updates must be positive')
    run(args)


if __name__ == '__main__':
    main()
