"""Predeclared learning snapshots and held-out development observations."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import random
import time

import numpy as np

from .q8_compute import lr_multiplier


@contextmanager
def preserve_observation_state(model):
    """Evaluation must not consume the training random stream or change modes."""
    import torch
    device = model.device
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state(device) if device.type == 'cuda' else
             torch.mps.get_rng_state() if device.type == 'mps' else None)
    modes = [(module, module.training) for module in model.modules()]
    try:
        yield
    finally:
        model.train(modes[0][1])
        for module, training in modes:
            module.training = training
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state[3], device)
        elif device.type == 'mps':
            torch.mps.set_rng_state(state[3])


def observation_plan(windows, config, phase, target_updates):
    requested = config.get('external_snapshot_updates', []) if phase == 'external' else []
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in requested):
        raise ValueError('Snapshot updates must be predeclared positive integers.')
    events = {value: ['external_update'] for value in sorted(set(requested)) if value <= target_updates}
    total = 0
    if phase == 'official' and config.get('snapshot_official_epochs', False):
        for epoch, items in enumerate(windows, 1):
            total += len(items)
            if total <= target_updates:
                events.setdefault(total, []).append(f'official_epoch_{epoch}_end')
    diagnostics = config.get('gradient_diagnostic_updates', [])
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in diagnostics):
        raise ValueError('Diagnostic updates must be predeclared positive integers.')
    return dict(phase=phase, total_research_updates=target_updates,
                full_layout_updates=sum(map(len, windows)),
                snapshot_events={str(key): value for key, value in events.items()},
                unavailable_snapshot_updates=[value for value in requested if value > target_updates],
                diagnostic_updates=sorted(set(value for value in diagnostics if value <= target_updates)),
                schedule_scope='Snapshots retain the original total update horizon; they do not reset or shorten cosine decay.')


def save_learning_snapshot(path, model, optimizer, scheduler, scaler, progress, metadata,
                           reasons, observation_count, evaluate=None):
    """Retain complete states separately from the rotating recovery checkpoints."""
    from .q8_checkpoint import save_training_checkpoint
    path = Path(path)
    if path.exists():
        old = json.loads((path / 'snapshot.json').read_text())
        if old['successful_updates'] != progress['step'] or old['target_updates'] != progress['target_updates']:
            raise ValueError('A saved learning-curve snapshot has another training horizon.')
        return old
    step, horizon = progress['step'], progress['target_updates']
    note = dict(successful_updates=step, target_updates=horizon,
                full_planned_updates=progress['planned_updates'], reasons=reasons,
                is_intermediate=step < horizon,
                last_lr_multiplier=lr_multiplier(step - 1, horizon, metadata['settings']['warmup_fraction']),
                next_lr_multiplier=lr_multiplier(step, horizon, metadata['settings']['warmup_fraction']),
                optimizer_group_lrs_next={group['name']: group['lr'] for group in optimizer.param_groups},
                interpretation='An intermediate point in this horizon is not a completed shorter-horizon experiment.')
    snapshot_metadata = copy.deepcopy(metadata)
    snapshot_metadata.update(status='research_snapshot', prepared_only=False, snapshot=note,
                             successful_updates=step)
    if 'actual_centered_coefficient_mass' in progress:
        snapshot_metadata['actual_centered_coefficient_mass'] = progress['actual_centered_coefficient_mass']
    if metadata.get('inner_split'):
        snapshot_metadata['conditioning_fold'] = metadata['inner_split']['outer_fold']
    started = time.perf_counter()
    with preserve_observation_state(model):
        save_training_checkpoint(path, model, optimizer, scheduler, scaler, progress,
                                 snapshot_metadata, keep=max(1, observation_count + 1))
        (path / 'config.json').write_text(json.dumps(snapshot_metadata, ensure_ascii=False, indent=2) + '\n')
        (path / 'snapshot.json').write_text(json.dumps(note, ensure_ascii=False, indent=2) + '\n')
        if metadata.get('model_assets'):
            (path / 'base_identity.json').write_text(json.dumps(metadata['model_assets']['content'], sort_keys=True) + '\n')
        if evaluate is not None:
            report = evaluate(path, step)
            (path / 'development_observation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    note['save_and_development_seconds'] = time.perf_counter() - started
    (path / 'snapshot.json').write_text(json.dumps(note, ensure_ascii=False, indent=2) + '\n')
    return note


def evaluate_inner_development(model, fit, dev, labels, split, settings, output, *,
                               phase, training_rows_path, step, centered_trained, centered_evidence=None):
    """Use no outer outcomes. Every fitted relation transform uses inner fit."""
    from .q8_data import file_sha
    from .q8_inner import checkpoint_identity
    from .q8_evaluation import evaluate_cohorts
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output.parent
    artifacts = {str(path.relative_to(checkpoint)): path for path in checkpoint.rglob('*.safetensors')
                 if 'checkpoints' not in path.relative_to(checkpoint).parts and 'snapshots' not in path.relative_to(checkpoint).parts}
    for name in ('model_config.json', 'base_identity.json'):
        if (checkpoint / name).exists(): artifacts[name] = checkpoint / name
    identity = checkpoint_identity(artifacts)
    with preserve_observation_state(model):
        query = model.predict_outputs(model.encode(dev.text.tolist()), settings['eval_batch_size'], keep_features=True)
        features = model.predict_outputs(model.encode(fit.text.tolist()), settings['eval_batch_size'], keep_features=True)['features']
        report = evaluate_cohorts(
            fit, dev.assign(fold=split['outer_fold']), labels, query['plain_logits'],
            query['raw_centered_logits'], query['centered_bias'], output,
            query_features=query['features'], fit_features=features, centered_trained=centered_trained,
            provenance=dict(role='inner_dev', phase=phase, checkpoint_updates=step,
                            split_sha256=split['split_sha256'],
                            model_identity_sha256=identity['model_identity_sha256'],
                            centered_training_evidence=centered_evidence,
                            model_scope='q8_inner_fit' if phase == 'official' else 'external_only',
                            official_optimizer_training_ids=fit.id.astype(str).tolist() if phase == 'official' else [],
                            public_external_history=split['public_external_history']),
            config=dict(grouping_protocol='train_fold', seeds=settings.get('eval_cohort_seeds', [42,43]),
                        cluster_threshold=settings.get('eval_cluster_threshold', .15)))
    evidence = dict(split_sha256=split['split_sha256'],
                    official_training_rows_file=str(Path(training_rows_path).resolve()),
                    official_training_rows_sha256=file_sha(training_rows_path),
                    representation_fit_ids=fit.id.astype(str).tolist(), head_fit_ids=fit.id.astype(str).tolist(),
                    model_scope='q8_inner_fit' if phase == 'official' else 'external_only', **identity)
    (output / 'training_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    return dict(split_sha256=split['split_sha256'], role='inner_dev', phase=phase, step=step,
                evaluation_dir=str(output.resolve()), full_views=report['cohorts']['full']['views'],
                selection_allowed=phase == 'official', outer_labels_used=False)
