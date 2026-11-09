"""Prepare one frozen Q8 job or run one resumable Kaggle training segment."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


REFERENCE_KEYS = (
    'settings', 'labels', 'plan_sha256', 'data_sha256', 'data_manifest_sha256',
    'relation_data_sha256', 'planned_token_identity', 'planned_relation_token_identity',
)
INPUT_COPIES = (
    'actual_exposure.parquet', 'token_rows.parquet', 'physical_token_batches.parquet',
    'step_token_metrics.parquet', 'plan.npz', 'centered_loss_weights.npy',
)
OUTPUT_LIMIT_BYTES = 20_000_000_000
SEGMENT_RESERVE_BYTES = 6_000_000_000


def _read(path):
    return json.loads(Path(path).read_text())


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def space_status(workspace):
    """Logical file bytes, including repeated hard links; use decimal GB."""
    total = 0
    for directory, _, names in os.walk(Path(workspace), followlinks=False):
        for name in names:
            path = Path(directory) / name
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                continue
    return dict(logical_bytes=total, limit_bytes=OUTPUT_LIMIT_BYTES,
                segment_reserve_bytes=SEGMENT_RESERVE_BYTES,
                can_start_segment=total + SEGMENT_RESERVE_BYTES <= OUTPUT_LIMIT_BYTES)


def _require_space(workspace):
    status = space_status(workspace)
    if not status['can_start_segment']:
        raise RuntimeError(
            f'Workspace contains {status["logical_bytes"] / 1e9:.2f} GB; '
            'the next segment needs 6 GB of headroom within the 20 GB output budget. '
            'Save the existing output and organize completed jobs before starting another segment.')
    return status


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def _inside(workspace, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Catalog paths must be relative to the workspace: {relative}')
    return workspace / path


def _catalog(workspace):
    return _read(workspace / 'kaggle_jobs.json')


def _job(catalog, job_id):
    if job_id in ('.', '..') or '/' in job_id or '\\' in job_id:
        raise ValueError('A job ID must be a single directory name.')
    job = catalog['jobs'][job_id]
    if set(job['reference']) != set(REFERENCE_KEYS):
        raise ValueError(f'{job_id}: incomplete frozen scientific reference.')
    if job['initial_group'] not in ('auxiliary', 'source'):
        raise ValueError(f'{job_id}: unknown common initialization group.')
    if job['phase'] not in ('external', 'official'):
        raise ValueError(f'{job_id}: unknown training phase.')
    if (job['phase'] == 'official') != bool(job.get('parent')):
        raise ValueError(f'{job_id}: official jobs need one external parent.')
    return job


def _verify(metadata, job, job_id):
    for key in REFERENCE_KEYS:
        if metadata.get(key) != job['reference'][key]:
            raise ValueError(f'{job_id}: rebuilt plan differs from the frozen reference: {key}.')
    for key in ('phase', 'variant', 'fold', 'max_updates'):
        if metadata.get(key) != job.get(key):
            raise ValueError(f'{job_id}: different {key} in the saved job.')


def _state_path(workspace, job_id):
    return workspace / 'runs' / 'jobs' / job_id / 'state.json'


def _update_state(workspace, job_id, **changes):
    path = _state_path(workspace, job_id)
    state = _read(path) if path.is_file() else dict(job_id=job_id)
    state.update(changes)
    _write(path, state)
    return state


def _run_directories(parent):
    return sorted((p.parent for p in parent.glob('*/config.json')),
                  key=lambda p: ((p / 'config.json').stat().st_mtime_ns, p.name))


def _prepared(workspace, job_id, job):
    candidates = _run_directories(workspace / 'runs' / 'prepared' / job_id)
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(f'{job_id}: several prepared plans exist; resolve the ambiguous directory.')
    path = candidates[0]
    metadata = _read(path / 'config.json')
    if metadata.get('status') != 'planned' or not metadata.get('prepared_only'):
        raise ValueError(f'{job_id}: the preparation directory is incomplete or already executed.')
    _verify(metadata, job, job_id)
    expected = dict(
        tokenizer_path=_inside(workspace, _catalog(workspace)['tokenizer_dir']),
        shared_initialization_path=workspace / 'runs' / 'common_initialization' / job['initial_group'])
    if Path(metadata['data_path']).parent.resolve() != _inside(workspace, job['data_dir']).resolve():
        raise ValueError(f'{job_id}: the saved data path belongs to another workspace.')
    for key, location in expected.items():
        if not metadata.get(key) or Path(metadata[key]).resolve() != location.resolve():
            raise ValueError(f'{job_id}: {key} belongs to another workspace; rebuild the plan here.')
    for filename in ('exposure_manifest.json', 'plan.npz', 'token_rows.parquet', 'token_profile.json'):
        if not (path / filename).is_file():
            raise FileNotFoundError(f'{job_id}: incomplete prepared plan: {filename}')
    return path


def prepare_job(workspace, job_id):
    """Rebuild machine paths while checking frozen logical inputs and settings."""
    workspace = Path(workspace).resolve()
    catalog = _catalog(workspace)
    job = _job(catalog, job_id)
    existing = _prepared(workspace, job_id, job)
    if existing is not None:
        _update_state(workspace, job_id, prepared=str(existing.relative_to(workspace)))
        return existing
    output = workspace / 'runs' / 'prepared' / job_id
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{job_id}: unfinished preparation exists at {output}; inspect it before retrying.')
    parent = None
    if job.get('parent'):
        parent_job = _job(catalog, job['parent'])
        if parent_job['phase'] != 'external' or parent_job['initial_group'] != job['initial_group']:
            raise ValueError(f'{job_id}: parent must be an external job with the same common start.')
        parent = prepare_job(workspace, job['parent'])
    command = [sys.executable, '-m', 'ntllm.train_lora',
               '--config', str(workspace / 'configs' / 'q8.toml'), '--experiment', job['profile'],
               '--phase', job['phase'], '--variant', job['variant'], '--fold', str(job['fold']),
               '--data-dir', str(_inside(workspace, job['data_dir'])),
               '--tokenizer', str(_inside(workspace, catalog['tokenizer_dir'])),
               '--initial-state', str(workspace / 'runs' / 'common_initialization' / job['initial_group']),
               '--output-dir', str(output), '--plan-only']
    if job.get('max_updates') is not None:
        command += ['--max-updates', str(job['max_updates'])]
    if parent is not None:
        command += ['--init', str(parent)]
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / 'prepare.log'
    print(f'Preparing {job_id} ...', flush=True)
    with log_path.open('w') as log:
        process = subprocess.run(command, cwd=workspace, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        tail = '\n'.join(log_path.read_text(errors='replace').splitlines()[-10:])[-2500:]
        raise RuntimeError(f'{job_id}: preparation failed; see {log_path}\n{tail}')
    result = _prepared(workspace, job_id, job)
    if result is None:
        raise RuntimeError(f'{job_id}: preparation returned without a saved plan.')
    _update_state(workspace, job_id, prepared=str(result.relative_to(workspace)))
    updates = _read(result / 'config.json')['supervision']['active_updates']
    print(f'Prepared {job_id}: {updates} updates; {result}', flush=True)
    return result


def _latest_run(workspace, job_id):
    candidates = _run_directories(workspace / 'runs' / 'jobs' / job_id)
    return candidates[-1] if candidates else None


def _complete(run, job, job_id):
    metadata = _read(run / 'config.json')
    if metadata.get('status') != 'complete':
        return False
    _verify(metadata, job, job_id)
    metrics = _read(run / 'metrics.json')
    target = metadata['supervision']['active_updates']
    if (metrics.get('successful_updates') != target or metrics.get('target_updates') != target
            or metrics.get('interrupted') or metrics.get('final_evaluation_pending')):
        raise ValueError(f'{job_id}: complete status disagrees with the actual training cursor/evaluation.')
    return True


def _checkpoint(run):
    """Use an actual complete save, including a save preceding a failed process."""
    candidates = []
    paths = {marker.parent for marker in (run / 'checkpoints').glob('*/checkpoint.json')}
    # A resumed process can fail before its first new save. Its input checkpoint
    # remains a valid recovery point; do not discard it with that failed segment.
    config = _read(run / 'config.json')
    for key in ('last_checkpoint', 'resume_from'):
        if config.get(key):
            path = Path(config[key]).resolve()
            if path.is_relative_to(run.parent.resolve()):
                paths.add(path)
    for path in paths:
        marker = path / 'checkpoint.json'
        if not all((path / name).is_file() for name in
                   ('checkpoint.json', 'run_config.json', 'progress.json', 'training_state.pt')):
            continue
        record = _read(marker)
        if record.get('format') == 'ntllm-q8-training-v1':
            if _sha256(path / 'run_config.json') != record.get('run_config_sha256'):
                raise ValueError(f'Checkpoint metadata changed after saving: {path}')
            candidates.append((int(_read(path / 'progress.json')['step']),
                               int(record['created_ns']), path))
    if candidates:
        return max(candidates, key=lambda item: item[:2])[2]
    return None


def _compact_input_copies(run, prepared, workspace, reference_hashes=None):
    """Remove only byte-identical inputs; the receipt identifies their stable copy."""
    reference_hashes = reference_hashes if reference_hashes is not None else {}
    matches = {}
    for prefix in (Path(), Path('execution_tokens')):
        for name in INPUT_COPIES:
            relative = prefix / name
            path, reference = run / relative, prepared / name
            if not path.is_file() or path.is_symlink() or not reference.is_file():
                continue
            size = path.stat().st_size
            if size != reference.stat().st_size:
                continue
            if name not in reference_hashes:
                reference_hashes[name] = _sha256(reference)
            digest = _sha256(path)
            if digest == reference_hashes[name]:
                matches[relative.as_posix()] = dict(
                    bytes=size, sha256=digest,
                    prepared_file=str(reference.relative_to(workspace)))
    if not matches:
        return dict(files=0, bytes=0)
    receipt_path = run / 'compact_artifacts.json'
    receipt = _read(receipt_path) if receipt_path.is_file() else dict(
        policy='byte_identical_prepared_inputs_v1', artifacts={},
        interpretation='An absent listed file is byte-identical to its stable prepared_file. '
                       'Training rows, model files, logs, metrics and snapshots remain in this run.')
    receipt['artifacts'].update(matches)
    # Persist the replacement locations before removing any copy. A process
    # interruption may leave some copies present, which is safe and retryable.
    _write(receipt_path, receipt)
    for relative in matches:
        (run / relative).unlink()
    return dict(files=len(matches), bytes=sum(item['bytes'] for item in matches.values()))


def compact_job(workspace, job_id):
    """Keep the latest run intact and two recovery points across older segments."""
    workspace = Path(workspace).resolve()
    job = _job(_catalog(workspace), job_id)
    runs = _run_directories(workspace / 'runs' / 'jobs' / job_id)
    if not runs:
        return dict(removed_duplicate_files=0, removed_duplicate_bytes=0,
                    removed_checkpoints=[], space=space_status(workspace))
    latest = runs[-1]
    config = _read(latest / 'config.json')
    _verify(config, job, job_id)
    checkpoint = _checkpoint(latest)
    if config.get('status') == 'complete':
        _complete(latest, job, job_id)
    elif config.get('status') == 'interrupted' and checkpoint is not None:
        _verify(_read(checkpoint / 'run_config.json'), job, job_id)
        if _read(latest / 'metrics.json')['successful_updates'] != _read(checkpoint / 'progress.json')['step']:
            raise ValueError(f'{job_id}: latest saved cursor does not cover the interrupted segment.')
    else:
        raise ValueError(f'{job_id}: compact only after a completed or recoverably interrupted segment.')
    prepared = _prepared(workspace, job_id, job)
    if prepared is None:
        raise FileNotFoundError(f'{job_id}: stable prepared inputs must remain available before compaction.')
    removed_files, removed_bytes, hashes = 0, 0, {}
    for run in runs[:-1]:
        result = _compact_input_copies(run, prepared, workspace, hashes)
        removed_files += result['files']
        removed_bytes += result['bytes']
    candidates = []
    for run in runs:
        for marker in (run / 'checkpoints').glob('*/checkpoint.json'):
            path = marker.parent
            if path.is_symlink():
                continue
            record = _read(marker)
            if record.get('format') != 'ntllm-q8-training-v1':
                continue
            if not all((path / name).is_file() for name in ('run_config.json', 'progress.json', 'training_state.pt')):
                continue
            if _sha256(path / 'run_config.json') != record.get('run_config_sha256'):
                raise ValueError(f'Checkpoint metadata changed after saving: {path}')
            candidates.append((int(record['created_ns']), path))
    state_path = _state_path(workspace, job_id)
    state = _read(state_path) if state_path.is_file() else {}
    protected = {checkpoint} if checkpoint is not None else set()
    if state.get('checkpoint'):
        protected.add(_inside(workspace, state['checkpoint']).resolve())
    existing = {path for _, path in candidates}
    kept = protected & existing
    for _, path in sorted(candidates, reverse=True):
        if len(kept) >= 2:
            break
        kept.add(path)
    removed = []
    for _, path in candidates:
        if path not in kept:
            removed.append(str(path.relative_to(workspace)))
            shutil.rmtree(path)
    report = dict(removed_duplicate_files=removed_files, removed_duplicate_bytes=removed_bytes,
                  removed_checkpoints=removed, remaining_checkpoints=len(kept),
                  space=space_status(workspace))
    _update_state(workspace, job_id, compaction=report)
    return report


def run_job(workspace, job_id, base, *, segment_seconds=10800,
            checkpoint_every=100, checkpoint_seconds=900, stop_after_updates=None):
    """Run exactly one segment; a second invocation resumes its actual cursor."""
    workspace, base = Path(workspace).resolve(), Path(base).resolve()
    catalog = _catalog(workspace)
    job = _job(catalog, job_id)
    previous = _latest_run(workspace, job_id)
    if previous is not None and _complete(previous, job, job_id):
        recovery = _checkpoint(previous)
        _update_state(workspace, job_id, status='complete', run=str(previous.relative_to(workspace)),
                      checkpoint=str(recovery.relative_to(workspace)) if recovery else None)
        compact_job(workspace, job_id)
        return _read(_state_path(workspace, job_id))
    if not base.is_dir():
        raise FileNotFoundError(f'Attach the existing complete base model directory: {base}')
    if job['phase'] == 'external':
        initializer = next(name for name, item in catalog['jobs'].items()
                           if item['phase'] == 'external' and item['initial_group'] == job['initial_group'])
        common = workspace / 'runs' / 'common_initialization' / job['initial_group'] / 'initialization.json'
        if job_id != initializer and not common.is_file():
            raise ValueError(f'{job_id}: restore the common initialization from {initializer!r} first; '
                             'paired arms must not independently materialize their starts.')
    parent = None
    if job.get('parent'):
        parent_job = _job(catalog, job['parent'])
        parent = _latest_run(workspace, job['parent'])
        if parent is None or not _complete(parent, parent_job, job['parent']):
            raise ValueError(f'{job_id}: finish external job {job["parent"]!r} before official adaptation.')
    checkpoint = _checkpoint(previous) if previous is not None else None
    if previous is not None and checkpoint is None:
        raise ValueError(f'{job_id}: an existing run has no complete recovery checkpoint: {previous}. '
                         'Inspect the failed or unfinished run; it will not be silently restarted.')
    _require_space(workspace)
    prepared = None if checkpoint is not None else prepare_job(workspace, job_id)
    output = workspace / 'runs' / 'jobs' / job_id
    output.mkdir(parents=True, exist_ok=True)
    before = set(_run_directories(output))
    command = [sys.executable, '-m', 'ntllm.train_lora', '--base', str(base),
               '--output-dir', str(output), '--config', str(workspace / 'configs' / 'q8.toml'),
               '--checkpoint-every', str(checkpoint_every), '--checkpoint-seconds', str(checkpoint_seconds),
               '--segment-seconds', str(segment_seconds)]
    if stop_after_updates is not None:
        command += ['--stop-after-updates', str(stop_after_updates)]
    if checkpoint is not None:
        _verify(_read(checkpoint / 'run_config.json'), job, job_id)
        command += ['--resume', str(checkpoint)]
    else:
        command += ['--prepared-plan', str(prepared)]
        if parent is not None:
            command += ['--init', str(parent)]
    _require_space(workspace)
    _update_state(workspace, job_id, status='starting', command=command)
    try:
        subprocess.run(command, cwd=workspace, check=True)
    finally:
        created = set(_run_directories(output)) - before
        if len(created) == 1:
            current = created.pop()
            config = _read(current / 'config.json')
            recovery = _checkpoint(current)
            _update_state(workspace, job_id, status=config.get('status', 'unknown'),
                          run=str(current.relative_to(workspace)),
                          checkpoint=str(recovery.relative_to(workspace)) if recovery else None)
        elif len(created) > 1:
            raise RuntimeError(f'{job_id}: concurrent runs detected; inspect {output}.')
    current = _latest_run(workspace, job_id)
    if current is None or current in before:
        raise RuntimeError(f'{job_id}: process returned without a new saved run.')
    metadata = _read(current / 'config.json')
    _verify(metadata, job, job_id)
    if metadata.get('status') == 'complete':
        _complete(current, job, job_id)
    elif metadata.get('status') != 'interrupted':
        raise RuntimeError(f'{job_id}: unexpected terminal status {metadata.get("status")!r}.')
    compact_job(workspace, job_id)
    return _read(_state_path(workspace, job_id))


def list_jobs(workspace):
    workspace = Path(workspace).resolve()
    catalog = _catalog(workspace)
    result = []
    for job_id in catalog['jobs']:
        job = _job(catalog, job_id)
        run = _latest_run(workspace, job_id)
        status = _read(run / 'config.json').get('status', 'unknown') if run else 'not_started'
        result.append(dict(job=job_id, phase=job['phase'], profile=job['profile'],
                           fold=job['fold'], parent=job.get('parent'), status=status))
    return dict(jobs=result, budget=catalog.get('budget'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('/kaggle/working/ntllm'))
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    prepared = commands.add_parser('prepare')
    prepared.add_argument('job')
    run = commands.add_parser('run')
    run.add_argument('job')
    run.add_argument('--base', type=Path, required=True)
    run.add_argument('--segment-seconds', type=float, default=10800)
    run.add_argument('--checkpoint-every', type=int, default=100)
    run.add_argument('--checkpoint-seconds', type=float, default=900)
    run.add_argument('--stop-after-updates', type=int)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = dict(job=args.job, prepared=str(prepare_job(args.workspace, args.job)))
    elif args.command == 'run':
        result = run_job(args.workspace, args.job, args.base,
                         segment_seconds=args.segment_seconds, checkpoint_every=args.checkpoint_every,
                         checkpoint_seconds=args.checkpoint_seconds, stop_after_updates=args.stop_after_updates)
    else:
        result = list_jobs(args.workspace)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
