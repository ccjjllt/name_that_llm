"""Save and restore Q8 training between optimizer updates.

Use a dedicated parent directory such as run/checkpoints/step_000200. Model
weights are loaded by Q8Classifier before restoring optimizer and RNG state.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
import tomllib

import numpy as np


FORMAT = 'ntllm-q8-training-v1'

# Operational limits are separate from the immutable research settings. Old
# prepared plans/checkpoints also receive these defaults when first executed.
EXECUTION_DEFAULTS = dict(checkpoint_every=100, checkpoint_seconds=900.0,
                          segment_seconds=10800.0, stop_after_updates=None)


def execution_policy(config_path=None, saved=None, **overrides):
    """Resolve defaults < config [execution] < saved policy < explicit options.

    Limits count successful updates / elapsed wall time, never change the
    optimizer horizon. Zero explicitly disables a time or checkpoint cadence.
    """
    policy = dict(EXECUTION_DEFAULTS)
    if config_path is not None:
        with Path(config_path).open('rb') as handle:
            policy.update(tomllib.load(handle).get('execution', {}))
    policy.update(saved or {})
    policy.update({key: value for key, value in overrides.items() if value is not None})
    if policy.keys() != EXECUTION_DEFAULTS.keys():
        raise ValueError('Unknown execution policy options.')
    for key in ('checkpoint_every', 'stop_after_updates'):
        value = policy[key]
        if key == 'stop_after_updates' and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{key} must be a nonnegative integer.')
    for key in ('checkpoint_seconds', 'segment_seconds'):
        value = policy[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative.')
    return policy


def save_training_checkpoint(path, model, optimizer, scheduler, scaler, progress, metadata, keep=2):
    """Save a new checkpoint directory atomically and return its Path.

    progress records the successful update, next epoch/window, exposed row
    indices and elapsed time. Call only after a completed optimizer update;
    gradients accumulated within an unfinished window are not saved.
    """
    import torch

    path = Path(path)
    if keep < 1:
        raise ValueError('Keep at least one training checkpoint.')
    if path.exists():
        raise FileExistsError(f'Checkpoint already exists; choose a new update directory: {path}')
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    progress_text = json.dumps(progress, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    config_sha = hashlib.sha256(metadata_text.encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{path.name}.tmp-', dir=path.parent))
    try:
        model.save(temporary)
        numpy_state = np.random.get_state()
        device = torch.device(model.device)
        cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
        state = dict(
            format=FORMAT,
            device_type=device.type,
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(), python_rng=random.getstate(),
            numpy_rng=dict(generator=numpy_state[0], keys=numpy_state[1].tolist(),
                           position=int(numpy_state[2]), has_gauss=int(numpy_state[3]),
                           cached_gaussian=float(numpy_state[4])),
            torch_rng=torch.get_rng_state(), cuda_rng=cuda_rng,
            mps_rng=torch.mps.get_rng_state() if device.type == 'mps' else None,
        )
        torch.save(state, temporary / 'training_state.pt')
        (temporary / 'run_config.json').write_text(metadata_text, encoding='utf-8')
        (temporary / 'progress.json').write_text(progress_text, encoding='utf-8')
        marker = dict(format=FORMAT, created_ns=time.time_ns(), run_config_sha256=config_sha)
        (temporary / 'checkpoint.json').write_text(json.dumps(marker, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    # The caller dedicates this parent to one run. Ordinary model/run folders
    # and incomplete temporary saves never carry this complete-checkpoint marker.
    previous = []
    for candidate in path.parent.iterdir():
        if not candidate.is_dir() or candidate.is_symlink() or candidate.name.startswith('.'):
            continue
        try:
            marker = json.loads((candidate / 'checkpoint.json').read_text())
        except (OSError, ValueError):
            continue
        if marker.get('format') == FORMAT:
            previous.append((int(marker['created_ns']), candidate))
    previous.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    for _, candidate in previous[keep:]:
        if candidate != path:
            shutil.rmtree(candidate)
    return path


def load_training_state(path, optimizer, scheduler, scaler, device):
    """Restore after model/optimizer construction; return progress and metadata.

    The checkpoint's adapter/heads must already be loaded by the caller. The
    saved state uses tensors and primitive containers, so weights_only=True
    remains available even for Python and NumPy RNG restoration.
    """
    import torch

    path = Path(path)
    marker = json.loads((path / 'checkpoint.json').read_text())
    if marker.get('format') != FORMAT:
        raise ValueError('Not a Q8 training checkpoint produced by this helper.')
    metadata_text = (path / 'run_config.json').read_text(encoding='utf-8')
    if hashlib.sha256(metadata_text.encode()).hexdigest() != marker['run_config_sha256']:
        raise ValueError('Checkpoint run metadata changed after the state was saved.')
    progress = json.loads((path / 'progress.json').read_text())
    metadata = json.loads(metadata_text)
    state = torch.load(path / 'training_state.pt', map_location='cpu', weights_only=True)
    if state.get('format') != FORMAT:
        raise ValueError('Unexpected training-state format.')
    device = torch.device(device)
    if state['device_type'] != device.type:
        raise ValueError('Resume on the same device type to preserve the training RNG.')
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    random.setstate(state['python_rng'])
    numpy_state = state['numpy_rng']
    np.random.set_state((numpy_state['generator'], np.asarray(numpy_state['keys'], dtype=np.uint32),
                         numpy_state['position'], numpy_state['has_gauss'], numpy_state['cached_gaussian']))
    torch.set_rng_state(state['torch_rng'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda_rng'], device)
    elif device.type == 'mps':
        torch.mps.set_rng_state(state['mps_rng'])
    return dict(progress=progress, metadata=metadata)
