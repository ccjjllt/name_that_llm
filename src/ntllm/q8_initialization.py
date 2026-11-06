"""A common materialized start, bound to actual base and tokenizer contents.

This snapshot contains trainable parameters and initial RNG, never optimizer
state. Each experiment creates its own fresh optimizer after loading it. It is
not an external-trained checkpoint and cannot replace a fold's external start.
"""

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

from .q8_data import file_sha


FORMAT = 'ntllm-paired-initialization-v1'
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                   'added_tokens.json', 'vocab.json', 'merges.txt', 'tokenizer.model',
                   'spiece.model', 'sentencepiece.bpe.model', 'chat_template.jinja')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _file_record(path):
    return dict(bytes=path.stat().st_size, sha256=file_sha(path))


def asset_identity(base, tokenizer_dir=None, require_weights=True):
    """Hash the actual HF weight index/shards and local tokenizer assets.

    Safetensors takes precedence, matching the normal Transformers loader. The
    filenames remain relative in the identity, so moving an unchanged snapshot
    does not change its content identity. Missing weights are explicit and may
    only be used for a plan; an initialization cannot bind to an incomplete base.
    """
    base = Path(base)
    tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else base
    config = base / 'config.json'
    if not config.is_file():
        raise FileNotFoundError(f'Base model config missing: {config}')
    selected_index, weight_files, weight_format = None, [], None
    for fmt, index_name, single_name in (
            ('safetensors', 'model.safetensors.index.json', 'model.safetensors'),
            ('pytorch_bin', 'pytorch_model.bin.index.json', 'pytorch_model.bin')):
        index_path = base / index_name
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            names = sorted(set(index.get('weight_map', {}).values()))
            if not names or not all(isinstance(name, str) for name in names):
                raise ValueError('The base weight index has no valid shard map.')
            for name in names:
                relative = Path(name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Weight-index shard names must stay inside the base directory.')
                if not (base / name).is_file():
                    if require_weights:
                        raise FileNotFoundError(f'Indexed weight shard missing: {base / name}')
            selected_index, weight_files, weight_format = index_path, names, fmt
            break
        if (base / single_name).is_file():
            weight_files, weight_format = [single_name], fmt
            break
    complete_weights = bool(weight_files) and all((base / name).is_file() for name in weight_files)
    if require_weights and not complete_weights:
        raise FileNotFoundError(f'No complete local base-model weights in {base}; nothing was downloaded.')
    tokenizer_files = {name: _file_record(tokenizer_dir / name) for name in TOKENIZER_FILES
                       if (tokenizer_dir / name).is_file()}
    # Also retain tokenizer vocabulary/template variations declared by the
    # local directory instead of silently accepting a different added-token set.
    for path in sorted(tokenizer_dir.glob('chat_templates/*.jinja')):
        tokenizer_files[str(path.relative_to(tokenizer_dir))] = _file_record(path)
    has_tokenizer = any(name in tokenizer_files for name in
                        ('tokenizer.json', 'vocab.json', 'tokenizer.model', 'spiece.model', 'sentencepiece.bpe.model'))
    if not has_tokenizer:
        raise FileNotFoundError(f'No local tokenizer vocabulary in {tokenizer_dir}.')
    content = dict(kind='transformer_base', model_config=_file_record(config),
                   weight_format=weight_format,
                   weight_index=({selected_index.name: _file_record(selected_index)} if selected_index else {}),
                   weights={name: (_file_record(base / name) if (base / name).is_file() else None)
                            for name in weight_files},
                   tokenizer_files=tokenizer_files, complete=complete_weights)
    return dict(content=content, identity_sha256=_digest(content),
                base_directory=str(base.resolve()), tokenizer_directory=str(tokenizer_dir.resolve()),
                weights_available=complete_weights)


def _validate_assets(assets):
    if assets.get('identity_sha256') != _digest(assets['content']):
        raise ValueError('Asset identity digest does not match its contents.')
    if not assets['content'].get('complete', False):
        raise ValueError('A real initialization requires complete frozen asset contents; a plan is not a model start.')


def _context(model, assets, labels, architecture, backend):
    _validate_assets(assets)
    if hasattr(model, 'labels') and list(model.labels) != list(labels):
        raise ValueError('Declared labels differ from the constructed model.')
    if hasattr(model, 'model_config') and model.model_config != architecture:
        raise ValueError('Declared architecture differs from the constructed model.')
    return dict(asset_identity_sha256=assets['identity_sha256'], labels=list(labels),
                architecture=architecture, backend=backend,
                compute_dtype=str(getattr(model, 'compute_dtype', 'not_declared')))


def _tensor_records(tensors):
    import torch

    records = {}
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        records[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                             sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    return records


def _rng_state(model):
    import numpy as np
    import torch

    device = torch.device(getattr(model, 'device', 'cpu'))
    numpy_state = np.random.get_state()
    return dict(device_type=device.type, python=random.getstate(),
                numpy=dict(generator=numpy_state[0], keys=numpy_state[1].tolist(),
                           position=int(numpy_state[2]), has_gauss=int(numpy_state[3]),
                           cached_gaussian=float(numpy_state[4])),
                torch_cpu=torch.get_rng_state(),
                accelerator=(torch.cuda.get_rng_state(device) if device.type == 'cuda' else
                             torch.mps.get_rng_state() if device.type == 'mps' else None))


def _restore_rng(state, model):
    import numpy as np
    import torch

    device = torch.device(getattr(model, 'device', 'cpu'))
    if device.type != state['device_type']:
        raise ValueError('Use the same device type for the common initial training RNG.')
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['generator'], np.asarray(n['keys'], dtype=np.uint32),
                         n['position'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['accelerator'], device)
    elif device.type == 'mps':
        torch.mps.set_rng_state(state['accelerator'])


def initialization_identity(directory):
    """Read a materialized identity without loading parameter tensors or GPUs."""
    directory = Path(directory)
    if not directory.exists():
        return dict(status='pending', directory=str(directory.resolve()),
                    reason='Common Q8 trainable parameters have not been materialized.')
    manifest = json.loads((directory / 'initialization.json').read_text())
    if manifest.get('format') != FORMAT:
        raise ValueError('Not a common initial-parameter snapshot.')
    _validate_assets(manifest['assets'])
    if manifest['assets']['identity_sha256'] != manifest['context']['asset_identity_sha256']:
        raise ValueError('Snapshot asset contents differ from its initialization context.')
    if set(manifest['files']) != {'trainables.safetensors', 'initial_rng.pt'}:
        raise ValueError('Unexpected files in the common initialization snapshot.')
    for filename, expected in manifest['files'].items():
        if _file_record(directory / filename) != expected:
            raise ValueError(f'Common initialization file changed: {filename}')
    expected = _digest({k: manifest[k] for k in ('format', 'context', 'tensor_records', 'files')})
    if expected != manifest['initialization_sha256']:
        raise ValueError('Common initialization manifest changed.')
    return dict(status='available', directory=str(directory.resolve()), **manifest)


def save_initialization(model, directory, *, assets, labels, architecture, backend):
    """Freeze one realized start. Optimizer/scheduler state is intentionally absent."""
    import torch
    from safetensors.torch import save_file

    directory = Path(directory)
    if directory.exists():
        raise FileExistsError('Common initialization already exists; load it rather than overwriting history.')
    context = _context(model, assets, labels, architecture, backend)
    tensors = {name: value.detach().cpu().contiguous().clone() for name, value in model.named_parameters()
               if value.requires_grad}
    if not tensors:
        raise ValueError('No trainable parameters to freeze.')
    records = _tensor_records(tensors)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{directory.name}.', dir=directory.parent))
    try:
        save_file(tensors, str(temporary / 'trainables.safetensors'))
        torch.save(_rng_state(model), temporary / 'initial_rng.pt')
        manifest = dict(format=FORMAT, context=context, assets=assets, tensor_records=records,
                        files={name: _file_record(temporary / name) for name in
                               ('trainables.safetensors', 'initial_rng.pt')},
                        optimizer_state='absent; each experiment creates a fresh optimizer',
                        rng_scope='Python, NumPy, Torch CPU and active accelerator after model construction')
        manifest['initialization_sha256'] = _digest({k: manifest[k] for k in ('format', 'context', 'tensor_records', 'files')})
        (temporary / 'initialization.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        os.rename(temporary, directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return initialization_identity(directory)


def load_initialization(model, directory, *, assets, labels, architecture, backend, restore_rng=True):
    """Load exact trainables into a freshly constructed compatible model."""
    import torch
    from safetensors.torch import load_file

    identity = initialization_identity(directory)
    if identity['status'] != 'available':
        raise FileNotFoundError('Materialize the common initial parameters before training this arm.')
    context = _context(model, assets, labels, architecture, backend)
    if context != identity['context']:
        raise ValueError('Common initialization differs in base/tokenizer, labels, architecture, backend or precision.')
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    tensors = load_file(str(Path(directory) / 'trainables.safetensors'), device='cpu')
    if set(parameters) != set(tensors):
        raise ValueError('Trainable parameter names differ from the common initialization.')
    records = _tensor_records(tensors)
    if records != identity['tensor_records']:
        raise ValueError('Saved parameter tensor values differ from their inventory.')
    for name, parameter in parameters.items():
        if list(parameter.shape) != records[name]['shape'] or str(parameter.dtype) != records[name]['dtype']:
            raise ValueError(f'Trainable parameter shape/dtype differs: {name}')
    # Validate the RNG device before changing parameters, not after half a load.
    state = torch.load(Path(directory) / 'initial_rng.pt', map_location='cpu', weights_only=True) if restore_rng else None
    if state and torch.device(getattr(model, 'device', 'cpu')).type != state['device_type']:
        raise ValueError('Initial training RNG has a different device type.')
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(tensors[name].to(parameter.device))
    if _tensor_records(parameters) != records:
        raise ValueError('Parameter read-back after loading does not match the common start.')
    if restore_rng:
        _restore_rng(state, model)
    return dict(identity, loaded_parameter_tensors=len(records), values_verified=True,
                optimizer_state='not loaded; use a newly constructed optimizer', rng_restored=restore_rng)
