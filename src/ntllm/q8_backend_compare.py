"""Explicit HF/Unsloth controls with one canonical trainable-parameter start.

Only the inspected extra SequenceClassification ``model`` wrapper is removed.
This is not a general fuzzy checkpoint converter, optimizer-state converter,
or evidence that the two backends have equal training quality.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import pandas as pd

from .q8_data import file_sha
from .q8_initialization import (_context, _digest, _file_record, _tensor_records,
                                _rng_state, _restore_rng, _validate_assets)

FORMAT = 'ntllm-canonical-q8-backend-initialization-v1'
PREFIXES = {'hf': 'backbone.base_model.model.layers.',
            'unsloth': 'backbone.base_model.model.model.layers.'}
PROJECTIONS = {'self_attn': ('q_proj', 'k_proj', 'v_proj', 'o_proj'),
               'mlp': ('gate_proj', 'up_proj', 'down_proj')}
LORA = re.compile(r'(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_([AB])\.default\.weight')
HEAD = re.compile(r'(?:norms\.\d+|plain|centered)\.(?:weight|bias)')


def canonical_name(name, backend):
    if backend not in PREFIXES:
        raise ValueError('This explicit comparison supports hf and unsloth only.')
    prefix = PREFIXES[backend]
    if name.startswith(prefix):
        tail = name[len(prefix):]
        match = LORA.fullmatch(tail)
        if match is None or match[3] not in PROJECTIONS[match[2]]:
            raise ValueError(f'Unrecognized LoRA parameter: {name}')
        return 'backbone.layers.' + tail.removesuffix('.default.weight') + '.weight'
    if HEAD.fullmatch(name):
        return name
    raise ValueError(f'Unexpected trainable parameter (including any unfrozen score head): {name}')


def expected_keys(architecture):
    result = {f'backbone.layers.{layer}.{block}.{projection}.lora_{side}.weight'
              for layer in range(int(architecture['keep'])) for block, projections in PROJECTIONS.items()
              for projection in projections for side in ('A', 'B')}
    result.update(f'norms.{i}.{part}' for i in range(len(architecture['layers'])) for part in ('weight', 'bias'))
    result.update(f'{head}.{part}' for head in ('plain', 'centered') for part in ('weight', 'bias'))
    return result


def canonical_parameters(model, backend, architecture=None):
    architecture = architecture or model.model_config
    parameters, native = {}, {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        key = canonical_name(name, backend)
        if key in parameters:
            raise ValueError(f'Canonical mapping collision: {key}')
        parameters[key], native[key] = parameter, name
    expected = expected_keys(architecture)
    if set(parameters) != expected:
        raise ValueError(f'Incomplete trainables: missing={sorted(expected-set(parameters))}, extra={sorted(set(parameters)-expected)}')
    rank = int(architecture['r'])
    for key, value in parameters.items():
        if '.lora_A.' in key and (value.ndim != 2 or value.shape[0] != rank):
            raise ValueError('LoRA A does not have the declared rank: ' + key)
        if '.lora_B.' in key and (value.ndim != 2 or value.shape[1] != rank):
            raise ValueError('LoRA B does not have the declared rank: ' + key)
    return parameters, native


def _canonical_context(model, assets, labels, architecture, backend):
    context = _context(model, assets, labels, architecture, backend)
    context.pop('backend')
    context['canonical_layout'] = 'qwen3_lora_7_projections_plus_pool_norms_and_two_heads_v1'
    return context


def backend_initialization_identity(directory):
    directory = Path(directory)
    if not directory.exists():
        return {'status': 'pending', 'directory': str(directory.resolve()),
                'reason': 'Canonical cross-backend Q8 tensors have not been materialized.'}
    manifest = json.loads((directory / 'initialization.json').read_text())
    if manifest.get('format') != FORMAT:
        raise ValueError('Not an explicitly canonical backend initialization.')
    _validate_assets(manifest['assets'])
    if manifest['assets']['identity_sha256'] != manifest['context']['asset_identity_sha256']:
        raise ValueError('Canonical initialization asset identity changed.')
    if set(manifest['tensor_records']) != expected_keys(manifest['context']['architecture']):
        raise ValueError('Canonical snapshot does not cover the complete Q8 trainable set.')
    if manifest.get('allowed_backends') != ['hf', 'unsloth'] or manifest.get('source_backend') not in PREFIXES:
        raise ValueError('Unsupported canonical backend definition.')
    native = manifest['native_name_map']
    if set(native) != set(manifest['tensor_records']) or len(set(native.values())) != len(native):
        raise ValueError('Native parameter mapping is incomplete or ambiguous.')
    if any(canonical_name(name, manifest['source_backend']) != key for key, name in native.items()):
        raise ValueError('Native parameter mapping no longer matches the explicit wrapper rule.')
    if set(manifest['files']) != {'trainables.safetensors', 'initial_rng.pt'}:
        raise ValueError('Unexpected canonical snapshot files.')
    for name, expected in manifest['files'].items():
        if _file_record(directory / name) != expected:
            raise ValueError('Canonical snapshot content changed: ' + name)
    if _digest({key: manifest[key] for key in ('format', 'context', 'tensor_records', 'files')}) != manifest['initialization_sha256']:
        raise ValueError('Canonical initialization manifest changed.')
    return dict(manifest, status='available', directory=str(directory.resolve()))


def save_backend_initialization(model, directory, *, assets, labels, architecture, backend):
    import torch
    from safetensors.torch import save_file
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError('Do not overwrite a common initial state.')
    parameters, native = canonical_parameters(model, backend, architecture)
    context = _canonical_context(model, assets, labels, architecture, backend)
    tensors = {key: value.detach().cpu().contiguous().clone() for key, value in parameters.items()}
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + directory.name, dir=directory.parent))
    try:
        save_file(tensors, str(temporary / 'trainables.safetensors'))
        torch.save(_rng_state(model), temporary / 'initial_rng.pt')
        manifest = dict(format=FORMAT, context=context, assets=assets,
                        source_backend=backend, allowed_backends=['hf', 'unsloth'], native_name_map=native,
                        tensor_records=_tensor_records(tensors),
                        files={name: _file_record(temporary / name) for name in ('trainables.safetensors', 'initial_rng.pt')},
                        optimizer_state='absent; construct a new optimizer independently in each backend',
                        exclusion='Frozen pretrained parameters are bound by complete base-file identity. Unused Unsloth container score must be frozen and is not a classifier head.')
        manifest['initialization_sha256'] = _digest({key: manifest[key] for key in ('format', 'context', 'tensor_records', 'files')})
        (temporary / 'initialization.json').write_text(json.dumps(manifest, indent=2) + '\n')
        os.rename(temporary, directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return backend_initialization_identity(directory)


def load_backend_initialization(model, directory, *, assets, labels, architecture, backend, restore_rng=True):
    import torch
    from safetensors.torch import load_file
    saved = backend_initialization_identity(directory)
    if saved['status'] != 'available':
        raise FileNotFoundError('Materialize one canonical initial state before backend training.')
    if backend not in saved['allowed_backends'] or _canonical_context(model, assets, labels, architecture, backend) != saved['context']:
        raise ValueError('Base/tokenizer, architecture, labels or precision differ across backends.')
    parameters, native = canonical_parameters(model, backend, architecture)
    tensors = load_file(str(Path(directory) / 'trainables.safetensors'), device='cpu')
    if _tensor_records(tensors) != saved['tensor_records'] or set(tensors) != set(parameters):
        raise ValueError('Canonical tensors or complete parameter coverage changed.')
    for key, parameter in parameters.items():
        expected = saved['tensor_records'][key]
        if list(parameter.shape) != expected['shape'] or str(parameter.dtype) != expected['dtype']:
            raise ValueError('Shape/dtype mismatch: ' + key)
    state = torch.load(Path(directory) / 'initial_rng.pt', map_location='cpu', weights_only=True) if restore_rng else None
    if state and torch.device(model.device).type != state['device_type']:
        raise ValueError('Compare initial RNG on the same device type.')
    with torch.no_grad():
        for key, parameter in parameters.items():
            parameter.copy_(tensors[key].to(parameter.device))
    if _tensor_records(parameters) != saved['tensor_records']:
        raise ValueError('Canonical parameter values changed during import.')
    if restore_rng:
        _restore_rng(state, model)
    return dict(saved, values_verified=True, loaded_backend=backend,
                loaded_native_names=native, loaded_parameter_tensors=len(parameters), rng_restored=restore_rng)


def derive_backend_plan(prepared, out, target_backend, canonical_initial_state):
    """Explicitly derive a backend-only arm while preserving scientific inputs."""
    from .q8_exposure import load_exposure_manifest, _sha, _json
    import numpy as np
    prepared, out, canonical_initial_state = map(Path, (prepared, out, canonical_initial_state))
    if target_backend not in PREFIXES or out.exists():
        raise ValueError('Choose hf/unsloth and a new output directory.')
    meta = json.loads((prepared / 'config.json').read_text())
    exposure = json.loads((prepared / 'exposure_manifest.json').read_text())
    if meta['phase'] != 'external' or meta['status'] != 'planned' or not meta['prepared_only']:
        raise ValueError('Backend arms derive from an unexecuted external plan.')
    if any((prepared / name).exists() for name in ('train_log.jsonl', 'metrics.json', 'executed_initialization.json', 'checkpoints')):
        raise ValueError('An executed run is not a new backend plan.')
    data_path = Path(meta['data_path'])
    if file_sha(data_path) != meta['data_sha256'] or file_sha(data_path.parent / 'manifest.json') != meta['data_manifest_sha256']:
        raise ValueError('Prepared data changed.')
    frame = pd.read_parquet(data_path)
    weights = pd.read_parquet(prepared / 'training_rows.parquet').loss_weight.to_numpy(dtype='float32')
    auxiliary = np.load(prepared / 'centered_loss_weights.npy', allow_pickle=False)
    if meta.get('relation_data_path'):
        from .q8_supervision import align_anchor_rows
        relations = align_anchor_rows(frame, pd.read_parquet(meta['relation_data_path']))
    else:
        relations = None
    load_exposure_manifest(prepared, frame, weights, auxiliary, relations,
                           meta['settings']['centered_weight'], expected_config=meta['settings'])
    identity = backend_initialization_identity(canonical_initial_state)
    reference = dict(status=identity['status'], model_kind='transformer_base',
                     directory=str(canonical_initial_state.resolve()), format=FORMAT)
    if identity['status'] == 'available':
        if identity['assets']['content']['kind'] != 'transformer_base':
            raise ValueError('A CPU study snapshot cannot stand in for Q8 initial weights.')
        reference.update(initialization_sha256=identity['initialization_sha256'],
                         manifest_path=str((canonical_initial_state / 'initialization.json').resolve()),
                         manifest_sha256=file_sha(canonical_initial_state / 'initialization.json'))
    shutil.copytree(prepared, out)
    source_settings = dict(meta['settings'])
    meta['settings'] = dict(source_settings, backend=target_backend, initialization_mode='canonical_backend')
    meta['shared_initialization_path'] = str(canonical_initial_state.resolve())
    meta['backend_initialization_path'] = str(canonical_initial_state.resolve())
    meta['initialization_proof'] = reference
    meta['backend_derivation'] = dict(source=str(prepared.resolve()), source_config_sha256=file_sha(prepared / 'config.json'),
                                    source_backend=source_settings.get('backend', 'hf'), target_backend=target_backend,
                                    allowed_change='Backend implementation only; canonical initialization is the common binding in both derived arms.')
    exposure['settings'] = meta['settings']
    exposure['settings_sha256'] = _sha(_json(meta['settings']))
    exposure['references']['initialization'] = reference
    exposure['references']['model_kind'] = 'transformer_base'
    (out / 'exposure_manifest.json').write_text(json.dumps(exposure, indent=2, ensure_ascii=False) + '\n')
    meta['exposure_manifest_sha256'] = file_sha(out / 'exposure_manifest.json')
    (out / 'config.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
    preserved = {p.name: file_sha(p) for p in prepared.iterdir() if p.is_file() and p.name not in ('config.json', 'exposure_manifest.json')}
    if any(file_sha(out / name) != digest for name, digest in preserved.items()):
        raise ValueError('Deriving backend changed scientific artifacts.')
    report = dict(source=str(prepared.resolve()), target_backend=target_backend, preserved_files=preserved,
                  status='derived_initialization_' + identity['status'], quality='not_measured')
    (out / 'backend_derivation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def _inner_quality(path, config, successful_updates):
    from .q8_inner import (content_identity, validate_split_manifest,
                           validate_training_evidence, checkpoint_identity)
    directory = path / 'inner_evaluation'
    required = ('query_rows.parquet', 'training_evidence.json', 'evaluation.json')
    if not all((directory / name).is_file() for name in required):
        return dict(status='not_measured', role='inner_dev')
    split = validate_split_manifest(config['inner_split'])
    rows = pd.read_parquet(directory / 'query_rows.parquet')
    if (not rows.id.is_unique or set(rows.id.astype(str)) != set(split['inner_dev']['ids'])
            or content_identity(rows.drop(columns='fold', errors='ignore')) != split['inner_dev']['content_sha256']):
        raise ValueError('Backend quality must use exactly the fixed inner-development content.')
    report = json.loads((directory / 'evaluation.json').read_text())
    evidence = json.loads((directory / 'training_evidence.json').read_text())
    if file_sha(path / 'training_rows.parquet') != evidence['official_training_rows_sha256']:
        raise ValueError('Evaluation fitting evidence is not bound to this run training membership.')
    artifacts = {str(item.relative_to(path)): item for item in path.rglob('*.safetensors')
                 if 'checkpoints' not in item.relative_to(path).parts and 'snapshots' not in item.relative_to(path).parts}
    for name in ('model_config.json', 'base_identity.json'):
        if (path / name).is_file():
            artifacts[name] = path / name
    actual_model = checkpoint_identity(artifacts)
    if actual_model['model_identity_sha256'] != evidence.get('model_identity_sha256'):
        raise ValueError('Inner-development evidence belongs to another saved model.')
    provenance = report.get('provenance', {})
    if (provenance.get('role') != 'inner_dev' or provenance.get('split_sha256') != split['split_sha256']
            or provenance.get('checkpoint_updates') != successful_updates or report['labels'] != split['labels']
            or provenance.get('model_identity_sha256') != evidence.get('model_identity_sha256')):
        raise ValueError('Backend evaluation is not bound to the same inner split and actual checkpoint.')
    if config['phase'] == 'official':
        membership = validate_training_evidence(evidence, split)
    else:
        # External-only snapshots have no official optimizer fitting rows. The
        # Gaussian/grouping representation fitting still uses inner_fit only.
        if (evidence.get('model_scope') != 'external_only' or provenance.get('model_scope') != 'external_only'
                or provenance.get('official_optimizer_training_ids') != []
                or evidence.get('split_sha256') != split['split_sha256']):
            raise ValueError('External development evidence contains official optimizer fitting.')
        for name in ('representation_fit_ids', 'head_fit_ids'):
            if set(evidence.get(name, [])) != set(split['inner_fit']['ids']):
                raise ValueError('External development transforms must use the frozen inner fit scope.')
        training = Path(evidence['official_training_rows_file'])
        if file_sha(training) != evidence['official_training_rows_sha256']:
            raise ValueError('External training-membership artifact changed.')
        saved = checkpoint_identity({name: item['path'] for name, item in evidence['model_artifacts'].items()})
        if (saved['model_identity_sha256'] != evidence['model_identity_sha256'] or
                any(saved['model_artifacts'][name]['sha256'] != item['sha256']
                    for name, item in evidence['model_artifacts'].items())):
            raise ValueError('External checkpoint artifacts changed after development evaluation.')
        membership = dict(scope='external_only', membership_verified=True,
                          model_identity_sha256=saved['model_identity_sha256'])
    return dict(status='verified_inner_dev', role='inner_dev', split_sha256=split['split_sha256'],
                content_sha256=split['inner_dev']['content_sha256'], rows=len(rows),
                checkpoint_updates=successful_updates, membership=membership,
                data_governance=report.get('data_governance', dict(status='unavailable')),
                validation_scope=report.get('validation_scope', dict(kind='unrecorded')),
                source_intervention=report.get('source_intervention', dict(status='unrecorded')),
                actual_source_intervention_exposure=report.get('actual_source_intervention_exposure'),
                evaluation_options=report['options'], full_views=report['cohorts']['full']['views'],
                outer_labels_used=False)


def _run_observations(path, config):
    import numpy as np
    measured = json.loads((path / 'metrics.json').read_text()) if (path / 'metrics.json').is_file() else {}
    logs = [json.loads(line) for line in (path / 'train_log.jsonl').read_text().splitlines() if line.strip()] if (path / 'train_log.jsonl').is_file() else []
    successful = measured.get('successful_updates')
    if logs:
        covered = [item['update'] for item in logs]
        if (covered != list(range(covered[0], covered[-1] + 1))
                or (successful is not None and covered[-1] != successful)):
            raise ValueError('Training log is not a contiguous completed update segment.')
    log_coverage = dict(first_update=logs[0]['update'] if logs else None,
                        last_update=logs[-1]['update'] if logs else None,
                        covers_entire_run=bool(logs) and logs[0]['update'] == 1,
                        policy='A resumed directory contains this segment only; missing earlier diagnostics/timing are not inferred.')
    snapshots = [json.loads(item.read_text()) for item in sorted((path / 'snapshots').glob('*/snapshot.json'))]
    diagnostics = {}
    for item in sorted((path / 'gradient_diagnostics').glob('update_*.json')):
        value = json.loads(item.read_text())
        update = value['context']['update']
        if (successful is not None and update > successful) or str(update) in diagnostics:
            raise ValueError('Gradient diagnostics do not name unique completed updates.')
        diagnostics[str(update)] = value
    times = dict(train_wall_seconds=measured.get('train_seconds'),
                 update_compute_seconds=sum(item.get('update_compute_seconds', 0.) for item in logs) if logs else None,
                 gradient_diagnostic_seconds=sum(item.get('gradient_diagnostic_seconds', 0.) for item in logs) if logs else None,
                 snapshot_save_and_development_seconds=sum(item['save_and_development_seconds'] for item in snapshots) if snapshots else None,
                 model_load_seconds=measured.get('model_load_seconds'),
                 final_evaluation_seconds=measured.get('evaluation_seconds'),
                 log_coverage=log_coverage,
                 interpretation='Training wall time is cumulative; compute/diagnostic sums cover the recorded segment and snapshot overhead covers this directory. Null means unrecorded, not zero.')
    physical = {}
    for field in ('physical_tokens', 'physical_tokens_including_retries'):
        if logs and all(field in item for item in logs):
            physical[field] = {key: sum(item[field][key] for item in logs)
                               for key in ('rows', 'input_tokens', 'padded_tokens', 'padding_tokens')}
    budget = {name: measured.get(name) for name in ('successful_updates', 'target_updates', 'row_exposures',
              'actual_plain_tokens', 'actual_centered_tokens')}
    actual_exposure_verified = False
    if successful is not None and all((path / name).is_file() for name in ('actual_row_indices.npy', 'actual_exposure.parquet')):
        expected = pd.read_parquet(path / 'actual_exposure.parquet')
        expected = expected.loc[expected['update'].le(successful)]
        actual = np.load(path / 'actual_row_indices.npy', allow_pickle=False)
        if len(actual) != measured['row_exposures'] or not np.array_equal(actual, expected.row_position.to_numpy()):
            raise ValueError('Executed row positions differ from the actual successful-update prefix.')
        if config.get('relation_data_path'):
            actual_relation = np.load(path / 'actual_relation_row_indices.npy', allow_pickle=False)
            if not np.array_equal(actual_relation, expected.loc[expected.role.eq('relation'), 'row_position'].to_numpy()):
                raise ValueError('Executed anchor positions differ from the actual successful-update prefix.')
        actual_exposure_verified = True
    return dict(backend=config['settings']['backend'], status=config['status'],
                executed=not config.get('prepared_only', False), actual_dtype=config.get('dtype'),
                interrupted=measured.get('interrupted'), training_budget=budget,
                actual_exposure_verified=actual_exposure_verified,
                executed_token_identity=config.get('executed_token_identity'),
                executed_relation_token_identity=config.get('executed_relation_token_identity'),
                executed_tokenizer_identity=config.get('executed_tokenizer_identity'),
                quality=_inner_quality(path, config, successful), timing=times,
                physical_work=physical, diagnostic_updates=diagnostics)


def _canonical_run_receipt(path, config):
    receipt_path = path / 'executed_initialization.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else config.get('initialization_proof', {})
    if receipt.get('status') == 'arm_specific_external_checkpoint':
        receipt = receipt.get('shared_root') or {}
    shared = config.get('shared_initialization') or {}
    directory = receipt.get('directory') or shared.get('directory') or config.get('shared_initialization_path')
    if not directory:
        return None, False
    identity = backend_initialization_identity(directory)
    expected = receipt.get('initialization_sha256')
    if identity['status'] != 'available' or expected is None:
        return None, False
    if expected != identity['initialization_sha256']:
        raise ValueError('Canonical initialization bound to this run changed.')
    if receipt.get('manifest_sha256') and file_sha(Path(directory) / 'initialization.json') != receipt['manifest_sha256']:
        raise ValueError('Canonical initialization manifest reference changed.')
    actual_import = (shared.get('values_verified') is True
                     and shared.get('initialization_sha256') == expected
                     and shared.get('loaded_backend') == config['settings']['backend'])
    return expected, actual_import


def compare_backend_runs(left, right, output=None):
    """Compare only equal-budget inner-dev quality, updates and observed time."""
    paths = list(map(Path, (left, right)))
    configs = [json.loads((path / 'config.json').read_text()) for path in paths]
    settings = [dict(c['settings']) for c in configs]
    backends = [s.pop('backend') for s in settings]
    if set(backends) != {'hf', 'unsloth'} or settings[0] != settings[1]:
        raise ValueError('Backend comparison changed another training setting.')
    if configs[0]['phase'] != configs[1]['phase']:
        raise ValueError('Compare the same training phase in both backends.')
    if configs[0].get('training_code_sha256') != configs[1].get('training_code_sha256'):
        raise ValueError('Backend arms used different recorded training source contents.')
    manifests = [json.loads((path / 'exposure_manifest.json').read_text()) for path in paths]
    for path, config, manifest in zip(paths, configs, manifests):
        if file_sha(path / 'exposure_manifest.json') != config['exposure_manifest_sha256']:
            raise ValueError('Frozen exposure manifest changed.')
        for name, digest in manifest['files'].items():
            if file_sha(path / name) != digest:
                raise ValueError('Frozen backend exposure artifact changed: ' + name)
    if manifests[0]['labels'] != manifests[1]['labels']:
        raise ValueError('Backend arms have different label order.')
    names = ['actual_exposure.parquet', 'update_schedule.parquet', 'plan.npz', 'centered_loss_weights.npy', 'training_rows.parquet']
    names += [name for name in ('token_rows.parquet', 'relation_token_rows.parquet', 'physical_token_batches.parquet')
              if any((path / name).exists() for path in paths)]
    equal = {name: all((path / name).is_file() for path in paths) and file_sha(paths[0] / name) == file_sha(paths[1] / name)
             for name in names}
    if not all(equal.values()):
        raise ValueError('Backend arms do not use the same actual exposure/weights/token inputs.')
    receipts = [_canonical_run_receipt(path, config) for path, config in zip(paths, configs)]
    initializations = [item[0] for item in receipts]
    metrics = [_run_observations(path, config) for path, config in zip(paths, configs)]
    same_initial = all(initializations) and initializations[0] == initializations[1]
    quality = [item['quality'] for item in metrics]
    budget = [item['training_budget'] for item in metrics]
    budget_equal = all(value is not None for value in budget[0].values()) and budget[0] == budget[1]
    inner_equal = all(item['status'] == 'verified_inner_dev' for item in quality) and all(
        quality[0][name] == quality[1][name] for name in ('split_sha256', 'content_sha256', 'evaluation_options'))
    executed_tokens_equal = (metrics[0]['executed_token_identity'] is not None and all(
        metrics[0][name] == metrics[1][name] for name in ('executed_token_identity', 'executed_relation_token_identity')))
    tokenizers = [item.get('executed_tokenizer_identity') or {} for item in metrics]
    tokenizer_files_equal = (tokenizers[0].get('files_identity_sha256') is not None
                             and tokenizers[0]['files_identity_sha256'] == tokenizers[1].get('files_identity_sha256'))
    diagnostics = [item['diagnostic_updates'] for item in metrics]
    diagnostic_updates_equal = sorted(diagnostics[0]) == sorted(diagnostics[1])
    common_diagnostics = sorted(set(diagnostics[0]) & set(diagnostics[1]), key=int)
    for update in common_diagnostics:
        for key in ('row_positions', 'target_updates', 'window', 'epoch_index'):
            if diagnostics[0][update]['context'][key] != diagnostics[1][update]['context'][key]:
                raise ValueError('Backend diagnostics do not use the same actual optimizer window.')
    external = None
    if configs[0]['phase'] == 'official':
        parents = [config.get('initialization_run') for config in configs]
        if not all(parents):
            raise ValueError('Official backend comparison needs the actual external-stage runs.')
        external = compare_backend_runs(*parents)
    execution_controls = (same_initial and all(item[1] for item in receipts) and budget_equal
                          and all(item['actual_exposure_verified'] for item in metrics)
                          and executed_tokens_equal and tokenizer_files_equal
                          and metrics[0]['actual_dtype'] is not None and metrics[0]['actual_dtype'] == metrics[1]['actual_dtype']
                          and (external is None or external['execution_controls_verified']))
    comparable = execution_controls and inner_equal
    deltas = {}
    if comparable:
        hf, unsloth = (quality[backends.index(name)]['full_views'] for name in ('hf', 'unsloth'))
        if set(hf) != set(unsloth):
            raise ValueError('Backend inner-development views differ.')
        deltas = {view: {metric: unsloth[view][metric] - hf[view][metric]
                         for metric in ('balanced_accuracy', 'nll', 'balanced_nll')
                         if hf[view].get(metric) is not None and unsloth[view].get(metric) is not None}
                  for view in hf}
    report = dict(actual_artifacts_equal=equal, allowed_setting_difference='backend',
                  initialization_status='same_canonical_values_bound' if same_initial else 'not_verified',
                  actual_initialization_imports_verified=[item[1] for item in receipts],
                  execution_controls_verified=bool(execution_controls),
                  actual_training_budget_equal=budget_equal, inner_development_equal=inner_equal,
                  executed_token_identity_equal=executed_tokens_equal,
                  tokenizer_files_equal=tokenizer_files_equal,
                  diagnostic_updates_equal=diagnostic_updates_equal,
                  common_diagnostic_updates=common_diagnostics,
                  unpaired_diagnostic_updates={backend: sorted(set(values)-set(common_diagnostics),key=int)
                                               for backend,values in zip(backends,diagnostics)},
                  external_stage=external,
                  runs=metrics, paired_quality_status='comparable_inner_dev' if comparable else 'missing_matched_execution_evidence',
                  quality_difference_unsloth_minus_hf=deltas,
                  interpretation='Equal tensors and exposure enable a comparison; they do not establish backend quality equivalence or bitwise training trajectories.')
    if output:
        Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backend', choices=tuple(PREFIXES), required=True)
    parser.add_argument('--initial-state', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(derive_backend_plan(args.prepared, args.out, args.backend, args.initial_state)))


if __name__ == '__main__':
    main()
