"""Describe the measured scope of a fixed-slot text replacement experiment.

This sidecar annotates immutable inputs. It does not select replacements, rename
legacy arms, change training, or treat evidence availability as text quality.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from .q8_data import file_sha


POLICY = 'fixed_slot_local_text_replacement_scope_v1'
DATA_POLICY = 'q8_protocol_fixed_relation_v1'
EVIDENCE_COLUMNS = ('research_condition', 'model_identity_status', 'protocol_evidence_level',
                    'generation_status', 'transformation_status')


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _bound_file(path, expected, label):
    if not Path(path).is_file() or file_sha(Path(path)) != expected:
        raise ValueError(f'{label} differs from its source-scope binding: {path}')


def _counts(frame, columns=EVIDENCE_COLUMNS):
    return {name: frame[name].fillna('unknown').value_counts().to_dict()
            for name in columns if name in frame}


def _transitions(left, right, columns):
    records = pd.DataFrame({f'before_{name}': left[name].to_numpy() for name in columns})
    for name in columns:
        records[f'after_{name}'] = right[name].to_numpy()
    names = list(records.columns)
    return records.groupby(names, dropna=False).size().rename('rows').reset_index().to_dict('records')


def build_source_scope(data_dir, *, exposure=None, out=None):
    """Bind actual differences and declared anchor exposure without editing data.

    ``exposure`` is the real anchor slot table used by q8_protocol_controls.
    Existing pairs read that table from their parent directory. It is a plan,
    not evidence that a model has executed those successful updates.
    """
    import hashlib
    data_dir = Path(data_dir).resolve()
    out = Path(out).resolve() if out is not None else data_dir / 'intervention_scope'
    if out.exists():
        raise FileExistsError(f'Use a new source-scope output directory: {out}')
    data_manifest_path = data_dir / 'manifest.json'
    data_manifest_sha = file_sha(data_manifest_path)
    manifest = json.loads(data_manifest_path.read_text())
    if manifest.get('policy') != DATA_POLICY:
        raise ValueError('Source scope requires an existing fixed-relation condition pair.')
    names = {'baseline': 'external_baseline.parquet', 'mix20': 'external_mix20.parquet',
             'anchor': 'external_anchor.parquet'}
    data_hashes = {key: file_sha(data_dir / name) for key, name in names.items()}
    for key, name in names.items():
        expected = manifest.get('files_sha256', {}).get(name)
        if expected is None or data_hashes[key] != expected:
            raise ValueError(f'The source manifest does not bind unchanged {name}.')
    left, right, anchor = (pd.read_parquet(data_dir / names[key]) for key in ('baseline', 'mix20', 'anchor'))
    n = len(anchor)
    if not n or len(left) != n or len(right) != n:
        raise ValueError('The two arms and anchor must retain the same nonempty slots.')
    for frame in (left, right):
        if not frame.training_slot_id.equals(anchor.training_slot_id) or not frame.id.is_unique:
            raise ValueError('Training slot order or unique row identities changed.')
        if not frame[['model', 'word_bin']].equals(anchor[['model', 'word_bin']]):
            raise ValueError('Class or word-bin matching changed.')
    if not anchor.training_slot_id.equals(anchor.id):
        raise ValueError('Anchor slots must identify the original anchor rows.')
    changed = left.text.ne(right.text).to_numpy()
    positions = np.flatnonzero(changed)
    before, after = left.iloc[positions].copy(), right.iloc[positions].copy()
    labels = list(manifest['labels'])
    direct = sorted(before.model.unique().tolist())
    other = sorted(set(labels) - set(direct))
    source_exposure_path = data_dir.parent / 'anchor_actual_slots.parquet'
    source_exposure_sha = file_sha(source_exposure_path) if source_exposure_path.is_file() else None
    if exposure is None:
        exposure = pd.read_parquet(source_exposure_path)
    else:
        exposure = exposure.copy()
        if source_exposure_sha and not exposure.equals(pd.read_parquet(source_exposure_path)):
            raise ValueError('Supplied anchor exposures differ from the saved source plan.')
    required = {'position', 'epoch', 'update', 'microbatch', 'centered', 'training_slot_id'}
    if not required <= set(exposure):
        raise ValueError('Anchor exposures lack position, order, role or slot identity.')
    if not pd.api.types.is_bool_dtype(exposure.centered):
        raise ValueError('Anchor relation roles must be Boolean.')
    index = exposure.position.to_numpy(dtype=np.int64)
    if not len(index) or (index < 0).any() or (index >= n).any():
        raise ValueError('Anchor exposure positions are outside the paired data.')
    if not np.array_equal(exposure.training_slot_id.to_numpy(), anchor.id.to_numpy()[index]):
        raise ValueError('Anchor exposure slot identities differ from paired inputs.')
    planned_updates = len(exposure[['epoch', 'update']].drop_duplicates())
    if planned_updates != manifest['anchor_plan']['max_updates']:
        raise ValueError('Exposure table does not cover the declared update budget.')
    if exposure.epoch.min() < 0 or exposure.epoch.max() >= manifest['anchor_plan']['epochs']:
        raise ValueError('Exposure epoch is outside the declared plan.')
    relation_positions = np.unique(index[exposure.centered.to_numpy()])
    separate = manifest['anchor_plan']['mode'] == 'anchored_separate'
    relation_columns = ['id', 'text', 'model', 'prompt_id']
    relation_inputs_unchanged = separate or all(
        frame.iloc[relation_positions][relation_columns].equals(anchor.iloc[relation_positions][relation_columns])
        for frame in (left, right))
    if not relation_inputs_unchanged:
        raise ValueError('A shared relation input changed between the paired arms.')
    changed_exposure = changed[index]
    relation_changed = int((changed_exposure & exposure.centered.to_numpy()).sum())
    if not separate and relation_changed:
        raise ValueError('Shared-forward replacements must remain in plain-only slots.')
    changed_rows = pd.DataFrame(dict(row_position=positions,
        training_slot_id=anchor.id.to_numpy()[positions], model=before.model.to_numpy(),
        word_bin=before.word_bin.to_numpy()))
    for prefix, frame in (('before', before), ('after', after)):
        for column in ('id', 'prompt_id', 'source_dataset', 'n_words', *EVIDENCE_COLUMNS):
            if column in frame:
                changed_rows[f'{prefix}_{column}'] = frame[column].to_numpy()
        changed_rows[f'{prefix}_text_sha256'] = frame.text.map(lambda text: hashlib.sha256(text.encode()).hexdigest()).to_numpy()
    exposure_counts = np.bincount(index, minlength=n)
    changed_rows['planned_exposures_per_arm'] = exposure_counts[positions]
    by_class = []
    for label in labels:
        pool_mask = left.model.eq(label).to_numpy()
        part = changed & pool_mask
        count = int(part.sum())
        by_class.append(dict(model=label, rows_per_arm=int(pool_mask.sum()), changed_text_rows=count,
            changed_fraction=count / int(pool_mask.sum()) if pool_mask.any() else None,
            directly_replaced=bool(count), planned_changed_exposures_per_arm=int(exposure_counts[part].sum()),
            before_evidence=_counts(left.loc[part]), after_evidence=_counts(right.loc[part])))
    by_length = []
    for (label, word_bin), group in left.groupby(['model', 'word_bin'], observed=True, sort=True):
        idx = group.index.to_numpy()
        count = int(changed[idx].sum())
        by_length.append(dict(model=label, word_bin=str(word_bin), rows_per_arm=len(idx),
                              changed_text_rows=count, changed_fraction=count / len(idx)))
    unknown_before = int(before.research_condition.eq('protocol_unknown').sum())
    summary = dict(policy=POLICY,
        display_name=f'固定关系的{len(direct)}类局部文本替换' if direct else '固定关系的未发生正文替换对照',
        legacy_comparison_name=manifest['name'], arm_roles=dict(baseline=manifest['left'], mix20=manifest['right']),
        rows_per_arm=n, changed_text_rows=len(positions), changed_fraction=len(positions) / n,
        unchanged_text_rows=n-len(positions), declared_replacement_slots=manifest['replacement_slots'],
        directly_replaced_labels=direct, not_directly_replaced_labels=other,
        changed_source_label_rows=int(before.source_dataset.ne(after.source_dataset).sum()),
        before_unknown_protocol_rows=unknown_before,
        before_unknown_protocol_fraction=unknown_before / len(positions) if len(positions) else None,
        before_evidence=_counts(before), after_evidence=_counts(after),
        evidence_transitions={name:_transitions(before, after, [name]) for name in EVIDENCE_COLUMNS
                              if name in before and name in after},
        per_class=by_class, per_class_word_bin=by_length,
        source_transitions=_transitions(before, after, ['source_dataset']),
        planned_exposure=dict(scope='Declared anchor update plan, not completed training or a successful-update prefix.',
            updates_per_arm=planned_updates, epochs=manifest['anchor_plan']['epochs'], row_exposures_per_arm=len(exposure),
            changed_text_exposures_per_arm=int(changed_exposure.sum()),
            changed_text_exposure_fraction=float(changed_exposure.mean()),
            changed_rows_exposure_counts=changed_rows.planned_exposures_per_arm.value_counts().sort_index().to_dict(),
            relation_exposures_per_arm=int(exposure.centered.sum()),
            relation_inputs_and_slot_partners_unchanged=bool(relation_inputs_unchanged),
            changed_rows_in_relation_slots=relation_changed,
            all_changed_rows_plain_only=relation_changed == 0,
            relation_mode=manifest['anchor_plan']['mode']),
        controls=dict(slot_order_unchanged=True, class_and_word_bin_unchanged=True,
                      exact_word_counts_unchanged=bool(left.n_words.equals(right.n_words)),
                      scope='Declared input and slot controls; not proof of equal actual initialization, successful updates or model execution.'),
        estimand='Joint effect of this finite set of text/source replacements in the fixed relation, class and length-bin slots, conditional on the declared training recipe and public-data history.',
        supported_claims=['Overall 18-class model effect of this local intervention.',
                          'Per-class effects, separated into directly replaced labels and other labels; both may change through shared parameters.'],
        unsupported_claims=['Reliable protocol is better or worse than unknown protocol.',
                            'All 18 labels received equivalent source or protocol interventions.',
                            'A general effect of any source, generation protocol, provider, reasoning setting or text transformation.',
                            'The other labels are independent untreated controls.'],
        interpretation='Historical anchor is a sampling reference, not a verified-protocol condition. protocol_unknown is a composite evidence gap, not a proven quality defect. A shared classifier can change predictions for labels with no directly replaced training text.')
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{out.name}.tmp-', dir=out.parent))
    try:
        changed_rows.to_parquet(temporary / 'changed_rows.parquet', index=False)
        exposure.to_parquet(temporary / 'planned_slots.parquet', index=False)
        _write(temporary / 'summary.json', summary)
        binding = dict(policy=POLICY, data_manifest_sha256=data_manifest_sha,
            data_sha256=data_hashes, data_files=names,
            source_exposure_path=str(source_exposure_path) if source_exposure_sha else None,
            source_exposure_sha256=source_exposure_sha,
            files_sha256={name:file_sha(temporary/name) for name in ('summary.json','changed_rows.parquet','planned_slots.parquet')})
        _write(temporary / 'manifest.json', binding)
        _bound_file(data_manifest_path, data_manifest_sha, 'Data manifest during build')
        for key, name in names.items():
            _bound_file(data_dir/name, data_hashes[key], 'Paired data during build')
        if source_exposure_sha:
            _bound_file(source_exposure_path, source_exposure_sha, 'Anchor exposure during build')
        os.replace(temporary, out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_source_scope(data_dir, directory=out)


def load_source_scope(data_dir, *, directory=None):
    """Return an immutable measured scope, or an explicit unavailable status."""
    data_dir = Path(data_dir).resolve()
    data_manifest_path = data_dir / 'manifest.json'
    if not data_manifest_path.is_file() or json.loads(data_manifest_path.read_text()).get('policy') != DATA_POLICY:
        return dict(status='not_applicable', reason='No declared fixed-relation source-pair dataset.', data_dir=str(data_dir))
    directory = Path(directory).resolve() if directory is not None else data_dir / 'intervention_scope'
    path = directory / 'manifest.json'
    if not directory.exists():
        return dict(status='unavailable', reason='The source pair has no measured intervention-scope sidecar.', data_dir=str(data_dir))
    if not path.is_file():
        raise ValueError('Source-scope directory has no completed manifest.')
    binding = json.loads(path.read_text())
    if binding.get('policy') != POLICY:
        raise ValueError('Unsupported source-scope policy.')
    _bound_file(data_manifest_path, binding['data_manifest_sha256'], 'Data manifest')
    for key, name in binding['data_files'].items():
        _bound_file(data_dir / name, binding['data_sha256'][key], 'Paired data')
    for name, digest in binding['files_sha256'].items():
        _bound_file(directory / name, digest, 'Source-scope artifact')
    summary = json.loads((directory / 'summary.json').read_text())
    return dict(status='available', policy=POLICY, display_name=summary['display_name'],
        manifest_path=str(path), manifest_sha256=file_sha(path),
        summary_path=str(directory/'summary.json'), summary_sha256=binding['files_sha256']['summary.json'],
        changed_rows_path=str(directory/'changed_rows.parquet'), changed_rows_sha256=binding['files_sha256']['changed_rows.parquet'],
        data_manifest_sha256=binding['data_manifest_sha256'], data_sha256=binding['data_sha256'], summary=summary)


def source_exposure_summary(scope, exposed_positions, *, changed_rows=None):
    """Summarize actual external-stage positions, including repeated exposures.

    The caller supplies the cumulative completed external-stage cursor. Official
    adaptation positions refer to another table and must never be passed here.
    """
    if scope.get('status') != 'available':
        return dict(status=scope.get('status', 'unavailable'), reason='No bound source replacement scope.')
    if changed_rows is None:
        _bound_file(scope['changed_rows_path'], scope['changed_rows_sha256'], 'Changed rows')
        changed_rows = pd.read_parquet(scope['changed_rows_path'])
    positions = np.asarray(exposed_positions, dtype=np.int64)
    n = scope['summary']['rows_per_arm']
    if positions.ndim != 1 or ((positions < 0) | (positions >= n)).any():
        raise ValueError('Actual external exposure positions are outside the source pool.')
    if not changed_rows.row_position.is_unique:
        raise ValueError('Changed rows must identify unique source slots.')
    frequencies = np.bincount(positions, minlength=n)
    measured = changed_rows[['row_position', 'model']].copy()
    measured['exposures'] = frequencies[measured.row_position.to_numpy(dtype=np.int64)]
    count = int(measured.exposures.sum())
    per_label = {str(label): int(value) for label, value in measured.groupby('model').exposures.sum().items()}
    return dict(status='measured_external_exposure',
        scope='Cumulative completed external-stage row positions supplied by the training cursor; not the full planned budget or official-stage positions.',
        total_row_exposures=len(positions), changed_text_exposures=count,
        changed_exposure_fraction=count / len(positions) if len(positions) else None,
        unique_changed_slots=int(measured.exposures.gt(0).sum()),
        actually_exposed_replacement_labels=sorted(label for label, value in per_label.items() if value),
        changed_exposures_by_label=per_label,
        source_scope_manifest_sha256=scope['manifest_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    report = build_source_scope(args.data_dir, out=args.out)
    print(json.dumps({key:report[key] for key in ('status','display_name','manifest_path')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
