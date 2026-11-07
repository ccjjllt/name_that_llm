"""Immutable evidence sidecars for existing external data, never admission rules.

Exact text/label identity binds evidence. Model aliases, version-shaped strings,
source-author claims and independently verified versions remain distinct.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import pandas as pd
import pyarrow.parquet as pq

from .q8_data import file_sha


POLICY = 'existing_data_governance_v1'
STATUSES = ('model_identity_status', 'research_condition', 'generation_status',
            'protocol_evidence_level', 'transformation_status', 'prompt_identity_status')
DIMENSIONS = ('generation_prompt_design', 'length_targets', 'output_selection', 'identity_pilot')


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _count(mask):
    return dict(rows=int(mask.sum()), fraction=float(mask.mean()) if len(mask) else None)


def _history_status(registry, sources, dimension, exact_pools):
    statuses = set()
    for source in sources:
        context = registry.get('sources', {}).get(source, {})
        binding = context.get('row_context_binding', {})
        required = set(binding.get('required_any_evidence_pool', []))
        if required and not required.intersection(exact_pools):
            statuses.add(binding.get('unmatched_status', 'scope_unresolved'))
        else:
            statuses.add(context.get(dimension, {}).get('status', 'unknown'))
    return '|'.join(sorted(statuses)) or 'unknown'


def _bound_file(path, expected, title):
    if not Path(path).is_file() or file_sha(Path(path)) != expected:
        raise ValueError(f'{title} differs from its governance binding: {path}')


def build_governance(data_dir, source_registry, out=None):
    """Build once beside original files; preserve every row and original status."""
    data_dir, registry_path = Path(data_dir).resolve(), Path(source_registry).resolve()
    out = Path(out).resolve() if out is not None else data_dir / 'governance'
    if out.exists():
        raise FileExistsError(f'Governance output already exists: {out}')
    data_manifest_path = data_dir / 'manifest.json'
    data_manifest_sha256 = file_sha(data_manifest_path)
    data_manifest = json.loads(data_manifest_path.read_text())
    ledger = Path(data_manifest['evidence_ledger'])
    ledger = ledger if ledger.is_absolute() else data_dir / ledger
    ledger = ledger.resolve()
    _bound_file(ledger, data_manifest['evidence_ledger_sha256'], 'Evidence ledger')
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    if registry.get('policy') != 'source_history_registry_v1':
        raise ValueError('Expected the explicit source_history_registry_v1 dimensions.')
    verified_registry_evidence = {}
    for name, reference in registry.get('evidence', {}).items():
        _bound_file(reference['path'], reference['sha256'], f'Source history evidence {name}')
        verified_registry_evidence[name] = reference['sha256']
    frames, needed = {}, {}
    source_hashes = {}
    columns = ['id', 'model', 'text', 'evidence_record_ids', 'variant_evidence_record_ids',
               'source_dataset', 'source_pool', *STATUSES]
    for path in sorted(data_dir.glob('external_*.parquet')):
        variant = path.stem.removeprefix('external_')
        source_hashes[variant] = file_sha(path)
        available = set(pq.read_schema(path).names)
        if not {'id', 'model', 'text', 'evidence_record_ids', *STATUSES} <= available:
            raise ValueError(f'{path} lacks the current evidence-bearing data schema.')
        frame = pd.read_parquet(path, columns=[name for name in columns if name in available])
        frame['exact_text_sha256'] = frame.pop('text').map(_text_sha)
        frame.insert(0, 'row_position', range(len(frame)))
        for row in frame.itertuples(index=False):
            key = (row.id, row.model, row.exact_text_sha256)
            for record in json.loads(row.evidence_record_ids):
                if record in needed and needed[record] != key:
                    raise ValueError('One evidence record was linked to different exact texts or labels.')
                needed[record] = key
        frames[variant] = frame
    if not frames:
        raise ValueError('No external data files were found.')
    # Read the ledger once in bounded batches. Retain small identifier tuples,
    # never raw response bodies or request-envelope JSON in the sidecar builder.
    evidence = {}
    cols = ['evidence_record_id', 'id', 'model', 'exact_text_sha256', 'source_dataset', 'source_pool', 'model_evidence_json']
    for batch in pq.ParquetFile(ledger).iter_batches(batch_size=8192, columns=cols):
        for row in batch.to_pylist():
            record = row['evidence_record_id']
            if record not in needed:
                continue
            key = (row['id'], row['model'], row['exact_text_sha256'])
            if key != needed[record] or record in evidence:
                raise ValueError('Evidence is duplicated or fails exact text/label/ID association.')
            model = json.loads(row['model_evidence_json'])
            evidence[record] = (model.get('requested_model'), model.get('returned_model'),
                                model.get('model_version'), row['source_dataset'], row['source_pool'])
    if set(evidence) != set(needed):
        raise ValueError('Some declared exact evidence records are missing from the ledger.')
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{out.name}.tmp-', dir=out.parent))
    try:
        summaries, datasets = {}, {}
        for variant, frame in frames.items():
            derived = []
            for row in frame.itertuples(index=False):
                records = json.loads(row.evidence_record_ids)
                values = [evidence[record] for record in records]
                requested, returned, versions = [sorted({item[i] for item in values if item[i]}) for i in range(3)]
                alias = [v for v in versions if v.rsplit('/', 1)[-1].removesuffix('@default') == row.model]
                nonalias = sorted(set(versions) - set(alias))
                dated = [v for v in returned if re.search(r'-\d{4}(?:-\d{2}-\d{2})?(?:@[^/]+)?$', v)]
                sources = sorted({item[3] for item in values if item[3]})
                exact_pools = {item[4] for item in values}
                selected_source = str(getattr(row, 'source_dataset', '') or '')
                if selected_source and selected_source not in sources:
                    sources.append(selected_source)
                pool = str(getattr(row, 'source_pool', '') or '')
                container = registry.get('containers', {}).get(pool, {}).get('output_selection', {})
                derived.append(dict(
                    requested_model_identifiers=_json(requested), returned_model_identifiers=_json(returned),
                    model_version_field_values=_json(versions), model_version_alias_values=_json(alias),
                    model_version_nonalias_values=_json(nonalias), date_style_returned_identifiers=_json(dated),
                    requested_identifier_recorded=bool(requested), returned_identifier_recorded=bool(returned),
                    version_field_recorded=bool(versions), version_field_alias_recorded=bool(alias),
                    version_field_nonalias_recorded=bool(nonalias), date_style_identifier_recorded=bool(dated),
                    version_verification_status='not_independently_confirmed',
                    source_history_refs=_json(sorted(sources)), selected_container=pool,
                    source_history_assignment_scope='Source-context evidence from exact export links; not per-row generation verification.',
                    container_selection_status=container.get('status', 'unknown'),
                    **{f'{dimension}_history_status': _history_status(registry, sources, dimension, exact_pools)
                       for dimension in DIMENSIONS}))
            rows = pd.concat([frame.reset_index(drop=True), pd.DataFrame(derived)], axis=1)
            filename = f'external_{variant}.governance.parquet'
            rows.to_parquet(temporary / filename, index=False)
            counts = {name: rows[name].value_counts(dropna=False).to_dict() for name in STATUSES}
            coverage = {name: _count(rows[name]) for name in (
                'requested_identifier_recorded', 'returned_identifier_recorded', 'version_field_recorded',
                'version_field_alias_recorded', 'version_field_nonalias_recorded', 'date_style_identifier_recorded')}
            coverage.update(label_only=_count(rows.model_identity_status.eq('label_only')),
                            protocol_condition_unknown=_count(rows.research_condition.eq('protocol_unknown')),
                            generation_status_unknown=_count(rows.generation_status.eq('unknown')))
            history = {}
            for dimension, column in [(d, f'{d}_history_status') for d in DIMENSIONS] + [('container_output_selection', 'container_selection_status')]:
                patterns = rows[column].value_counts().to_dict()
                history[dimension] = dict(status_patterns=patterns,
                    documented_official_dependency_in_source_context=_count(rows[column].str.split('|').map(lambda v: 'documented_official_dependency' in v)),
                    unknown_present=_count(rows[column].str.split('|').map(lambda v: 'unknown' in v)))
            summary = dict(variant=variant, rows=len(rows), original_rows_preserved=True,
                coverage_scope='Full bound external file; not the successful-update exposure prefix.',
                status_counts=counts, evidence_coverage=coverage, source_history=history,
                unknown_protocol_generation_breakdown=rows.loc[rows.research_condition.eq('protocol_unknown'), 'generation_status'].value_counts().to_dict(),
                validation_scope=dict(independent_confirmation=False,
                    statement='Official prompt holdout does not remove historical official-informed generation or upstream selection. These are development results conditional on fixed public external data.',
                    unknown_is_not_mislabeled=True, model_identifiers_do_not_confirm_immutable_versions=True,
                    source_history_scope='Source-context descriptions are not per-row generation verification. Registry evidence basis and artifact scope remain explicit; selected-container selection is separate from original generation history.'))
            summaries[variant] = summary
            datasets[variant] = dict(data_file=f'external_{variant}.parquet', data_sha256=source_hashes[variant],
                                     rows_file=filename, rows_sha256=file_sha(temporary / filename), rows=len(rows))
        _write(temporary / 'summary.json', dict(policy=POLICY, datasets=summaries))
        (temporary / 'source_history_registry.json').write_bytes(registry_bytes)
        manifest = dict(policy=POLICY, data_manifest_sha256=data_manifest_sha256,
            evidence_ledger_path=str(ledger), evidence_ledger_sha256=data_manifest['evidence_ledger_sha256'],
            source_registry_origin=str(registry_path), source_registry_file='source_history_registry.json',
            source_registry_sha256=file_sha(temporary / 'source_history_registry.json'),
            registry_evidence_verified_at_build=verified_registry_evidence,
            summary_file='summary.json', summary_sha256=file_sha(temporary / 'summary.json'), datasets=datasets,
            row_identity='Original row order, ID, label and exact UTF-8 text SHA256; only declared exact evidence records, never normalized-text variants.',
            interpretation='Evidence governance only: no admission, exclusion, relabeling, training or score improvement.')
        _write(temporary / 'manifest.json', manifest)
        for variant, expected in source_hashes.items():
            _bound_file(data_dir / f'external_{variant}.parquet', expected, 'Source data during build')
        _bound_file(data_manifest_path, data_manifest_sha256, 'Data manifest during build')
        _bound_file(ledger, data_manifest['evidence_ledger_sha256'], 'Evidence ledger during build')
        os.replace(temporary, out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return dict(status='built', directory=str(out), datasets=list(datasets), policy=POLICY)


def load_governance(data_dir, variant=None, *, directory=None):
    """Read bound summaries without reading external rows or full sidecar tables."""
    data_dir = Path(data_dir).resolve()
    directory = Path(directory).resolve() if directory is not None else data_dir / 'governance'
    manifest_path = directory / 'manifest.json'
    if not directory.exists():
        return dict(status='unavailable', reason='No governance sidecar exists for this data directory.', data_dir=str(data_dir))
    if not manifest_path.is_file():
        raise ValueError('Governance directory exists without a completed manifest.')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('policy') != POLICY:
        raise ValueError('Unsupported data governance policy.')
    _bound_file(data_dir / 'manifest.json', manifest['data_manifest_sha256'], 'Data manifest')
    current = json.loads((data_dir / 'manifest.json').read_text())
    ledger = Path(current['evidence_ledger'])
    ledger = ledger if ledger.is_absolute() else data_dir / ledger
    if current['evidence_ledger_sha256'] != manifest['evidence_ledger_sha256']:
        raise ValueError('Governance and data disagree on the evidence ledger.')
    _bound_file(ledger, manifest['evidence_ledger_sha256'], 'Evidence ledger')
    summary_path = directory / manifest['summary_file']
    _bound_file(summary_path, manifest['summary_sha256'], 'Governance summary')
    _bound_file(directory / manifest['source_registry_file'], manifest['source_registry_sha256'], 'Source history registry')
    summaries = json.loads(summary_path.read_text())['datasets']
    selected = (str(variant).removeprefix('external_').removesuffix('.parquet') if variant is not None else None)
    if selected is not None and selected not in manifest['datasets']:
        raise ValueError(f'No governance binding for variant {selected!r}.')
    names = [selected] if selected is not None else list(manifest['datasets'])
    for name in names:
        binding = manifest['datasets'][name]
        _bound_file(data_dir / binding['data_file'], binding['data_sha256'], 'External data')
        _bound_file(directory / binding['rows_file'], binding['rows_sha256'], 'Governance rows')
        if summaries[name]['rows'] != binding['rows']:
            raise ValueError('Governance summary and row binding disagree.')
    result = dict(status='available', policy=POLICY, variant=selected,
        manifest_path=str(manifest_path), manifest_sha256=file_sha(manifest_path),
        summary_path=str(summary_path), summary_sha256=manifest['summary_sha256'],
        evidence_ledger_sha256=manifest['evidence_ledger_sha256'],
        source_registry_sha256=manifest['source_registry_sha256'])
    if selected is None:
        result['datasets'] = summaries
    else:
        binding = manifest['datasets'][selected]
        result.update(data_path=str(data_dir / binding['data_file']), data_sha256=binding['data_sha256'],
                      rows_path=str(directory / binding['rows_file']), rows_sha256=binding['rows_sha256'],
                      summary=summaries[selected])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--source-registry', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    print(json.dumps(build_governance(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == '__main__':
    main()
