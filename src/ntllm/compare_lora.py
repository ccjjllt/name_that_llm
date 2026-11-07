"""Compare aligned prediction views/cohorts and fixed log-probability mixtures."""

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.special import softmax


KEYS = ['prompt_id', 'fold', 'true_model', 'n_words']


def infer_grouping_protocol(evaluation, manifest):
    """Resolve explicit protocol metadata or a documented historical fit scope.

    TF-IDF parameters alone cannot identify where the vectorizer was fitted.
    Unknown artifacts therefore stay unknown rather than inheriting defaults.
    """
    declared = {
        'evaluation_options': (evaluation.get('options') or {}).get('grouping_protocol'),
        'cohort_manifest': (manifest.get('cluster') or {}).get('protocol'),
    }
    declared = {source: value for source, value in declared.items() if value is not None}
    if any(value not in ('train_fold', 'query_transductive') for value in declared.values()):
        raise ValueError(f'Unknown grouping protocol declaration: {declared}.')
    if len(set(declared.values())) > 1:
        raise ValueError(f'Evaluation index and cohort manifest grouping protocols conflict: {declared}.')
    scope = (manifest.get('cluster') or {}).get('vocabulary_scope')
    historical = {
        "this cohort's unlabelled query text": 'query_transductive',
        'training-fold text only': 'train_fold',
    }.get(scope)
    if declared:
        protocol = next(iter(declared.values()))
        if historical is not None and historical != protocol:
            raise ValueError('Declared grouping protocol conflicts with the recorded vocabulary fit scope.')
        return dict(protocol=protocol, inferred=False, source='+'.join(declared))
    if historical is not None:
        return dict(protocol=historical, inferred=True, source='legacy_vocabulary_scope')
    return dict(protocol=None, inferred=False, source='unrecorded')


def _prediction_source(path, view='plain', cohort='full'):
    path = Path(path)
    index = path / 'evaluation' / 'evaluation.json'
    source = dict(run=str(path.resolve()), view=view, cohort=cohort)
    evaluation = {}
    if index.exists():
        content = index.read_bytes()
        evaluation = json.loads(content)
        entry = evaluation.get('cohorts', {}).get(cohort, {})
        if not entry.get('available') or view not in entry.get('views', {}):
            raise ValueError(f'{path}: {cohort}/{view} is not in the current evaluation index; '
                             'stale outputs or untrained branches cannot be selected.')
        source.update(evaluation_index=str(index.resolve()),
                      evaluation_index_sha256=hashlib.sha256(content).hexdigest(),
                      evaluation_options=evaluation.get('options', {}),
                      data_governance=evaluation.get('data_governance', dict(status='unavailable')),
                      validation_scope=evaluation.get('validation_scope', dict(kind='unrecorded')),
                      source_intervention=evaluation.get('source_intervention', dict(status='unrecorded')),
                      actual_source_intervention_exposure=evaluation.get('actual_source_intervention_exposure'),
                      condition_signature=evaluation.get('provenance', {}).get('condition_signature'))
    candidate = path / 'evaluation' / cohort / f'{view}_predictions.parquet'
    if candidate.exists():
        manifest_path = candidate.parent / 'manifest.json'
        manifest = {}
        if manifest_path.exists():
            content = manifest_path.read_bytes()
            manifest = json.loads(content)
            source.update(cohort_manifest=str(manifest_path.resolve()),
                          cohort_manifest_sha256=hashlib.sha256(content).hexdigest())
        protocol = infer_grouping_protocol(evaluation, manifest)
        source.update(grouping_protocol=protocol['protocol'],
                      grouping_protocol_inferred=protocol['inferred'],
                      grouping_protocol_source=protocol['source'])
        return candidate, dict(source, layout='evaluation')
    if index.exists():
        raise FileNotFoundError(f'{candidate}: the current evaluation index declares this prediction, '
                                'but its file is missing; refusing a stale root-file fallback.')
    if view == 'plain' and cohort == 'full' and (path / 'valid_predictions.parquet').exists():
        return path / 'valid_predictions.parquet', dict(source, layout='legacy_root',
                                                        grouping_protocol=None,
                                                        grouping_protocol_inferred=False,
                                                        grouping_protocol_source='unrecorded')
    raise FileNotFoundError(f'{path}: prediction view {view!r} for cohort {cohort!r} is unavailable; '
                            'evaluate the requested trained branches/cohort first.')


def prediction_path(path, view='plain', cohort='full'):
    return _prediction_source(path, view, cohort)[0]


def read_prediction(path, view='plain', cohort='full'):
    """Read one selected file once; its hash identifies the bytes being scored."""
    selected, source = _prediction_source(path, view, cohort)
    content = selected.read_bytes()
    rows = pd.read_parquet(BytesIO(content))
    return rows, dict(source, path=str(selected.resolve()),
                      sha256=hashlib.sha256(content).hexdigest(), rows=len(rows))


def combine_predictions(frames):
    """Validate and score the supplied in-memory snapshots without rereading files."""
    rows = pd.concat(frames)
    rows = rows.set_index('id').sort_index()
    if not rows.index.is_unique:
        raise ValueError('Each arm must contain one prediction per validation ID.')
    columns = sorted(column for column in rows if column.startswith('prob__'))
    if not columns:
        raise ValueError('Probability columns are needed to evaluate calibration and mixtures.')
    labels = [column.removeprefix('prob__') for column in columns]
    probability = rows[columns].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError('Probabilities must be finite and nonnegative.')
    if not np.allclose(probability.sum(axis=1), 1, rtol=1e-5, atol=1e-7):
        raise ValueError('Each probability row must sum to one.')
    if not rows.true_model.isin(labels).all():
        raise ValueError('A true label has no probability column.')
    probability /= probability.sum(axis=1, keepdims=True)
    return rows, labels, probability


def read_runs(paths, view='plain', cohort='full', return_sources=False):
    snapshots = [read_prediction(path, view, cohort) for path in paths]
    result = combine_predictions([rows for rows, _ in snapshots])
    return (*result, [source for _, source in snapshots]) if return_sources else result


def metrics(rows, labels, probability):
    y = rows.true_model.to_numpy()
    prediction = np.asarray(labels)[probability.argmax(axis=1)]
    correct = pd.Series(y == prediction)
    label_ids = {label: index for index, label in enumerate(labels)}
    targets = np.asarray([label_ids[value] for value in y])
    nll = -np.log(np.clip(probability[np.arange(len(y)), targets], 1e-30, 1))
    one_hot = np.eye(len(labels))[targets]
    observed = sorted(set(y))
    ba = float(correct.groupby(y).mean().mean())
    return dict(rows=len(rows), prompts=int(rows.prompt_id.nunique()),
                balanced_accuracy=ba, balanced_accuracy_18=ba if len(observed) == 18 else None,
                represented_classes=len(observed), missing_classes=sorted(set(labels) - set(observed)),
                class_counts={label: int(np.sum(y == label)) for label in labels},
                nll=float(nll.mean()), balanced_nll=float(pd.Series(nll).groupby(y).mean().mean()),
                brier=float(np.square(probability - one_hot).sum(axis=1).mean()))


def outcomes(rows, first, second):
    if not len(rows):
        return dict(rows=0)
    y = rows.true_model.to_numpy()
    a, b = first == y, second == y
    ba_a = float(pd.Series(a).groupby(y).mean().mean())
    ba_b = float(pd.Series(b).groupby(y).mean().mean())
    return dict(rows=len(rows), baseline_ba=ba_a, candidate_ba=ba_b, delta=ba_b - ba_a,
                changed=int((first != second).sum()), rescued=int((~a & b).sum()), lost=int((a & ~b).sum()))


def paired_interval(rows, first, second):
    group, groups = pd.factorize(rows.prompt_id)
    target, labels = pd.factorize(rows.true_model)
    count = np.zeros((len(groups), len(labels)), dtype=int)
    difference = np.zeros_like(count)
    np.add.at(count, (group, target), 1)
    np.add.at(difference, (group, target), (second == rows.true_model.to_numpy()).astype(int)
              - (first == rows.true_model.to_numpy()).astype(int))
    rng = np.random.default_rng(42)
    bootstrap = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=2000)
    denominator = bootstrap @ count
    rate = (bootstrap @ difference) / np.where(denominator, denominator, np.nan)
    return np.quantile(np.nanmean(rate, axis=1), [.025, .975]).tolist()


def compare(paths_a, paths_b, fractions, view='plain', cohort='full'):
    rows, labels, first_probability, first_sources = read_runs(paths_a, view, cohort, return_sources=True)
    other, other_labels, second_probability, second_sources = read_runs(paths_b, view, cohort, return_sources=True)
    if labels != other_labels or not rows[KEYS].equals(other[KEYS]):
        raise ValueError('Both arms must use the same IDs, labels, folds and text lengths.')
    if rows.groupby('prompt_id').fold.nunique().max() != 1:
        raise ValueError('A validation prompt spans multiple folds.')
    grouping_equal = None
    if 'inferred_group' in rows and 'inferred_group' in other:
        first_groups = pd.factorize(pd.MultiIndex.from_frame(rows[['fold', 'inferred_group']]))[0]
        second_groups = pd.factorize(pd.MultiIndex.from_frame(other[['fold', 'inferred_group']]))[0]
        grouping_equal = bool(np.array_equal(first_groups, second_groups))
    first = np.asarray(labels)[first_probability.argmax(axis=1)]
    second = np.asarray(labels)[second_probability.argmax(axis=1)]
    family = rows.true_model.str.split('-', n=1).str[0]
    family = family.where(~family.str.startswith('qwen'), 'qwen')
    masks = {f'class/{label}': rows.true_model.eq(label).to_numpy() for label in labels}
    masks.update({f'family/{value}': family.eq(value).to_numpy() for value in sorted(family.unique())})
    lengths = pd.cut(rows.n_words, [-1, 59, 99, 159, np.inf], labels=['<60', '60-99', '100-159', '>=160'])
    masks.update({f'words/{value}': lengths.eq(value).to_numpy() for value in lengths.cat.categories})
    if 'group_size' in rows:
        for name, lower, upper in [('singleton', 1, 1), ('pair', 2, 2), ('triple', 3, 3),
                                    ('4-6', 4, 6), ('7-18', 7, 18)]:
            masks[f'group/{name}'] = rows.group_size.between(lower, upper).to_numpy()
    fold_metrics = {}
    for fold in sorted(rows.fold.unique()):
        mask = rows.fold.eq(fold).to_numpy()
        fold_metrics[str(fold)] = outcomes(rows.loc[mask], first[mask], second[mask])
    report = dict(scope='Paired named-view OOF probabilities. The fixed diagnostic q8 scale is not a complete-system replay.',
                  view=view, cohort=cohort,
                  grouping_equal=grouping_equal, slice_group_source='baseline inferred groups',
                  prediction_sources=dict(baseline=first_sources, candidate=second_sources),
                  labels=labels, baseline=metrics(rows, labels, first_probability),
                  candidate=metrics(rows, labels, second_probability),
                  paired=outcomes(rows, first, second),
                  descriptive_prompt_bootstrap_95pct=paired_interval(rows, first, second),
                  slices={name: outcomes(rows.loc[mask], first[mask], second[mask]) for name, mask in masks.items()},
                  folds=fold_metrics,
                  baseline_runs=[str(path.resolve()) for path in paths_a],
                  candidate_runs=[str(path.resolve()) for path in paths_b], mixtures=[])
    from .q8_evaluation import shared_intervention_scope, intervention_label_groups, score_intervention_groups
    source_intervention = shared_intervention_scope([source.get('source_intervention')
                                                     for source in first_sources + second_sources])
    report['source_intervention'] = source_intervention or dict(status='unrecorded_or_incomplete')
    if source_intervention:
        arm_roles = {name: sorted({(source.get('source_intervention') or {}).get('arm_variant', 'unrecorded')
                                   for source in sources})
                     for name, sources in (('baseline', first_sources), ('candidate', second_sources))}
        report['source_intervention_comparison'] = dict(
            arm_variants=arm_roles,
            declared_pair_order=arm_roles == dict(baseline=['baseline'], candidate=['mix20']),
            interpretation='Data scope identifies the intended replacement. Matching initialization, training recipe and completed exposure require separate execution evidence; subgroup scores do not establish protocol reliability or per-replacement causal yield.')
        evaluation_rows = rows.rename(columns={'true_model': 'model'})
        before = score_intervention_groups(evaluation_rows, labels, first_probability, source_intervention)
        after = score_intervention_groups(evaluation_rows, labels, second_probability, source_intervention)
        grouped = {}
        for group, members in intervention_label_groups(source_intervention).items():
            mask = rows.true_model.isin(members).to_numpy()
            grouped[group] = dict(baseline=before[group], candidate=after[group],
                paired=outcomes(rows.loc[mask], first[mask], second[mask]),
                descriptive_prompt_bootstrap_95pct=(paired_interval(rows.loc[mask], first[mask], second[mask])
                                                   if mask.any() else None),
                uncertainty_scope='Prompt-paired descriptive interval conditional on fitted models; not training-seed uncertainty or a protocol-only causal estimate.')
        report['by_source_intervention_label_coverage'] = grouped
    for fraction in fractions:
        if not 0 <= fraction <= 1:
            raise ValueError('Mixture fractions must be in [0, 1].')
        scores = (1 - fraction) * np.log(np.clip(first_probability, 1e-30, 1))
        scores += fraction * np.log(np.clip(second_probability, 1e-30, 1))
        probability = softmax(scores, axis=1)
        prediction = np.asarray(labels)[probability.argmax(axis=1)]
        report['mixtures'].append(dict(candidate_fraction=fraction, **metrics(rows, labels, probability),
                                      paired=outcomes(rows, first, prediction)))
    changed = rows[KEYS].copy()
    changed['baseline'], changed['candidate'] = first, second
    return report, changed.loc[first != second].reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', nargs='+', type=Path, required=True)
    parser.add_argument('--candidate', nargs='+', type=Path, required=True)
    parser.add_argument('--mix', nargs='+', type=float, default=[0, 0.25, 1])
    parser.add_argument('--view', default='plain', help='Named evaluated branch view, e.g. plain_centered')
    parser.add_argument('--cohort', default='full', help='Exact evaluated cohort name from evaluation/evaluation.json')
    parser.add_argument('--output-dir', type=Path, default=Path('runs'))
    args = parser.parse_args()
    report, changed = compare(args.baseline, args.candidate, args.mix, args.view, args.cohort)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output_dir / f'{stamp}_probability_comparison_{uuid4().hex[:6]}'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    changed.to_parquet(output / 'changed_predictions.parquet', index=False)
    print(json.dumps(report['paired']), flush=True)
    print(output.resolve(), flush=True)


if __name__ == '__main__':
    main()
