"""Audit frozen peer-context GPU predictions against two unchanged Sibling Odds traces."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .sibling_odds_replay import replay_predictions

ARMS = ('single', 'unrelated', 'retrieved')
METADATA = ['model', 'prompt_id', 'n_words', 'partition', 'cohort', 'fold', 'source_dataset']


def read_ids(path):
    frame = pd.read_parquet(path)
    if 'id' in frame:
        frame = frame.set_index('id')
    if frame.index.name != 'id' or not frame.index.is_unique or frame.index.hasnans:
        raise ValueError(f'Missing or duplicate id in {path}')
    return frame


def same_ids(actual, expected, name, ordered=False):
    if (not actual.is_unique or set(actual) != set(expected)
            or (ordered and not actual.equals(expected))):
        raise ValueError(f'{name}: ids differ from frozen inputs ({len(actual)} vs {len(expected)})')


def same_meta(actual, expected, columns, name):
    actual = actual.loc[expected.index]
    for column in columns:
        a, b = actual[column], expected[column]
        equal = a.eq(b).fillna(False) | (a.isna() & b.isna())
        if not equal.all():
            raise ValueError(f'{name}: {column} differs for {equal.index[~equal][0]}')


def read_panel(path):
    ids = read_ids(path / 'visible_ids.parquet').index
    truth, query = read_ids(path / 'truth.parquet'), read_ids(path / 'queries.parquet')
    info = json.loads((path / 'panel.json').read_text())
    digest = hashlib.sha256(('\n'.join(ids) + '\n').encode()).hexdigest()
    if len(ids) != 191 or digest != info['ordered_ids_sha256'] or len(ids) != info['visible_rows']:
        raise ValueError('The independent experiment requires the frozen 191-row GEN003 layout')
    for table in (truth, query):
        same_ids(table.index, ids, 'Independent panel', ordered=True)
    if not query.text.map(lambda x: hashlib.sha256(x.encode()).hexdigest()).eq(truth.body_sha256).all():
        raise ValueError('GEN003 archived bodies changed')
    return truth.join(query['text'])


def check_manifest(manifest, public, independent):
    for stage, expected in [('trace', public), ('blind', independent)]:
        part = manifest[manifest.partition.eq(stage)]
        same_ids(part.index, expected.index, stage, ordered=stage == 'blind')
        same_meta(part, expected, ['model', 'prompt_id', 'n_words'], stage)
        if stage == 'trace':
            same_meta(part, expected, ['cohort'], stage)
        elif part.cohort.nunique(dropna=False) != 1:
            raise ValueError('The independent queries must form one sparse batch')
    off = manifest[manifest.partition.eq('official')]
    if set(off.fold) != set(range(5)) or not off.groupby('prompt_id').fold.nunique().eq(1).all():
        raise ValueError('Official prompts must have one of five distinct held-out fold assignments')
    if not manifest.peer_id.isna().equals(manifest.sham_id.isna()):
        raise ValueError('The two context arms have different query coverage')
    for name in ('peer_id', 'sham_id'):
        targets = manifest[manifest[name].notna()]
        refs = manifest.loc[targets[name]].copy()
        if np.any(refs.index.to_numpy() == targets.index.to_numpy()):
            raise ValueError(f'Self-reference in {name}')
        refs.index = targets.index
        same_meta(refs, targets, ['partition', 'cohort', 'fold', 'source_dataset'], name)


def read_trace(directory, truth):
    with np.load(directory / 'stage_scores.npz', allow_pickle=False) as data:
        trace = dict(data)
    ids = pd.Index(trace['ids'], name='id')
    same_ids(ids, truth.index, 'Trace', ordered=True)
    labels = pd.Index(trace['labels'])
    if len(labels) != 18 or not labels.is_unique or not truth.model.isin(labels).all():
        raise ValueError('Invalid candidate classes or ground truth')
    for key in ('stack_logp', 'glm_logp', 'sibling_logp'):
        x = trace[key]
        if x.shape != (len(ids), len(labels)) or not np.isfinite(x).all():
            raise ValueError(f'Incomplete trace scores in {key}')
    old = replay_predictions(trace, truth.text, sibling_odds=False)
    if not np.array_equal(old, trace['final']):
        raise ValueError('Dennis decisions disagree with the original trace')
    return trace, replay_predictions(trace, truth.text)


def read_heads(directory, manifest, labels, arms=ARMS):
    expected = manifest.loc[manifest.partition.ne('external')]
    heads = {}
    for name in arms:
        table = read_ids(directory / (name + '.parquet'))
        same_ids(table.index, expected.index, name)
        if set(table.columns) != set(METADATA) | set(labels):
            raise ValueError(f'{name}: expected metadata and exactly 18 candidate scores')
        same_meta(table, expected, METADATA, name)
        scores = table.loc[expected.index, list(labels)]
        if not np.isfinite(scores.to_numpy()).all():
            raise ValueError(f'Missing or invalid prediction scores for {name}')
        heads[name] = scores
    return heads


def correction(single, contextual, has_peer):
    values = contextual - single
    values = np.clip(values - values.mean(axis=1, keepdims=True), -3., 3.)
    values[~has_peer] = 0.
    return values


def all_predictions(trace, truth, baseline, heads, manifest, weights):
    result = truth[['model', 'prompt_id', 'cohort', 'n_words']].assign(baseline=baseline)
    ids = truth.index
    result[['peer_id', 'sham_id', 'similarity']] = manifest.loc[ids, ['peer_id', 'sham_id', 'similarity']]
    has_peer = result.peer_id.notna().to_numpy()
    single = heads['single'].loc[ids].to_numpy()
    for arm in ARMS[1:]:
        delta = correction(single, heads[arm].loc[ids].to_numpy(), has_peer)
        for w in weights:
            prediction = replay_predictions({**trace, 'sibling_logp': trace['sibling_logp'] + w * delta}, truth.text)
            check = np.ones(len(ids), dtype=bool) if w == 0 else ~has_peer
            if not np.array_equal(prediction[check], baseline[check]):
                raise ValueError(f'Identity control failed at {arm} weight {w}')
            result[f'{arm}_{w}'] = prediction
    return result


def balanced_accuracy(truth, prediction):
    return float(pd.Series(np.asarray(truth) == np.asarray(prediction)).groupby(np.asarray(truth)).mean().mean())


def paired_stats(part, before, after, draws):
    old, new = (part[which].eq(part.model).to_numpy() for which in (before, after))
    p, groups = pd.factorize(part.prompt_id, sort=True)
    c, classes = pd.factorize(part.model, sort=True)
    denom = np.zeros((len(groups), len(classes)))
    difference = np.zeros_like(denom)
    np.add.at(denom, (p, c), 1)
    np.add.at(difference, (p, c), new.astype(int) - old.astype(int))
    rng = np.random.default_rng(2411)
    counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=draws)
    sampled = counts @ denom
    deltas = np.divide(counts @ difference, sampled, out=np.full_like(sampled, np.nan), where=sampled > 0)
    fixed, harmed = int((new & ~old).sum()), int((old & ~new).sum())
    return dict(rows=len(part), prompts=len(groups), classes=len(classes), before=before, after=after,
                ba_before=balanced_accuracy(part.model, part[before]), ba_after=balanced_accuracy(part.model, part[after]),
                ba_delta_pp=float(100 * (difference.sum(0) / denom.sum(0)).mean()),
                prompt_ci95_pp=list(np.percentile(100 * np.nanmean(deltas, axis=1), [2.5, 97.5])),
                changed=int(part[before].ne(part[after]).sum()), fixed=fixed, harmed=harmed, net=fixed - harmed)


def lock_weights(public, weights, gpu_metrics):
    development = public.loc[public.cohort.eq('development')]
    if development.empty or not weights or 0 not in weights:
        raise ValueError('C1 and a zero-weight control are required')
    values, chosen = {}, {}
    for name in ARMS[1:]:
        values[name] = {w: balanced_accuracy(development.model, development[f'{name}_{w}']) for w in weights}
        chosen[name] = max(sorted(weights), key=values[name].get)
        if chosen[name] != gpu_metrics[name]['selected_weight']:
            raise ValueError(f'Reconstructed C1 choice differs from GPU for {name}')
    return chosen, values


def standalone_results(manifest, heads):
    records = []
    data = manifest.loc[manifest.partition.isin(['official', 'blind', 'held'])].copy()
    for name in heads:
        data[name] = heads[name].loc[data.index].idxmax(axis=1)
    partitions = [('official_all_oof', data[data.partition.eq('official')]),
                  ('gen003_sparse', data[data.partition.eq('blind')])]
    partitions.extend((str(name), part) for name, part in data[data.partition.eq('held')].groupby('source_dataset'))
    for name, part in partitions:
        for subset, take in [('all', part), ('short', part[part.n_words < 100])]:
            if len(take):
                records.append(dict(panel=name, subset=subset, rows=len(take), prompts=take.prompt_id.nunique(),
                                    classes=take.model.nunique(),
                                    **{arm: balanced_accuracy(take.model, take[arm]) for arm in heads}))
    return records, data


def blind_lexical_report(trace, panel, official, known, output, draws):
    from .peer_collision import collision_predictions, original_decisions, LEXICAL_THRESHOLD
    from .q8_evaluation import fit_grouping_vectorizer

    vectorizer = fit_grouping_vectorizer(pd.read_csv(official, usecols=['text']).text)['vectorizer']
    vec = vectorizer.transform(panel.text)
    affinity = (vec @ vec.T).toarray()
    records = panel[['prompt_id', 'model', 'n_words']].copy()
    records['group_size'] = records.groupby('prompt_id').model.transform('size')
    # These two weights were fixed by the original C1 experiment before this trace ran.
    comparisons = []
    for name, weight, restore in [('sibling', 1., False), ('dash', 2., True)]:
        old, _ = original_decisions(trace, panel.text, restore)
        records[name] = old
        records[name + '_peer'] = collision_predictions(trace, panel.text, affinity,
                                                        threshold=LEXICAL_THRESHOLD, weight=weight,
                                                        restore_dash=restore)
        subsets = {'all': records, 'short': records[records.n_words < 100],
                   'singleton': records[records.group_size == 1], 'grouped': records[records.group_size > 1]}
        if known:
            subsets['without_known_overlap'] = records[~records.index.isin(known)]
        for subset, rows in subsets.items():
            if len(rows):
                comparisons.append(dict(subset=subset, **paired_stats(rows, name, name + '_peer', draws)))
    records.to_parquet(output / 'blind_lexical_predictions.parquet')
    return comparisons


def run(gpu, public_trace, independent_trace, independent_panel, output, draws, official):
    if draws < 1:
        raise ValueError('At least one bootstrap draw is required')
    manifest = read_ids(gpu / 'manifest.parquet')
    panel = read_panel(independent_panel)
    truth = read_ids(public_trace / 'public_truth.parquet')
    check_manifest(manifest, truth, panel)
    trace, baseline = read_trace(public_trace, truth)
    blind_trace, blind_baseline = read_trace(independent_trace, panel)
    if not np.array_equal(blind_trace['labels'], trace['labels']):
        raise ValueError('Trace label orders differ')
    frozen = read_ids(independent_trace / 'frozen_predictions.parquet')
    same_ids(frozen.index, panel.index, 'Frozen Sibling Odds decisions', ordered=True)
    if not np.array_equal(blind_baseline, frozen.sibling_odds) or not np.array_equal(blind_trace['final'], frozen.dennis):
        raise ValueError('The independent baseline failed to reproduce Kaggle decisions')
    exposure = json.loads((independent_trace / 'body_overlap.json').read_text())
    known = set(exposure['active_pool_ids']) | set(exposure['official_train_ids'])
    if not known.issubset(panel.index):
        raise ValueError('Unknown ids in GEN003 training-corpus overlap audit')
    metrics = json.loads((gpu / 'metrics.json').read_text())
    weights = sorted(float(v) for v in metrics['retrieved']['arms'])
    if set(weights) != {float(v) for v in metrics['unrelated']['arms']}:
        raise ValueError('Control and candidate used different weight grids')
    heads = read_heads(gpu, manifest, trace['labels'])
    public = all_predictions(trace, truth, baseline, heads, manifest, weights)
    recorded = read_ids(gpu / 'paired.parquet')
    same_ids(recorded.index, public.index, 'GPU public replay')
    same_meta(recorded, public, [c for c in public if c not in ('peer_id', 'sham_id', 'similarity')], 'GPU replay')
    chosen, development_ba = lock_weights(public, weights, metrics)
    blind = all_predictions(blind_trace, panel, blind_baseline, heads, manifest, weights)
    pred = pd.concat([public, blind], verify_integrity=True)
    for arm, w in chosen.items():
        pred[arm] = pred[f'{arm}_{w}']
    pred['unrelated_at_retrieved_weight'] = pred[f"unrelated_{chosen['retrieved']}"]
    pred['known_overlap'] = pred.index.isin(known)
    selected = pred[['model','prompt_id','cohort','n_words','peer_id','sham_id','similarity','known_overlap',
                     'baseline','unrelated','retrieved','unrelated_at_retrieved_weight']]
    comparisons = [('baseline', 'unrelated'), ('baseline', 'retrieved'), ('unrelated', 'retrieved'),
                   ('unrelated_at_retrieved_weight', 'retrieved')]
    results = []
    for cohort, part in selected.groupby('cohort'):
        subsets = [('all', part), ('short', part[part.n_words < 100])]
        if known and part.known_overlap.any():
            subsets.append(('without_known_training_overlap', part[~part.known_overlap]))
        for subset, take in subsets:
            if not len(take):
                continue
            for a, b in comparisons:
                results.append(dict(cohort=cohort, subset=subset, **paired_stats(take, a, b, draws)))
    standalone, individual = standalone_results(manifest, heads)
    output.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output / 'paired_predictions.parquet')
    calls = ['baseline', 'unrelated', 'retrieved', 'unrelated_at_retrieved_weight']
    changed = selected[calls].nunique(axis=1) > 1
    selected.loc[changed].to_parquet(output / 'changed_rows.parquet')
    individual.to_parquet(output / 'standalone_predictions.parquet')
    lexical = blind_lexical_report(blind_trace, panel, official, known, output, draws)
    report = dict(c1_selected_weights=chosen, c1_ba_by_weight=development_ba, gen003_lexical=lexical,
                  interval='Paired bootstrap over prompts. Each draw averages recalls over true classes represented in that draw. Fixed heads and C1 choices.',
                  draws=draws, panels=dict(legacy=len(truth), independent=len(panel)), gen003_exposure=exposure,
                  paired=results, standalone=standalone)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('gpu', 'public-trace', 'independent-trace', 'independent-panel', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--official', type=Path, default=Path('data/train.csv'))
    args = parser.parse_args()
    run(args.gpu, args.public_trace, args.independent_trace, args.independent_panel, args.output, args.draws,
        args.official)
