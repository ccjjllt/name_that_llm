"""Review competing author predictions using text-similar peers on the same inference batch."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from .sibling_odds_replay import replay_predictions

WEIGHTS = (0., .5, 1., 2.)
LEXICAL_THRESHOLD = .20


def peer_edges(similarity, threshold, neighbors=3):
    affinity = np.asarray(similarity)
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError('Expected a square matrix of query similarities')
    if not np.isfinite(affinity).all() or not 0 < threshold <= 1 or neighbors < 1:
        raise ValueError('Invalid similarities, threshold, or neighbor count')
    work = affinity.copy().astype(np.float32)
    np.fill_diagonal(work, -np.inf)
    indices = np.argsort(-work, axis=1, kind='stable')[:, :min(neighbors, len(work))]
    rows = np.broadcast_to(np.arange(len(work))[:, None], indices.shape)
    strengths = work[rows, indices]
    keep = strengths >= threshold
    return rows[keep], indices[keep], np.minimum(1., strengths[keep])


def original_decisions(trace, texts, restore_dash=False):
    baseline = replay_predictions(trace, texts)
    pre_typography = replay_predictions(trace, texts, typography=False)
    if restore_dash:
        restore = ((baseline != pre_typography) & np.char.startswith(pre_typography.astype(str), 'gemini-')
                   & texts.loc[trace['ids']].str.contains('—', regex=False).to_numpy())
        baseline = np.where(restore, pre_typography, baseline)
    return baseline, pre_typography


def collision_predictions(trace, texts, similarity, *, threshold, weight, restore_dash=False, neighbors=3):
    """Scores, texts and both matrix axes must refer to the same trace ids. No labels or group ids are read."""
    baseline, pre_typography = original_decisions(trace, texts, restore_dash)
    if weight < 0 or len(similarity) != len(baseline):
        raise ValueError('Expected a nonnegative weight and a query similarity matrix aligned to the trace')
    if not weight:
        return baseline
    rows, peers, affinity = peer_edges(similarity, threshold, neighbors)
    labels = pd.Index(trace['labels'])
    peer_class = labels.get_indexer(baseline[peers])
    scores = trace['sibling_logp']
    probability = softmax(scores, axis=1)
    peer_prob = probability[peers, peer_class]
    stronger = peer_prob > probability[rows, peer_class]
    discount = np.zeros_like(scores, dtype=np.float64)
    # Several neighbors voting for the same author must not accumulate a hard exclusion.
    np.maximum.at(discount, (rows[stronger], peer_class[stronger]), (affinity * peer_prob)[stronger])
    adjusted = scores + weight * np.log1p(-np.minimum(discount, .999))
    proposal, _ = original_decisions({**trace, 'sibling_logp': adjusted}, texts, restore_dash)
    # Freeze all original GLM/Fable handoffs and already-triggered typography rules.
    unchanged = ((trace['glm_winner'] | trace['fable_guard']) | (baseline != pre_typography)
                 | (discount.max(axis=1) == 0))
    if restore_dash:
        unchanged |= replay_predictions(trace, texts) != baseline
    return np.where(unchanged, baseline, proposal)


def metrics(frame, old, new):
    rows = []
    for name, part in frame.groupby('cohort'):
        for subset, pick in [('all', part), ('short', part[part.n_words < 100])]:
            p = pick.pos.to_numpy()
            truth = pick.model.to_numpy()
            before, after = old[p] == truth, new[p] == truth
            ba0 = float(pd.Series(before).groupby(truth).mean().mean())
            ba1 = float(pd.Series(after).groupby(truth).mean().mean())
            rows.append(dict(cohort=name, subset=subset, rows=len(p), fixed=int((~before & after).sum()),
                             harmed=int((before & ~after).sum()), net=int(after.sum() - before.sum()),
                             changed=int((old[p] != new[p]).sum()), ba_before=ba0, ba_after=ba1))
    return rows


def neighbor_diagnostics(frame, similarity, threshold, neighbors):
    rows, peers, strengths = peer_edges(similarity, threshold, neighbors)
    group = frame.prompt_id.to_numpy()
    correct = group[rows] == group[peers]
    details = pd.DataFrame({'id': frame.index.to_numpy()[rows], 'peer': frame.index.to_numpy()[peers],
                            'affinity': strengths, 'true_same_prompt': correct})
    stats = []
    for cohort, part in frame.groupby('cohort'):
        selected = np.isin(rows, part.pos)
        relevant = sum(n * (n - 1) for n in part.groupby('prompt_id').size())
        found = selected & correct
        stats.append(dict(cohort=cohort, queries=len(part), edges=int(selected.sum()),
                          queries_covered=int(len(np.unique(rows[selected]))),
                          true_edges=int(found.sum()), all_same_prompt_directed_pairs=int(relevant),
                          precision=float(correct[selected].mean()) if selected.any() else None,
                          recall=float(found.sum() / relevant) if relevant else None))
    return stats, details


def compare(frame, trace, matrices, out, restore_dash=False):
    baseline, _ = original_decisions(trace, frame.text, restore_dash)
    configurations, predictions = [], pd.DataFrame({'model': frame.model, 'prompt_id': frame.prompt_id,
                                                     'cohort': frame.cohort, 'n_words': frame.n_words,
                                                     'baseline': baseline})
    for name, (matrix, threshold, k) in matrices.items():
        for weight in WEIGHTS:
            pred = collision_predictions(trace, frame.text, matrix, threshold=threshold, weight=weight,
                                         restore_dash=restore_dash, neighbors=k)
            col = f'{name}_{weight:g}'
            predictions[col] = pred
            configurations.append(dict(view=name, weight=weight, metrics=metrics(frame, baseline, pred)))
    dev = [x for x in configurations if x['view'] == 'lexical']
    chosen = max(dev, key=lambda x: next(r['ba_after'] for r in x['metrics']
                                        if (r['cohort'], r['subset']) == ('development', 'all')))['weight']
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out / 'predictions.parquet')
    (out / 'comparison.json').write_text(json.dumps(dict(selected_weight=chosen, arms=configurations),
                                                     ensure_ascii=False, indent=2) + '\n')
    return chosen


def run(official, trace_dir, output):
    from .q8_evaluation import fit_grouping_vectorizer

    with np.load(trace_dir / 'stage_scores.npz', allow_pickle=False) as bundle:
        trace = dict(bundle)
    frame = pd.read_parquet(trace_dir / 'public_truth.parquet').set_index('id').loc[trace['ids']].copy()
    frame['pos'] = np.arange(len(frame))
    training = pd.read_csv(official, usecols=['text'])
    fit = fit_grouping_vectorizer(training.text)
    if fit['vectorizer'] is None:
        raise ValueError(fit['diagnostics']['fallback'])
    features = fit['vectorizer'].transform(frame.text)
    affinity = (features @ features.T).toarray().astype(np.float32)
    cohort = frame.cohort.to_numpy()
    affinity[cohort[:, None] != cohort[None, :]] = 0.
    # True groups are only consulted for descriptive diagnostics, after lexical affinity is fixed.
    group = frame.prompt_id.to_numpy()
    true_pair = group[:, None] == group[None, :]
    matrices = dict(lexical=(affinity, LEXICAL_THRESHOLD, 3),
                    true_group=(true_pair.astype(np.float32), 1., 18))
    output.mkdir(parents=True, exist_ok=True)
    stats, edges = neighbor_diagnostics(frame, *matrices['lexical'])
    edges.to_parquet(output / 'edges.parquet', index=False)
    np.savez_compressed(output / 'similarity.npz', ids=trace['ids'], similarity=affinity)
    experiment = dict(fit=fit['diagnostics'], threshold=LEXICAL_THRESHOLD, neighbors=3, weights=WEIGHTS,
                      retrieval=stats, oracle='All same-prompt neighbors receive unit affinity; diagnostic only.')
    for name, restore in [('sibling_odds', False), ('dash_only', True)]:
        experiment[name + '_chosen_weight'] = compare(frame, trace, matrices, output / name, restore)
    (output / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official', type=Path, default=Path('data/train.csv'))
    parser.add_argument('--trace', type=Path, default=Path('runs/atlas_0719_20261123/trace/atlas_baseline'))
    parser.add_argument('--output', type=Path, default=Path('runs/peer_retrieval_20261124/lexical'))
    args = parser.parse_args()
    run(args.official, args.trace, args.output)
