"""Learn expected gain inside the fixed Low/user GLM shortlist."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax, logsumexp
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from glm_prefix_window import BLEND_WEIGHT, MODELS, ORIGINAL_BA, read_traces, token_cost
from glm_user_prefix import feature_block, held_out_member, metrics, select_rows
from likelihood_refine import transfer_glm_posterior
from likelihood_train import GLM, fit_head, head_logp


COVERAGE = 0.30
BUDGET = 130_000
RIDGE_ALPHA = 32.0
BENEFIT_SHARE = 0.25
ROUTER_FEATURES = ('glm_log_odds', 'glm_vs_best_other', 'glm_sibling_gap', 'claude_vs_best_other', 'log_words')


def router_features(base, glm, labels, words):
    other = ~np.isin(labels, GLM)
    claude = np.array([name.startswith('claude-') for name in labels])
    group = logsumexp(base[:, glm], axis=1)
    best_other = base[:, other].max(axis=1)
    return np.column_stack((group - logsumexp(base[:, other], axis=1),
                            base[:, glm].max(axis=1) - best_other,
                            np.abs(base[:, glm[0]] - base[:, glm[1]]),
                            base[:, claude].max(axis=1) - best_other, np.log1p(words)))


def fused_decision(base, ll, labels):
    updated = transfer_glm_posterior(base, base + BLEND_WEIGHT * ll, labels)
    return np.asarray(labels)[updated.argmax(axis=1)]


def router_targets(block, base, target, labels, folds, outer, meta):
    nested = np.empty((len(meta), len(labels)))
    for inner in np.unique(folds[meta]):
        fit = (folds != outer) & (folds != inner)
        held = folds[meta] == inner
        head = fit_head(block.iloc[fit], target[fit])
        if head['classes'] != labels:
            raise ValueError('An inner fold likelihood fit does not contain every class in order')
        nested[held] = head_logp(head, block.iloc[meta[held]])
    before = np.asarray(labels)[base[meta].argmax(axis=1)]
    after = fused_decision(base[meta], nested, labels)
    net = (after == target[meta]).astype(int) - (before == target[meta]).astype(int)
    classes = pd.Series(target[folds != outer]).value_counts()
    weights = (folds != outer).sum() / len(labels) / classes.loc[target[meta]].to_numpy()
    return net, weights


def rerank(rows, benefit):
    prior = np.arange(len(rows), 0, -1) / len(rows)
    learned = np.empty_like(prior)
    learned[np.argsort(-benefit, kind='stable')] = prior
    score = (1 - BENEFIT_SHARE) * prior + BENEFIT_SHARE * learned
    return rows[np.argsort(-score, kind='stable')]


def compare(train_path, reference_path, cache):
    train = pd.read_csv(train_path, dtype={'id': str, 'prompt_id': str})
    labels = sorted(train.model.unique())
    columns = [f'reference__{name}' for name in labels]
    reference = pd.read_parquet(reference_path, columns=['id', 'fold', *columns])
    reference['id'] = reference.id.astype(str)
    if len(labels) != 18 or set(train.id) != set(reference.id):
        raise ValueError('Reference IDs and official classes differ')
    train = train.merge(reference, on='id', validate='one_to_one', sort=False)
    folds = train.fold.to_numpy()
    if len(np.unique(folds)) != 5 or train.groupby('prompt_id').fold.nunique().max() != 1:
        raise ValueError('Use the original five prompt folds')
    target = train.model.to_numpy()
    base = log_softmax(train[columns].to_numpy(), axis=1)
    baseline = np.asarray(labels)[base.argmax(axis=1)]
    words = train.text.fillna('').str.split().str.len().to_numpy()
    glm = [labels.index(name) for name in GLM]
    block, costs = feature_block(cache, 'low_user_160', train)
    all_candidates = select_rows(base, folds, glm, costs, COVERAGE, None)
    original = select_rows(base, folds, glm, costs, COVERAGE, BUDGET)
    if (len(all_candidates) != 809 or len(original) != 587 or int(costs[original].sum()) != 87409
            or block.shape[1] != 54):
        raise ValueError('The baseline does not reproduce the 0.716 fixed shortlist and 130k-budget selection')

    oof_post = held_out_member(block, target, folds)
    candidate = fused_decision(base, oof_post, labels)
    features = router_features(base, glm, labels, words)
    estimates = np.zeros(len(folds))
    selected = []
    fit_counts = {}
    for outer in np.unique(folds):
        meta = all_candidates[folds[all_candidates] != outer]
        holdout = all_candidates[folds[all_candidates] == outer]
        net, weights = router_targets(block, base, target, labels, folds, outer, meta)
        model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        model.fit(features[meta], net * weights)
        estimates[holdout] = model.predict(features[holdout])
        ranking = rerank(holdout, estimates[holdout])
        ceiling = BUDGET * (folds == outer).sum() / 4000
        selected.extend(ranking[np.cumsum(costs[ranking]) <= ceiling])
        fit_counts[str(outer)] = {'fit_rows': len(meta), 'fit_gains': int((net > 0).sum()),
                                  'fit_harms': int((net < 0).sum()),
                                  'standardized_coefficients': dict(zip(ROUTER_FEATURES, model[-1].coef_.tolist()))}
    selected = np.asarray(selected, dtype=np.intp)
    traces = {tag: read_traces(cache, tag, train.id) for tag in MODELS}

    def summarize(rows):
        prediction = baseline.copy()
        prediction[rows] = candidate[rows]
        short = words < 100
        result = metrics(target, prediction, baseline, labels, folds)
        result.update(rows=len(rows), budget_positions=int(costs[rows].sum()), tokens=token_cost(traces, rows, 160),
                      under_100_words=float(balanced_accuracy_score(target[short], prediction[short])),
                      selected_short=int(short[rows].sum()),
                      allocation_by_fold={str(fold): {'rows': int((folds[rows] == fold).sum()),
                                                     'budget_positions': int(costs[rows[folds[rows] == fold]].sum())}
                                          for fold in np.unique(folds)})
        return prediction, result

    mass_prediction, mass_result = summarize(original)
    learned_prediction, learned_result = summarize(selected)
    if not np.isclose(mass_result['balanced_accuracy'], ORIGINAL_BA, rtol=0, atol=1e-9):
        raise ValueError(f"Original pure-mass route BA differs: {mass_result['balanced_accuracy']}")
    first, second = mass_prediction == target, learned_prediction == target
    shortlist_net = (candidate[all_candidates] == target[all_candidates]).astype(int) - (baseline[all_candidates] == target[all_candidates]).astype(int)
    predicted = estimates[all_candidates]
    return {
        'shortlist_rows': len(all_candidates), 'coverage': COVERAGE, 'budget_per_4000': BUDGET,
        'glm_blend_weight': BLEND_WEIGHT, 'ridge_alpha': RIDGE_ALPHA, 'benefit_rank_share': BENEFIT_SHARE,
        'scope': 'Conditional on the fixed upstream Native OOF. GLM likelihood targets are internally cross-fitted '
                 'inside every outer training side, and each router excludes the entire evaluation fold.',
        'router_fit_by_fold': fit_counts,
        'oof_benefit_correlation': float(np.corrcoef(predicted, shortlist_net)[0, 1]) if predicted.std() > 0 else None,
        'glm_mass': mass_result, 'learned': learned_result,
        'versus_glm_mass': {
            'balanced_accuracy': learned_result['balanced_accuracy'] - mass_result['balanced_accuracy'],
            'under_100_words': learned_result['under_100_words'] - mass_result['under_100_words'],
            'by_fold': {fold: value - mass_result['by_fold'][fold] for fold, value in learned_result['by_fold'].items()},
            'recall': {name: value - mass_result['recall'][name] for name, value in learned_result['recall'].items()},
            'common_selected': len(np.intersect1d(original, selected)),
            'changed': int((learned_prediction != mass_prediction).sum()),
            'fixed': int((~first & second).sum()), 'lost': int((first & ~second).sum()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/glm_prefix_learned_routing_metrics.json'))
    args = parser.parse_args()
    report = compare(args.train, args.reference, args.cache)
    output = json.dumps(report, indent=2, allow_nan=False)
    args.output.write_text(output + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
