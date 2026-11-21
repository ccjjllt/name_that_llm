"""Measure which passages give a candidate-likelihood model room to improve."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax


def candidate_cost(words):
    return np.minimum(128, np.maximum(16, 1.45 * np.asarray(words)))


def select_candidate_rows(logits, words, coverage=0.25, candidate_count=3):
    probability = softmax(logits, axis=1)
    candidates = np.argsort(-probability, axis=1, kind='stable')[:, :candidate_count]
    best = np.take_along_axis(probability, candidates, axis=1)
    priority = best[:, 1:].sum(axis=1) / np.sqrt(candidate_cost(words))
    selected = np.sort(np.argsort(-priority, kind='stable')[:int(coverage * len(logits))])
    return selected, candidates[selected]


def measure_routing(frame, logits, labels, coverage=(0.15, 0.25, 0.40)):
    target = pd.Categorical(frame.model, categories=labels).codes
    weights = 1.0 / np.bincount(target)[target] / len(labels)
    probability = softmax(logits, axis=1)
    top = np.argsort(-probability, axis=1, kind='stable')[:, :3]
    top_probability = np.take_along_axis(probability, top, axis=1)
    hit = top == target[:, None]
    current = hit[:, 0]
    can_fix = hit[:, 1:].any(axis=1)
    words = frame.text.str.split().str.len().to_numpy()
    costs = candidate_cost(words)
    entropy = -(probability * np.log(np.maximum(probability, 1e-15))).sum(axis=1)
    alternative_mass = top_probability[:, 1:].sum(axis=1)
    margins = top_probability[:, 0] - top_probability[:, 1]
    ranking = {
        'entropy': entropy,
        'small_margin': -margins,
        'alternative_mass': alternative_mass,
        'alternative_mass_per_cost': alternative_mass / np.sqrt(costs),
    }
    folds = frame.fold.to_numpy()
    results = {'score': float(weights[current].sum()),
               'top2_upper_bound': float(weights[hit[:, :2].any(1)].sum()),
               'top3_upper_bound': float(weights[hit.any(1)].sum()),
               'routing': []}
    for fraction in coverage:
        for method, priority in ranking.items():
            selected = []
            for fold in np.unique(folds):
                rows = np.flatnonzero(folds == fold)
                order = np.argsort(-priority[rows], kind='stable')
                selected.extend(rows[order[:int(fraction * len(rows))]])
            selected = np.asarray(selected, dtype=int)
            fix = selected[can_fix[selected]]
            class_gain = np.bincount(target[fix], weights=weights[fix], minlength=len(labels))
            class_gain *= len(labels)
            confusions = pd.DataFrame({'true': np.asarray(labels)[target[fix]],
                                       'winner': np.asarray(labels)[top[fix, 0]]})
            errors = confusions.value_counts().head(8)
            results['routing'].append({
                'coverage': fraction,
                'method': method,
                'upper_bound_gain': float(weights[fix].sum()),
                'eligible_wrong_or_correct': len(selected),
                'selected_with_correct_in_candidates_and_wrong_top1': len(fix),
                'mean_token_cost': float(costs[selected].mean()),
                'by_class_recall_gain': {label: float(gain) for label, gain in zip(labels, class_gain)},
                'top_confusions': [dict(true=true, winner=winner, count=int(count))
                                   for (true, winner), count in errors.items()],
            })
    return results


def main():
    root = Path('/kaggle/input')
    path = next(root.glob('**/name-that-llm/train.csv'))
    oof_path = next(root.rglob('glm_dynamics_oof.parquet'))
    train = pd.read_csv(path, dtype={'id': str})
    frame = train[['id', 'text', 'model']].merge(pd.read_parquet(oof_path).drop(columns=['model', 'prompt_id']),
                                                on='id', how='inner', validate='one_to_one')
    if len(frame) != len(train):
        raise ValueError('Official rows and held-out probabilities do not match')
    labels = sorted(train.model.unique())
    sources = ('reference', 'family_mass_glm_posterior')
    result = {source: measure_routing(frame, frame[[f'{source}__{label}' for label in labels]].to_numpy(), labels)
              for source in sources}
    (Path('/kaggle/working') / 'source_routing.json').write_text(json.dumps(result, indent=2))
    for source, values in result.items():
        print(source, {k: value for k, value in values.items() if k != 'routing'}, flush=True)
        for row in values['routing']:
            print(row['coverage'], row['method'], round(row['upper_bound_gain'], 5),
                  row['selected_with_correct_in_candidates_and_wrong_top1'], flush=True)


if __name__ == '__main__':
    main()
