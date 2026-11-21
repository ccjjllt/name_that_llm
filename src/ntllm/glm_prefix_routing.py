"""Compare GLM mass and Fable-boosted routing on fixed prefix OOF scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score

if __package__:
    from .likelihood_refine import transfer_glm_posterior
    from .likelihood_train import FABLE, GLM, normalize_temperature
else:
    from likelihood_refine import transfer_glm_posterior
    from likelihood_train import FABLE, GLM, normalize_temperature


def find_input(filename):
    matches = sorted(Path('/kaggle/input').rglob(filename))
    if len(matches) != 1:
        raise ValueError(f'Expected one Kaggle input named {filename}, found {matches}')
    return matches[0]


def select_rows(priority, folds, costs, coverage, budget):
    selected = np.zeros(len(folds), dtype=bool)
    for fold in np.unique(folds):
        rows = np.flatnonzero(folds == fold)
        rank = rows[np.argsort(-priority[rows], kind='stable')][:int(coverage * len(rows))]
        if budget is not None:
            rank = rank[np.cumsum(costs[rank]) <= budget * len(rows) / 4000]
        selected[rank] = True
    return selected


def metrics(target, prediction, baseline, folds):
    correct = prediction == target
    old_correct = baseline == target
    return {
        'balanced_accuracy': float(balanced_accuracy_score(target, prediction)),
        'recall': {label: float(correct[target == label].mean()) for label in (*GLM, FABLE)},
        'by_fold': {str(fold): float(balanced_accuracy_score(target[folds == fold], prediction[folds == fold]))
                    for fold in np.unique(folds)},
        'changed_from_reference': int((prediction != baseline).sum()),
        'fixed': int((correct & ~old_correct).sum()),
        'lost': int((~correct & old_correct).sum()),
    }


def compare(prefix, reference):
    with np.load(prefix, allow_pickle=False) as archive:
        ids = archive['ids'].astype(str)
        folds = archive['folds']
        likelihood = archive['low_user_160_logp']
        costs = archive['low_user_160_token_positions']
    frame = pd.read_parquet(reference)
    frame['id'] = frame.id.astype(str)
    frame = frame.set_index('id', verify_integrity=True)
    if len(ids) != len(frame) or set(ids) != set(frame.index):
        raise ValueError('Prefix and reference OOF IDs differ')
    frame = frame.loc[ids]
    if not np.array_equal(frame.fold.to_numpy(), folds) or frame.groupby('prompt_id').fold.nunique().max() != 1:
        raise ValueError('Prefix and reference OOF prompt folds differ')

    target = frame.model.to_numpy()
    labels = np.sort(frame.model.unique())
    base = log_softmax(frame[[f'reference__{label}' for label in labels]].to_numpy(), axis=1)
    if likelihood.shape != base.shape or costs.shape != (len(base),):
        raise ValueError('Prefix OOF score or token budget dimensions differ')
    baseline = labels[base.argmax(axis=1)]
    weight = 1.0 / normalize_temperature(base)
    updated = transfer_glm_posterior(base, base + weight * likelihood, labels)
    candidate = labels[updated.argmax(axis=1)]

    prob = np.exp(base)
    mass = prob[:, np.isin(labels, GLM)].sum(axis=1)
    fable = prob[:, labels == FABLE].ravel()
    priorities = {'glm_mass': mass, 'fable_bonus': mass + 0.35 * np.minimum(fable, mass)}
    cases = {}
    for case, coverage, budget in (('same_130k_tokens_per_4000', 0.30, 130_000), ('same_top15', 0.15, None)):
        selections = {}
        predictions = {}
        strategies = {}
        for name, priority in priorities.items():
            selected = select_rows(priority, folds, costs, coverage, budget)
            prediction = np.where(selected, candidate, baseline)
            selections[name] = selected
            predictions[name] = prediction
            strategies[name] = {'rows': int(selected.sum()), 'token_positions': int(costs[selected].sum()),
                                **metrics(target, prediction, baseline, folds)}

        mass_prediction, fable_prediction = predictions['glm_mass'], predictions['fable_bonus']
        mass_correct, fable_correct = mass_prediction == target, fable_prediction == target
        changed = mass_prediction != fable_prediction
        paired = {
            'common_selected': int((selections['glm_mass'] & selections['fable_bonus']).sum()),
            'changed_between_routes': int(changed.sum()),
            'changed_between_routes_by_fold': {str(fold): int(changed[folds == fold].sum()) for fold in np.unique(folds)},
            'fable_bonus_only_correct': int((fable_correct & ~mass_correct).sum()),
            'glm_mass_only_correct': int((mass_correct & ~fable_correct).sum()),
            'fable_bonus_minus_glm_mass_ba': (strategies['fable_bonus']['balanced_accuracy']
                                               - strategies['glm_mass']['balanced_accuracy']),
        }
        cases[case] = {'coverage_limit': coverage, 'token_budget_per_4000': budget, **strategies, 'paired': paired}
    return {'blend_weight': weight, 'reference': metrics(target, baseline, baseline, folds), 'cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/glm_prefix_routing_metrics.json'))
    args = parser.parse_args()
    result = compare(args.prefix or find_input('glm_user_prefix_oof.npz'),
                     args.reference or find_input('glm_dynamics_oof.parquet'))
    report = json.dumps(result, indent=2)
    args.output.write_text(report + '\n')
    print(report, flush=True)


if __name__ == '__main__':
    main()
