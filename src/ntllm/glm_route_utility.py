"""Compare native GLM scheduling policies with equal per-fold token budgets."""

import json
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .kiro_pb115_oof import GLM, combine_scores, fit_component, load_features, load_native_oof, select_original_route
from .peer_context_report import paired_stats, read_ids, same_ids


def routing_features(base, glm, words, cost):
    probability = np.exp(base)
    siblings = np.sort(probability[:, glm], axis=1)
    non_glm = np.delete(probability, glm, axis=1).max(axis=1)
    top = np.sort(probability, axis=1)[:, -2:]
    mass = siblings.sum(axis=1)
    return np.column_stack((mass, siblings[:, 0], siblings[:, 0] / np.maximum(mass, 1e-12), siblings[:, 1] - non_glm,
                            non_glm, top[:, 1], top[:, 1] - top[:, 0], -(probability * base).sum(axis=1),
                            np.log1p(words), cost / 190, np.isin(base.argmax(axis=1), glm),
                            mass * (1 - top[:, 1]), mass * (cost / 190)))


def nested_actions(x, y, folds, base, glm):
    """For each pair of outer and calibration folds, train the same PB115 head on the other three."""
    outputs = {}
    y3 = np.where(y == glm[0], 1, np.where(y == glm[1], 2, 0))
    for a, b in combinations(np.unique(folds), 2):
        predict = (folds == a) | (folds == b)
        scale = StandardScaler().fit(x[~predict])
        learn, test = scale.transform(x[~predict]), scale.transform(x[predict])
        likelihood = fit_component(learn, y[~predict], test, 18)
        three = fit_component(learn, y3[~predict], test, 3)
        marginal = np.zeros_like(likelihood)
        marginal[:, glm] = three[:, 1:3] - three[:, [0]] + np.log(16.)
        score = combine_scores(base[predict], folds[predict], np.ones(len(test), dtype=bool), likelihood, marginal)
        new = score.argmax(axis=1)
        for outer, inner in ((a, b), (b, a)):
            outputs[outer, inner] = new[folds[predict] == inner]
        print(f'Finished nested likelihood fits excluding folds {a} and {b}', flush=True)
    return outputs


def spend(priority, costs, budget):
    """Greedily allocate positions and fill any remainder with a fitting passage."""
    chosen = np.zeros(len(priority), dtype=bool)
    for row in np.argsort(-priority, kind='stable'):
        if costs[row] <= budget:
            chosen[row] = True
            budget -= costs[row]
    return chosen


def run(official, native, coverage_path, features, output, draws=4000):
    train, labels, folds, base = load_native_oof(official, native)
    archived = read_ids(coverage_path)
    same_ids(archived.index, train.index, 'GLM coverage archive')
    archived = archived.loc[train.index]
    _, x, tokens = load_features(features, train.index)
    cost = tokens + 30
    label_index = pd.Index(labels)
    glm = label_index.get_indexer(GLM)
    target = label_index.get_indexer(train.model)
    before = base.argmax(axis=1)
    expert = label_index.get_indexer(archived['full__pb115_hybrid'])
    original = select_original_route(base, folds, glm, cost)
    if (not np.array_equal(folds, archived.fold) or not np.array_equal(cost, archived.estimated_positions)
            or not np.array_equal(original, archived.routed__original)
            or not np.array_equal(labels[before], archived.native)
            or (expert < 0).any() or int(cost[original].sum()) != 87409
            or not np.array_equal(labels[np.where(original, expert, before)], archived.original__pb115_hybrid)):
        raise ValueError('The fixed head, original route or token costs differ from the coverage audit')

    words = train.text.str.split().str.len().to_numpy()
    signal = routing_features(base, glm, words, cost)
    nested = nested_actions(x, target, folds, base, glm)
    gain = (expert == target).astype(int) - (before == target)
    routes = {name: original.copy() if name == 'original' else np.zeros(len(train), dtype=bool)
              for name in ('original', 'glm_mass_per_position', 'learned_per_position', 'oracle_per_position')}
    prediction = np.empty(len(train))
    for outer in np.unique(folds):
        held = folds == outer
        calibration = ~held
        # Each calibration target excludes its own prompt fold and the outer evaluation fold.
        nested_prediction = np.full(len(train), -1)
        for inner in np.unique(folds[calibration]):
            nested_prediction[folds == inner] = nested[outer, inner]
        observed = ((nested_prediction == target).astype(float) - (before == target))[calibration]
        counts = np.bincount(target[calibration], minlength=len(labels))
        weights = calibration.sum() / (len(labels) * counts[target[calibration]])
        scale = StandardScaler().fit(signal[calibration])
        model = Ridge(alpha=40.).fit(scale.transform(signal[calibration]), observed, sample_weight=weights)
        prediction[held] = model.predict(scale.transform(signal[held]))
        budget = int(cost[held & original].sum())
        priorities = {'glm_mass_per_position': signal[held, 0] / cost[held],
                      'learned_per_position': prediction[held] / cost[held],
                      'oracle_per_position': np.where(gain[held] > 0, 1 / (counts[target[held]] * cost[held]), -np.inf)}
        for name, priority in priorities.items():
            mask = spend(priority, cost[held], budget)
            routes[name][held] = mask if name != 'oracle_per_position' else mask & np.isfinite(priority)
    rows = train[['model', 'prompt_id']].assign(fold=folds, n_words=words, estimated_positions=cost,
                                               cheap=labels[before], if_routed=labels[expert],
                                               gain_if_routed=gain, predicted_gain=prediction)
    report = dict(rows=len(rows), budget_by_fold={}, arms={})
    for fold in np.unique(folds):
        report['budget_by_fold'][int(fold)] = int(cost[(folds == fold) & original].sum())
    for name, route in routes.items():
        rows['routed__' + name] = route
        rows[name] = labels[np.where(route, expert, before)]
        if any(cost[(folds == fold) & route].sum() > limit for fold, limit in report['budget_by_fold'].items()):
            raise ValueError(f'{name} exceeds a fixed token budget')
    for name, route in routes.items():
        report['arms'][name] = dict(
            routed=int(route.sum()), original_overlap=int((original & route).sum()), token_positions=int(cost[route].sum()),
            all=paired_stats(rows, 'original', name, draws),
            short=paired_stats(rows.loc[rows.n_words.lt(100)], 'original', name, draws),
            focal=paired_stats(rows.loc[rows.model.isin((*GLM, 'claude-fable-5'))], 'original', name, draws),
            versus_cheap=paired_stats(rows, 'cheap', name, draws),
        )
    report['limitations'] = ('Native scores are frozen from their original outer folds. The observed action targets '
                             'cross-fit PB115 a second time, but cannot cross-fit the upstream Native training again. '
                             'Token positions are cost estimates on the official data, not measured runtime.')
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Compared {len(routes)} schedules on {len(rows)} passages', flush=True)
