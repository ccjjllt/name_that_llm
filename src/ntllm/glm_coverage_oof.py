"""Measure the benefit of wider native GLM coverage using fixed out-of-fold heads."""

import json

import numpy as np
import pandas as pd
from scipy.special import log_softmax

from .kiro_pb115_oof import GLM, fit_predictions, load_features, select_original_route
from .member_comp_replay import match_temperature
from .peer_context_report import paired_stats, read_ids, same_ids


BUDGET_MULTIPLIERS = (1.5, 2., 3.)
HEADS = ('legacy54', 'pb115_hybrid')


def coverage_routes(base, folds, glm, costs):
    original = select_original_route(base, folds, glm, costs)
    masks = {'original': original, **{f'budget_{factor:g}x': np.zeros(len(base), dtype=bool)
                                    for factor in BUDGET_MULTIPLIERS},
             'full': np.ones(len(base), dtype=bool)}
    mass = np.exp(base[:, glm]).sum(1)
    for fold in sorted(set(folds)):
        rows = np.flatnonzero(folds == fold)
        ordered = rows[np.argsort(-mass[rows], kind='stable')]
        cumulative = costs[ordered].cumsum()
        actual_original = costs[rows[original[rows]]].sum()
        for factor in BUDGET_MULTIPLIERS:
            budget = int(actual_original * factor)
            masks[f'budget_{factor:g}x'][ordered[cumulative <= budget]] = True
    for before, after in zip(masks.values(), list(masks.values())[1:]):
        if (before & ~after).any():
            raise ValueError('Expanded route must include each earlier routed row')
    return masks


def predicted_if_routed(base, folds, labels, scores, marginal):
    result = {name: np.empty(len(base), dtype=labels.dtype) for name in HEADS}
    for fold in sorted(set(folds)):
        held = folds == fold
        w18 = 1.5 / match_temperature(base[held], .5692304418233374)
        w3 = 1.2 / match_temperature(base[held], .90)
        legacy = base[held] + w18 * scores['legacy54'][held]
        hybrid_boost = .75 * w18 * scores['pb115_hybrid'][held] + .25 * w3 * marginal['pb115'][held]
        hybrid = base[held] + hybrid_boost
        result['legacy54'][held] = labels[legacy.argmax(1)]
        result['pb115_hybrid'][held] = labels[hybrid.argmax(1)]
    return result


def report_route(rows, head, route, previous, masks, draws):
    probes = {'all': rows, 'short': rows[rows.n_words.lt(100)], 'extra_rows': rows[masks[route] & ~masks['original']]}
    out = {}
    for scope, part in probes.items():
        if part.empty:
            continue
        origin = f'original__{head}'
        arm = f'{route}__{head}'
        prev = f'{previous}__{head}'
        start = part[origin].eq(part.model)
        finish = part[arm].eq(part.model)
        by_label = part.assign(fixed=~start & finish, harmed=start & ~finish).groupby('model').agg(
            rows=('model', 'size'), fixed=('fixed', 'sum'), harmed=('harmed', 'sum'))
        out[scope] = {'against_original': paired_stats(part, origin, arm, draws),
                      'against_previous': paired_stats(part, prev, arm, draws),
                      'by_label': by_label.reset_index().to_dict('records')}
    return out


def run(official, native_file, prior_file, features, output, draws=3000):
    train = pd.read_csv(official, dtype={'id': str, 'prompt_id': str}).set_index('id', verify_integrity=True)
    labels = np.sort(train.model.unique())
    if len(train) != 2699 or len(labels) != 18:
        raise ValueError('Expected 2,699 rows and 18 official labels')
    ref, prior = (read_ids(path) for path in (native_file, prior_file))
    for name, table in (('Native', ref), ('Kiro', prior)):
        same_ids(table.index, train.index, name)
    ref, prior = ref.loc[train.index], prior.loc[train.index]
    folds = ref.fold.to_numpy()
    if (len(set(folds)) != 5 or not np.array_equal(folds, prior.fold.to_numpy())
            or not pd.Series(folds, index=train.index).groupby(train.prompt_id).nunique().eq(1).all()):
        raise ValueError('Expected the same five whole-prompt folds as the Kiro audit')
    for key in ('model', 'prompt_id'):
        if not prior[key].astype(str).equals(train[key].astype(str)):
            raise ValueError(f'Prior audit differs in {key}')
    features54, features115, length = load_features(features, train.index)
    costs = length + 30
    base = log_softmax(ref[[f'reference__{label}' for label in labels]].to_numpy(), axis=1)
    glm = pd.Index(labels).get_indexer(GLM)
    target = pd.Index(labels).get_indexer(train.model)
    masks = coverage_routes(base, folds, glm, costs)
    original = masks['original']
    if original.sum() != 587 or costs[original].sum() != 87409 or not np.array_equal(original, prior.routed):
        raise ValueError('The reconstructed original route disagrees with the frozen audit')
    ll, marginal = fit_predictions((features54, features115), target, folds, glm)
    candidate = predicted_if_routed(base, folds, labels, ll, marginal)
    rows = train[['model', 'prompt_id']].assign(n_words=train.text.str.split().str.len(), fold=folds,
                                               estimated_positions=costs, native=labels[base.argmax(1)])
    result = {}
    for route, covered in masks.items():
        rows['routed__' + route] = covered
        for name in HEADS:
            new = np.where(covered, candidate[name], rows.native)
            if route == 'original' and not np.array_equal(new, prior[name]):
                raise ValueError(f'{name}: first-stage decisions differ from the frozen audit')
            rows[f'{route}__{name}'] = new
    for previous, route in zip(masks, list(masks)[1:]):
        covered = masks[route]
        result[route] = dict(covered=int(covered.sum()), short_covered=int((covered & rows.n_words.lt(100)).sum()),
                             estimated_positions=int(costs[covered].sum()), heads={})
        for name in HEADS:
            result[route]['heads'][name] = report_route(rows, name, route, previous, masks, draws)
    oracle = {}
    for name in HEADS:
        original_label = rows[f'original__{name}']
        helpful = (~original) & original_label.ne(rows.model) & rows[f'full__{name}'].eq(rows.model)
        column = f'oracle__{name}'
        rows[column] = np.where(helpful, rows[f'full__{name}'], original_label)
        oracle[name] = paired_stats(rows, f'original__{name}', column, draws)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'predictions.parquet')
    report = dict(original_rows=int(original.sum()), original_positions=int(costs[original].sum()),
                  widened=result, oracle_truth_only=oracle,
                  limitation='Estimated positions on official folds, without measuring additional inference time')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
