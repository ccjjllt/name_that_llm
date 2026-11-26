"""Compare Flash-first scoring and paired PB115 at the original variable-position budget."""

import json

import numpy as np
import pandas as pd
from scipy.special import log_softmax, xlogy
from sklearn.preprocessing import StandardScaler

from .kiro_pb115_oof import FOCAL, GLM, combine_scores, fit_component, load_features, load_native_oof, select_original_route
from .peer_context_report import paired_stats, read_ids, same_ids, same_meta


FLASH_RATE = .01172
FULL_RATE = .019489
SHARES = (.55, .65, .75)
PRIORITIES = ('native_mass', 'flash_entropy', 'tag_change_then_entropy')


def read_flash(features, ids):
    spec = json.loads((features / 'spec.json').read_text())
    keys = [name.removeprefix('flash:') for name in spec['features_54'] if name.startswith('flash:')]
    raw = pd.read_csv(features / 'flash_features.csv', dtype={'id': str}).set_index('id', verify_integrity=True)
    same_ids(raw.index, ids, 'Cached Flash evidence')
    if not keys or len(keys) != len(set(keys)) or not set(keys).issubset(raw.columns):
        raise ValueError('The published Flash-only feature columns changed')
    matrix = raw.loc[ids, keys].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError('Flash-only features must be finite')
    return matrix, keys


def fit_flash(x, target, folds, glm):
    logp = np.empty((len(target), 18), dtype=np.float64)
    marginal = np.zeros_like(logp)
    y3 = np.where(target == glm[0], 1, np.where(target == glm[1], 2, 0))
    for fold in sorted(set(folds)):
        held = folds == fold
        scale = StandardScaler().fit(x[~held])
        train, test = scale.transform(x[~held]), scale.transform(x[held])
        logp[held] = fit_component(train, target[~held], test, 18)
        glm_logp = fit_component(train, y3[~held], test, 3)
        marginal[np.ix_(held, glm)] = glm_logp[:, 1:3] - glm_logp[:, [0]] + np.log(16.)
        print('Finished Flash-only prompt fold', int(fold), flush=True)
    return logp, marginal


def simplex(scores, glm):
    normalized = np.exp(log_softmax(scores, axis=1))
    focal = normalized[:, glm]
    rest = normalized[:, np.setdiff1d(np.arange(scores.shape[1]), glm)].sum(1)
    return np.column_stack([focal, rest])


def prefix(order, costs, budget):
    return order[costs[order].cumsum() <= budget + 1e-9]


def route_arms(base, flash_fused, folds, original, positions, glm):
    shape = len(base)
    flash_cost = positions * FLASH_RATE
    full_cost = positions * FULL_RATE
    native_tag = simplex(base, glm)
    flash_tag = simplex(flash_fused, glm)
    native_mass = np.exp(base[:, glm]).sum(1)
    flash_entropy = -xlogy(flash_tag, flash_tag).sum(1)
    changed = native_tag.argmax(1) != flash_tag.argmax(1)
    arms = {'flash_all_budget': (np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool))}
    for share in SHARES:
        arms[f'flash_{share:g}_no_full'] = (np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool))
        for name in PRIORITIES:
            arms[f'cascade_{share:g}_{name}'] = (np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool))

    summary = {}
    for fold in sorted(set(folds)):
        rows = np.flatnonzero(folds == fold)
        quota = float((flash_cost[rows] + full_cost[rows])[original[rows]].sum())
        summary[int(fold)] = dict(original_rows=int(original[rows].sum()), original_positions=int(positions[rows][original[rows]].sum()),
                                  original_variable_budget=quota, arms={})
        order = rows[np.argsort(-native_mass[rows], kind='stable')]
        arms['flash_all_budget'][0][prefix(order, flash_cost, quota)] = True
        for share in SHARES:
            screen = prefix(order, flash_cost, share * quota)
            left = quota - flash_cost[screen].sum()
            candidates = {
                'native_mass': screen,
                'flash_entropy': screen[np.argsort(-flash_entropy[screen], kind='stable')],
                'tag_change_then_entropy': screen[np.lexsort((screen, -flash_entropy[screen], -changed[screen].astype(int)))],
            }
            arms[f'flash_{share:g}_no_full'][0][screen] = True
            for kind, picked in candidates.items():
                first, second = arms[f'cascade_{share:g}_{kind}']
                first[screen] = True
                second[prefix(picked, full_cost, left)] = True
        for name, (first, second) in arms.items():
            f, g = first[rows], second[rows]
            spend = flash_cost[rows][f].sum() + full_cost[rows][g].sum()
            if (g & ~f).any() or spend > quota + 1e-7:
                raise ValueError('Full must be inside the Flash tier and fit its original fold variable budget')
            summary[int(fold)]['arms'][name] = dict(flash_rows=int(f.sum()), full_rows=int(g.sum()),
                    flash_positions=int(positions[rows][f].sum()), full_positions=int(positions[rows][g].sum()),
                    variable_cost=float(spend), unused=float(quota - spend))
    return arms, summary, changed, flash_entropy


def summarize(rows, arms, draws):
    results = {}
    for name in arms:
        old_ok = rows.original_paired.eq(rows.model)
        new_ok = rows[name].eq(rows.model)
        by_class = rows.assign(fixed=~old_ok & new_ok, harmed=old_ok & ~new_ok,
                               old_ok=old_ok, new_ok=new_ok).groupby('model').agg(
            rows=('model', 'size'), fixed=('fixed', 'sum'), harmed=('harmed', 'sum'),
            original_correct=('old_ok', 'sum'), cascade_correct=('new_ok', 'sum'))
        results[name] = dict(
            versus_original_pair=paired_stats(rows, 'original_paired', name, draws),
            versus_native=paired_stats(rows, 'native', name, draws),
            flash_rows=int(rows[f'flash__{name}'].sum()), full_rows=int(rows[f'full__{name}'].sum()),
            additional_flash_outside_original=int((rows[f'flash__{name}'] & ~rows.original_routed).sum()),
            old_pairs_not_full=int((rows.original_routed & ~rows[f'full__{name}']).sum()),
            by_class=by_class.reset_index().to_dict('records'),
        )
        if name.startswith('cascade_'):
            share = name.split('_', 2)[1]
            results[name]['versus_same_flash_tier_without_full'] = paired_stats(rows, f'flash_{share}_no_full', name, draws)
    return results


def run(official, native, features, coverage_file, original_file, output, draws=3000):
    train, labels, folds, base = load_native_oof(official, native)
    glm = pd.Index(labels).get_indexer(GLM)
    target = pd.Index(labels).get_indexer(train.model)
    _, _, token_count = load_features(features, train.index)
    positions = token_count + 30
    route = select_original_route(base, folds, glm, positions)
    reference, cover = [read_ids(at) for at in (original_file, coverage_file)]
    rows = train[['model', 'prompt_id']].assign(n_words=train.text.str.split().str.len(), fold=folds,
                                               native=labels[base.argmax(1)], original_routed=route)
    for table, name in ((reference, 'PB115 original OOF'), (cover, 'PB115 coverage archive')):
        same_ids(table.index, rows.index, name, ordered=True)
        same_meta(table, rows, ['model', 'prompt_id', 'n_words', 'fold'], name)
    full = cover['full__pb115_hybrid']
    original = np.where(route, full, rows.native)
    if (int(route.sum()) != 587 or int(positions[route].sum()) != 87409
            or not cover.estimated_positions.eq(positions).all() or not cover.native.eq(rows.native).all()
            or not np.array_equal(route, reference.routed) or not np.array_equal(route, cover.routed__original)
            or not np.array_equal(original, reference.pb115_hybrid)
            or not np.array_equal(original, cover.original__pb115_hybrid)
            or not full.isin(labels).all()):
        raise ValueError('Archived pure pairing fails to replay the original route, budget or decision on every row')
    x, cols = read_flash(features, rows.index)
    likelihood, marginal = fit_flash(x, target, folds, glm)
    flash_score = combine_scores(base, folds, np.ones(len(rows), dtype=bool), likelihood, marginal)
    flash_only = labels[flash_score.argmax(1)]
    arms, fold_summary, tag_change, ambiguity = route_arms(base, flash_score, folds, route, positions, glm)
    rows = rows.assign(original_paired=original, full_if_called=full, flash_if_called=flash_only,
                       estimated_pair_positions=positions, flash_tag_changed=tag_change, flash_entropy=ambiguity)
    decisions = {}
    for name, (flash, paired) in arms.items():
        decisions[name] = np.where(paired, full, np.where(flash, flash_only, rows.native))
        decisions[f'flash__{name}'], decisions[f'full__{name}'] = flash, paired
        if not np.array_equal(decisions[name][~flash], rows.native.loc[~flash]):
            raise ValueError('Rows skipped by the cascade must exactly preserve the Native call')
    rows = rows.assign(**decisions)
    scopes = {'all': rows, 'short': rows.loc[rows.n_words.lt(100)], 'true_glm_fable': rows.loc[rows.model.isin(FOCAL)],
              'short_true_glm_fable': rows.loc[rows.model.isin(FOCAL) & rows.n_words.lt(100)]}
    budget = (FLASH_RATE + FULL_RATE) * positions[route].sum()
    report = dict(rows=len(rows), original_reproduced=len(rows), original_rows=int(route.sum()),
                  original_paired_positions=int(positions[route].sum()), original_paired_variable_cost=float(budget),
                  flash_feature_columns=cols, flash_rate=FLASH_RATE, full_rate=FULL_RATE,
                  flash_budget_shares=list(SHARES), fold_allocation=fold_summary,
                  full_priorities='Native GLM mass, post-Flash 3-way entropy, or changed Full/Flash/other tag then entropy.',
                  scope='Both model calls charge the previous paired (max tokens + 30) proxy per row. No fixed launch costs, measured runtime, or deadline advantage are claimed.',
                  evaluation='Untouched official whole-prompt folds. No route priority, threshold, or share is fit to the evaluation results.',
                  results={name: summarize(part, arms, draws) for name, part in scopes.items()})
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Verified archived paired calls and scored Flash-first tiers on', len(rows), 'official rows', flush=True)
    return report
