"""Compare PB115 GLM likelihood fusion and routing on the official prompt folds."""

import json

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler

from .kiro_pb115_oof import FOCAL, GLM, combine_scores, fit_component, load_features, load_native_oof, select_original_route
from .member_comp_replay import match_temperature
from .peer_context_report import paired_stats, read_ids


def select_route(base, folds, glm, costs, control, multiplier):
    confidence = np.exp(base[:, glm]).sum(axis=1)
    route = np.zeros(len(base), dtype=bool)
    limits = {}
    for fold in sorted(set(folds)):
        rows = np.flatnonzero(folds == fold)
        budget = multiplier * int(costs[rows[control[rows]]].sum())
        cap = int(len(rows) * .40)
        ranked = rows[np.argsort(-confidence[rows], kind='stable')][:cap]
        selected = ranked[np.cumsum(costs[ranked]) <= budget]
        route[selected] = True
        limits[int(fold)] = dict(rows=len(rows), row_limit=cap, position_limit=budget,
                                 selected=len(selected), estimated_positions=int(costs[selected].sum()))
    return route, limits


def fit_heads(x, target, folds, glm):
    logp = np.empty((len(target), 18))
    dedicated = np.zeros_like(logp)
    y3 = np.where(target == glm[0], 1, np.where(target == glm[1], 2, 0))
    for fold in sorted(set(folds)):
        held = folds == fold
        scale = StandardScaler().fit(x[~held])
        learn, test = scale.transform(x[~held]), scale.transform(x[held])
        logp[held] = fit_component(learn, target[~held], test, 18)
        lg3 = fit_component(learn, y3[~held], test, 3)
        dedicated[np.ix_(held, glm)] = lg3[:, 1:3] - lg3[:, [0]] + np.log(16.)
        print('Finished prompt fold', int(fold), flush=True)
    other = np.ones(18, dtype=bool)
    other[glm] = False
    marginalized = np.zeros_like(logp)
    marginalized[:, glm] = logp[:, glm] - logsumexp(logp[:, other], axis=1, keepdims=True) + np.log(16.)
    return logp, dedicated, marginalized


def fuse(base, folds, route, logp, correction, alpha):
    if alpha == .25:
        return combine_scores(base, folds, route, logp, correction)
    boost = np.zeros_like(base)
    for fold in sorted(set(folds)):
        whole = folds == fold
        chosen = whole & route
        w18 = 1.5 / match_temperature(base[whole], .5692304418233374)
        w3 = 1.2 / match_temperature(base[whole], .90)
        boost[chosen] = (1 - alpha) * w18 * logp[chosen] + alpha * w3 * correction[chosen]
    return base + boost


def route_report(rows, route, costs, details, control):
    return dict(coverage=float(route.mean()), selected=int(route.sum()),
                estimated_positions=int(costs[route].sum()), short_selected=int((route & rows.n_words.lt(100)).sum()),
                selected_by_label=rows.loc[route, 'model'].value_counts().to_dict(),
                added_rows=int((route & ~control).sum()), removed_rows=int((control & ~route).sum()), folds=details)


def run(official, native, features, reference, output, draws=4000):
    train, labels, folds, base = load_native_oof(official, native)
    _, x115, token_count = load_features(features, train.index)
    glm = pd.Index(labels).get_indexer(GLM)
    target = pd.Index(labels).get_indexer(train.model)
    costs = token_count + 30
    control = select_original_route(base, folds, glm, costs)
    previous = read_ids(reference).loc[train.index]
    rows = train[['model', 'prompt_id']].assign(n_words=train.text.str.split().str.len(), fold=folds,
                                               baseline=labels[base.argmax(axis=1)])
    for column in ('model', 'prompt_id', 'n_words', 'fold', 'baseline'):
        if not np.array_equal(previous[column], rows[column]):
            raise ValueError(f'The archived Kiro control disagrees in {column}')
    if (int(control.sum()) != 587 or int(costs[control].sum()) != 87409
            or not np.array_equal(control, previous.routed)):
        raise ValueError('Expected the exact archived 587-row, 87,409-position control route')
    routes = {'original': control}
    limits = {'original': {int(f): dict(rows=int((folds == f).sum()), row_limit=int((folds == f).sum() * .30),
                                       position_limit=130000 * int((folds == f).sum()) / 4000,
                                       selected=int((control & (folds == f)).sum()),
                                       estimated_positions=int(costs[control & (folds == f)].sum()))
                           for f in sorted(set(folds))}}
    for name, multiplier in (('same_positions', 1.), ('more_positions', 1.5)):
        routes[name], limits[name] = select_route(base, folds, glm, costs, control, multiplier)
    for name, route in routes.items():
        rows[f'routed_{name}'] = route
    print('Routes', {name: (int(route.sum()), int(costs[route].sum())) for name, route in routes.items()}, flush=True)

    logp, dedicated, marginalized = fit_heads(x115, target, folds, glm)
    report = dict(rows=len(rows), position_estimator='max(pb115 n_tok across two models) + 30', route={}, arms={}, scopes={})
    comparisons = []
    for name, route in routes.items():
        report['route'][name] = route_report(rows, route, costs, limits[name], control)
        suffix = '' if name == 'original' else '_' + name
        for head, correction, alpha in (('dedicated25', dedicated, .25), ('marginal25', marginalized, .25),
                                        ('marginal35', marginalized, .35)):
            arm = head + suffix
            rows[arm] = labels[fuse(base, folds, route, logp, correction, alpha).argmax(axis=1)]
            if not rows.loc[~route, arm].equals(rows.loc[~route, 'baseline']):
                raise ValueError(f'Unrouted predictions changed in {arm}')
            report['arms'][arm] = dict(route=name, alpha=alpha, head=head[:-2])
            comparisons.append(('baseline', arm))
            if suffix:
                comparisons.append((head, arm))
        comparisons.extend([('dedicated25' + suffix, 'marginal25' + suffix),
                            ('marginal25' + suffix, 'marginal35' + suffix),
                            ('dedicated25' + suffix, 'marginal35' + suffix)])
    if not np.array_equal(rows.dedicated25, previous.pb115_hybrid):
        raise ValueError('The fitted control disagrees with archived standalone PB115 hybrid decisions')
    report['archive_exact_prediction_match'] = True

    short = rows.n_words.lt(100)
    focal = rows.model.isin(FOCAL)
    for scope, piece in (('all', rows), ('short', rows[short]), ('glm_fable', rows[focal]),
                         ('short_glm_fable', rows[short & focal])):
        correct = {arm: piece[arm].eq(piece.model) for arm in ('baseline', *report['arms'])}
        report['scopes'][scope] = dict(rows=len(piece), counts=piece.model.value_counts().to_dict(),
                                      comparisons=[paired_stats(piece, a, b, draws) for a, b in comparisons],
                                      correct_by_label={arm: ok.groupby(piece.model).sum().to_dict()
                                                        for arm, ok in correct.items()})
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
