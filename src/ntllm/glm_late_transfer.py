"""Replay budget rollbacks within the completed archived Sibling Odds route."""

import json

import numpy as np
import pandas as pd
from scipy.special import log_softmax

from .counterfactual_native_guard import verify_route
from .glm_prefix_response_replay import FABLE, GLM, with_glm_scores
from .member_evidence_transfer import read_capture, read_prvsiyan
from .peer_context_report import paired_stats, read_ids
from .sibling_odds_replay import replay_predictions


FRACTIONS = {'keep_two_thirds': (2, 3), 'keep_three_quarters': (3, 4)}
FOCAL = (*GLM, FABLE)
RESTORES = {'restore_glm': GLM, 'restore_focal': FOCAL}


def replay_budget(truth, capture):
    trace, baseline, _, _, _ = read_capture(capture, truth)
    completed, weight = verify_route(capture, truth, trace, baseline)
    if not completed.any():
        raise ValueError('Expected a completed archived GLM route')
    labels = pd.Index(trace['labels'])
    pre = trace['stack_logp']
    post = trace['glm_logp']
    mass = np.exp(log_softmax(pre.astype(np.float64), axis=1)[:, labels.get_indexer(GLM)]).sum(axis=1)
    scored = np.flatnonzero(completed)
    ranked = scored[np.argsort(-mass[scored], kind='stable')]
    rank = np.zeros(len(truth), dtype=int)
    rank[ranked] = np.arange(1, len(ranked) + 1)
    prediction = np.asarray(baseline)
    rows = truth[['model', 'prompt_id', 'cohort', 'n_words']].assign(
        strict=truth.get('strict_topic_holdout', False),
        archived=prediction, completed_route=completed, route_rank=rank,
        pre_glm_mass=mass, pre_glm_winner=labels.to_numpy()[pre.argmax(axis=1)])
    if not rows.n_words.eq(truth.text.str.split().str.len()).all():
        raise ValueError('Short-text length must agree with the archived text')
    route_details = dict(rows=len(rows), completed=int(completed.sum()), original_blend_weight=weight, budgets={})
    for budget, (numerator, denominator) in FRACTIONS.items():
        kept = len(ranked) * numerator // denominator
        late = rank > kept
        if (late & ~completed).any() or int(late.sum()) != len(ranked) - kept:
            raise ValueError('Expected a ranked suffix of the completed native route')
        # Restore the exact original scores for selected rows and rerun the original sibling and typography rules.
        withheld = {'rollback': late, 'restore_all': np.zeros(len(rows), dtype=bool)}
        for action, allowed in RESTORES.items():
            withheld[action] = late & ~np.isin(prediction, allowed)
        rows[f'{budget}__late'] = late
        for action, drop in withheld.items():
            updated = post.copy()
            updated[drop] = pre[drop]
            replay = replay_predictions(with_glm_scores(trace, updated), truth.text)
            if not np.array_equal(replay[~drop], prediction[~drop]):
                raise ValueError(f'{budget} {action} changed a row with unchanged likelihood evidence')
            column = f'{budget}__{action}'
            rows[column] = replay
            if action != 'rollback' and not np.array_equal(replay[drop], rows[f'{budget}__rollback'].to_numpy()[drop]):
                raise ValueError(f'{budget} {action} changed a row where the gate withheld its original evidence')
        if not rows[f'{budget}__restore_all'].equals(rows.archived):
            raise ValueError('Restoring all completed score rows must reproduce the exact frozen decisions')
        route_details['budgets'][budget] = dict(kept=kept, late=int(late.sum()))
    return rows, route_details


def per_label(rows, before, after):
    old = rows[before].eq(rows.model)
    new = rows[after].eq(rows.model)
    summary = rows.assign(fixed=~old & new, harmed=old & ~new, before=old, after=new)
    return summary.groupby('model')[['fixed', 'harmed', 'before', 'after']].sum().to_dict('index')


def scope_report(rows, draws):
    detail = dict(rows=len(rows), label_count=rows.model.value_counts().to_dict(),
                  completed=int(rows.completed_route.sum()), budgets={})
    for budget in FRACTIONS:
        columns = {a: f'{budget}__{a}' for a in ('rollback', 'restore_all', *RESTORES)}
        late = rows[f'{budget}__late']
        comparisons = [('archived', columns['rollback'])]
        comparisons.extend((columns['rollback'], columns[a]) for a in ('restore_all', *RESTORES))
        comparisons.extend(('archived', columns[a]) for a in RESTORES)
        comparisons.append((columns['restore_focal'], columns['restore_glm']))
        detail['budgets'][budget] = dict(
            retained=int((rows.completed_route & ~late).sum()), late=int(late.sum()),
            late_archived_predicted_class=rows.loc[late, 'archived'].value_counts().to_dict(),
            late_changed_from_rollback=int(rows[columns['rollback']].ne(rows.archived).sum()),
            paired=[paired_stats(rows, a, b, draws) for a, b in comparisons],
            by_label={a: per_label(rows, columns['rollback'], columns[a]) for a in ('restore_all', *RESTORES)},
        )
    return detail


def run(historical, historical_truth, independent, source, prior_index, output, draws=4000):
    public = read_ids(historical_truth / 'public_truth.parquet')
    held = read_prvsiyan(source, prior_index)
    if not public.index.intersection(held.index).empty:
        raise ValueError('Historical and independent passages must be disjoint')
    public_rows, public_route = replay_budget(public, historical)
    held_rows, held_route = replay_budget(held, independent)
    c2 = public_rows.loc[public_rows.cohort.eq('evaluation')]
    if len(c2) != 300 or c2.model.isin(GLM).any():
        raise ValueError('Expected the frozen C2 harm check without true GLM labels')
    scopes = {'c2': c2, 'prvsiyan_895': held_rows, 'prvsiyan_316': held_rows.loc[held_rows.strict]}
    report = dict(ranking='normalized pre-GLM GLM + Flash mass, ties in archived row order',
                  budget_unit='fraction of each archived batch’s completed score rows',
                  historical_route=public_route, independent_route=held_route, scopes={})
    for name, panel in scopes.items():
        parts = {'all': panel, 'short': panel.loc[panel.n_words.lt(100)]}
        report['scopes'][name] = {subset: scope_report(piece, draws) for subset, piece in parts.items()}
    output.mkdir(parents=True, exist_ok=True)
    rows = pd.concat([c2, held_rows], verify_integrity=True)
    rows.to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Historical and independent archived route counts', public_route, held_route, flush=True)
