"""Veto fixed proposals using the original native GLM comparison."""

import json

import numpy as np
import pandas as pd

from .counterfactual_selection import FABLE, SPACED_DASH
from .glm_prefix_response_replay import GLM, restored_route
from .sibling_odds_replay import replay_predictions


TOLERANCE = 1e-8


def verify_route(folder, truth, trace, baseline):
    """Return the completed native route and its exact observed blend weight."""
    receipt = json.loads((folder / 'glm_report.json').read_text())
    return verify_route_receipt(receipt, truth, trace, baseline)


def verify_route_receipt(receipt, truth, trace, baseline):
    ids = pd.Index(trace['ids'], name=truth.index.name)
    if not ids.equals(truth.index) or not ids.is_unique:
        raise ValueError('The complete trace and truth must share ordered passage IDs')
    if isinstance(baseline, pd.Series) and not baseline.index.equals(ids):
        raise ValueError('The baseline series does not follow the complete trace order')
    if receipt.get('n_texts', len(ids)) != len(ids):
        raise ValueError('The native GLM receipt covers a different query count')
    if not np.array_equal(replay_predictions(trace, truth.text), np.asarray(baseline)):
        raise ValueError('The unmodified replay must reproduce the original calls')
    route, weight = restored_route(trace, truth.text, baseline, receipt)
    if route.any() and (not np.isfinite(weight) or weight <= 0.):
        raise ValueError('A completed likelihood route requires a positive finite blend weight')
    if receipt.get('n_blended', 0) != int(route.sum()):
        raise ValueError('Completed GLM route count differs from the saved receipt')
    return route, weight


def guard_frame(truth, trace, baseline, fitted, folder=None, *, receipt=None):
    """Keep or veto each already thresholded open call in its original row order."""
    route, weight = (
        verify_route(folder, truth, trace, baseline) if receipt is None
        else verify_route_receipt(receipt, truth, trace, baseline)
    )
    row = truth.index.get_indexer(fitted.index)
    if not fitted.index.is_unique or (row < 0).any():
        raise ValueError('Fixed proposals must cover unique IDs in the complete trace')
    labels = pd.Index(trace['labels'])
    anchor = np.asarray(baseline)[row]
    opened = fitted['open'].to_numpy()
    incumbent = labels.get_indexer(anchor)
    candidate = labels.get_indexer(opened)
    if (incumbent < 0).any() or (candidate < 0).any():
        raise ValueError('A fixed proposal or incumbent is outside the traced classes')
    if 'baseline' in fitted and not np.array_equal(fitted.baseline, anchor):
        raise ValueError('Fixed proposals refer to a different incumbent')

    selected = route[row]
    post = np.asarray(trace['glm_logp'], dtype=np.float64)
    pre = np.asarray(trace['stack_logp'], dtype=np.float64)
    increment = ((post[row, incumbent] - post[row, candidate])
                 - (pre[row, incumbent] - pre[row, candidate]))
    weighted_increment = np.where(selected, increment, 0.)
    relative_logodds = weighted_increment / weight if weight > 0. else weighted_increment
    support = selected & (relative_logodds > TOLERANCE)

    proposed_change = opened != anchor
    text = truth.text.iloc[row].fillna('').astype(str)
    typographic_before = replay_predictions(trace, truth.text, typography=False)[row]
    prior_typography = typographic_before != anchor
    family = np.array([label.split('-', 1)[0] for label in opened])
    invalid = text.str.contains(' — ', regex=False).to_numpy() & ~np.isin(opened, SPACED_DASH)
    invalid |= text.str.contains('’', regex=False).to_numpy() & np.isin(family, ('claude', 'glm'))
    invalid |= text.str.contains('—', regex=False).to_numpy() & (family == 'gemini')
    typography = prior_typography | invalid
    fable = np.asarray(trace['fable_guard'], dtype=bool)[row]
    glm_winner = np.asarray(trace['glm_winner'], dtype=bool)[row]
    incumbent_glm = np.isin(anchor, GLM)
    native_veto = (incumbent_glm | glm_winner) & support
    native_or_fable = (anchor == FABLE) | fable | native_veto

    return pd.DataFrame(dict(
        baseline=anchor,
        open=opened,
        typography_fixed=np.where(typography, anchor, opened),
        fable_typography_fixed=np.where(typography | fable, anchor, opened),
        native_guard=np.where(typography | fable | native_veto, anchor, opened),
        native_and_fable=np.where(native_or_fable, anchor, opened),
        strict_fixed=np.where(typography | fable | glm_winner, anchor, opened),
        route=selected,
        likelihood_support_incumbent=support,
        explicit_fable=fable,
        incumbent_glm=incumbent_glm,
        glm_stage_winner=glm_winner,
        original_typography_change=prior_typography,
        proposed_typography_violation=invalid,
        typography_block=typography,
        typography_veto=proposed_change & typography,
        fable_veto=proposed_change & fable,
        native_veto=proposed_change & native_veto,
        proposed_change=proposed_change,
        relative_logodds=relative_logodds,
        weighted_logodds_increment=weighted_increment,
        native_blend_weight=weight,
    ), index=fitted.index)
