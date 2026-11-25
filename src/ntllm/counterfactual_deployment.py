"""Apply a fitted stage-only candidate selector to the existing final decisions."""

import numpy as np
import pandas as pd

from .counterfactual_native_guard import guard_frame
from .counterfactual_selection import build_actions, replay_scoring_model


POLICIES = ('native_guard', 'strict_fixed', 'typography_fixed', 'open')
FABLE_AUDIT_POLICY = 'native_and_fable'


def score_predictions(queries, trace, incumbent, receipt, model, policy):
    if policy not in POLICIES and policy != FABLE_AUDIT_POLICY:
        raise ValueError('Unknown decision protection policy')
    if receipt.get('status') != 'applied':
        return np.asarray(incumbent).copy()
    if list(queries) != ['id', 'text']:
        raise ValueError('Deployment queries must contain only passage ID and body')
    body = queries.set_index('id', verify_integrity=True).text.fillna('').astype(str)
    rows = body.to_frame().assign(n_words=body.str.split().str.len())
    if model['label_order'] != list(trace['labels']):
        raise ValueError('The runtime and fitted capsule use different class orders')
    actions = build_actions(rows, trace, incumbent)
    if 'outcome' in actions:
        raise ValueError('An unlabeled deployment must not contain outcomes')
    results = replay_scoring_model(actions, model, rows.index)
    with_guard = guard_frame(rows, trace, incumbent, results.rename(columns={'prediction': 'open'}), receipt=receipt)
    return with_guard[policy].to_numpy()
