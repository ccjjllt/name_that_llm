"""Cross-family attribution from changes across segments of a passage."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Two recurring cross-family confusions in the project's prompt-grouped Q8 results.
PAIRS = (
    ('kimi-k3', 'gpt-5.5'),
    ('claude-fable-5', 'glm-5.3'),
    ('claude-fable-5', 'glm-5.3-flash'),
)


def pool_trajectory(hidden_states, attention, depths=(4, 12, 20, 24)):
    """Differences in unit-normalized early, middle and late hidden-state means."""
    import torch
    from torch.nn.functional import normalize

    # Keep token segmentation identical for left or right padding.
    positions = attention.cumsum(1) - 1
    valid = attention.bool() & (positions > 0)
    positions = (positions - 1).clamp_min(0)
    lengths = valid.sum(1, keepdim=True).clamp_min(1)
    thirds = (3 * positions // lengths).clamp_max(2)
    band_ids = torch.arange(3, device=attention.device)
    bands = valid[:, None, :] & (thirds[:, None, :] == band_ids[None, :, None])
    denominators = bands.sum(2, keepdim=True).clamp_min(1)
    weights = bands / denominators
    features = []
    for depth in depths:
        hidden = hidden_states[depth]
        means = torch.bmm(weights.to(hidden.device, dtype=hidden.dtype), hidden).float()
        means = normalize(means, dim=-1)
        change = means[:, 1:, :] - means[:, :-1, :]
        features.append(change.flatten(1).cpu().numpy().astype(np.float16))
    return np.concatenate(features, axis=1)


def _paired_rows(train, left, right):
    rows = train.reset_index(drop=True)[['prompt_id', 'model']].copy()
    rows['position'] = np.arange(len(rows))
    rows = rows[rows.model.isin((left, right))]
    groups = rows.pivot_table(index='prompt_id', columns='model', values='position', aggfunc='first')
    groups = groups.dropna(subset=[left, right])
    return groups[left].to_numpy(dtype=int), groups[right].to_numpy(dtype=int)


def pair_log_odds(train, train_features, query_features):
    """Estimate log odds for each left author using matched right-author answers."""
    scores = {}
    for left, right in PAIRS:
        lhs, rhs = _paired_rows(train, left, right)
        a = train_features[lhs].astype(np.float32)
        b = train_features[rhs].astype(np.float32)
        difference = a - b
        midpoint = ((a + b) * .5).mean(axis=0)
        scale = max(np.sqrt(np.square(difference).sum(1).mean()), 1e-6)
        difference /= scale
        head = LogisticRegression(C=2.0, fit_intercept=False, max_iter=200)
        head.fit(np.concatenate([difference, -difference]), np.repeat([1, 0], len(difference)))
        weight = (head.coef_[0] / scale).astype(np.float32)
        # Multiply the compact learned direction into the input in chunks so
        # that float16 stored arrays never require a full float32 copy.
        offset = midpoint @ weight
        scores[left, right] = np.concatenate([
            query_features[start:start + 256].astype(np.float32) @ weight - offset
            for start in range(0, len(query_features), 256)
        ])
    return scores


def rerank(logits, labels, current, pair_scores, *, strength=.65, max_margin=1.25):
    """Reconsider these pairs when they already lead the full ensemble."""
    labels = np.asarray(labels)
    logit_order = np.argsort(logits, axis=1)
    left_pred = labels[logit_order[:, -1]]
    right_pred = labels[logit_order[:, -2]]
    rows = np.arange(len(logits))
    margin = logits[rows, logit_order[:, -1]] - logits[rows, logit_order[:, -2]]
    result = np.array(current, dtype=object, copy=True)
    for (left, right), personal_margin in pair_scores.items():
        eligible = (left_pred == current) & (margin <= max_margin)
        move_right = eligible & (left_pred == left) & (right_pred == right)
        move_left = eligible & (left_pred == right) & (right_pred == left)
        result[move_right & (margin + strength * personal_margin < 0)] = right
        result[move_left & (margin - strength * personal_margin < 0)] = left
    return result


def refine_submission(train, query, output, logits_path, trajectory_dir):
    """Run the pair heads on artifacts from the ensemble's existing forwards."""
    with np.load(logits_path, allow_pickle=False) as saved:
        ids = saved['ids'].astype(str)
        labels = saved['labels'].astype(str)
        logits = saved['logits']
    current = pd.read_parquet(output)
    if not (np.array_equal(ids, query.id.astype(str)) and np.array_equal(ids, current.id.astype(str))):
        raise ValueError('Ensemble and trajectory rows disagree on submission order')
    root = Path(trajectory_dir)
    scores = {pair: np.zeros(len(query), dtype=np.float32) for pair in PAIRS}
    members = list(root.glob('*_train_trajectory.npy'))
    if not members:
        raise ValueError('No ensemble trajectory features were generated')
    for train_path in members:
        test_path = train_path.with_name(train_path.name.replace('_train_', '_test_'))
        train_features = np.load(train_path, mmap_mode='r')
        query_features = np.load(test_path, mmap_mode='r')
        for pair, values in pair_log_odds(train, train_features, query_features).items():
            scores[pair] += values / len(members)
    previous = current.model.to_numpy()
    revised = rerank(logits, labels, previous, scores)
    # The original final stage downweights Fable on spaced em dashes. Keep
    # that decision when the trajectory model would move a GLM label to Fable.
    prose = query.text.fillna('').str.contains(' — ', regex=False).to_numpy()
    typography = prose & (revised == 'claude-fable-5') & (previous != revised)
    revised[typography] = previous[typography]
    print('cross-family trajectory changes', int((revised != previous).sum()))
    current['model'] = revised
    current.to_parquet(output, index=False)
    return current
