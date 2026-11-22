"""Transfer same-prompt Claude function-word transitions to official short answers."""

import json
from collections import Counter
from itertools import combinations
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from scipy.special import log_softmax, logsumexp
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from likelihood_refine import transfer_glm_posterior
from likelihood_train import GLM, normalize_temperature


CLAUDE = ('claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5')
ANCHORS = frozenset('a an the and but or yet nor if then though although because while whereas as when where '
                    'which that who whose what how why whether i me my we us our you your they them their he him his '
                    'she her it its this these those here there to at in of by on for with from not no can could may '
                    'might must will would shall should be is are was were been being am do does did have has had '
                    'so too only also even more than rather just still thus once both each any however instead'.split())
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def transitions(text, crop=False):
    tokens = WORD.findall(str(text).lower().replace('’', "'"))
    if crop and len(tokens) > 80:
        start = (len(tokens) - 80) // 2
        tokens = tokens[start:start + 80]
    anchors = [(token, i) for i, token in enumerate(tokens) if token in ANCHORS]
    counts = Counter(f'{left}>{min(4, j - i - 1)}>{right}' for (left, i), (right, j) in zip(anchors, anchors[1:]))
    return {key: count / max(1, len(anchors) - 1) for key, count in counts.items()}


def input_path(root, pattern):
    return next(root.glob(pattern))


def official_scores(root):
    train = pd.read_csv(input_path(root, '**/name-that-llm/train.csv'), dtype={'id': str, 'prompt_id': str})
    labels = sorted(train.model.unique())
    columns = [f'reference__{label}' for label in labels]
    oof = pd.read_parquet(input_path(root, '**/glm_dynamics_oof.parquet'), columns=['id', 'fold', *columns])
    oof['id'] = oof.id.astype(str)
    with np.load(input_path(root, '**/glm_user_prefix_oof.npz'), allow_pickle=False) as saved:
        ids, folds = saved['ids'].astype(str), saved['folds']
        scores, costs = saved['low_user_160_logp'], saved['low_user_160_token_positions']
    frame = train.merge(oof, on='id', validate='one_to_one').set_index('id').loc[ids]
    if (len(frame) != len(train) or not np.array_equal(frame.fold.to_numpy(), folds)
            or frame.groupby('prompt_id').fold.nunique().max() != 1):
        raise ValueError('The saved GLM and official prompt folds differ')
    base = log_softmax(frame[columns].to_numpy(), axis=1)
    mass = np.exp(base[:, [labels.index(label) for label in GLM]]).sum(1)
    weight = 1 / normalize_temperature(base)
    for fold in np.unique(folds):
        rows = np.flatnonzero(folds == fold)
        order = rows[np.argsort(-mass[rows], kind='stable')][:int(0.30 * len(rows))]
        keep = order[np.cumsum(costs[order]) <= 130_000 * len(rows) / 4000]
        base[keep] = transfer_glm_posterior(base[keep], base[keep] + weight * scores[keep], labels)
    return frame, np.asarray(labels), base


def external_pairs(root, official):
    path = input_path(root, '**/ntl_extra_passages_claude.csv')
    ext = pd.read_csv(path, usecols=['prompt_id', 'text', 'model'], dtype={'prompt_id': str})
    ext = ext.loc[ext.model.isin(CLAUDE)].dropna(subset=['text', 'prompt_id'])
    normalized = lambda texts: texts.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    keep = ~normalized(ext.text).isin(normalized(official.text.fillna('')))
    ext = ext.loc[keep].reset_index(drop=True)
    groups = ext.assign(row=np.arange(len(ext))).pivot(index='prompt_id', columns='model', values='row')
    return ext, groups


def fit_pair_directions(ext, groups, frame, labels, short):
    vector = DictVectorizer()
    x = vector.fit_transform(transitions(text, crop=True) for text in ext.text)
    frequent = np.asarray((x > 0).sum(axis=0)).ravel() >= 20
    x = x[:, frequent]
    target_x = vector.transform(transitions(text) for text in frame.text.fillna(''))[:, frequent]
    effect = np.zeros((len(frame), len(labels)))
    metrics = {}
    for left, right in combinations(CLAUDE, 2):
        rows = groups[[left, right]].dropna().to_numpy(dtype=int)
        diff = x[rows[:, 0]] - x[rows[:, 1]]
        scale = 1 / np.maximum(np.sqrt(np.asarray(diff.power(2).mean(axis=0)).ravel()), 0.02)
        diff = diff.multiply(scale).tocsr()
        head = LogisticRegression(C=0.05, fit_intercept=False, solver='liblinear', max_iter=150)
        head.fit(vstack((diff, -diff)), np.repeat([1, 0], len(rows)),
                 sample_weight=np.full(2 * len(rows), 150 / len(rows)))
        raw = head.decision_function(target_x.multiply(scale))
        centered = raw.copy()
        for fold in sorted(frame.fold.unique()):
            fit = frame.fold.ne(fold).to_numpy() & short
            midpoint = np.mean([raw[fit & frame.model.eq(label).to_numpy()].mean() for label in (left, right)])
            centered[frame.fold.eq(fold)] -= midpoint
        i, j = np.flatnonzero(labels == left)[0], np.flatnonzero(labels == right)[0]
        evidence = np.clip(centered, -6, 6) / len(CLAUDE)
        effect[:, i] += evidence
        effect[:, j] -= evidence
        selected = short & frame.model.isin([left, right]).to_numpy()
        metrics[f'{left}/{right}'] = {
            'external_pairs': len(rows), 'official_short_rows': int(selected.sum()),
            'official_short_auc': float(roc_auc_score(frame.model.eq(left).to_numpy()[selected], centered[selected])),
        }
    return effect, metrics, int(frequent.sum())


def outcome(truth, prediction, original, rows):
    classes = np.unique(truth[rows])
    by_class = [rows & (truth == label) for label in classes]
    return {'rows': int(rows.sum()),
            'balanced_accuracy': float(np.mean([(prediction[keep] == truth[keep]).mean() for keep in by_class])),
            'base_balanced_accuracy': float(np.mean([(original[keep] == truth[keep]).mean() for keep in by_class])),
            'changed': int((rows & (prediction != original)).sum()),
            'fixed': int((rows & (prediction == truth) & (original != truth)).sum()),
            'harm': int((rows & (prediction != truth) & (original == truth)).sum())}


def main():
    root = Path('/kaggle/input')
    frame, labels, base = official_scores(root)
    ext, groups = external_pairs(root, frame)
    short = frame.text.fillna('').str.split().str.len().to_numpy() < 100
    truth = frame.model.to_numpy()
    score, pair_metrics, features = fit_pair_directions(ext, groups, frame, labels, short)
    top = np.argsort(-base, axis=1, kind='stable')[:, :3]
    family = np.isin(labels[top], CLAUDE)
    eligible = short & family[:, 0] & (family.sum(1) >= 2)
    original = labels[top[:, 0]]
    prediction = original.copy()
    for row in np.flatnonzero(eligible):
        allowed = top[row, family[row]]
        z = base[row].copy()
        z[allowed] = logsumexp(z[allowed]) + log_softmax(z[allowed] + 0.15 * score[row, allowed])
        if z.argmax() in allowed:
            prediction[row] = labels[z.argmax()]

    claude = np.isin(truth, CLAUDE)
    missed = short & claude & (truth != original)
    reachable = eligible & missed & (labels[top] == truth[:, None]).any(1)
    summary = {
        'external_rows': len(ext), 'external_prompts': len(groups), 'features': features, 'weight': 0.15,
        'all': outcome(truth, prediction, original, np.ones(len(truth), dtype=bool)),
        'claude_short': outcome(truth, prediction, original, short & claude),
        'short_eligible': int(eligible.sum()), 'short_claude_misses': int(missed.sum()),
        'short_claude_misses_with_nonclaude_winner': int((missed & ~family[:, 0]).sum()),
        'short_claude_misses_reachable': int(reachable.sum()),
        'by_fold': {str(fold): outcome(truth, prediction, original, short & claude & frame.fold.eq(fold).to_numpy())
                    for fold in sorted(frame.fold.unique())},
        'short_recall_by_class': {label: outcome(truth, prediction, original, short & (truth == label)) for label in CLAUDE},
        'pairs': pair_metrics,
    }
    report = json.dumps(summary, indent=2)
    Path('/kaggle/working/claude_anchor_pairs_metrics.json').write_text(report + '\n')
    print(report, flush=True)


if __name__ == '__main__':
    main()
