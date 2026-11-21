"""Compare frozen Native sibling readouts on exact-route same-prompt pairs."""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import log_softmax, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from native_q8 import DEPTHS, INSTRUCTION, NativeQ8, windows
from sibling_pair_data import GEMINI_FLASH, GPT56, load_sibling_panels


GROUPS = {'gemini_flash': GEMINI_FLASH, 'gpt56': GPT56}
TRAJECTORY_DEPTHS = (6, 18, 30)
VIEWS = ('last_token', 'trajectory')
SCORE_SOURCE = 'family_mass_glm_posterior'
WEIGHT = 0.15


def official_scores(path, oof):
    original = pd.read_csv(path, dtype={'id': str, 'prompt_id': str})
    labels = sorted(original.model.unique())
    columns = [f'{SCORE_SOURCE}__{label}' for label in labels]
    scores = pd.read_parquet(oof, columns=['id', 'prompt_id', 'model', 'fold', *columns])
    scores[['id', 'prompt_id']] = scores[['id', 'prompt_id']].astype(str)
    frame = original.merge(scores, on=['id', 'prompt_id', 'model'], validate='one_to_one')
    if len(frame) != len(original) or frame.fold.nunique() != 5 or frame.groupby('prompt_id').fold.nunique().max() != 1:
        raise ValueError('Expected the existing five prompt-disjoint OOF folds')
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('The saved GLM posterior scores must be finite')
    return frame, labels, log_softmax(values, axis=1)


@torch.inference_mode()
def window_views(expert, seqs, indices, prompt_tokens):
    batch = expert.tokenizer.pad({'input_ids': [seqs[i] for i in indices]}, padding=True,
                                 return_tensors='pt').to(expert.head_device)
    out = expert.model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
    last = batch['attention_mask'].sum(1) - 1
    last_token, trajectory = [], []
    for depth, norm in zip(DEPTHS, expert.norms):
        state = out.last_hidden_state if depth == expert.layers else out.hidden_states[depth]
        rows = torch.arange(len(indices), device=state.device)
        position = last.to(state.device)
        pooled = state[rows, position].float()
        if depth == expert.layers:
            pooled = F.normalize(pooled, dim=-1)
        last_token.append(norm(pooled.to(expert.head_device)))
        if depth in TRAJECTORY_DEPTHS:
            # Pool thirds of the body only, excluding the task instruction and end token.
            token = torch.arange(state.shape[1], device=state.device)[None]
            n_body = (position - prompt_tokens).clamp_min(1)
            body = (token >= prompt_tokens) & (token < position[:, None])
            segment = ((token - prompt_tokens) * 3 // n_body[:, None]).clamp(0, 2)
            mask = torch.stack([body & segment.eq(i) for i in range(3)], dim=1).float()
            means = torch.bmm(mask, state.float()) / mask.sum(dim=-1, keepdim=True).clamp_min(1)
            means = norm(means.to(expert.head_device))
            trajectory.extend((means[:, 1] - means[:, 0], means[:, 2] - means[:, 1]))
    return {'last_token': torch.cat(last_token, dim=1).cpu().numpy(),
            'trajectory': torch.cat(trajectory, dim=1).cpu().numpy()}


def representations(expert, texts, batch_size):
    start = time.monotonic()
    seqs, owners, weights = windows(expert.tokenizer, texts)
    prefix_length = len(expert.tokenizer(INSTRUCTION, add_special_tokens=False)['input_ids'])
    result = {'last_token': np.zeros((len(texts), len(DEPTHS) * expert.hidden), dtype=np.float32),
              'trajectory': np.zeros((len(texts), 2 * len(TRAJECTORY_DEPTHS) * expert.hidden), dtype=np.float32)}
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    offset, batch = 0, batch_size
    while offset < len(order):
        indices = order[offset:offset + batch]
        try:
            values = window_views(expert, seqs, indices, prefix_length)
        except torch.cuda.OutOfMemoryError:
            if batch == 1:
                raise
            batch = (batch + 1) // 2
        else:
            for name, features in values.items():
                np.add.at(result[name], owners[indices], features * weights[indices, None].astype(np.float32))
            offset += len(indices)
            continue
        torch.cuda.empty_cache()
    denom = np.bincount(owners, weights=weights, minlength=len(texts)).astype(np.float32)
    for features in result.values():
        features /= denom[:, None]
        if not np.isfinite(features).all():
            raise ValueError('Non-finite frozen Native features')
    return result, {'rows': len(texts), 'windows': len(seqs), 'seconds': round(time.monotonic() - start, 1)}


def pair_indices(frame, left, right):
    indexed = frame[['prompt_id', 'model']].assign(row=np.arange(len(frame)))
    pairs = indexed.loc[indexed.model.isin((left, right))].pivot(index='prompt_id', columns='model', values='row')
    return pairs.reindex(columns=[left, right]).dropna().to_numpy(dtype=int)


def fit_pair(features, pairs):
    first, second = features[pairs[:, 0]], features[pairs[:, 1]]
    difference = first - second
    scale = np.sqrt(np.mean(difference ** 2, axis=0)).clip(0.1)
    midpoint = (first.mean(axis=0) + second.mean(axis=0)) / 2
    standardized = np.clip(difference / scale, -6, 6)
    oriented = np.concatenate((standardized, -standardized))
    target = np.repeat([1, 0], len(pairs))
    loss_weight = np.full(len(target), 150 / len(pairs))
    head = LogisticRegression(C=0.05, fit_intercept=False, solver='liblinear', max_iter=150)
    head.fit(oriented, target, sample_weight=loss_weight)
    return head.coef_[0].astype(np.float32), midpoint, scale


def evidence(features, head):
    coef, midpoint, scale = head
    return np.clip((features - midpoint) / scale, -6, 6) @ coef


def pair_report(frame, left, right, odds):
    relevant = frame.model.isin((left, right)).to_numpy()
    pairs = pair_indices(frame, left, right)
    gap = odds[pairs[:, 0]] - odds[pairs[:, 1]]
    return {'rows': int(relevant.sum()), 'auc': float(roc_auc_score(frame.model.to_numpy()[relevant] == left, odds[relevant])),
            'matched_prompts': len(pairs), 'matched_order_accuracy': float(np.mean((gap > 0) + 0.5 * (gap == 0)))}


def top_group(logp, labels):
    top = np.argsort(-logp, axis=1, kind='stable')[:, :3]
    groups = {}
    for track, siblings in GROUPS.items():
        group = np.array([labels.index(label) for label in siblings])
        match = np.isin(top, group)
        groups[track] = match[:, 0] & (match.sum(axis=1) >= 2)
    return top, groups


def rerank(logp, effects, top, active, labels):
    revised = logp.copy()
    original = top[:, 0]
    for track, siblings in GROUPS.items():
        group = [labels.index(label) for label in siblings]
        for row in np.flatnonzero(active[track]):
            allowed = top[row, np.isin(top[row], group)]
            scores = logp[row, allowed]
            revised[row, allowed] = logsumexp(scores) + log_softmax(scores + WEIGHT * effects[row, allowed])
            if revised[row].argmax() not in allowed:
                revised[row] = logp[row]
    return revised.argmax(axis=1)


def balanced_accuracy(y, predicted):
    return float(np.mean([(y[y == label] == predicted[y == label]).mean() for label in np.unique(y)]))


def outcome(y, old, new):
    return {'balanced_accuracy': balanced_accuracy(y, new),
            'delta_ba': balanced_accuracy(y, new) - balanced_accuracy(y, old),
            'changed': int((new != old).sum()), 'fixed': int(((old != y) & (new == y)).sum()),
            'harmed': int(((old == y) & (new != y)).sum())}


def outcomes(frame, labels, top, active, predictions):
    y = pd.Categorical(frame.model, categories=labels).codes
    old = top[:, 0]
    words = frame.text.str.split().str.len().to_numpy()
    results = {}
    for name, new in predictions.items():
        results[name] = {
            **outcome(y, old, new),
            'length': {label: outcome(y[keep], old[keep], new[keep]) for label, keep in
                       (('<100', words < 100), ('100+', words >= 100))},
            'folds': {str(fold): outcome(y[keep], old[keep], new[keep]) for fold in sorted(frame.fold.unique())
                      if (keep := frame.fold.to_numpy() == fold).any()},
            'tracks': {track: {'eligible': int(active[track].sum()),
                               **outcome(y[keep], old[keep], new[keep])}
                       for track, siblings in GROUPS.items() if (keep := frame.model.isin(siblings).to_numpy()).any()},
        }
    return results


def evaluate(args):
    mount = Path('/kaggle/input')
    official, labels, logp = official_scores(args.train or next(mount.glob('**/name-that-llm/train.csv')),
                                            args.oof or next(mount.rglob('glm_dynamics_oof.parquet')))
    assets = args.assets or next(mount.rglob('ntl-native-q8-multidepth-fivefold-0679'))
    base = args.base or next(mount.glob('**/qwen-3-embedding/[Tt]ransformers/8b/1/config.json')).parent
    with np.load(assets / 'tokens.npz') as tok:
        indices = pd.Index(tok['ids'].astype(str)).get_indexer(official.id.to_numpy())
        if (indices < 0).any() or not np.array_equal(tok['folds'][indices], official.fold.to_numpy()):
            raise ValueError('Original Native adapter folds and input OOF folds do not align')
    external = load_sibling_panels(args.pool or next(mount.rglob('ntl_open_pool.parquet')), official.text,
                                   gemini_prompts=args.prompts_per_track, gpt_prompts=args.prompts_per_track)
    pair_labels = [(track, siblings, left, right) for track, siblings in GROUPS.items()
                   for left, right in combinations(siblings, 2)]
    external_pairs = {(left, right): pair_indices(external, left, right) for _, _, left, right in pair_labels}
    if any(len(rows) < 2 for rows in external_pairs.values()):
        raise ValueError('Missing true same-route external sibling pairs')
    top, active = top_group(logp, labels)
    for_features = np.logical_or.reduce(list(active.values())) | official.model.isin(GEMINI_FLASH + GPT56).to_numpy()
    effects = {name: np.zeros_like(logp) for name in VIEWS}
    odds = {name: {(left, right): np.zeros(len(official)) for _, _, left, right in pair_labels} for name in VIEWS}
    reference = {(left, right): logp[:, labels.index(left)] - logp[:, labels.index(right)]
                 for _, _, left, right in pair_labels}
    expert = NativeQ8(base, assets, '/kaggle/temp/ntl-native-frozen', log=lambda msg: print(msg, flush=True))
    progress, saved_heads = {}, {}
    for fold in sorted(official.fold.unique()):
        idx = np.flatnonzero((official.fold.to_numpy() == fold) & for_features)
        expert.load_fold(int(fold))
        all_texts = external.text.astype(str).tolist() + official.iloc[idx].text.astype(str).tolist()
        features, stats = representations(expert, all_texts, args.batch)
        with threadpool_limits(limits=2):
            for name, representation in features.items():
                for pair_index, (_, siblings, left, right) in enumerate(pair_labels):
                    pair = external_pairs[left, right]
                    head = fit_pair(representation[:len(external)], pair)
                    key = f'fold{fold}_{name}_pair{pair_index}'
                    for field, value in zip(('direction', 'midpoint', 'scale'), head):
                        saved_heads[f'{key}_{field}'] = value
                    values = evidence(representation[len(external):], head)
                    odds[name][left, right][idx] = values
                    values = np.clip(values, -6, 6) / len(siblings)
                    effects[name][idx, labels.index(left)] += values
                    effects[name][idx, labels.index(right)] -= values
        del features, representation
        progress[str(fold)] = stats
        print(f'fold {fold}: {stats}', flush=True)
    pairwise = {}
    for _, _, left, right in pair_labels:
        all_odds = {'original': reference[left, right], **{name: odds[name][left, right] for name in VIEWS}}
        pairwise[f'{left} vs {right}'] = {'external_pairs': len(external_pairs[left, right]),
                                        **{name: pair_report(official, left, right, values)
                                           for name, values in all_odds.items()}}
    predictions = {'original': top[:, 0], **{name: rerank(logp, effects[name], top, active, labels) for name in VIEWS}}
    result = {'comparison': SCORE_SOURCE, 'weight': WEIGHT,
              'external_prompts': external.groupby('track').prompt_id.nunique().to_dict(),
              'external_routes': external.groupby(['track', 'matched_route']).prompt_id.nunique().unstack(fill_value=0).to_dict('index'),
              'extraction': progress, 'metrics': outcomes(official, labels, top, active, predictions),
              'pairwise': pairwise,
              'head_layout': {'labels': labels, 'pair_order': [[left, right] for _, _, left, right in pair_labels],
                              'last_token_depth_order': DEPTHS, 'trajectory_depth_order': TRAJECTORY_DEPTHS,
                              'trajectory_parts_per_depth': ['middle_minus_early', 'late_minus_middle'],
                              'score': 'clip((features - midpoint) / scale, -6, 6) @ direction',
                              'intercept_in_centered_space': 0}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'native_frozen_siblings.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(args.output / 'native_frozen_sibling_heads.npz', **saved_heads)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'oof', 'base', 'assets', 'pool'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    parser.add_argument('--prompts-per-track', type=int, choices=range(2, 901), default=900, metavar='2-900')
    parser.add_argument('--batch', type=int, default=8)
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
