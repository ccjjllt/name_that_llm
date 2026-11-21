"""Adapt the public Native 8B expert to matched prompts and sibling errors."""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import log_softmax
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from torch.nn import functional as F

from native_q8 import INSTRUCTION, TERMINAL, NativeQ8, calibrate
from sibling_pair_data import load_sibling_panels


CONFUSIONS = (
    ('gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'),
    ('gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra'),
    ('glm-5.3', 'glm-5.3-flash', 'claude-fable-5'),
    ('kimi-k3', 'gpt-5.5'),
    ('claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5'),
)


def matching_pool(path, official, labels):
    columns = ['id', 'text', 'model', 'prompt_canon', 'prompt_evidence', 'gen_reasoning',
               'flag_train_prompt_regen', 'flag_near_train_prompt', 'n_words']
    frame = pd.read_parquet(path, columns=columns).dropna(subset=['text', 'prompt_canon', 'model'])
    frame = frame.loc[
        frame.model.isin(labels) & frame.prompt_evidence.isin(['own_text', 'own_bank', 'other_text'])
        & ~frame.text.isin(official.text) & frame.n_words.between(40, 250)
        & ~frame.flag_train_prompt_regen.fillna(False) & ~frame.flag_near_train_prompt.fillna(False)
    ].rename(columns={'prompt_canon': 'prompt_id'})
    frame = frame.loc[frame.prompt_id.astype(str).str.len().gt(0)].copy()
    frame['priority'] = (frame.model.eq('kimi-k3') & frame.gen_reasoning.eq('high')).astype(int)
    frame['tie'] = np.random.default_rng(42).random(len(frame))
    frame = frame.sort_values(['priority', 'tie'], ascending=False).drop_duplicates(['prompt_id', 'model'])
    sizes = frame.groupby('prompt_id').size()
    groups = sizes.index[sizes >= 8]
    if len(groups) > 1_800:
        weights = sizes.loc[groups].to_numpy(dtype=float) ** 2
        groups = np.random.default_rng(17).choice(groups, 1_800, replace=False, p=weights / weights.sum())
    small = ['id', 'prompt_id', 'model', 'text']
    matched = load_sibling_panels(path, official.text)
    generic = frame.loc[frame.prompt_id.isin(groups) & ~frame.text.isin(matched.text), small].copy()
    generic['prompt_id'] = 'mixed:' + generic.prompt_id.astype(str)
    print(f'Trusted sibling pairs: {matched.groupby("track").prompt_id.nunique().to_dict()}', flush=True)
    return pd.concat([generic, matched[small]], ignore_index=True)


def examples(expert, frame):
    labels = {name: i for i, name in enumerate(expert.labels)}
    sequences = []
    text = frame.text.astype(str).tolist()
    for start in range(0, len(text), 512):
        encoded = expert.tokenizer([INSTRUCTION + t for t in text[start:start + 512]],
                                   add_special_tokens=False, truncation=True, max_length=383)['input_ids']
        sequences.extend(ids + [TERMINAL] for ids in encoded)
    target = frame.model.map(labels).to_numpy()
    episodes = []
    for rows in frame.groupby('prompt_id', sort=False).indices.values():
        if len(rows) < 2:
            continue
        by_class = {target[i]: i for i in rows}
        related = []
        for group in CONFUSIONS:
            near = [by_class[labels[name]] for name in group if labels[name] in by_class]
            if len(near) >= 2:
                related.append(np.asarray(near))
        episodes.append((rows, related))
    return sequences, target, episodes


def draw_episode(episodes, rng):
    rows, related = episodes[rng.integers(len(episodes))]
    count = min(4, len(rows))
    if related and rng.random() < 0.70:
        focus = rng.choice(related[rng.integers(len(related))], 2, replace=False)
        # Retain other labels in the prompt so learning a sibling need not discard family identity.
        others = rows[~np.isin(rows, focus)]
        return np.concatenate((focus, rng.choice(others, count - 2, replace=False)))
    return rng.choice(rows, count, replace=False)


def shorten(ids, prefix, rng):
    length = len(ids) - prefix - 1
    if length <= 100 or rng.random() > 0.33:
        return ids
    keep = int(rng.integers(75, min(length, 225)))
    start = int(rng.integers(0, length - keep + 1))
    return ids[:prefix] + ids[prefix + start:prefix + start + keep] + [TERMINAL]


def adaptation_parameters(expert, layers):
    selected = {}
    for name, parameter in expert.expected.items():
        match = re.search(r'\.layers\.(\d+)\.', name)
        enabled = match is None or int(match[1]) >= expert.layers - layers
        parameter.requires_grad_(enabled)
        if enabled:
            selected[name] = parameter
    return selected


def fit(expert, data, trainable, steps, lr, seed, phase):
    if not steps:
        return
    sequences, labels, episodes = data
    if not episodes:
        raise ValueError(f'No matched prompt episodes for {phase}')
    prefix = len(expert.tokenizer(INSTRUCTION, add_special_tokens=False)['input_ids'])
    confusion = torch.eye(len(expert.labels), device=expert.head_device, dtype=torch.bool)
    for group in CONFUSIONS:
        cols = torch.tensor([expert.labels.index(name) for name in group], device=expert.head_device)
        confusion[cols[:, None], cols] = True
    backbone = [param for name, param in trainable.items() if name.startswith('encoder.')]
    heads = [param for name, param in trainable.items() if not name.startswith('encoder.')]
    parameter_groups = [{'params': heads, 'lr': 4 * lr}]
    if backbone:
        parameter_groups.append({'params': backbone, 'lr': lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=0.08 * lr)
    scaler = torch.amp.GradScaler('cuda', init_scale=128)
    rng = np.random.default_rng(seed)
    expert.model.train()
    total = 0.0
    start = time.monotonic()
    for step in range(steps):
        rows = draw_episode(episodes, rng)
        tokens = [shorten(sequences[i], prefix, rng) for i in rows]
        y = torch.as_tensor(labels[rows], dtype=torch.long, device=expert.head_device)
        optimizer.zero_grad(set_to_none=True)
        features = expert._features(tokens, list(range(len(tokens))))
        plain = expert.plain(features)
        relative = expert.centered(features - features.mean(0))
        # Keep the 18-way objective while giving close siblings a direct gradient.
        restricted = (0.7 * plain + 0.3 * relative).masked_fill(~confusion[y], -torch.inf)
        loss = F.cross_entropy(plain, y) + 0.30 * F.cross_entropy(relative, y) + 0.40 * F.cross_entropy(restricted, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += float(loss.detach())
        if step in (0, steps - 1) or (step + 1) % 50 == 0:
            print(f'{phase}: {step + 1}/{steps}, loss {total / (step + 1):.4f}, '
                  f'{time.monotonic() - start:.0f}s', flush=True)
    expert.model.eval()


def report(frame, labels, predictions):
    scores = {}
    classes = frame.model.to_numpy()
    words = frame.text.str.split().str.len().to_numpy()
    for name, logits in predictions.items():
        guessed = np.asarray(labels)[logits.argmax(1)]
        scores[name] = {'balanced_accuracy': float(balanced_accuracy_score(classes, guessed)),
                        'recall': {c: float((guessed[classes == c] == c).mean()) for c in np.unique(classes)},
                        'short': float(balanced_accuracy_score(classes[words < 100], guessed[words < 100]))}
    return scores


def inferred_groups(train, query):
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.05, sublinear_tf=True)
    vectorizer.fit(train.tolist())
    x = vectorizer.transform(query.tolist())
    distance = (1 - (x @ x.T).toarray()).clip(0, 1)
    np.fill_diagonal(distance, 0)
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.85, metric='precomputed', linkage='average')
    return model.fit_predict(distance)


def study(args):
    from safetensors.torch import save_file

    mount = Path('/kaggle/input')
    official = pd.read_csv(args.train or next(mount.glob('**/name-that-llm/train.csv')),
                           dtype={'id': str, 'prompt_id': str})
    assets = args.assets or next(mount.rglob('ntl-native-q8-multidepth-fivefold-0679'))
    base = args.base or next(mount.glob('**/qwen-3-embedding/[Tt]ransformers/8b/1/config.json')).parent
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    expert = NativeQ8(base, assets, '/kaggle/temp/native-training', log=lambda message: print(message, flush=True))
    if args.last_layers:
        expert.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    selected = adaptation_parameters(expert, args.last_layers)
    external = matching_pool(args.pool or next(mount.rglob('ntl_open_pool.parquet')), official, expert.labels)
    external_examples = examples(expert, external)
    print(f'{len(external)} matched external answers, {external.prompt_id.nunique()} prompts, '
          f'{sum(p.numel() for p in selected.values()):,} trainable parameters', flush=True)
    with np.load(assets / 'tokens.npz') as data, np.load(assets / 'oof_reference.npz') as original:
        indexer = pd.Index(data['ids'].astype(str)).get_indexer(official.id)
        if (indexer < 0).any():
            raise ValueError('Public Native fold IDs do not match official train')
        folds = data['folds'][indexer]
        raw_plain = original['plain'][indexer]
        raw_center = original['centered'][indexer]
    sparse_rng = np.random.default_rng(112)
    sparse = np.sort(np.concatenate([sparse_rng.choice(rows, min(2, len(rows)), replace=False)
                                     for rows in official.groupby('prompt_id').indices.values()]))
    sparse = np.isin(np.arange(len(official)), sparse)
    old_full, new_full, old_sparse, new_sparse = [np.full_like(raw_plain, np.nan) for _ in range(4)]
    results = []
    for fold in args.folds:
        train = official.loc[folds != fold]
        query = np.flatnonzero(folds == fold)
        if set(train.prompt_id) & set(official.loc[query, 'prompt_id']):
            raise ValueError(f'Prompt overlap in fold {fold}')
        expert.load_fold(fold)
        # Save the bias used with the corresponding public out-of-fold centered logits.
        previous_bias = expert.centered.bias.detach().cpu().numpy().copy()
        state = json.loads((assets / 'calibration' / f'fold{fold}.json').read_text())['models']['qwen8']
        fit(expert, external_examples, selected, args.external_steps, args.lr, 29, f'fold {fold} external')
        fit(expert, examples(expert, train), selected, args.official_steps, args.lr * 0.45, 173 + fold,
            f'fold {fold} official')
        new, finite = expert.predict(official.loc[query, 'text'].tolist(), folds=[fold], batch_size=8)
        if not finite.all():
            raise ValueError(f'Non-finite held-out predictions in fold {fold}')
        old_full[query] = log_softmax(raw_plain[query] * state['plain_scale'], axis=1)
        new_full[query] = log_softmax(new['plain'][0] * state['plain_scale'], axis=1)
        positions = np.flatnonzero(sparse[query])
        rows = query[positions]
        groups = inferred_groups(train.text, official.loc[rows, 'text'])
        old_sparse[rows] = calibrate(raw_plain[rows][None], raw_center[rows][None], previous_bias[None], groups, [state])[0]
        new_sparse[rows] = calibrate(new['plain'][:, positions], new['centered'][:, positions],
                                     new['centered_bias'], groups, [state])[0]
        save_file({name: param.detach().cpu().contiguous() for name, param in selected.items()},
                  str(output / f'native-adapt-fold{fold}.safetensors'))
        fold_result = report(official.iloc[query], expert.labels,
                             {'native_plain': old_full[query], 'adapted_plain': new_full[query]})
        results.append({'fold': fold, 'scores': fold_result})
        print(f'fold {fold} OOF:', {k: v['balanced_accuracy'] for k, v in fold_result.items()}, flush=True)
        taken = np.isin(folds, args.folds[:len(results)])
        summary = {
            'folds': results,
            'full': report(official[taken], expert.labels, {'original': old_full[taken], 'adapted': new_full[taken]}),
            'two_per_prompt': report(official[taken & sparse], expert.labels,
                                     {'original': old_sparse[taken & sparse], 'adapted': new_sparse[taken & sparse]}),
        }
        (output / 'native_adapt_metrics.json').write_text(json.dumps(summary, indent=2))
    np.savez_compressed(output / 'native_adapt_oof.npz', ids=official.id.to_numpy(dtype=str), labels=expert.labels,
                        folds=folds, old_full=old_full, new_full=new_full, old_sparse=old_sparse, new_sparse=new_sparse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'base', 'assets', 'pool'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--external-steps', type=int, default=360)
    parser.add_argument('--official-steps', type=int, default=180)
    parser.add_argument('--last-layers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    study(parser.parse_args())


if __name__ == '__main__':
    main()
