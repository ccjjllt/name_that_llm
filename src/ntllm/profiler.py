"""Compare next-token loss statistics with PROFILER's six-position statistics."""

import argparse
import copy
import hashlib
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .peer_context import fit_head, make_frame
from .peer_context_report import read_panel

# Offsets relative to the token whose hidden state feeds the LM head.
OFFSETS = (-2, -1, 0, 1, 2, 3)
NEXT = OFFSETS.index(1)
PAIR_LEFT, PAIR_RIGHT = np.asarray(list(combinations(range(len(OFFSETS)), 2))).T
CONTINUOUS = 10 * len(OFFSETS)
FEATURES = CONTINUOUS + len(PAIR_LEFT)


def targets(tokens, lengths):
    offsets = torch.tensor(OFFSETS, device=tokens.device)
    anchors = torch.arange(2, max(2, tokens.shape[1] - 3), device=tokens.device)
    return anchors, anchors[None, :] + 3 < lengths[:, None], tokens[:, anchors[:, None] + offsets]


def offset_losses(logits, token_ids):
    scores = logits.float()
    return scores.logsumexp(dim=1, keepdim=True) - scores.gather(1, token_ids)


def describe(losses):
    """Summarize each offset, its token-wise gradient and pairwise symmetric KL."""
    def stats(x):
        return np.stack((x.mean(0), x.max(0), x.min(0), x.std(0), np.median(x, axis=0)), axis=1)

    grad = np.gradient(losses, axis=0) if len(losses) > 1 else np.zeros_like(losses)
    profiles = losses.clip(1e-8)
    profiles /= profiles.sum(axis=0, keepdims=True)
    p, q = profiles[:, PAIR_LEFT], profiles[:, PAIR_RIGHT]
    sym_kl = ((p - q) * (np.log(p) - np.log(q))).sum(axis=0)
    return np.concatenate([np.concatenate([stats(losses), stats(grad)], axis=1).ravel(), sym_kl])


@torch.inference_mode()
def extract_shard(rank, model_path, texts, config, cache, devices):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_device(rank)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, local_files_only=True,
        attn_implementation='sdpa', device_map={'': rank},
    ).eval()
    owned = np.arange(rank, len(texts), devices)
    seq = tokenizer([texts[i] for i in owned], add_special_tokens=False, truncation=True,
                    max_length=config['max_tokens'])['input_ids']
    lengths = np.array([len(ids) for ids in seq])
    # At least one shared anchor is needed to compute statistics for all six positions.
    if lengths.min() < 6:
        raise ValueError('At least one input tokenizes to fewer than six tokens')
    result = np.empty((len(seq), FEATURES), dtype=np.float32)
    order = np.argsort(lengths, kind='stable')
    first = 0
    started = time.monotonic()
    head = model.get_output_embeddings()
    while first < len(order):
        width = min(64, len(order) - first)
        while width * lengths[order[first + width - 1]] > config['tokens_per_forward'] and width > 1:
            width -= 1
        rows = order[first:first + width]
        tokens = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq[i]) for i in rows], batch_first=True,
                                                 padding_value=tokenizer.eos_token_id).cuda()
        sizes = torch.as_tensor(lengths[rows], device=tokens.device)
        anchors, valid, target_ids = targets(tokens, sizes)
        # Causality makes right padding invisible to all retained positions.
        states = model.model(input_ids=tokens, use_cache=False).last_hidden_state[:, anchors][valid]
        observed = target_ids[valid]
        packed = torch.cat([offset_losses(head(h), ids)
                            for h, ids in zip(states.split(128), observed.split(128))]).cpu().numpy()
        lengths_in_batch = lengths[rows] - 5
        for row, losses in zip(rows, np.split(packed, np.cumsum(lengths_in_batch)[:-1])):
            result[row] = describe(losses)
        first += width
        if first == len(order) or first // 2000 != (first - width) // 2000:
            print(f'GPU {rank}: {first}/{len(order)} in {time.monotonic() - started:.0f}s', flush=True)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite PROFILER statistics')
    np.savez(cache / f'shard{rank}.npz', rows=owned, features=result, token_counts=lengths)


def save_scores(frame, features, config, out):
    labels = sorted(frame.model.unique())
    external = np.flatnonzero(frame.partition.eq('external'))
    official = np.flatnonzero(frame.partition.eq('official'))
    held_out = np.flatnonzero(~frame.partition.isin(['external', 'official']))
    report_rows = np.concatenate([official, held_out])
    folds = frame.fold.to_numpy()
    codes = torch.as_tensor(pd.Categorical(frame.model, categories=labels).codes.astype(np.int64)).cuda()
    table = frame.drop(columns='text')
    for arm in ('next_token', 'multi_position'):
        fit_features = features.copy()
        if arm == 'next_token':
            keep = np.arange(NEXT * 10, (NEXT + 1) * 10)
            fit_features[:, ~np.isin(np.arange(FEATURES), keep)] = 0
        center = fit_features[external].mean(axis=0)
        scale = fit_features[external].std(axis=0).clip(.001)
        x = torch.from_numpy(np.clip((fit_features - center) / scale, -10, 10)).cuda()
        torch.manual_seed(config['seed'])
        head = nn.Sequential(nn.Linear(FEATURES, 256), nn.LayerNorm(256), nn.GELU(),
                             nn.Dropout(.15), nn.Linear(256, len(labels))).cuda()
        fit_head(head, x, codes, external, config)
        scores = np.full((len(frame), len(labels)), np.nan, dtype=np.float32)
        for fold in range(5):
            eval_rows = official[folds[official] == fold]
            train_rows = official[folds[official] != fold]
            adapted = copy.deepcopy(head)
            fit_head(adapted, x, codes, train_rows, config, adapt=True)
            with torch.no_grad():
                scores[eval_rows] = adapted.eval()(x[eval_rows]).cpu().numpy()
        fit_head(head, x, codes, official, config, adapt=True)
        with torch.no_grad():
            scores[held_out] = head.eval()(x[held_out]).cpu().numpy()
        evaluated = table.iloc[report_rows].join(pd.DataFrame(scores, columns=labels))
        if not np.isfinite(evaluated[labels].to_numpy()).all():
            raise ValueError(f'Missing or invalid scores in {arm}')
        evaluated.to_parquet(out / (arm + '.parquet'), index=False)
        torch.save(dict(head=head.cpu().state_dict(), center=torch.from_numpy(center),
                        scale=torch.from_numpy(scale), labels=labels), out / (arm + '.pt'))
        print(f'{arm}: {len(evaluated)} scored rows', flush=True)


def run(args):
    config = json.loads(args.config.read_text())
    official = pd.read_csv(args.official, dtype={'id': str, 'prompt_id': str})
    public = pd.read_parquet(args.trace / 'public_truth.parquet')
    blind = read_panel(args.blind).reset_index()
    frame = make_frame(args.pool, official, public, blind, json.loads(args.folds.read_text()), config)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = frame.drop(columns='text')
    manifest.to_parquet(args.out / 'manifest.parquet', index=False)
    corpus = hashlib.sha256(b''.join(hashlib.sha256(t.encode()).digest() for t in frame.text)).hexdigest()
    print(manifest.groupby('partition').size().to_dict(), 'ordered body hash', corpus, flush=True)
    (args.out / 'extraction.json').write_text(json.dumps(
        dict(config=config, offsets=OFFSETS, features=FEATURES, ordered_body_digest=corpus,
             rows=manifest.partition.value_counts().to_dict()), indent=2) + '\n')
    if args.prepare_only:
        return
    devices = torch.cuda.device_count()
    if devices != 2:
        raise ValueError(f'Expected two GPUs, found {devices}')
    args.cache.mkdir(parents=True, exist_ok=True)
    torch.multiprocessing.spawn(extract_shard, args=(args.base, frame.text.tolist(), config, args.cache, devices),
                                nprocs=devices, join=True)
    with np.load(args.cache / 'shard0.npz') as first, np.load(args.cache / 'shard1.npz') as second:
        order = np.concatenate([first['rows'], second['rows']])
        align = np.argsort(order)
        if not np.array_equal(order[align], np.arange(len(frame))):
            raise ValueError('Token statistics must cover all input rows exactly once')
        features = np.concatenate([first['features'], second['features']])[align]
        token_counts = np.concatenate([first['token_counts'], second['token_counts']])[align]
    manifest.assign(scored_tokens=token_counts - 5).to_parquet(args.out / 'token_counts.parquet', index=False)
    save_scores(frame, features, config, args.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('base', 'pool', 'official', 'trace', 'blind', 'folds', 'config', 'out'):
        p.add_argument('--' + arg, type=Path, required=True)
    p.add_argument('--cache', type=Path, default=Path('/kaggle/temp/profiler'))
    p.add_argument('--prepare-only', action='store_true')
    run(p.parse_args())
