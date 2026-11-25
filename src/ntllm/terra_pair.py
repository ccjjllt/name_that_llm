"""Compare cross-entropy and matched Terra/GPT-5.5 assignment training."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .paired_attribution import swap_loss, text_key
from .sibling_specialist import Specialist, feed_batch, predict, tokenize, write_json


LABELS = ('gpt-5.5', 'gpt-5.6-terra')
ARMS = ('classification', 'pairwise')


def prepare_rows(training, queries):
    needed = {'id', 'text', 'model', 'pair_id', 'prompt_id', 'source_dataset', 'n_words'}
    if not needed.issubset(training) or not training.id.is_unique or not queries.id.is_unique:
        raise ValueError('Expected frozen pair IDs and unique query IDs')
    ordered = training.sort_values(['pair_id', 'model']).reset_index(drop=True)
    counts = ordered.groupby('pair_id').model.agg(['size', 'nunique'])
    widths = ordered.n_words.to_numpy().reshape(-1, 2)
    if (set(ordered.model) != set(LABELS) or not counts.eq(2).all().all()
            or not ordered.text.str.split().str.len().eq(ordered.n_words).all()
            or not ordered.n_words.between(40, 99).all()
            or np.any(widths.max(1) > 1.5 * widths.min(1))
            or ordered.groupby('pair_id')[['prompt_id', 'source_dataset']].nunique().ne(1).any().any()):
        raise ValueError('Expected two different authors per same-source, near-length short pair')
    keys = text_key(ordered.text)
    if not keys.is_unique or not set(keys).isdisjoint(text_key(queries.text)) or not set(ordered.id).isdisjoint(queries.id):
        raise ValueError('Training passages overlap or contain duplicated bodies')
    return ordered


def fit(model, tokenizer, frame, config, swap_weight):
    rng = np.random.default_rng(config['seed'])
    tokens = tokenize(tokenizer, frame, config)
    codes = pd.Categorical(frame.model, categories=LABELS).codes.astype(np.int64)
    pairs = np.arange(len(frame)).reshape(-1, 2)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'])
    scaler = torch.amp.GradScaler('cuda', init_scale=256)
    steps = 0
    steps_skipped = 0
    start = time.monotonic()
    model.train()
    model.trunk.eval()
    for epoch in range(config['epochs']):
        order = rng.permutation(len(pairs))
        losses = np.zeros(2)
        for offset in range(0, len(order), config['pairs_per_batch']):
            batch = pairs[order[offset:offset + config['pairs_per_batch']]].ravel()
            target = torch.as_tensor(codes[batch], device='cuda:0')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(feed_batch(tokenizer, tokens, batch, 'cuda:0'))
                ce = F.cross_entropy(logits, target)
                ranking = swap_loss(logits, target)
                loss = ce + swap_weight * ranking
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters, 1.)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            steps_skipped += scaler.get_scale() < scale_before
            losses += (ce.item(), ranking.item())
        num_batches = (len(order) + config['pairs_per_batch'] - 1) // config['pairs_per_batch']
        print(f'epoch={epoch + 1} steps={steps} skipped={steps_skipped} '
              f'ce={losses[0]/num_batches:.4f} swap={losses[1]/num_batches:.4f} '
              f'elapsed={time.monotonic()-start:.1f}s', flush=True)
    return dict(steps=steps, steps_skipped=int(steps_skipped), swap_weight=swap_weight)


def run(args):
    from transformers import AutoTokenizer

    config = json.loads(args.config.read_text())
    public = pd.read_parquet(args.public, columns=['id', 'text'])
    holdout = pd.read_parquet(args.holdout, columns=['id', 'text'])
    queries = pd.concat([public, holdout], ignore_index=True)
    frame = prepare_rows(pd.read_parquet(args.training), queries)
    if args.prepare_only:
        print(len(frame), 'training passages;', len(public), 'public;', len(holdout), 'held-source', flush=True)
        return
    if torch.cuda.device_count() != 1:
        raise ValueError('Each matched arm needs exactly one assigned GPU')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    tokenizer.padding_side = 'right'
    model = Specialist(args.base, config, len(LABELS), 'cuda:0')
    weight = config['swap_weight'] if args.arm == 'pairwise' else 0.
    receipt = fit(model, tokenizer, frame, config, weight)
    logits = predict(model, tokenizer, queries, list(LABELS), config, 'cuda:0')
    predictions = queries[['id']].assign(margin=logits[:, 1] - logits[:, 0])
    predictions.to_parquet(args.out / f'{args.arm}.parquet', index=False)
    torch.save({k: p.detach().cpu() for k, p in model.named_parameters() if p.requires_grad},
               args.out / f'{args.arm}.pt')
    receipt.update(train_pairs=frame.pair_id.nunique(), public_rows=len(public), holdout_rows=len(holdout),
                   training_file_sha256=hashlib.sha256(args.training.read_bytes()).hexdigest())
    write_json(args.out / f'{args.arm}.json', receipt)
    print(args.arm, receipt, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('base', 'config', 'training', 'public', 'holdout', 'out'):
        parser.add_argument('--' + arg, type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    run(parser.parse_args())
