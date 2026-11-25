"""Train matched Kimi/GLM classifiers with and without across-effort consistency."""

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

from .paired_attribution import text_key
from .sibling_specialist import Specialist, feed_batch, predict, tokenize, write_json


LABELS = ('glm-5.3', 'glm-5.3-flash', 'kimi-k3')
ARMS = ('classification', 'consistent')


def prepare_rows(training, queries):
    columns = {'id', 'text', 'model', 'pair_id', 'prompt_id', 'source_dataset', 'n_words',
               'effort', 'provider_route', 'served_model'}
    if not columns.issubset(training) or not training.id.is_unique or not queries.id.is_unique:
        raise ValueError('Expected distinct passages with explicit matched-pair metadata')
    ordered = training.sort_values(['pair_id', 'effort']).reset_index(drop=True)
    groups = ordered.groupby('pair_id', sort=False)
    match = ['prompt_id', 'source_dataset', 'model', 'provider_route', 'served_model']
    if (not groups.size().eq(2).all() or not groups.effort.nunique().eq(2).all()
            or not groups[match].nunique().eq(1).all().all()
            or set(ordered.model) != set(LABELS) or ordered[list(columns)].isna().any().any()):
        raise ValueError('Expected two effort settings within the same source, prompt, model and route')
    lengths = ordered.text.str.split().str.len().to_numpy()
    widths = lengths.reshape(-1, 2)
    if (not np.array_equal(lengths, ordered.n_words.to_numpy()) or not ordered.n_words.between(40, 250).all()
            or np.any(widths.max(1) - widths.min(1) > 12)
            or np.any(widths.max(1) > 1.15 * widths.min(1))):
        raise ValueError('Unexpected text lengths in the matched pairs')
    keys = text_key(ordered.text)
    if not keys.is_unique or not set(keys).isdisjoint(text_key(queries.text)) or not set(ordered.id).isdisjoint(queries.id):
        raise ValueError('Matched pairs overlap held-out or public passages')
    return ordered


def balanced_pairs(frame, rng):
    groups = [np.flatnonzero(frame.model.iloc[::2].to_numpy() == label) for label in LABELS]
    limit = max(map(len, groups))
    draws = []
    for candidates in groups:
        extra = rng.choice(candidates, size=limit - len(candidates), replace=True)
        draws.append(np.concatenate([candidates, extra]))
    return rng.permutation(np.concatenate(draws))


def consistency_loss(logits):
    log_prob = F.log_softmax(logits.float(), dim=1).reshape(-1, 2, len(LABELS))
    prob = log_prob.exp()
    return 0.5 * ((prob[:, 0] - prob[:, 1]) * (log_prob[:, 0] - log_prob[:, 1])).sum(1).mean()


def fit(model, tokenizer, frame, config, consistency_weight):
    rng = np.random.default_rng(config['seed'])
    tokens = tokenize(tokenizer, frame, config)
    targets = pd.Categorical(frame.model, categories=LABELS).codes.astype(np.int64)
    pair_rows = np.arange(len(frame)).reshape(-1, 2)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'])
    scaler = torch.amp.GradScaler('cuda', init_scale=256)
    model.train()
    model.trunk.eval()
    steps = skipped = 0
    started = time.monotonic()
    for epoch in range(config['epochs']):
        order = balanced_pairs(frame, rng)
        total = np.zeros(2)
        for offset in range(0, len(order), config['pairs_per_batch']):
            rows = pair_rows[order[offset:offset + config['pairs_per_batch']]].ravel()
            truth = torch.as_tensor(targets[rows], device='cuda:0')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                scores = model(feed_batch(tokenizer, tokens, rows, 'cuda:0'))
                ce = F.cross_entropy(scores, truth)
                agreement = consistency_loss(scores)
                loss = ce + consistency_weight * agreement
            scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters, 1.)
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            skipped += scaler.get_scale() < scale
            total += (ce.item(), agreement.item())
        count = (len(order) + config['pairs_per_batch'] - 1) // config['pairs_per_batch']
        print(f'epoch={epoch + 1} steps={steps} skipped={skipped} ce={total[0]/count:.4f} '
              f'consistency={total[1]/count:.4f} elapsed={time.monotonic()-started:.1f}s', flush=True)
    return dict(steps=steps, steps_skipped=int(skipped), consistency_weight=consistency_weight)


def run(args):
    from transformers import AutoTokenizer

    config = json.loads(args.config.read_text())
    public = pd.read_parquet(args.public, columns=['id', 'text'])
    holdout = pd.read_parquet(args.holdout, columns=['id', 'text'])
    queries = pd.concat([public, holdout], ignore_index=True)
    train = prepare_rows(pd.read_parquet(args.training), queries)
    if args.prepare_only:
        print('Pairs by label:', train.iloc[::2].model.value_counts().to_dict(),
              'public:', len(public), 'source holdout:', len(holdout), flush=True)
        return
    if torch.cuda.device_count() != 1:
        raise ValueError('Each matched arm needs exactly one assigned GPU')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    tokenizer.padding_side = 'right'
    model = Specialist(args.base, config, len(LABELS), 'cuda:0')
    weight = config['consistency_weight'] if args.arm == 'consistent' else 0.
    receipt = fit(model, tokenizer, train, config, weight)
    scores = predict(model, tokenizer, queries, list(LABELS), config, 'cuda:0')
    predictions = queries[['id']].join(pd.DataFrame(scores, columns=LABELS))
    predictions.to_parquet(args.out / f'{args.arm}.parquet', index=False)
    torch.save({name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad},
               args.out / f'{args.arm}.pt')
    receipt.update(train_pairs=train.pair_id.nunique(), public_rows=len(public), holdout_rows=len(holdout),
                   training_file_sha256=hashlib.sha256(args.training.read_bytes()).hexdigest())
    write_json(args.out / f'{args.arm}.json', receipt)
    print(args.arm, receipt, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('base', 'config', 'training', 'public', 'holdout', 'out'):
        p.add_argument('--' + arg, type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--prepare-only', action='store_true')
    run(p.parse_args())
