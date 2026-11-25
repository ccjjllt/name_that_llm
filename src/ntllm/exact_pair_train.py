"""Train matched 18-class readers with an optional verified assignment loss."""

import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .sibling_specialist import batches, classification_loss, feed_batch, tokenize


GEMINI = ('gemini-3.7-flash', 'gemini-3.8-flash')
PAIR_WEIGHT = .40
TOKEN_BIN_WIDTH = 16


def frozen_rows(frame):
    if 'id' in frame.columns:
        ids = pd.Index(frame.id)
    elif frame.index.name == 'id':
        ids = frame.index
    else:
        raise ValueError('Training requires the frozen sample IDs')
    if len(frame) != 44522 or not ids.is_unique or ids.hasnans:
        raise ValueError('Expected 44,522 unique training passage IDs')
    order = np.argsort(ids.astype(str).to_numpy(), kind='stable')
    rows = frame.iloc[order].reset_index(drop=True)
    verified = rows.loc[rows.vertex_pair_id.notna(), ['vertex_pair_id', 'model']]
    groups = verified.groupby('vertex_pair_id').model.agg(['size', 'nunique'])
    if (len(groups) != 941 or not groups.eq(2).all().all() or set(verified.model) != set(GEMINI)
            or verified.vertex_pair_id.astype(str).str.strip().eq('').any()):
        raise ValueError('Expected 941 disjoint verified Gemini 3.7/3.8 pairs')
    pairs = verified.sort_values(['vertex_pair_id', 'model']).index.to_numpy().reshape(-1, 2)
    return rows, ids.take(order).astype(str).to_numpy(), pairs


def paired_batches(tokens, pairs, batch_size, rng):
    """Anchor each pair in one batch and fill with length-near unpaired rows."""
    if batch_size < 2 or len(tokens) < len(pairs) * batch_size:
        raise ValueError('Cannot fit each verified pair in a separate full batch')
    lengths = np.array([len(row) for row in tokens])
    token_bins = lengths // TOKEN_BIN_WIDTH
    unpaired = np.ones(len(tokens), dtype=bool)
    unpaired[pairs.ravel()] = False
    buckets = [[] for _ in range(int(token_bins.max()) + 1)]
    for row in np.flatnonzero(unpaired):
        buckets[token_bins[row]].append(row)
    for bucket in buckets:
        rng.shuffle(bucket)

    result = []
    for a, b in pairs[rng.permutation(len(pairs))]:
        batch = [a, b]
        pair_bin = max(token_bins[a], token_bins[b])
        nearest = sorted(range(len(buckets)), key=lambda i: (abs(i - pair_bin), i))
        for i in nearest:
            needed = batch_size - len(batch)
            if not needed:
                break
            take = min(needed, len(buckets[i]))
            if take:
                batch.extend(buckets[i][-take:])
                del buckets[i][-take:]
        if len(batch) != batch_size:
            raise ValueError('Could not fill a paired batch without repeating a row')
        result.append((np.asarray(batch, dtype=np.int64), True))

    remaining = np.asarray([row for bucket in buckets for row in bucket], dtype=np.int64)
    result.extend((remaining[take], False) for take in batches([tokens[i] for i in remaining], batch_size, rng))
    result = [result[i] for i in rng.permutation(len(result))]
    exposures = np.concatenate([rows for rows, _ in result])
    if not np.array_equal(np.sort(exposures), np.arange(len(tokens))):
        raise ValueError('A paired schedule repeated or omitted a passage')
    return result


def assignment_margin(scores, target):
    a, b = target[:2]
    return scores[0, a] + scores[1, b] - scores[0, b] - scores[1, a]


def train_exact(model, tokenizer, frame, labels, config, device, paired):
    """Use the same row exposures, order, and AdamW setup for either arm."""
    if torch.device(device).type != 'cuda' or torch.cuda.device_count() != 1:
        raise ValueError('This matched arm requires exactly one assigned CUDA GPU')
    if config['external_epochs'] != 2 or len(labels) != 18:
        raise ValueError('Expected the original two epochs and 18 classes')
    frame, ids, exact_pairs = frozen_rows(frame)
    tokens = tokenize(tokenizer, frame, config)
    codes = pd.Categorical(frame.model, categories=labels).codes.astype(np.int64)
    if (codes < 0).any():
        raise ValueError('Found an unknown model label in the frozen sample')
    counts = np.bincount(codes, minlength=len(labels))
    if (counts == 0).any():
        raise ValueError('Every class must be represented in the matched sample')
    class_weights = torch.as_tensor(len(frame) / (len(labels) * counts), device=device, dtype=torch.float32)
    rng = np.random.default_rng(config['seed'])
    torch.manual_seed(config['seed'])
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'])
    scaler = torch.amp.GradScaler('cuda', init_scale=256)
    pair_weight = PAIR_WEIGHT if paired else 0.
    digest = hashlib.sha256()
    history = []
    started = time.monotonic()
    model.train()
    for epoch in range(1, config['external_epochs'] + 1):
        schedule = paired_batches(tokens, exact_pairs, config['batch_size'], rng)
        ce_total = torch.zeros((), device=device)
        assignment_total = torch.zeros((), device=device)
        ordered_correct = torch.zeros((), device=device)
        row_exposures = pair_exposures = optimizer_steps = 0
        for step, (rows, has_pair) in enumerate(schedule, 1):
            digest.update((json.dumps([epoch, ids[rows].tolist(), has_pair], separators=(',', ':')) + '\n').encode())
            target = torch.as_tensor(codes[rows], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                scores = model(feed_batch(tokenizer, tokens, rows, device))
                ce = classification_loss(scores, target, class_weights)
                loss = ce
                if has_pair:
                    margin = assignment_margin(scores, target)
                    assignment = F.softplus(-margin)
                    if paired:
                        loss = ce + pair_weight * assignment
                    assignment_total += assignment.detach()
                    ordered_correct += margin.detach().gt(0)
            ce_total += ce.detach()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters, 1.)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += int(scaler.get_scale() >= scale)
            row_exposures += len(rows)
            pair_exposures += int(has_pair)
            if step == 1 or step % 100 == 0:
                print('external', epoch, step, 'loss', round(loss.item(), 4), 'elapsed',
                      round(time.monotonic() - started), flush=True)
        if optimizer_steps < len(schedule) * .95:
            raise RuntimeError('Too many mixed precision optimizer steps were skipped')
        if row_exposures != len(frame) or pair_exposures != len(exact_pairs):
            raise ValueError('A training epoch changed the frozen row or pair exposures')
        history.append(dict(epoch=epoch, scheduled_batches=len(schedule), optimizer_steps=optimizer_steps,
                            skipped=len(schedule) - optimizer_steps, row_exposures=row_exposures,
                            pair_exposures=pair_exposures, mean_ce=ce_total.item() / len(schedule),
                            assignment_nll=assignment_total.item() / pair_exposures,
                            correct_assignments=int(ordered_correct.item())))
        print('external complete', history[-1], flush=True)
    totals = {key: sum(epoch[key] for epoch in history)
              for key in ('scheduled_batches', 'optimizer_steps', 'skipped', 'row_exposures', 'pair_exposures')}
    return dict(**totals, seed=config['seed'], pair_weight=pair_weight, schedule_sha256=digest.hexdigest(),
                train_rows=len(frame), train_pairs=len(exact_pairs), epoch_logs=history)
