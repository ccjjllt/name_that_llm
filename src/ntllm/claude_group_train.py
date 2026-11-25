"""Compare CE with two- and three-author assignment on the same training batches."""

import hashlib
from itertools import permutations
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .exact_pair_train import PAIR_WEIGHT, TOKEN_BIN_WIDTH
from .sibling_specialist import batches, classification_loss, feed_batch, tokenize


CLAUDE = ('claude-opus-4-6', 'claude-opus-4-8', 'claude-sonnet-5')
GROUP_SIZES = (2, 3)


def frozen_rows(frame):
    if 'id' in frame.columns:
        ids = pd.Index(frame.id)
    elif frame.index.name == 'id':
        ids = frame.index
    else:
        raise ValueError('Training requires the frozen sample IDs')
    if len(frame) != 44522 or not ids.is_unique or ids.hasnans:
        raise ValueError('Expected 44,522 distinct passages')
    order = np.argsort(ids.astype(str).to_numpy(), kind='stable')
    rows = frame.iloc[order].reset_index(drop=True)
    matched = rows.loc[rows.match_group.notna()]
    widths = matched.groupby('match_group', observed=True).model.agg(['size', 'nunique'])
    if (widths.empty or not widths['size'].isin(GROUP_SIZES).all() or not widths['size'].eq(widths['nunique']).all()
            or not matched.model.isin(CLAUDE).all() or matched.match_group.astype(str).str.strip().eq('').any()
            or not matched.text.str.split().str.len().between(40, 99).all()):
        raise ValueError('Matched groups need two or three distinct supported Claude models and 40–99 word bodies')
    groups = tuple(group.index.to_numpy() for _, group in matched.sort_values('model').groupby('match_group', observed=True))
    return rows, ids.take(order).astype(str).to_numpy(), groups


def grouped_batches(tokens, groups, batch_size, rng):
    if batch_size < max(map(len, groups)) or len(tokens) < len(groups) * batch_size:
        raise ValueError('Each matched group must fit in its own full batch')
    bins = np.array([len(sequence) // TOKEN_BIN_WIDTH for sequence in tokens])
    free = np.ones(len(tokens), dtype=bool)
    free[np.concatenate(groups)] = False
    buckets = [[] for _ in range(int(bins.max()) + 1)]
    for row in np.flatnonzero(free):
        buckets[bins[row]].append(row)
    for bucket in buckets:
        rng.shuffle(bucket)

    schedule = []
    for index in rng.permutation(len(groups)):
        group = groups[index]
        batch = list(group)
        anchor_bin = bins[group].max()
        nearest = sorted(range(len(buckets)), key=lambda value: (abs(value - anchor_bin), value))
        for index in nearest:
            needed = batch_size - len(batch)
            if not needed:
                break
            take = min(needed, len(buckets[index]))
            if take:
                batch.extend(buckets[index][-take:])
                del buckets[index][-take:]
        if len(batch) != batch_size:
            raise ValueError('Could not fill a matched batch without reusing a row')
        schedule.append((np.asarray(batch, dtype=np.int64), len(group)))
    remaining = np.asarray([row for bucket in buckets for row in bucket], dtype=np.int64)
    schedule.extend((remaining[take], 0) for take in batches([tokens[i] for i in remaining], batch_size, rng))
    order = rng.permutation(len(schedule))
    schedule = [schedule[i] for i in order]
    if not np.array_equal(np.sort(np.concatenate([rows for rows, _ in schedule])), np.arange(len(tokens))):
        raise ValueError('A group schedule repeated or omitted a training passage')
    return schedule


def assignment(scores, target, width, order, position):
    """Return scaled NLL and the gold advantage over the best alternative assignment."""
    local = scores[:width].index_select(1, target[:width])
    alternatives = local[position, order].sum(1)
    normalizer = math.log(2.) / math.log(math.factorial(width))
    loss = (torch.logsumexp(alternatives, dim=0) - alternatives[0]) * normalizer
    return loss, alternatives[0] - alternatives[1:].max()


def train_group(model, tokenizer, frame, labels, config, device, paired):
    """Train either arm and return its ordered schedule and optimizer receipt."""
    if torch.device(device).type != 'cuda' or torch.cuda.device_count() != 1:
        raise ValueError('This matched arm requires exactly one assigned CUDA GPU')
    if len(labels) != 18 or config['external_epochs'] < 1:
        raise ValueError('Expected training epochs and an 18-class reader')
    frame, ids, groups = frozen_rows(frame)
    tokens = tokenize(tokenizer, frame, config)
    target_codes = pd.Categorical(frame.model, categories=labels).codes.astype(np.int64)
    if (target_codes < 0).any():
        raise ValueError('Unknown training model label')
    class_counts = np.bincount(target_codes, minlength=len(labels))
    if (class_counts == 0).any():
        raise ValueError('Every class must be represented in the matched sample')
    class_weights = torch.as_tensor(len(frame) / (len(labels) * class_counts), device=device, dtype=torch.float32)
    order = {width: torch.as_tensor(list(permutations(range(width))), device=device) for width in GROUP_SIZES}
    position = {width: torch.arange(width, device=device)[None, :] for width in GROUP_SIZES}
    expected = {width: sum(len(group) == width for group in groups) for width in GROUP_SIZES}
    rng = np.random.default_rng(config['seed'])
    torch.manual_seed(config['seed'])
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'])
    scaler = torch.amp.GradScaler('cuda', init_scale=256)
    weight = PAIR_WEIGHT if paired else 0.
    digest = hashlib.sha256()
    history = []
    start = time.monotonic()
    model.train()
    for epoch in range(1, config['external_epochs'] + 1):
        schedule = grouped_batches(tokens, groups, config['batch_size'], rng)
        sums = {width: torch.zeros(3, device=device) for width in GROUP_SIZES}
        observed = dict.fromkeys(GROUP_SIZES, 0)
        ce_total = torch.zeros((), device=device)
        rows_seen = optimizer_steps = 0
        for step, (rows, width) in enumerate(schedule, 1):
            digest.update((json.dumps([epoch, ids[rows].tolist(), width], separators=(',', ':')) + '\n').encode())
            target = torch.as_tensor(target_codes[rows], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                scores = model(feed_batch(tokenizer, tokens, rows, device))
                ce = classification_loss(scores, target, class_weights)
                loss = ce
                if width:
                    nll, gap = assignment(scores, target, width, order[width], position[width])
                    if paired:
                        loss = ce + weight * nll
                    detached = gap.detach()
                    sums[width] += torch.stack((nll.detach(), detached.gt(0).float(), detached.eq(0).float()))
                    observed[width] += 1
            ce_total += ce.detach()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters, 1.)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += int(scaler.get_scale() >= previous_scale)
            rows_seen += len(rows)
            if step == 1 or step % 100 == 0:
                print('external', epoch, step, 'loss', round(loss.item(), 4), 'elapsed',
                      round(time.monotonic() - start), flush=True)
        if optimizer_steps < len(schedule) * .95:
            raise RuntimeError('Too many mixed precision optimizer steps were skipped')
        if observed != expected or rows_seen != len(frame):
            raise ValueError('Training changed the frozen group or row exposures')
        outcomes = {}
        for width in GROUP_SIZES:
            count = observed[width]
            total_nll, correct, tied = sums[width].tolist()
            outcomes[str(width)] = dict(groups=count, correct=int(correct), tied=int(tied),
                                        mean_scaled_nll=total_nll / count if count else None)
        history.append(dict(epoch=epoch, scheduled_batches=len(schedule), optimizer_steps=optimizer_steps,
                            skipped=len(schedule) - optimizer_steps, row_exposures=rows_seen,
                            matched_group_exposures=sum(observed.values()),
                            matched_row_exposures=sum(width * count for width, count in observed.items()),
                            mean_ce=ce_total.item() / len(schedule), assignment_by_size=outcomes))
        print('external complete', history[-1], flush=True)
    totals = {key: sum(epoch[key] for epoch in history) for key in (
        'scheduled_batches', 'optimizer_steps', 'skipped', 'row_exposures', 'matched_group_exposures', 'matched_row_exposures',
    )}
    return dict(**totals, seed=config['seed'], pair_weight=weight, schedule_sha256=digest.hexdigest(),
                train_rows=len(frame), train_groups=len(groups), train_groups_by_size=expected, epoch_logs=history)
