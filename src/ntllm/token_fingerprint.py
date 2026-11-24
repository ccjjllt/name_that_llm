"""Train an 18-author classifier on the sequence of a frozen LM's token forecasts."""

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from .paired_attribution import read_pool, text_key


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def corpus(pool_path, official, public, config):
    # Reuse the existing disjoint external prompt/source split. Keep benchmark bodies
    # out of sampling and drop benchmark publishers, including their other passages.
    excluded = pd.concat([official[['model', 'text']], public[['model', 'text']]], ignore_index=True)
    external, held = read_pool(pool_path, excluded)
    excluded_publishers = ('beroli/', 'markbarney/')
    external = external.loc[~external.source_dataset.str.startswith(excluded_publishers)].copy()
    rng = np.random.default_rng(config['seed'])
    frequency = external.groupby(['model', 'source_dataset']).id.transform('size')
    external['draw'] = -np.log(rng.random(len(external))) * np.sqrt(frequency)
    external = external.sort_values('draw').groupby('model').head(config['external_per_class'])
    # Keep IDs and raw text for auditing. Use a different column for diagnostic cohorts.
    held = (held.sample(frac=1, random_state=config['seed']).groupby(['source_dataset', 'model'])
            .head(config['source_holdout_per_class']))
    assert set(external.model) == set(official.model) and set(held.prompt_canon).isdisjoint(external.prompt_canon)
    assert set(text_key(external.text)).isdisjoint(text_key(excluded.text))
    ext = external.assign(partition='external', prompt_id=external.prompt_canon)
    unseen = held.assign(partition='held_source', prompt_id=held.prompt_canon)
    panel = public.assign(partition='public')
    official = official.assign(partition='official')
    frame = pd.concat([ext, official, panel, unseen], ignore_index=True)
    frame['n_words'] = frame.text.str.split().str.len()
    return frame[['partition', 'id', 'text', 'model', 'prompt_id', 'n_words', 'cohort', 'source_dataset']]


def forecast(logits, target):
    # Float32 statistics of the full vocabulary, including tail rank and uncertainty.
    score = logits.float()
    chosen = score.gather(1, target[:, None]).squeeze(1)
    logz = score.logsumexp(1)
    best = score.topk(2, dim=1).values
    logp = score - logz[:, None]
    entropy = -(logp.exp() * logp).sum(1)
    rank = (score > chosen[:, None]).sum(1).float().log1p()
    return torch.stack([logz - chosen, rank, entropy, logz - best[:, 0],
                        best[:, 0] - best[:, 1], best[:, 0] - chosen], dim=1).clamp(0, 50)


def make_projections(width, config):
    generator = torch.Generator().manual_seed(config['seed'])
    return [torch.randn(width, config['projection_width'], generator=generator) / math.sqrt(width)
            for _ in range(2)]


@torch.inference_mode()
def extract_shard(rank, base, texts, config, cache, devices):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_device(rank)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, local_files_only=True,
                                                  attn_implementation='sdpa', device_map={'': rank}).eval()
    model.config.use_cache = False
    projections = [p.to('cuda', dtype=torch.float16) for p in make_projections(model.config.hidden_size, config)]
    limit = config['max_tokens']
    indices = np.arange(rank, len(texts), devices)
    seq = tokenizer([texts[i] for i in indices], add_special_tokens=False, max_length=limit, truncation=True)['input_ids']
    lengths = np.array([len(row) for row in seq], dtype=np.int16)
    if lengths.min() == 0:
        raise ValueError('Encountered an empty token sequence')
    channels = 2 * config['projection_width'] + 6
    cache.mkdir(parents=True, exist_ok=True)
    values = np.memmap(cache / 'features.f16', mode='r+', shape=(len(texts), limit, channels), dtype=np.float16)
    # Right-pad on sorted length buckets. A fixed EOS starting token defines the
    # forecast of the first token even for a checkpoint with no BOS token.
    order = np.argsort(lengths, kind='stable')
    start = time.monotonic()
    first = 0
    while first < len(order):
        width = min(64, len(order) - first)
        while width > 1 and width * int(lengths[order[first + width - 1]] + 1) > config['tokens_per_forward']:
            width -= 1
        rows = order[first:first + width]
        longest = int(lengths[rows].max())
        tokens = torch.full((width, longest + 1), tokenizer.eos_token_id, dtype=torch.long)
        valid = torch.arange(longest)[None] < torch.as_tensor(lengths[rows, None])
        for i, row in enumerate(rows):
            tokens[i, 1:1 + len(seq[row])] = torch.tensor(seq[row])
        tokens, valid = tokens.cuda(), valid.cuda()
        # Right padding is always in the future of real tokens under causal SDPA.
        output = model.model(input_ids=tokens, use_cache=False, output_hidden_states=True)
        selected = [output.hidden_states[model.config.num_hidden_layers // 2], output.last_hidden_state]
        context = output.last_hidden_state[:, :-1][valid]
        observed = tokens[:, 1:][valid]
        stats = torch.cat([forecast(model.get_output_embeddings()(h), y)
                           for h, y in zip(context.split(128), observed.split(128))])
        parts = [F.rms_norm(h[:, 1:][valid].float(), (h.size(-1),)).half() @ p
                 for h, p in zip(selected, projections)]
        batch = torch.zeros((width, longest, channels), device='cuda', dtype=torch.float16)
        batch[valid] = torch.cat([*parts, stats.half()], dim=-1)
        array = batch.cpu().numpy()
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite fingerprint')
        values[indices[rows], :longest] = array
        first += width
        if first == len(seq) or first // 2000 != (first - width) // 2000:
            print(f'GPU {rank} extracted {first}/{len(seq)} rows in {time.monotonic() - start:.0f}s', flush=True)
    values.flush()
    del model, output, selected, context
    gc.collect()
    torch.cuda.empty_cache()
    np.save(cache / f'lengths_{rank}.npy', lengths)


def extract(base, frame, config, cache):
    cache.mkdir(parents=True, exist_ok=True)
    dims = (len(frame), config['max_tokens'], 2 * config['projection_width'] + 6)
    values = np.memmap(cache / 'features.f16', mode='w+', shape=dims, dtype=np.float16)
    devices = min(2, torch.cuda.device_count(), len(frame))
    if not devices:
        raise RuntimeError('A Kaggle GPU is required for extraction')
    inputs = (base, frame.text.tolist(), config, cache, devices)
    print(f'Extracting features on {devices} GPUs for {len(frame)} rows', flush=True)
    if devices == 1:
        extract_shard(0, *inputs)
    else:
        torch.multiprocessing.spawn(extract_shard, args=inputs, nprocs=devices, join=True)
    lengths = np.empty(len(frame), dtype=np.int16)
    for rank in range(devices):
        lengths[rank::devices] = np.load(cache / f'lengths_{rank}.npy')
    return values, lengths


class FingerprintHead(nn.Module):
    def __init__(self, config, nlabels, forecast_enabled):
        super().__init__()
        self.forecast_enabled = forecast_enabled
        channels = 2 * config['projection_width'] + 6
        self.project = nn.Linear(channels, 160)
        self.context = nn.Sequential(nn.Conv1d(160, 160, 5, padding=2), nn.GELU(),
                                     nn.Conv1d(160, 160, 3, padding=1))
        self.norm = nn.LayerNorm(160)
        self.attention = nn.Linear(160, 4)
        self.classify = nn.Sequential(nn.Dropout(.18), nn.Linear(160 * 5, nlabels))

    def forward(self, feat, valid):
        if not self.forecast_enabled:
            feat = torch.cat([feat[:, :, :-6], torch.zeros_like(feat[:, :, -6:])], dim=-1)
        expanded = valid.unsqueeze(-1)
        x = F.gelu(self.project(feat)) * expanded
        local = self.context[0](x.transpose(1, 2))
        local = self.context[1](local) * valid[:, None, :]
        local = self.context[2](local).transpose(1, 2)
        h = self.norm(x + local)
        attn = self.attention(h).masked_fill(~expanded, -1e4).softmax(dim=1)
        pool = (attn.transpose(1, 2) @ h).flatten(1)
        avg = (h * expanded).sum(1) / expanded.sum(1)
        return self.classify(torch.cat([pool, avg], dim=1))


def inputs(values, lengths, rows, device):
    sizes = torch.as_tensor(lengths[rows].astype(np.int64), device=device)
    width = int(lengths[rows].max())
    valid = torch.arange(width, device=device)[None] < sizes[:, None]
    return torch.from_numpy(values[rows, :width]).to(device=device, dtype=torch.float32), valid


def fit(model, values, lengths, codes, rows, config, epochs, lr, seed, device):
    rng = np.random.default_rng(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    counts = np.bincount(codes[rows], minlength=18)
    weights = torch.tensor(len(rows) / (18 * counts), device=device, dtype=torch.float32)
    for epoch in range(epochs):
        opt.param_groups[0]['lr'] = lr * (.2 + .8 * (1 + math.cos(math.pi * epoch / epochs)) / 2)
        order = rng.permutation(rows)
        for k in range(0, len(order), config['batch_size']):
            batch = order[k:k + config['batch_size']]
            y = torch.as_tensor(codes[batch], dtype=torch.long, device=device)
            loss = F.cross_entropy(model(*inputs(values, lengths, batch, device)), y, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step()
        print(f'trained {len(rows)} rows epoch {epoch + 1}/{epochs}, loss {loss.item():.3f}', flush=True)
    return model


@torch.inference_mode()
def predict(model, values, lengths, rows, config, device):
    model.eval()
    return np.concatenate([model(*inputs(values, lengths, batch, device)).cpu().numpy()
                           for batch in np.array_split(rows, math.ceil(len(rows) / config['batch_size']))])


def evaluate_fusion(public, trace_dir, table, labels, name, output):
    from .sibling_odds_replay import replay_predictions

    indexed = table.set_index('id')
    logits = indexed[['score__' + label for label in labels]].copy()
    logits.columns = labels
    with np.load(trace_dir / 'stage_scores.npz', allow_pickle=False) as trace:
        truth = public.set_index('id').loc[trace['ids']].copy()
        texts = truth.text
        if not np.array_equal(replay_predictions(trace, texts, sibling_odds=False), trace['final']):
            raise ValueError('Unmodified Dennis replay differs from the measured reference')
        settings = (0., .1, .25, .40)
        predictions = {w: replay_predictions(trace, texts, external_logits=logits, weight=w) for w in settings}
    dev = truth.cohort.eq('development')
    def ba(correct, models):
        return float(correct.groupby(models).mean().mean())
    selected = max(settings, key=lambda w: ba(truth.model.eq(predictions[w])[dev], truth.model[dev]))
    result = truth.drop(columns='text').assign(baseline=predictions[0], fused=predictions[selected])
    result.to_parquet(output / f'{name}_paired.parquet')
    summary = {}
    for cohort, rows in result.groupby('cohort'):
        before, after = rows.model.eq(rows.baseline), rows.model.eq(rows.fused)
        summary[cohort] = {'rows': len(rows), 'ba_before': ba(before, rows.model), 'ba_after': ba(after, rows.model),
                           'fixed': int((~before & after).sum()), 'harmed': int((before & ~after).sum()),
                           'changed': int(rows.fused.ne(rows.baseline).sum())}
    write_json(output / f'{name}_fusion.json', {'selected_weight_on_c1': selected, 'cohorts': summary,
               'development_ba': {str(w): ba(truth.model.eq(p)[dev], truth.model[dev]) for w, p in predictions.items()}})


def run(args):
    config = json.loads(args.config.read_text())
    official = pd.read_csv(args.official, dtype={'id': str, 'prompt_id': str})
    public = pd.read_parquet(args.trace / 'public_truth.parquet')
    frame = corpus(args.pool, official, public, config)
    args.out.mkdir(parents=True, exist_ok=True)
    labels = sorted(official.model.unique())
    folds = frame.prompt_id.map(json.loads(args.folds.read_text())).where(frame.partition.eq('official'), -1)
    if folds.isna().any() or set(folds.unique()) != {-1, 0, 1, 2, 3, 4}:
        raise ValueError('Official prompt fold assignment is incomplete')
    rows = frame.drop(columns='text').assign(fold=folds.astype(int))
    rows.to_parquet(args.out / 'sample_manifest.parquet', index=False)
    train = frame.partition.eq('external')
    write_json(args.out / 'sample.json', {'config': config, 'labels': labels, 'rows': frame.partition.value_counts().to_dict(),
               'external_class_counts': frame.loc[train, 'model'].value_counts().to_dict(),
               'external_sources': int(frame.loc[train, 'source_dataset'].nunique()),
               'external_prompts': int(frame.loc[train, 'prompt_id'].nunique())})
    if args.prepare_only:
        return
    values, lengths = extract(args.base, frame, config, args.cache)
    labels_array = np.array(labels)
    codes = pd.Categorical(frame.model, categories=labels).codes.astype(np.int64)
    ext = np.flatnonzero(train)
    off = np.flatnonzero(frame.partition.eq('official'))
    rest = np.flatnonzero(~train & ~frame.partition.eq('official'))
    fold = folds.to_numpy()
    device = 'cuda'
    for name, use_forecasts in [('states_only', False), ('states_forecasts', True)]:
        torch.manual_seed(config['seed'])
        head = FingerprintHead(config, len(labels), use_forecasts).to(device)
        fit(head, values, lengths, codes, ext, config, config['epochs'], config['learning_rate'], config['seed'], device)
        output = np.zeros((len(frame), len(labels)), np.float32)
        raw_rows = np.concatenate([off, rest])
        raw = predict(head, values, lengths, raw_rows, config, device)
        np.savez_compressed(args.out / f'{name}_external_only.npz', logits=raw, rows=raw_rows, labels=labels_array)
        for f in range(5):
            train_ids, eval_ids = off[fold[off] != f], off[fold[off] == f]
            adapted = copy.deepcopy(head)
            fit(adapted, values, lengths, codes, train_ids, config, config['adapt_epochs'],
                config['adapt_learning_rate'], config['seed'] + f, device)
            output[eval_ids] = predict(adapted, values, lengths, eval_ids, config, device)
        fit(head, values, lengths, codes, off, config, config['adapt_epochs'], config['adapt_learning_rate'],
            config['seed'], device)
        output[rest] = predict(head, values, lengths, rest, config, device)
        torch.save({'state': head.cpu().state_dict(), 'labels': labels, 'config': config, 'forecast_enabled': use_forecasts},
                   args.out / f'{name}.pt')
        table = rows.iloc[raw_rows].reset_index(drop=True)
        table = table.join(pd.DataFrame(output[raw_rows], columns=['score__' + lab for lab in labels]))
        table['prediction'] = labels_array[output[raw_rows].argmax(1)]
        table.to_parquet(args.out / f'{name}_predictions.parquet', index=False)
        evaluate_fusion(public, args.trace, table.loc[table.partition.eq('public')], labels, name, args.out)
        table['panel'] = table.cohort.fillna(table.source_dataset).fillna('official_oof')
        table['correct'] = table.model.eq(table.prediction)
        report = {}
        for panel, subset in table.groupby('panel'):
            short = subset.loc[subset.n_words.lt(100)]
            report[panel] = {'rows': len(subset), 'ba': subset.groupby('model').correct.mean().mean(),
                             'short_ba': short.groupby('model').correct.mean().mean(),
                             'recall': subset.groupby('model').correct.mean().to_dict()}
        write_json(args.out / f'{name}_metrics.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'official', 'trace', 'pool', 'folds', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--cache', type=Path, default=Path('/kaggle/temp/fingerprint-cache'))
    parser.add_argument('--prepare-only', action='store_true')
    run(parser.parse_args())
