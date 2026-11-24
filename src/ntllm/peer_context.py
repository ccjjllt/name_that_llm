"""Predict authorship from how a second passage changes a frozen LM's reading of the first."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
import torch.nn.functional as F

from .paired_attribution import read_pool
from .sibling_odds_replay import replay_predictions
from .sibling_specialist import summarize, write_json


def make_frame(pool, official, public, blind, folds, cfg):
    exclusions = pd.concat([official, public, blind], ignore_index=True)
    train, held = read_pool(pool, exclusions)
    keep = ~train.source_dataset.str.startswith(('beroli/', 'markbarney/', 'tadeuhayashida/'))
    train = train.loc[keep & ~train.prompt_canon.isin(blind.prompt_id)].copy()
    rng = np.random.default_rng(cfg['seed'])
    freq = train.groupby(['source_dataset', 'model']).id.transform('size')
    train['priority'] = -np.log(rng.random(len(train))) * np.sqrt(freq)
    train = train.sort_values('priority').groupby('model').head(cfg['external_per_class'])
    held = held.sample(frac=1, random_state=cfg['seed']).groupby(['source_dataset', 'model']).head(
        cfg['source_holdout_per_class'])
    parts = [train.assign(partition='external', prompt_id=train.prompt_canon),
             official.assign(partition='official', fold=official.prompt_id.map(folds)),
             public.assign(partition='trace'), blind.assign(partition='blind', cohort='gen003'),
             held.assign(partition='held', prompt_id=held.prompt_canon, cohort=held.source_dataset)]
    cols = ['partition', 'cohort', 'id', 'model', 'prompt_id', 'source_dataset', 'fold', 'text']
    frame = pd.concat(parts, ignore_index=True)[cols]
    frame['n_words'] = frame.text.str.split().str.len()
    if (not frame.id.is_unique or set(frame.loc[frame.partition.eq('official'), 'fold']) != set(range(5))
            or set(train.model) != set(official.model)):
        raise ValueError('Incomplete fold/label coverage or repeated ids')
    return frame


def panel_batches(frame, cfg):
    # Training and official evaluation simulate sparse query panels. Different slices
    # of the same prompt are put in separate batches, so the retriever cannot rejoin them.
    rng = np.random.default_rng(cfg['seed'])
    for (stage, source, fold), part in frame.groupby(['partition', 'source_dataset', 'fold'],
                                                      sort=True, dropna=False):
        if stage in ('trace', 'blind', 'held'):
            yield from part.groupby('cohort', sort=True).groups.values()
            continue
        prompts = list(part.groupby('prompt_id', sort=True).groups.values())
        rng.shuffle(prompts)
        for start in range(0, len(prompts), 90):
            batches = {}
            for group in prompts[start:start + 90]:
                order = rng.permutation(group)
                width = int(rng.choice([1, 2, 4, 8]))
                for k, offset in enumerate(range(0, len(order), width)):
                    batches.setdefault(k, []).extend(order[offset:offset + width])
            yield from batches.values()


def retrieve(frame, cfg):
    train = frame.partition.eq('external')
    tfidf = TfidfVectorizer(stop_words='english', sublinear_tf=True, dtype=np.float32)
    matrix = tfidf.fit(frame.loc[train, 'text']).transform(frame.text)
    peer = np.full(len(frame), -1)
    sham = peer.copy()
    strength = np.zeros(len(frame), np.float32)
    covered = np.zeros(len(frame), dtype=int)
    words = frame.n_words.to_numpy()
    rng = np.random.default_rng(cfg['seed'] + 1)
    for group in panel_batches(frame, cfg):
        ids = np.asarray(group, dtype=int)
        covered[ids] += 1
        if len(ids) < 2:
            continue
        scores = (matrix[ids] @ matrix[ids].T).toarray()
        np.fill_diagonal(scores, -1.)
        best = scores.argmax(axis=1)
        conf = scores[np.arange(len(ids)), best]
        for i in np.flatnonzero(conf >= cfg['peer_threshold']):
            # Choose a lexically unrelated reference close in length to the true
            # reference so that the extra input tokens alone cannot explain a gain.
            other = np.flatnonzero((scores[i] >= 0) & (scores[i] < cfg['peer_threshold'] / 2))
            if not len(other):
                continue
            distance = abs(words[ids[other]] - words[ids[best[i]]])
            choices = other[np.argsort(distance, kind='stable')[:8]]
            peer[ids[i]], sham[ids[i]] = ids[best[i]], ids[rng.choice(choices)]
            strength[ids[i]] = conf[i]
    if not np.all(covered == 1) or np.any(peer == np.arange(len(frame))):
        raise ValueError('Invalid query batch layout')
    return peer, sham, strength


@torch.inference_mode()
def extract_shard(rank, model_path, frame, peers, shams, cfg, out):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_device(rank)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, local_files_only=True,
                                                attn_implementation='sdpa', device_map={'': rank}).eval()
    encode = lambda text: tok.encode(text, add_special_tokens=False)
    heading, separator = encode('Reference passage:\n'), encode('\n\nAnother passage:\n')
    tokens = tok(frame.text.tolist(), add_special_tokens=False, truncation=True, max_length=cfg['max_tokens'])['input_ids']
    gen = torch.Generator().manual_seed(cfg['seed'])
    width = model.config.hidden_size
    proj = torch.randn(width, cfg['projection_width'], generator=gen).div_(width ** .5).half().cuda()
    owned = np.arange(rank, len(frame), 2)
    lengths = np.array([len(t) for t in tokens])
    if lengths.min() == 0:
        raise ValueError('Empty target text')
    order = owned[np.argsort(lengths[owned], kind='stable')]
    indices, chunks = [], []
    def measure(batch, ref):
        contexts = [heading + ([] if j < 0 else tokens[j][:cfg['reference_tokens']]) + separator for j in ref]
        sequences = [c + tokens[i] for c, i in zip(contexts, batch)]
        inputs = torch.nn.utils.rnn.pad_sequence([torch.tensor(s) for s in sequences], batch_first=True,
                                                 padding_value=tok.eos_token_id).cuda()
        h = model.model(input_ids=inputs, use_cache=False, output_hidden_states=True)
        features = []
        for row, (c, i) in enumerate(zip(contexts, batch)):
            start, end = len(c), len(c) + len(tokens[i])
            target = inputs[row, start:end]
            before = h.last_hidden_state[row, start - 1:end - 1]
            # Each row in this slice precedes the token being scored.
            loss = torch.cat([F.cross_entropy(model.lm_head(a).float(), b, reduction='none')
                              for a, b in zip(before.split(64), target.split(64))])
            states = [h.hidden_states[model.config.num_hidden_layers // 2], h.last_hidden_state]
            projected = [F.rms_norm(state[row, start:end].float(), (width,)).half() @ proj for state in states]
            features.append(torch.cat([p.float() for p in projected] + [loss[:, None]], 1))
        return features

    def pool(x):
        # Position, dispersion and typical token evidence retain different style cues.
        return torch.cat([x.mean(0), x.std(0, unbiased=False), x[:max(1, len(x) // 3)].mean(0),
                          x[-max(1, len(x) // 3):].mean(0)])

    batch_size = max(1, cfg['tokens_per_forward'] // (cfg['max_tokens'] + cfg['reference_tokens'] + 20))
    for step, start in enumerate(range(0, len(order), batch_size)):
        batch = order[start:start + batch_size]
        plain = measure(batch, np.full(len(batch), -1))
        views = [torch.stack([pool(t) for t in plain])]
        for refs in [shams, peers]:
            active = np.flatnonzero(refs[batch] >= 0)
            change = torch.zeros_like(views[0])
            if len(active):
                context = measure(batch[active], refs[batch[active]])
                change[active] = torch.stack([pool(b - plain[i]) for i, b in zip(active, context)])
            views.append(change)
        indices.extend(batch)
        chunks.append(torch.stack(views, 1).cpu().numpy())
        if step % 200 == 0:
            print(f'GPU {rank} extracted {start}/{len(order)}', flush=True)
    values = np.concatenate(chunks)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite peer-conditioned features')
    np.savez(out / f'shard{rank}.npz', rows=indices, values=values.astype(np.float32))


def fit_head(head, x, y, rows, cfg, adapt=False):
    rng = np.random.default_rng(cfg['seed'] + int(adapt))
    count = torch.bincount(y[rows], minlength=18)
    weights = len(rows) / (len(count) * count)
    opt = torch.optim.AdamW(head.parameters(), lr=.00035 if adapt else .001)
    head.train()
    for _ in range(cfg['official_epochs' if adapt else 'external_epochs']):
        for batch in np.array_split(rng.permutation(rows), (len(rows) + 127) // 128):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(head(x[batch]), y[batch], weight=weights).backward()
            opt.step()


def train_heads(frame, features, cfg, out):
    labels = sorted(frame.model.unique())
    ext = np.flatnonzero(frame.partition.eq('external'))
    off = np.flatnonzero(frame.partition.eq('official'))
    rest = np.flatnonzero(~frame.partition.isin(['external', 'official']))
    y = torch.tensor(pd.Categorical(frame.model, categories=labels).codes.astype('int64')).cuda()
    table = frame.drop(columns='text').copy()
    result = {}
    for col, arm in enumerate(('single', 'unrelated', 'retrieved')):
        target = features[:, 0]
        effect = np.zeros_like(target) if col == 0 else features[:, col]
        a = np.concatenate([target, effect], axis=1)
        mu, sd = a[ext].mean(0), a[ext].std(0).clip(.01)
        x = torch.tensor(np.clip((a - mu) / sd, -10, 10)).cuda()
        torch.manual_seed(cfg['seed'])
        head = nn.Sequential(nn.Linear(x.shape[1], 256), nn.LayerNorm(256), nn.GELU(),
                             nn.Dropout(.15), nn.Linear(256, len(labels))).cuda()
        fit_head(head, x, y, ext, cfg)
        scores = np.full((len(frame), len(labels)), np.nan, np.float32)
        for fold in range(5):
            take = frame.fold.iloc[off].eq(fold).to_numpy()
            adapted = copy.deepcopy(head)
            fit_head(adapted, x, y, off[~take], cfg, adapt=True)
            scores[off[take]] = adapted.eval()(x[off[take]]).detach().cpu().numpy()
        fit_head(head, x, y, off, cfg, adapt=True)
        scores[rest] = head.eval()(x[rest]).detach().cpu().numpy()
        torch.save({'head': head.cpu().state_dict(), 'mean': torch.from_numpy(mu), 'scale': torch.from_numpy(sd),
                    'labels': labels, 'config': cfg}, out / (arm + '.pt'))
        take = np.concatenate([off, rest])
        table.iloc[take].join(pd.DataFrame(scores, columns=labels)).to_parquet(out / (arm + '.parquet'), index=False)
        result[arm] = scores
    return labels, result


def evaluate(frame, peers, bundle, cfg, labels, logits, out):
    pos = frame.reset_index().set_index('id').loc[bundle['ids'], 'index'].to_numpy()
    truth = frame.iloc[pos].set_index('id')
    assert list(bundle['labels']) == labels
    base = replay_predictions(bundle, truth.text)
    predictions = truth[['prompt_id', 'model', 'cohort', 'n_words']].assign(baseline=base)
    metrics = {}
    for arm in ('unrelated', 'retrieved'):
        # Subtract the matched single-passage head to isolate the context readout.
        delta = (logits[arm][pos] - logits['single'][pos])
        delta = np.clip(delta - delta.mean(1, keepdims=True), -3., 3.)
        delta[peers[pos] < 0] = 0.
        scores = {}
        for w in cfg['weights']:
            prediction = replay_predictions({**bundle, 'sibling_logp': bundle['sibling_logp'] + w * delta}, truth.text)
            predictions[f'{arm}_{w}'] = prediction
            scores[str(w)] = {name: summarize(truth.iloc[rows], base[rows], prediction[rows])
                              for name, rows in truth.groupby('cohort').indices.items()}
        chosen = max(cfg['weights'], key=lambda w: scores[str(w)]['development']['all']['ba_after'])
        metrics[arm] = {'selected_weight': chosen, 'arms': scores}
    predictions.to_parquet(out / 'paired.parquet')
    write_json(out / 'metrics.json', metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pool', 'official', 'trace', 'blind', 'blind_ids', 'folds', 'cfg', 'base', 'out'):
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    cfg = json.loads(args.cfg.read_text())
    official = pd.read_csv(args.official, dtype={'id': str, 'prompt_id': str})
    public = pd.read_parquet(args.trace / 'public_truth.parquet')
    hashes = json.loads(args.blind_ids.read_text())
    blind = pd.read_parquet(args.blind).set_index('id').loc[list(hashes)]
    if any(hashlib.sha256(t.encode()).hexdigest() != hashes[i] for i, t in blind.text.items()):
        raise ValueError('GEN003 passages differ from the archived evaluation inputs')
    blind = blind.reset_index().rename(columns={'prompt_canon': 'prompt_id'})
    frame = make_frame(args.pool, official, public, blind, json.loads(args.folds.read_text()), cfg)
    peers, shams, conf = retrieve(frame, cfg)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    ids = frame.id.to_numpy()
    pred = frame.drop(columns='text').assign(peer_id=[ids[j] if j >= 0 else None for j in peers],
                                              sham_id=[ids[j] if j >= 0 else None for j in shams], similarity=conf)
    pred.to_parquet(out / 'manifest.parquet', index=False)
    print(pred.groupby('partition').peer_id.agg(['size', 'count']), flush=True)
    if args.prepare_only:
        return
    if torch.cuda.device_count() != 2:
        raise ValueError('This extraction is sized for two T4 GPUs')
    cache = Path('/kaggle/temp/peer_context')
    cache.mkdir(parents=True, exist_ok=True)
    torch.multiprocessing.spawn(extract_shard, args=(args.base, frame[['text']], peers, shams, cfg, cache), nprocs=2)
    shards = [np.load(cache / f'shard{i}.npz') for i in range(2)]
    order = np.concatenate([s['rows'] for s in shards])
    assert np.array_equal(np.sort(order), np.arange(len(frame)))
    features = np.concatenate([s['values'] for s in shards])[np.argsort(order)]
    labels, scores = train_heads(frame, features, cfg, out)
    with np.load(args.trace / 'stage_scores.npz', allow_pickle=False) as b:
        evaluate(frame, peers, dict(b), cfg, labels, scores, out)


if __name__ == '__main__':
    main()
