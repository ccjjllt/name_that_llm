"""Compare three fixed Qwen3.5 READER depths in a single frozen pass."""

import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from glm_fable_typography import GLM, compare, find_artifact, inputs
from glm_reader_apex_ab import READER_WEIGHT, reader_scores
from padilla_apex import apex_decisions
from pngy_residual import Residual
from reader_margin import rank_reader_rows
from reader_qwen35 import load_proxy


DEPTHS = (8, 13, 18)
MAX_TOKENS = 384
BATCH_SIZE = 12
CHUNK_TOKENS = 64


def pool_body(hidden, weights):
    # Project at the layer output. Keeping only 64 positions in FP32 at a time avoids
    # retaining or copying full FP32 activations across the three layers.
    pooled = weights.new_zeros((hidden.shape[0], 2, hidden.shape[2]))
    for start in range(0, hidden.shape[1], CHUNK_TOKENS):
        use = slice(start, start + CHUNK_TOKENS)
        pooled.add_(torch.bmm(weights[:, :, use], hidden[:, use].float()))
    return pooled


@torch.inference_mode()
def extract_depths(model_path, texts, device):
    model = load_proxy(model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side='right')
    tokens = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=MAX_TOKENS)['input_ids']
    tokens = [row or [tokenizer.eos_token_id] for row in tokens]
    order = sorted(range(len(tokens)), key=lambda row: len(tokens[row]))
    result = np.empty((len(texts), len(DEPTHS), 2, model.config.hidden_size), dtype=np.float32)
    pools = {}
    weights = None

    def hook_for(depth):
        def capture(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            pools[depth] = pool_body(hidden, weights)
        return capture

    handles = [model.layers[depth - 1].register_forward_hook(hook_for(depth)) for depth in DEPTHS]
    start = time.monotonic()
    offset = 0
    batch_size = BATCH_SIZE
    forwards = 0
    try:
        while offset < len(order):
            rows = order[offset:offset + batch_size]
            batch = tokenizer.pad({'input_ids': [tokens[row] for row in rows]}, padding=True,
                                  pad_to_multiple_of=8, return_tensors='pt').to(device)
            mask = batch.attention_mask.bool()
            length = mask.sum(dim=1, keepdim=True)
            position = torch.arange(mask.shape[1], device=device) + 0.5
            dc = mask / length
            ac = math.sqrt(2) * (math.pi * position / length).cos() * dc
            weights = torch.stack((dc, torch.where(length > 1, ac, 0.0)), dim=1)
            pools.clear()
            try:
                model(**batch, use_cache=False)
                result[rows] = torch.stack([pools[depth] for depth in DEPTHS], dim=1).cpu().numpy()
                offset += len(rows)
                forwards += 1
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                pools.clear()
                torch.cuda.empty_cache()
                continue
            if offset == len(order) or offset % 120 == 0:
                print(f'{device}: {offset}/{len(order)} rows, {time.monotonic() - start:.0f}s', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return result, {'rows': len(result), 'forwards': forwards, 'seconds': round(time.monotonic() - start, 2)}


def gather_features(train, root):
    rank = int(os.environ['LOCAL_RANK'])
    size = int(os.environ['WORLD_SIZE'])
    if size != 2 or torch.cuda.device_count() != 2:
        raise ValueError('This extraction requires two T4 workers')
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    if 'T4' not in torch.cuda.get_device_name(device):
        raise ValueError(f'Expected T4, received {torch.cuda.get_device_name(device)}')
    model_path = next(root.glob('**/qwen3.5-9b/1/model.safetensors.index.json')).parent
    directory = Path('/kaggle/temp/reader_qwen35_depth_ab')
    directory.mkdir(parents=True, exist_ok=True)
    torch.distributed.init_process_group('gloo', timeout=timedelta(minutes=30))
    values, stats = extract_depths(model_path, train.text.iloc[rank::size].tolist(), device)
    np.save(directory / f'rank{rank}.npy', values)
    (directory / f'rank{rank}.json').write_text(json.dumps(stats))
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    if rank != 0:
        return None, None
    out = np.empty((len(train), *values.shape[1:]), dtype=values.dtype)
    timing = {}
    for worker in range(size):
        out[worker::size] = np.load(directory / f'rank{worker}.npy')
        timing[f'cuda:{worker}'] = json.loads((directory / f'rank{worker}.json').read_text())
    if not np.isfinite(out).all():
        raise ValueError('Extracted features are not finite')
    return out, timing


def reference_features(ids, observed):
    with np.load(find_artifact('reader_qwen35_features.npz'), allow_pickle=False) as stored:
        index = pd.Index(stored['ids'].astype(str))
        order = index.get_indexer(ids)
        if len(index) != len(ids) or (order < 0).any():
            raise ValueError('Cached depth-18 features have different official IDs')
        old = stored['values'][order, 1:]
    if old.shape != observed.shape:
        raise ValueError('Cached depth-18 DC+AC shape differs')
    error = observed - old
    return {'max_abs': float(np.max(np.abs(error))), 'mean_abs': float(np.mean(np.abs(error))),
            'relative_rms': float(np.sqrt(np.mean(error ** 2) / np.mean(old ** 2)))}


def depth_readouts(frame, labels, values):
    folds = frame.fold.to_numpy()
    targets = pd.Categorical(frame.model, categories=labels).codes
    result = {}
    with threadpool_limits(limits=4):
        for index, depth in enumerate(DEPTHS):
            features = values[:, index].reshape(len(frame), -1)
            scores = np.empty((len(frame), len(labels)))
            for fold in np.unique(folds):
                hold = folds == fold
                scaler = StandardScaler()
                width = math.sqrt(features.shape[1])
                train = scaler.fit_transform(features[~hold]) / width
                validation = scaler.transform(features[hold]) / width
                head = LogisticRegression(C=16, max_iter=200)
                head.fit(train, targets[~hold])
                scores[hold] = head.decision_function(validation)
            result[f'depth{depth}'] = scores
    return result


def replay(frame, labels, base, post, logits):
    target = frame.model.to_numpy()
    texts = frame.text.fillna('').astype(str).to_numpy()
    folds = frame.fold.to_numpy()
    words = frame.text.fillna('').str.split().str.len().to_numpy()
    names = np.asarray(labels)
    current = names[post.argmax(axis=1)]
    glm_winners = np.isin(current, GLM)
    restore = np.isin(names[base.argmax(axis=1)], GLM) & (current == 'claude-fable-5')
    refined = np.empty_like(base)
    for fold in np.unique(folds):
        keep = folds != fold
        residual = Residual(labels).fit(texts[keep], target[keep])
        refined[~keep] = residual.predict_log_probability(base[~keep], texts[~keep])
    original = apex_decisions(refined, labels, texts, current)
    predicted = {name: apex_decisions((1 - READER_WEIGHT) * refined + READER_WEIGHT * scores,
                                       labels, texts, current) for name, scores in logits.items()}
    for guess in (original, *predicted.values()):
        guess[restore] = current[restore]

    chosen = rank_reader_rows(refined, glm_winners, restore)[:len(frame) // 2]
    half = {name: original.copy() for name in predicted}
    for name, guess in predicted.items():
        half[name][chosen] = guess[chosen]
    masks = {'all': np.ones(len(frame), dtype=bool),
             'short': words < 100, 'long': words >= 100,
             **{f'fold{fold}': folds == fold for fold in np.unique(folds)}}
    comparisons = {}
    for scope, guesses in (('full', predicted), ('margin50', half)):
        control = guesses['stored_depth18']
        initial = compare(target, original, control, masks['all'])
        if not np.allclose([initial['original'], initial['selective']],
                           [0.6946060154113846, 0.7031270196370868], rtol=0, atol=1e-9):
            raise ValueError(f'{scope} did not reproduce the existing depth-18 control: {initial}')
        comparisons[scope] = {name: {k: compare(target, control, guess, mask) for k, mask in masks.items()}
                              for name, guess in guesses.items() if name != 'stored_depth18'}
        comparisons[scope]['control'] = {k: compare(target, original, control, mask) for k, mask in masks.items()}
    return comparisons, int(restore.sum()), len(chosen)


def main():
    root = Path('/kaggle/input')
    output = Path('/kaggle/working')
    train = pd.read_csv('/kaggle/input/competitions/name-that-llm/train.csv', dtype={'id': str})
    train['text'] = train.text.fillna('')
    start = time.monotonic()
    values, timing = gather_features(train, root)
    if values is None:
        return

    frame, labels, base, post, selection, budget, weight = inputs()
    indices = pd.Index(train.id).get_indexer(frame.index)
    if len(train) != len(frame) or (indices < 0).any():
        raise ValueError('Extracted and OOF replay IDs differ')
    features = values[indices]
    feature_parity = reference_features(frame.index, features[:, -1])
    fresh = depth_readouts(frame, labels, features)
    original_depth18 = reader_scores(frame, labels)['dc_ac']
    depth_equal = np.mean(list(fresh.values()), axis=0)
    logits = {'stored_depth18': original_depth18, **fresh, 'equal3': depth_equal}
    outcomes, preserve, route_rows = replay(frame, labels, base, post, logits)

    np.savez(output / 'reader_qwen35_depth_features.npz', ids=train.id.to_numpy(dtype=str),
             depths=np.asarray(DEPTHS), values=values)
    saved = frame[['prompt_id', 'model', 'fold']].reset_index()
    for name, scores in fresh.items():
        saved[[f'{name}__{label}' for label in labels]] = scores
    saved.to_parquet(output / 'reader_qwen35_depth_oof.parquet', index=False)
    report = {'depths': DEPTHS, 'max_tokens': MAX_TOKENS, 'readout_c': 16, 'reader_weight': READER_WEIGHT,
              'rows': len(frame), 'workers': timing, 'seconds': time.monotonic() - start,
              'glm_rows': len(selection), 'glm_budget_positions': budget, 'glm_weight': weight,
              'fable_restoration_rows': preserve, 'margin50_rows': route_rows,
              'stored_to_new_depth18_features': feature_parity, 'outcomes': outcomes}
    encoded = json.dumps(report, indent=2, allow_nan=False)
    (output / 'reader_qwen35_depth_ab_metrics.json').write_text(encoded + '\n')
    print(encoded, flush=True)


if __name__ == '__main__':
    main()
