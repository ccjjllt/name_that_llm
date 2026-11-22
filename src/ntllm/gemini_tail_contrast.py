"""Evaluate a terminal hidden-state contrast from cached frozen Qwen states."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

if __package__:
    from .gemini_sentence_order import measure_scores, official_oof, official_pairs, paired_score
    from .sibling_pair_data import GEMINI_FLASH
else:
    from gemini_sentence_order import measure_scores, official_oof, official_pairs, paired_score
    from sibling_pair_data import GEMINI_FLASH


def cached_features(root, ids):
    archive = next(root.glob('**/ntl-qwen8-tail-features/qwen8_tail_train.npz'))
    official_train = next(root.glob('**/name-that-llm/train.csv'))
    with np.load(archive, allow_pickle=False) as cache:
        if cache['train_sha256'].item() != sha256(official_train.read_bytes()).hexdigest():
            raise ValueError('Cached features were not extracted from the same official training file')
        cached_ids = pd.Index(cache['ids'].astype(str))
        order = cached_ids.get_indexer(ids)
        if len(cached_ids) != len(ids) or not cached_ids.is_unique or (order < 0).any():
            raise ValueError('Cached features and official held-out scores have different IDs')
        layer_ids = cache['layers'].tolist()
        windows = cache['tail_windows'].tolist()
        if layer_ids != [4, 6, 8, 22] or windows != [32, 64]:
            raise ValueError(f'Unexpected cached configuration: {layer_ids}, {windows}')
        mean32 = cache['mean'][order].astype(np.float32).reshape(len(ids), -1)
        mean64 = cache['mean64'][order].astype(np.float32).reshape(len(ids), -1)
        width = np.minimum(np.ravel(cache['lengths'][order]) - 1, 64)
    if mean32.shape != mean64.shape or not np.isfinite(mean32).all() or not np.isfinite(mean64).all():
        raise ValueError('Incompatible pooled feature arrays')
    if (width <= 32).any():
        raise ValueError('Cached spans are too short to reconstruct a preceding token block')
    # The cached pools exclude the first real token. Below 64 usable tokens the preceding
    # block is shorter, so undo the weighted mean with the effective window width.
    scale = (width / (width - 32))[:, None].astype(np.float32)
    contrast = scale * (mean32 - mean64)
    return {'tail_dc': mean64, 'tail_dc_contrast': np.concatenate((mean64, contrast), axis=1)}, layer_ids


def evaluate(root):
    frame, labels, logits = official_oof(root)
    features, layers = cached_features(root, frame.id)
    folds = frame.fold.to_numpy()
    gemini = frame.model.isin(GEMINI_FLASH).to_numpy()
    margin = {name: np.zeros(len(frame)) for name in features}
    fold_pair_count = {}
    with threadpool_limits(limits=2):
        for fold in np.unique(folds):
            training = official_pairs(frame, gemini & (folds != fold))
            keep = folds == fold
            for name, values in features.items():
                margin[name][keep] = paired_score(values[training[:, 0]], values[training[:, 1]], values[keep])
            fold_pair_count[str(fold)] = len(training)
            print('Fold', fold, 'matched Gemini training prompts', len(training), flush=True)
    diagnostics = measure_scores(frame, labels, logits, margin)
    return {'layers': layers, 'fit_pairs_per_fold': fold_pair_count, **diagnostics}


if __name__ == '__main__':
    results = json.dumps(evaluate(Path('/kaggle/input')), indent=2, allow_nan=False)
    Path('/kaggle/working/gemini_tail_contrast.json').write_text(results + '\n')
    print(results, flush=True)
