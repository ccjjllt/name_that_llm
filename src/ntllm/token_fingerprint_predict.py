"""Apply a fitted Q4 token-fingerprint head to a CSV or Parquet containing id and text."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from .token_fingerprint import FingerprintHead, extract, predict


def read_texts(path):
    if path.suffix.lower() == '.parquet':
        frame = pd.read_parquet(path, columns=['id', 'text'])
    elif path.suffix.lower() == '.csv':
        frame = pd.read_csv(path, usecols=['id', 'text'], dtype={'id': str}, keep_default_na=False)
    else:
        raise ValueError('Provide a CSV or Parquet file')
    if frame.empty or frame.id.isna().any():
        raise ValueError('The input must contain passages with IDs')
    frame['id'] = frame.id.astype(str)
    if not frame.id.is_unique or frame.id.str.strip().eq('').any():
        raise ValueError('Passage IDs must be unique and nonempty')
    if any(not isinstance(value, str) or not value.strip() for value in frame.text):
        raise ValueError('Passages must contain nonempty text')
    return frame[['id', 'text']]


def score_table(ids, scores, labels):
    if scores.shape != (len(ids), len(labels)) or not np.isfinite(scores).all():
        raise ValueError('Expected one finite score per passage and label')
    table = pd.DataFrame(scores, columns=['score__' + lab for lab in labels])
    table.insert(0, 'id', list(ids))
    return table


def score_file(args):
    frame = read_texts(args.input)
    checkpoint = torch.load(args.head, map_location='cpu', weights_only=True)
    labels = checkpoint['labels']
    if len(set(labels)) != 18 or len(labels) != 18:
        raise ValueError('Expected the 18 unique training labels')
    config = checkpoint['config']
    head = FingerprintHead(config, len(labels), checkpoint['forecast_enabled'])
    head.load_state_dict(checkpoint['state'])
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.cache_dir) as temporary:
        values, lengths = extract(args.base, frame, config, Path(temporary))
        scores = predict(head.to('cuda'), values, lengths, np.arange(len(frame)), config, 'cuda')
        del values
    table = score_table(frame.id, scores, labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('input', 'head', 'base', 'out'):
        parser.add_argument('--' + key, required=True, type=Path)
    parser.add_argument('--cache-dir', type=Path)
    score_file(parser.parse_args())
