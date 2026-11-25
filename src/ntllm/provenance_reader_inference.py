"""Score an unlabeled CSV with a saved provenance reader on offline Kaggle."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd


def read_queries(path):
    queries = pd.read_csv(path, dtype={'id': str}, usecols=['id', 'text'])
    if queries.empty or queries.id.isna().any() or not queries.id.is_unique:
        raise ValueError('Queries must have one nonmissing, unique ID per passage')
    if queries.text.isna().any() or queries.text.str.strip().eq('').any():
        raise ValueError('Each query must have a nonempty body')
    return queries.set_index('id')


def predict_shard(query_file, base, artifacts, arm, deadline, shard, shards, output):
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer

    from .sibling_specialist import Specialist, predict, write_json

    started = time.monotonic()
    settings = json.loads((artifacts / 'data_report.json').read_text())['settings']
    names = pq.read_schema(artifacts / f'{arm}_logits.parquet').names
    labels = [name for name in names if name != 'id']
    if len(labels) != 18 or len(set(labels)) != 18 or names.count('id') != 1:
        raise ValueError('Expected the saved training-time 18-label order')
    queries = read_queries(query_file).iloc[shard::shards]
    if time.time() >= deadline:
        raise TimeoutError('No time remaining for the reader')

    model = Specialist(base, settings, len(labels), 'cuda:0')
    checkpoint = torch.load(artifacts / f'{arm}.pt', map_location='cpu', weights_only=True)
    trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
    if checkpoint.keys() != trainable.keys():
        raise ValueError('Checkpoint does not match the reader adapter and head')
    with torch.no_grad():
        for name, parameter in trainable.items():
            value = checkpoint[name]
            if value.shape != parameter.shape:
                raise ValueError(f'Checkpoint tensor shape mismatch for {name}')
            parameter.copy_(value)
    del checkpoint
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, padding_side='right')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.cuda.synchronize()
    ready = time.monotonic()
    arrays = []
    for start in range(0, len(queries), 768):
        if time.time() >= deadline:
            raise TimeoutError('Reader deadline reached before all queries were scored')
        arrays.append(predict(model, tokenizer, queries.iloc[start:start + 768],
                              labels, settings, 'cuda:0'))
    logits = pd.DataFrame(np.concatenate(arrays), index=queries.index, columns=labels)
    logits.to_parquet(output)
    write_json(output.with_suffix('.json'), dict(rows=len(logits), startup_seconds=ready - started,
               prediction_seconds=time.monotonic() - ready,
               gpu=torch.cuda.get_device_name(0), peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3))


def predict_reader(query_file, base, artifacts, arm, deadline, *, devices=(0, 1), scratch=Path('/kaggle/temp')):
    """Return one finite row per query in input order and the shard timings."""
    queries = read_queries(query_file)
    started = time.monotonic()
    devices = devices[:len(queries)]
    if not devices or deadline <= time.time():
        raise ValueError('Reader requires a GPU and a future wall-clock deadline')
    with tempfile.TemporaryDirectory(dir=scratch, prefix='ntl-provenance-') as temporary:
        root = Path(temporary)
        paths = [root / f'part-{i}.parquet' for i in range(len(devices))]
        jobs = []
        try:
            for i, (gpu, destination) in enumerate(zip(devices, paths)):
                env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                           CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2')
                command = [sys.executable, '-u', '-m', 'ntllm.provenance_reader_inference',
                           '--query', str(query_file), '--base', str(base), '--artifacts', str(artifacts),
                           '--arm', arm, '--deadline', repr(deadline), '--output', str(destination),
                           '--shard', str(i), '--shards', str(len(devices))]
                jobs.append(subprocess.Popen(command, env=env))
            for job in jobs:
                status = job.wait(timeout=max(0., deadline - time.time()))
                if status:
                    raise RuntimeError(f'Reader subprocess failed with exit code {status}')
        finally:
            for job in jobs:
                if job.poll() is None:
                    job.kill()
                job.wait()
        parts = [pd.read_parquet(path) for path in paths]
        logits = pd.concat(parts, verify_integrity=True)
        if set(logits.index) != set(queries.index) or not np.isfinite(logits.to_numpy()).all():
            raise ValueError('Reader must produce finite 18-class scores for every input ID')
        if any(not part.columns.equals(logits.columns) for part in parts):
            raise ValueError('Reader workers disagree about class order')
        timing = dict(rows=len(logits), seconds=time.monotonic() - started,
                      shards=[json.loads(path.with_suffix('.json').read_text()) for path in paths])
        return logits.loc[queries.index], timing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('query', 'base', 'artifacts', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--arm', choices=('length_matched', 'provenance_balanced'), required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    predict_shard(args.query, args.base, args.artifacts, args.arm, args.deadline, args.shard,
                  args.shards, args.output)
