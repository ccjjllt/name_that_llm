"""Benchmark the cached StyleDistance encoder on the same real CPU inputs."""

from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize


ROWS = 96
BATCH = 24
MAX_TOKENS = 192
THREADS = 4


def cpu_info():
    info = Path('/proc/cpuinfo').read_text().splitlines()
    name = next((line.split(':', 1)[1].strip() for line in info if line.startswith('model name')), 'unknown')
    flags = [set(line.split(':', 1)[1].split()) for line in info if line.startswith('flags')]
    common = set.intersection(*flags) if flags else set()
    return dict(model=name, flags=[flag for flag in sorted(common) if 'avx' in flag or 'bf16' in flag or 'amx' in flag],
                torch_cpu_capability=torch.backends.cpu.get_cpu_capability(),
                mkldnn_available=torch.backends.mkldnn.is_available(), mkldnn_enabled=torch.backends.mkldnn.enabled,
                threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())


def parameter_dtypes(model):
    result = Counter()
    for parameter in model.parameters():
        result[str(parameter.dtype)] += parameter.numel()
    return dict(result)


def measure(model, inputs):
    @torch.inference_mode()
    def embed(batch):
        features = model(**batch).last_hidden_state.float()
        mask = batch['attention_mask'][..., None]
        mean = (features * mask).sum(1) / mask.sum(1)
        return normalize(mean, dim=1).numpy()

    start = perf_counter()
    embed(inputs[-1])
    warmup = perf_counter() - start
    batches, outputs = [], []
    for batch in inputs:
        start = perf_counter()
        outputs.append(embed(batch))
        elapsed = perf_counter() - start
        batches.append(elapsed)
        print(str(model.dtype), 'timed batch', len(batches), round(elapsed, 3), 's', flush=True)
    values = np.concatenate(outputs)
    if values.shape[0] != ROWS or not np.isfinite(values).all():
        raise ValueError('Expected a finite embedding for every sampled passage')
    total = sum(batches)
    return values, dict(parameter_elements_by_dtype=parameter_dtypes(model), warmup_seconds=warmup,
                        encoding_seconds=total, batch_seconds=batches, rows_per_second=ROWS / total)


def describe(values):
    return dict(min=float(np.min(values)), median=float(np.median(values)), mean=float(np.mean(values)),
                p95=float(np.percentile(values, 95)), max=float(np.max(values)))


def run(official, encoder, output):
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    train = pd.read_csv(official, usecols=['id', 'text'], dtype={'id': str})
    if len(train) != 2699 or not train.id.is_unique or train.text.isna().any():
        raise ValueError('Expected the fixed official training text and IDs')
    ordered = train.assign(chars=train.text.str.len()).sort_values(['chars', 'id'])
    positions = np.linspace(0, len(ordered) - 1, ROWS, dtype=int)
    sample = ordered.iloc[positions]
    tokenizer = AutoTokenizer.from_pretrained(encoder, local_files_only=True)
    start = perf_counter()
    inputs = [tokenizer(sample.text.iloc[lo:lo + BATCH].tolist(), padding=True, truncation=True,
                        max_length=MAX_TOKENS, return_tensors='pt') for lo in range(0, ROWS, BATCH)]
    tokenizer_seconds = perf_counter() - start
    start = perf_counter()
    model = AutoModel.from_pretrained(encoder, local_files_only=True, use_safetensors=True).eval()
    load_seconds = perf_counter() - start
    native_dtypes = parameter_dtypes(model)
    native_is_bf16 = set(native_dtypes) == {'torch.bfloat16'}
    if not native_is_bf16:
        model.to(dtype=torch.bfloat16)
    print('Cached tokenizer and CPU model loaded', native_dtypes, cpu_info(), flush=True)
    bf16, bf16_timing = measure(model, inputs)
    start = perf_counter()
    model.float()
    fp32_conversion_seconds = perf_counter() - start
    fp32, fp32_timing = measure(model, inputs)

    bf16 = bf16.astype(np.float64)
    fp32 = fp32.astype(np.float64)
    fp32_norm = np.linalg.norm(fp32, axis=1)
    cosine = np.clip((bf16 * fp32).sum(1) / (np.linalg.norm(bf16, axis=1) * fp32_norm), -1., 1.)
    relative_l2 = np.linalg.norm(bf16 - fp32, axis=1) / fp32_norm
    token_lengths = torch.cat([batch['attention_mask'].sum(1) for batch in inputs]).numpy()
    report = dict(rows=ROWS, batch_size=BATCH, max_tokens=MAX_TOKENS, cpu=cpu_info(),
                  sample_ordered_ids_sha256=hashlib.sha256(('\n'.join(sample.id) + '\n').encode()).hexdigest(),
                  raw_chars=describe(sample.chars), effective_input_tokens=describe(token_lengths),
                  padded_batch_widths=[int(batch['input_ids'].shape[1]) for batch in inputs],
                  tokenizer_seconds=tokenizer_seconds, model_load_seconds=load_seconds,
                  native_parameter_elements_by_dtype=native_dtypes, native_load_was_bf16=native_is_bf16,
                  fp32_conversion_seconds=fp32_conversion_seconds, bf16=bf16_timing, fp32=fp32_timing,
                  fp32_speedup_vs_bf16=bf16_timing['encoding_seconds'] / fp32_timing['encoding_seconds'],
                  embedding_cosine=describe(cosine), one_minus_cosine=describe(1. - cosine),
                  relative_l2_to_fp32=describe(relative_l2))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Measured CPU precision comparison', report, flush=True)
