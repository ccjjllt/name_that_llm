"""Evaluate single-response READER fingerprints with frozen Qwen3.5-9B."""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from transformers import AutoConfig, AutoTokenizer, Qwen3_5TextModel


DEPTH = 18
BLEND_WEIGHT = 0.20
REFERENCE = "family_mass_glm_posterior"


def load_proxy(path, device):
    config = AutoConfig.from_pretrained(path, local_files_only=True).text_config
    config.num_hidden_layers = DEPTH
    config.layer_types = config.layer_types[:DEPTH]
    config.use_cache = False
    config._attn_implementation = "sdpa"
    with init_empty_weights():
        model = Qwen3_5TextModel(config)

    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    names_by_shard = defaultdict(list)
    for name in model.state_dict():
        names_by_shard[index[f"model.language_model.{name}"]].append(name)
    weights = {}
    for shard, names in names_by_shard.items():
        with safe_open(path / shard, framework="pt", device="cpu") as tensors:
            for name in names:
                weight = tensors.get_tensor(f"model.language_model.{name}")
                weights[name] = weight.to(device=device, dtype=torch.float16)
    model.load_state_dict(weights, assign=True)
    # The paper pools the block output before the final language-model norm.
    model.norm = torch.nn.Identity()
    return model.to(device).eval()


@torch.inference_mode()
def extract(path, texts, device, max_tokens, batch_size):
    model = load_proxy(path, device)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side="right")
    tokens = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"]
    tokens = [row or [tokenizer.eos_token_id] for row in tokens]
    ordering = sorted(range(len(tokens)), key=lambda row: len(tokens[row]))
    fingerprints = np.empty((len(texts), 3, model.config.hidden_size), dtype=np.float32)
    start = time.monotonic()
    offset = 0
    while offset < len(ordering):
        rows = ordering[offset:offset + batch_size]
        batch = tokenizer.pad({"input_ids": [tokens[row] for row in rows]},
                              padding=True, pad_to_multiple_of=8, return_tensors="pt").to(device)
        try:
            hidden = model(**batch, use_cache=False).last_hidden_state.float()
            mask = batch.attention_mask.bool()
            count = mask.sum(dim=1, keepdim=True)
            position = torch.arange(hidden.size(1), device=device) + 0.5
            dc_weights = mask / count
            ac_weights = math.sqrt(2) * (math.pi * position / count).cos() * dc_weights
            ac_weights = torch.where(count > 1, ac_weights, 0.0)
            last = hidden[torch.arange(len(rows), device=device), count[:, 0] - 1]
            dct = torch.bmm(torch.stack((dc_weights, ac_weights), dim=1), hidden)
            fingerprints[rows] = torch.cat((last[:, None], dct), dim=1).cpu().numpy()
            offset += len(rows)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
            continue
        if offset == len(rows) or offset == len(ordering) or offset % 256 == 0:
            print(f"{device}: {offset}/{len(ordering)} responses, {time.monotonic() - start:.0f}s", flush=True)
    return fingerprints


def report(y, logits, folds, words, labels, reference):
    pred = logits.argmax(axis=1)
    old = reference.argmax(axis=1)
    correct = pred == y
    short = words < 100
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, np.exp(log_softmax(logits, axis=1)), labels=np.arange(len(labels)))),
        "short_balanced_accuracy": float(balanced_accuracy_score(y[short], pred[short])),
        "long_balanced_accuracy": float(balanced_accuracy_score(y[~short], pred[~short])),
        "recall": {label: float(correct[y == index].mean()) for index, label in enumerate(labels)},
        "folds": {str(fold): float(balanced_accuracy_score(y[folds == fold], pred[folds == fold]))
                  for fold in np.unique(folds)},
        "fixed": int((correct & (old != y)).sum()),
        "harmed": int((~correct & (old == y)).sum()),
    }


def readout(training, y):
    scaler = StandardScaler()
    standardized = scaler.fit_transform(training) / math.sqrt(training.shape[1])
    head = LogisticRegression(C=16, max_iter=200).fit(standardized, y)
    coef = head.coef_ / (math.sqrt(training.shape[1]) * scaler.scale_)
    return coef, head.intercept_ - coef @ scaler.mean_


def views(values):
    return {"last": values[:, 0], "dc": values[:, 1], "dc_ac": values[:, 1:].reshape(len(values), -1)}


def evaluate(frame, values, labels, output):
    folds = frame.fold.to_numpy()
    y = pd.Categorical(frame.model, categories=labels).codes
    reference = frame[[f"{REFERENCE}__{label}" for label in labels]].to_numpy()
    words = frame.text.str.split().str.len().to_numpy()
    details = {"reference": report(y, reference, folds, words, labels, reference)}
    predictions = frame[["id", "prompt_id", "model", "fold"]].copy()
    for name, array in views(values).items():
        logits = np.empty_like(reference)
        for fold in np.unique(folds):
            held = folds == fold
            coef, intercept = readout(array[~held], y[~held])
            logits[held] = array[held] @ coef.T + intercept
        fused = (1 - BLEND_WEIGHT) * reference + BLEND_WEIGHT * logits
        details[name] = report(y, logits, folds, words, labels, reference)
        details[f"blend_{name}"] = report(y, fused, folds, words, labels, reference)
        predictions[[f"{name}__{label}" for label in labels]] = logits
    predictions.to_parquet(output / "reader_qwen35_oof.parquet", index=False)
    return details


def distributed_extract(root, texts, args):
    rank, world = int(os.getenv("LOCAL_RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    model_path = next(root.glob("**/qwen3.5-9b/1/model.safetensors.index.json")).parent
    gathered = Path("/kaggle/temp/reader_qwen35")
    gathered.mkdir(parents=True, exist_ok=True)
    if world > 1:
        from datetime import timedelta
        torch.distributed.init_process_group("gloo", timeout=timedelta(hours=4))
    part = extract(model_path, texts.iloc[rank::world].tolist(), device, args.max_tokens, args.batch_size)
    np.save(gathered / f"rank{rank}.npy", part)
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if rank != 0:
        return
    values = np.empty((len(texts), *part.shape[1:]), dtype=np.float32)
    for worker in range(world):
        values[worker::world] = np.load(gathered / f"rank{worker}.npy")
    if not np.isfinite(values).all():
        raise ValueError("The proxy produced non-finite fingerprints")
    return values


def export_head(train, features, output):
    training = train.set_index("id", verify_integrity=True)
    labels = sorted(training.model.unique())
    with np.load(features, allow_pickle=False) as archive:
        ids = pd.Index(archive["ids"].astype(str))
        if not ids.is_unique or set(ids) != set(training.index):
            raise ValueError("The READER features must cover the official training rows")
        y = pd.Categorical(training.loc[ids, "model"], categories=labels).codes
        values = archive["values"]
    params = {"labels": np.asarray(labels)}
    for name, array in views(values).items():
        if name == "last":
            continue
        coef, intercept = readout(array, y)
        params[f"{name}_coef"], params[f"{name}_intercept"] = coef, intercept
    np.savez_compressed(output / "reader_qwen35_head.npz", **params)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export-features", type=Path)
    mode.add_argument("--test", type=Path)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--view", choices=("dc", "dc_ac"), default="dc_ac")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    root = Path("/kaggle/input")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.test:
        if args.head is None:
            parser.error("--test requires --head")
        frame = pd.read_csv(args.test, dtype={"id": str})
        values = distributed_extract(root, frame.text.fillna(""), args)
        if values is not None:
            with np.load(args.head, allow_pickle=False) as head:
                selected = views(values)[args.view]
                logits = selected @ head[f"{args.view}_coef"].T + head[f"{args.view}_intercept"]
                if not np.isfinite(logits).all():
                    raise ValueError("The READER predictions contain nonfinite scores")
                np.savez(args.output / "reader_qwen35_test.npz", ids=frame.id.to_numpy(dtype=str),
                         labels=head["labels"], logits=logits)
        return

    train = pd.read_csv(next(root.glob("**/name-that-llm/train.csv")), dtype={"id": str, "prompt_id": str})
    if args.export_features:
        with threadpool_limits(limits=4):
            export_head(train, args.export_features, args.output)
        return
    train["text"] = train.text.fillna("")
    labels = sorted(train.model.unique())
    columns = [f"{REFERENCE}__{label}" for label in labels]
    reference = pd.read_parquet(next(root.rglob("glm_dynamics_oof.parquet")),
                                columns=["id", "prompt_id", "model", "fold", *columns])
    reference[["id", "prompt_id"]] = reference[["id", "prompt_id"]].astype(str)
    frame = train.merge(reference, on=["id", "prompt_id", "model"], validate="one_to_one")
    if len(frame) != len(train) or frame.fold.nunique() != 5 or frame.groupby("prompt_id").fold.nunique().max() != 1:
        raise ValueError("Expected the official five prompt-disjoint folds")
    started = time.monotonic()
    values = distributed_extract(root, frame.text, args)
    if values is None:
        return
    np.savez(args.output / "reader_qwen35_features.npz", ids=frame.id.to_numpy(dtype=str), values=values)
    with threadpool_limits(limits=4):
        metrics = evaluate(frame, values, labels, args.output)
    result = {"depth": DEPTH, "max_tokens": args.max_tokens, "blend_weight": BLEND_WEIGHT,
              "rows": len(frame), "seconds": time.monotonic() - started, "metrics": metrics}
    (args.output / "reader_qwen35_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, metrics in metrics.items():
        print(name, metrics["balanced_accuracy"], metrics["fixed"], metrics["harmed"], flush=True)


if __name__ == "__main__":
    main()
