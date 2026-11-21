"""Fit author-conditioned token likelihoods and compare them on held-out prompts."""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import log_softmax, logsumexp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__:
    from .source_routing import select_candidate_rows
else:
    from source_routing import select_candidate_rows


class SourceProjection(nn.Module):
    def __init__(self, base, n_classes, rank, route):
        super().__init__()
        self.base = base
        self.route = route
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(n_classes, base.out_features, rank, device=base.weight.device))
        nn.init.normal_(self.a, std=base.in_features ** -0.5)

    def forward(self, x):
        update = F.linear(x, self.a.to(x.dtype))
        right = self.b.index_select(0, self.route[0]).transpose(1, 2).to(x.dtype)
        return self.base(x) + 2 * torch.bmm(update, right)


class SourceLikelihood(nn.Module):
    def __init__(self, model_dir, n_classes, device, rank=8):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=torch.float16,
            attn_implementation="sdpa", device_map={"": str(device)},
        ).requires_grad_(False)
        self.model.config.use_cache = False
        self.route = [None]
        embeddings = self.model.get_input_embeddings().weight
        self.control = nn.Parameter(torch.zeros(n_classes, 1, embeddings.shape[1], device=device))
        for layer in self.model.model.layers:
            for name in ("q_proj", "v_proj"):
                projection = getattr(layer.self_attn, name)
                setattr(layer.self_attn, name, SourceProjection(projection, n_classes, rank, self.route))
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward(self, ids, attention, sources, token_rows, positions, weights):
        self.route[0] = sources
        embeddings = self.model.get_input_embeddings()(ids)
        prefix = self.control[sources].to(embeddings.dtype)
        attention = torch.cat((torch.ones_like(attention[:, :1]), attention), dim=1)
        hidden = self.model.model(
            inputs_embeds=torch.cat((prefix, embeddings), dim=1),
            attention_mask=attention, use_cache=False,
        ).last_hidden_state
        vectors = hidden[token_rows, positions]
        target = ids[token_rows, positions]
        head = self.model.get_output_embeddings()
        losses = [
            F.cross_entropy(head(part).float(), y, reduction="none")
            for part, y in zip(vectors.split(192), target.split(192))
        ]
        logp = -torch.cat(losses) * weights
        total = logp.new_zeros(len(ids)).scatter_add(0, token_rows, logp)
        count = logp.new_zeros(len(ids)).scatter_add(0, token_rows, weights)
        return total / count


def encode(tokenizer, texts, limit, score_tokens):
    encoded = tokenizer(
        texts, add_special_tokens=False, truncation=True, max_length=limit - 1,
        return_offsets_mapping=True,
    )
    by_text = []
    for text, ids, offsets in zip(texts, encoded["input_ids"], encoded["offset_mapping"]):
        ids = [tokenizer.eos_token_id] + (ids or [tokenizer.eos_token_id])
        shape = [
            any(not ch.isalnum() and not ch.isspace() or ch == "\n" for ch in text[first:last])
            for first, last in offsets
        ] or [False]
        positions = np.arange(1, len(ids))
        style = np.flatnonzero(shape) + 1
        plain = np.flatnonzero(~np.asarray(shape)) + 1
        if len(positions) > score_tokens:
            style = style[np.linspace(0, len(style) - 1, min(len(style), score_tokens // 2), dtype=int)]
            plain_limit = score_tokens - len(style)
            plain = plain[np.linspace(0, len(plain) - 1, min(len(plain), plain_limit), dtype=int)]
            positions = np.sort(np.concatenate((style, plain)))
        weights = 1.0 + np.asarray(shape)[positions - 1].astype(np.float32)
        by_text.append((ids, positions, weights))
    return by_text


def candidate_batches(encoded, rows, candidates, pad, device):
    width = max(len(encoded[i][0]) for i in rows)
    k = candidates.shape[1]
    ids = torch.full((len(rows) * k, width), pad, dtype=torch.long)
    attention = torch.zeros_like(ids)
    positions, token_rows, token_weights = [], [], []
    for i, row in enumerate(rows):
        passage, position, weight = encoded[row]
        start = i * k
        ids[start:start + k, :len(passage)] = torch.as_tensor(passage)
        attention[start:start + k, :len(passage)] = 1
        token_rows.extend(np.repeat(np.arange(start, start + k), len(position)))
        positions.extend(np.tile(position, k))
        token_weights.extend(np.tile(weight, k))
    return (
        ids.to(device), attention.to(device), torch.as_tensor(candidates.ravel(), device=device),
        torch.as_tensor(token_rows, device=device), torch.as_tensor(positions, device=device),
        torch.as_tensor(token_weights, device=device, dtype=torch.float32),
    )


def training_candidates(frame, fit, target, labels):
    """Choose same- and cross-family decoys from other prompts in this fold."""
    from scipy.sparse import csr_matrix

    subset = frame.iloc[fit]
    x = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=5, max_features=16000,
                        lowercase=False, dtype=np.float32).fit_transform(subset.text.fillna(""))
    y = target[fit]
    indicator = csr_matrix((np.ones(len(fit), dtype=np.float32), (np.arange(len(fit)), y)),
                            shape=(len(fit), len(labels)))
    class_sum = (indicator.T @ x).toarray()
    family = np.array([label.split("-", 1)[0] for label in labels])
    result = np.empty((len(frame), 3), dtype=np.int64)
    for group in subset.groupby("prompt_id", sort=False).indices.values():
        centers = class_sum - (indicator[group].T @ x[group]).toarray()
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
        similarity = np.asarray(x[group] @ centers.T)
        for row, scores, actual in zip(group, similarity, y[group]):
            peer = (family == family[actual]) & (np.arange(len(labels)) != actual)
            other = family != family[actual]
            first = np.where(peer if peer.any() else other, scores, -np.inf).argmax()
            other[first] = False
            second = np.where(other, scores, -np.inf).argmax()
            result[fit[row]] = (actual, first, second)
    return result


def native_reference(directory, ids, labels):
    directory = Path(directory)
    source = json.loads((directory / "manifest.json").read_text())["classes"]
    with np.load(directory / "tokens.npz") as index, np.load(directory / "oof_reference.npz") as probabilities:
        rows = pd.Index(index["ids"].astype(str)).get_indexer(ids)
        if (rows < 0).any():
            raise ValueError("Some official texts are missing from the native predictions")
        folds = index["folds"][rows].astype(int)
        logits = probabilities["plain"][rows][:, [source.index(c) for c in labels]]
    scale = np.empty(len(ids))
    for fold in np.unique(folds):
        cal = json.loads((directory / "calibration" / f"fold{fold}.json").read_text())
        scale[folds == fold] = cal["models"]["qwen8"]["plain_scale"]
    return log_softmax(logits * scale[:, None], axis=1), folds


@torch.no_grad()
def score_candidates(model, encoded, rows, candidates, pad, device, batch_size):
    model.eval()
    candidates = candidates.astype(np.int64)
    result = np.empty(candidates.shape, dtype=np.float32)
    lengths = np.array([len(encoded[row][0]) for row in rows])
    order = np.argsort(-lengths, kind="stable")
    for start in range(0, len(order), batch_size):
        item = order[start:start + batch_size]
        batch = candidate_batches(encoded, rows[item], candidates[item], pad, device)
        result[item] = model(*batch).reshape(len(item), -1).cpu().numpy()
    return 12.0 * (result - result[:, :1])


def fit_fold(args, fold, fit, valid, encoded, frame, target, labels, native, pad, device):
    rng = np.random.default_rng(8765 + fold)
    torch.manual_seed(8765 + fold)
    model = SourceLikelihood(args.model, len(labels), device)
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer = torch.optim.AdamW(trainable.values(), lr=8e-4, weight_decay=0.015)
    scaler = torch.amp.GradScaler("cuda", init_scale=4096.0)
    train_classes = training_candidates(frame, fit, target, labels)
    lengths = np.array([len(row[0]) for row in encoded])
    step = 0
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        randomized = rng.permutation(fit)
        blocks = np.array_split(randomized, max(1, len(fit) // 192))
        epoch_loss = []
        for block in blocks:
            block = block[np.argsort(-lengths[block], kind="stable")]
            for first in range(0, len(block), args.batch_size):
                rows = block[first:first + args.batch_size]
                batch = candidate_batches(encoded, rows, train_classes[rows], pad, device)
                likelihood = model(*batch).reshape(len(rows), -1)
                loss = F.cross_entropy(12.0 * likelihood, torch.zeros(len(rows), device=device, dtype=torch.long))
                loss = loss - 0.075 * likelihood[:, 0].mean()
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable.values(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                step += 1
                epoch_loss.append(float(loss.detach()))
                if step >= args.max_steps:
                    break
            if step >= args.max_steps:
                break
        print(f"fold {fold} epoch {epoch + 1}: {step} updates, loss {np.mean(epoch_loss):.4f}, "
              f"{time.monotonic() - started:.0f}s", flush=True)
        if step >= args.max_steps:
            break

    positions, candidates = fold_candidates(args, frame, native, valid)
    query = valid[positions]
    ratios = score_candidates(model, encoded, query, candidates, pad, device, args.batch_size)
    output = Path(args.output)
    np.savez(
        output / f"native_candidates_fold{fold}.npz",
        ids=frame.id.iloc[query].to_numpy(dtype=str),
        candidates=np.asarray(labels)[candidates],
        log_ratios=ratios,
    )
    torch.save(
        {"weights": {name: parameter.detach().cpu() for name, parameter in trainable.items()},
         "labels": labels, "max_tokens": args.max_tokens, "score_tokens": args.score_tokens},
        output / f"source_likelihood_fold{fold}.pt",
    )


def read_data(args):
    frame = pd.read_csv(args.train, dtype={"id": str, "prompt_id": str})
    labels, target = np.unique(frame.model, return_inverse=True)
    native, folds = native_reference(args.native_reference, frame.id, labels.tolist())
    if pd.Series(folds).groupby(frame.prompt_id).nunique().max() != 1:
        raise ValueError("Native folds split a shared prompt")
    if args.post_glm_reference:
        oof = pd.read_parquet(args.post_glm_reference).set_index("id").loc[frame.id]
        if not np.array_equal(folds, oof.fold.to_numpy()):
            raise ValueError("Post-GLM and Native folds disagree")
        native = oof[[f"family_mass_glm_posterior__{name}" for name in labels]].to_numpy()
    return frame, labels.tolist(), target, native, folds


def fold_candidates(args, frame, native, rows):
    words = frame.text.iloc[rows].fillna("").str.split().str.len().to_numpy()
    return select_candidate_rows(native[rows], words, args.coverage, args.candidates)


def run_worker(args, wanted, gpu):
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    frame, labels, target, native, folds = read_data(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoded = encode(tokenizer, frame.text.fillna("").tolist(), args.max_tokens, args.score_tokens)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for fold in wanted:
        fit = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        fit_fold(args, fold, fit, valid, encoded, frame, target, labels, native, pad, device)
        torch.cuda.empty_cache()


def report(args, wanted):
    frame, labels, target, native, folds = read_data(args)
    selected = np.flatnonzero(np.isin(folds, wanted))
    logits = {scale: native.copy() for scale in (0.0, 0.25, 0.50, 1.0)}
    lookup = {label: i for i, label in enumerate(labels)}
    for fold in wanted:
        valid = np.flatnonzero(folds == fold)
        positions, _ = fold_candidates(args, frame, native, valid)
        rows = valid[positions]
        with np.load(Path(args.output) / f"native_candidates_fold{fold}.npz") as result:
            if not np.array_equal(result["ids"], frame.id.iloc[rows].to_numpy(dtype=str)):
                raise ValueError("Fold predictions and native ids are out of order")
            candidates = np.vectorize(lookup.__getitem__)(result["candidates"])
            row = rows[:, None]
            initial = native[row, candidates]
            for scale in logits:
                updated = initial + scale * result["log_ratios"]
                logits[scale][row, candidates] = (
                    updated - logsumexp(updated, axis=1, keepdims=True)
                    + logsumexp(initial, axis=1, keepdims=True)
                )
    words = frame.text.fillna("").str.split().str.len().to_numpy()[selected]
    truth = target[selected]
    metrics = {}
    for scale, matrix in logits.items():
        pred = matrix[selected].argmax(axis=1)
        metrics[str(scale)] = {
            "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
            "per_model": {name: float((pred[truth == i] == i).mean()) for i, name in enumerate(labels) if (truth == i).any()},
            "by_length": {
                f"{low}-{high - 1}": float(balanced_accuracy_score(truth[r], pred[r]))
                for low, high in ((0, 70), (70, 120), (120, 200), (200, 1000))
                if (r := (words >= low) & (words < high)).any()
            },
        }
    (Path(args.output) / "source_likelihood_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--native-reference", type=Path, required=True)
    parser.add_argument("--post-glm-reference", type=Path)
    parser.add_argument("--train", default="/kaggle/input/competitions/name-that-llm/train.csv")
    parser.add_argument("--output", default="/kaggle/working")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--score-tokens", type=int, default=40)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--coverage", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    gpu_count = min(torch.cuda.device_count(), len(args.folds))
    if not gpu_count:
        raise RuntimeError("This experiment requires a CUDA GPU")
    cpu = mp.get_context("spawn")
    workers = [
        cpu.Process(target=run_worker, args=(args, args.folds[i::gpu_count], i))
        for i in range(gpu_count)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if any(worker.exitcode != 0 for worker in workers):
        raise RuntimeError("At least one source likelihood fold failed")
    report(args, args.folds)


if __name__ == "__main__":
    main()
