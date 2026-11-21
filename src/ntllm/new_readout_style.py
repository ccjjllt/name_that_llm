"""Train and evaluate a prompt-contrastive StyleDistance readout on Kaggle."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import log_softmax
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from torch import nn
from transformers import AutoModel, AutoTokenizer


CONFUSIONS = (
    ("gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"),
    ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"),
    ("glm-5.3", "glm-5.3-flash", "claude-fable-5"),
    ("kimi-k3", "gpt-5.5"),
    ("claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5"),
)


class StyleEncoder(nn.Module):
    def __init__(self, model_path, class_count, prototypes=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        depth = self.encoder.config.num_hidden_layers
        self.middle = depth // 2
        self.encoder.embeddings.requires_grad_(False)
        self.encoder.encoder.layer[:self.middle].requires_grad_(False)
        width = self.encoder.config.hidden_size
        self.middle_norm = nn.LayerNorm(width)
        self.final_norm = nn.LayerNorm(width)
        self.dispersion_norm = nn.LayerNorm(width)
        self.project = nn.Sequential(nn.Linear(3 * width, 384), nn.GELU(),
                                     nn.Dropout(0.10), nn.Linear(384, 192))
        self.prototypes = nn.Parameter(torch.randn(class_count, prototypes, 192) * 0.04)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
        mask = attention_mask.unsqueeze(-1).float()
        count = mask.sum(dim=1).clamp_min(1)
        middle = (out.hidden_states[self.middle].float() * mask).sum(dim=1) / count
        final_tokens = out.last_hidden_state.float()
        final = (final_tokens * mask).sum(dim=1) / count
        dispersion = (((final_tokens - final[:, None]).square() * mask).sum(dim=1)
                      / count).clamp_min(1e-6).sqrt()
        pooled = torch.cat([self.middle_norm(middle), self.final_norm(final),
                            self.dispersion_norm(dispersion)], dim=-1)
        embedding = F.normalize(self.project(pooled).float(), dim=-1)
        prototypes = F.normalize(self.prototypes.float(), dim=-1)
        scores = 20.0 * torch.einsum("bd,cpd->bcp", embedding, prototypes)
        logits = torch.logsumexp(scores, dim=2) - math.log(self.prototypes.shape[1])
        return logits, embedding


def contrastive_loss(embedding, labels, prompts, confusions):
    same_class = labels[:, None].eq(labels[None, :])
    same_prompt = prompts[:, None].eq(prompts[None, :])
    positive = same_class & ~same_prompt
    negative = ~same_class
    weights = 1.0 + 2.0 * (same_prompt & negative) + confusions[labels[:, None], labels[None, :]]
    similarity = embedding @ embedding.T / 0.12
    # Same-prompt passages from different generators are informative negatives.
    allowed = positive | negative
    denominator = torch.logsumexp((similarity + weights.log()).masked_fill(~allowed, -1e4), dim=1)
    count = positive.sum(dim=1)
    positive_mean = (similarity * positive).sum(dim=1) / count.clamp_min(1)
    return ((denominator - positive_mean) * (count > 0)).sum() / (count > 0).sum().clamp_min(1)


def prompt_batches(frame, rng, groups_per_batch):
    groups = list(frame.groupby("prompt_id", sort=False).indices.values())
    order = rng.permutation(len(groups))
    for start in range(0, len(order), groups_per_batch):
        yield np.concatenate([groups[i] for i in order[start:start + groups_per_batch]])


def external_panel(path, official, labels, prompts, seed):
    columns = ["id", "text", "model", "prompt_canon", "prompt_evidence", "gen_reasoning",
               "flag_train_prompt_regen", "flag_near_train_prompt"]
    frame = pd.read_parquet(path, columns=columns).dropna(subset=["text", "model", "prompt_canon"])
    frame = frame.loc[frame["model"].isin(labels) & ~frame["text"].isin(official["text"])
                      & frame["prompt_evidence"].isin(["own_text", "own_bank", "other_text"])
                      & ~frame["flag_train_prompt_regen"].fillna(False)
                      & ~frame["flag_near_train_prompt"].fillna(False)]
    frame = frame.rename(columns={"prompt_canon": "prompt_id"})
    frame = frame.loc[frame["prompt_id"].astype(str).str.len().gt(0)].copy()
    frame["kimihigh"] = frame["model"].eq("kimi-k3") & frame["gen_reasoning"].eq("high")
    rng = np.random.default_rng(seed)
    # Keep one response per model per prompt and favor documented thinking-on Kimi.
    frame["draw"] = rng.random(len(frame))
    frame = frame.sort_values(["kimihigh", "draw"], ascending=False)
    frame = frame.drop_duplicates(["prompt_id", "model"])
    counts = frame.groupby("prompt_id", sort=False)["model"].size()
    candidates = counts[counts >= 10]
    weights = candidates.to_numpy(dtype=float) ** 2
    selected = rng.choice(candidates.index, min(prompts, len(candidates)), replace=False,
                          p=weights / weights.sum())
    return frame.loc[frame["prompt_id"].isin(selected), ["id", "text", "model", "prompt_id"]]


def encode(tokenizer, frame, length):
    return tokenizer(frame["text"].astype(str).tolist(), truncation=True, max_length=length,
                     return_attention_mask=False)


def tensors(tokenizer, encodings, rows, device):
    inputs = [{"input_ids": encodings["input_ids"][i]} for i in rows]
    return tokenizer.pad(inputs, padding=True, pad_to_multiple_of=8, return_tensors="pt").to(device)


def fit(model, tokenizer, frame, labels, args, *, epochs, lr, seed):
    encodings = encode(tokenizer, frame, args.max_length)
    label_ids = {label: i for i, label in enumerate(labels)}
    y = torch.tensor(frame["model"].map(label_ids).to_numpy(), device="cuda")
    p = torch.tensor(pd.factorize(frame["prompt_id"])[0], device="cuda")
    confusions = torch.zeros((len(labels), len(labels)), device="cuda")
    for group in CONFUSIONS:
        indices = torch.tensor([label_ids[label] for label in group], device="cuda")
        confusions[indices[:, None], indices[None, :]] = 1.0
    trainable_encoder = [param for param in model.encoder.parameters() if param.requires_grad]
    trainable_head = [param for name, param in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": trainable_encoder, "lr": lr},
                                   {"params": trainable_head, "lr": 12 * lr}], weight_decay=0.04)
    steps_per_epoch = math.ceil(frame["prompt_id"].nunique() / args.groups_per_batch)
    total_steps = epochs * steps_per_epoch
    warmup = max(1, total_steps // 12)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, 1.0)
        * (0.2 + 0.8 * (1 + math.cos(math.pi * max(step - warmup, 0) / max(total_steps - warmup, 1))) / 2)
    )
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(epochs):
        started = time.monotonic()
        loss_sum, total = 0.0, 0
        for rows in prompt_batches(frame, rng, args.groups_per_batch):
            idx = torch.as_tensor(rows, device="cuda")
            inputs = tensors(tokenizer, encodings, rows, "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, embedding = model(**inputs)
                loss = F.cross_entropy(logits, y[idx], label_smoothing=0.04)
                if args.contrastive_weight:
                    loss = loss + args.contrastive_weight * contrastive_loss(embedding, y[idx], p[idx], confusions)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_encoder + trainable_head, 1.0)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            loss_sum += loss.detach().item() * len(rows)
            total += len(rows)
        print(f"epoch {epoch + 1}/{epochs}: {total} passages, loss {loss_sum / total:.4f}, "
              f"{time.monotonic() - started:.0f}s", flush=True)


@torch.inference_mode()
def predict(model, tokenizer, frame, args):
    encodings = encode(tokenizer, frame, args.max_length)
    row_order = np.argsort([len(tokens) for tokens in encodings["input_ids"]])
    logits = np.empty((len(frame), model.prototypes.shape[0]), dtype=np.float32)
    model.eval()
    for start in range(0, len(frame), args.inference_batch_size):
        rows = row_order[start:start + args.inference_batch_size]
        with torch.autocast("cuda", dtype=torch.float16):
            scores, _ = model(**tensors(tokenizer, encodings, rows, "cuda"))
        logits[rows] = scores.float().cpu().numpy()
    return logits


def restricted_mix(reference, style, labels, weight, confidence=0.85):
    """Blend scores on uncertain decisions inside modeled confusions, retaining their mass."""
    probability = np.exp(log_softmax(reference, axis=1))
    winner = probability.argmax(axis=1)
    for group in CONFUSIONS:
        columns = np.array([labels.index(label) for label in group])
        current = probability[:, columns]
        mass = current.sum(axis=1, keepdims=True)
        relative = current / mass.clip(1e-12)
        rows = np.flatnonzero(np.isin(winner, columns) & (relative.max(axis=1) < confidence))
        conditional = (1 - weight) * np.log(relative[rows].clip(1e-12)) + weight * style[np.ix_(rows, columns)]
        probability[np.ix_(rows, columns)] = mass[rows] * np.exp(log_softmax(conditional, axis=1))
    return probability


def native_reference(path, official, labels):
    manifest = json.loads((path / "manifest.json").read_text())
    order = [manifest["classes"].index(label) for label in labels]
    with np.load(path / "tokens.npz") as tokens:
        by_id = pd.Index(tokens["ids"].astype(str))
        indices = by_id.get_indexer(official["id"].astype(str))
        if (indices < 0).any():
            raise ValueError("Public native OOF is missing official rows")
        folds = tokens["folds"][indices]
    with np.load(path / "oof_reference.npz") as scores:
        plain = scores["plain"][indices][:, order].astype(float)
        centered = scores["centered"][indices][:, order].astype(float)
    states, biases = {}, {}
    from safetensors import safe_open
    for fold in np.unique(folds):
        states[fold] = json.loads((path / f"calibration/fold{fold}.json").read_text())["models"]["qwen8"]
        with safe_open(str(path / f"fold{fold}/trainables.safetensors"), framework="np") as state:
            biases[fold] = state.get_tensor("centered.bias")[order]
    return folds, plain, centered, states, biases


def replay_native(native, frame, rows):
    folds, plain, centered, states, biases = native
    result = np.empty((len(rows), plain.shape[1]), dtype=float)
    for fold, state in states.items():
        positions = np.flatnonzero(folds[rows] == fold)
        current = rows[positions]
        if not len(current):
            continue
        scales = state["plain_scale"], state["centered_scale"]
        result[positions] = log_softmax(plain[current] * scales[0], axis=1)
        if len(current) < 2:
            continue
        vectorizer = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.05, sublinear_tf=True)
        vectorizer.fit(frame.loc[folds != fold, "text"])
        X = vectorizer.transform(frame.iloc[current]["text"])
        distance = (1.0 - (X @ X.T).toarray()).clip(0, 1)
        np.fill_diagonal(distance, 0.0)
        groups = AgglomerativeClustering(n_clusters=None, distance_threshold=0.85,
                                         metric="precomputed", linkage="average").fit_predict(distance)
        for group in np.unique(groups):
            inside = np.flatnonzero(groups == group)
            if 2 <= len(inside) <= 18:
                raw = centered[current[inside]]
                residual = (raw - raw.mean(axis=0) + biases[fold]) * scales[1]
                weight = state["selected_centered_weight"]
                result[positions[inside]] = ((1 - weight) * result[positions[inside]]
                                             + weight * log_softmax(residual, axis=1))
    return log_softmax(result, axis=1)


def metrics(logits, frame, labels):
    predicted = np.asarray(labels)[logits.argmax(axis=1)]
    actual = frame["model"].to_numpy()
    words = frame["text"].str.split().str.len().to_numpy()
    recalls = {}
    for label in labels:
        matches = actual == label
        recalls[label] = float(np.mean(predicted[matches] == label)) if matches.any() else None
    return {"balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "recall": recalls,
            "short_balanced_accuracy": float(balanced_accuracy_score(actual[words < 100], predicted[words < 100])),
            "long_balanced_accuracy": float(balanced_accuracy_score(actual[words >= 100], predicted[words >= 100]))}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--open-pool", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--external-prompts", type=int, default=1200)
    parser.add_argument("--external-epochs", type=int, default=2)
    parser.add_argument("--official-epochs", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--groups-per-batch", type=int, default=2)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--contrastive-weight", type=float, default=0.25)
    parser.add_argument("--prototypes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-folds", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path("/kaggle/input")
    if args.train is None:
        args.train = next(input_root.glob("**/name-that-llm/train.csv"), None)
        if args.train is None:
            raise FileNotFoundError(f"Name That LLM training data is missing from {list(input_root.iterdir())}")
    if args.model is None:
        args.model = next(path.parent for path in input_root.rglob("model.safetensors")
                          if path.parent.name == "styledistance")
    if args.open_pool is None and args.external_epochs:
        args.open_pool = next(input_root.rglob("ntl_open_pool.parquet"))
    if args.native is None:
        args.native = next(input_root.rglob("ntl-native-q8-multidepth-fivefold-0679"), None)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    official = pd.read_csv(args.train, dtype={"id": str, "prompt_id": str})
    labels = sorted(official["model"].unique())
    native = native_reference(args.native, official, labels) if args.native is not None else None
    if native is None:
        folds = np.empty(len(official), dtype=int)
        for fold, (_, heldout) in enumerate(GroupKFold(5).split(official, groups=official["prompt_id"])):
            folds[heldout] = fold
    else:
        folds = native[0]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = StyleEncoder(args.model, len(labels), args.prototypes).float().cuda()
    if args.external_epochs:
        external = external_panel(args.open_pool, official, labels, args.external_prompts, args.seed)
        print(f"External pretraining on {len(external)} passages across {external['prompt_id'].nunique()} prompts", flush=True)
        fit(model, tokenizer, external, labels, args, epochs=args.external_epochs, lr=3e-5, seed=args.seed)
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    seed_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in trainable}
    oof = np.empty((len(official), len(labels)), dtype=np.float32)
    for fold in np.unique(folds):
        model.load_state_dict(seed_weights, strict=False)
        fit(model, tokenizer, official.loc[folds != fold], labels, args,
            epochs=args.official_epochs, lr=1.5e-5, seed=args.seed + int(fold))
        oof[folds == fold] = predict(model, tokenizer, official.loc[folds == fold], args)
        selected = folds == fold
        print(f"fold {fold}: {metrics(oof[selected], official.loc[selected], labels)['balanced_accuracy']:.4f}", flush=True)
        if args.save_folds:
            from safetensors.torch import save_file
            adapted = {name: value.detach().to(device="cpu", dtype=torch.float16)
                       for name, value in model.state_dict().items() if name in trainable}
            save_file(adapted, str(args.output / f"style-fold{fold}.safetensors"))
    np.savez_compressed(args.output / "style_oof.npz", ids=official["id"].to_numpy(dtype=str),
                        folds=folds, labels=labels, logits=oof)
    results = {"style": metrics(oof, official, labels),
               "model": str(args.model), "prototypes": args.prototypes, "max_length": args.max_length}
    if native is not None:
        sparse = []
        for seed in (42, 43):
            rng = np.random.default_rng(seed)
            sparse.append(np.sort(np.concatenate([
                rng.choice(group, min(len(group), 2), replace=False)
                for group in official.groupby("prompt_id").indices.values()
            ])))
        for name, rows in [("full", np.arange(len(official))), ("pairs_42", sparse[0]), ("pairs_43", sparse[1])]:
            reference = replay_native(native, official, rows)
            frame = official.iloc[rows]
            results[name] = {"native_readout": metrics(reference, frame, labels),
                             "native_plus_style": metrics(restricted_mix(reference, oof[rows], labels, 0.25), frame, labels)}
    (args.output / "style_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"OOF balanced accuracy: {results['style']['balanced_accuracy']:.4f}", flush=True)
    for cohort in ("full", "pairs_42", "pairs_43"):
        if cohort in results:
            scores = {method: values["balanced_accuracy"] for method, values in results[cohort].items()}
            print(cohort, scores, flush=True)


if __name__ == "__main__":
    main()
