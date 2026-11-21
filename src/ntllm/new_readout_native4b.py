"""Train a Qwen3-Embedding-4B replacement for the native attribution readout."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModel, AutoTokenizer

from new_readout_style import CONFUSIONS, external_panel, metrics, native_reference, replay_native


INSTRUCTION = "Instruct: Identify the language model that generated the text based on its writing style.\nQuery: "
DEPTHS = (3, 6, 9, 12, 18, 24, 30, 36)
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MAX_LENGTH = 512
CENTER_WEIGHT = 0.35


class LoRAProjection(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device)
        self.b = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device)
        nn.init.zeros_(self.b.weight)

    def forward(self, hidden):
        residual = self.b(self.a(F.dropout(hidden.float(), 0.05, self.training)))
        output = self.base(hidden)
        return output + (2 * residual).to(output.dtype)


class Native4B(nn.Module):
    def __init__(self, path, device, label_count, last_layers=12, rank=8):
        super().__init__()
        base = AutoModel.from_pretrained(path, local_files_only=True, torch_dtype=torch.float16,
                                        attn_implementation="sdpa", device_map={"": str(device)})
        base.requires_grad_(False)
        n_layers = base.config.num_hidden_layers
        targets = [(name, layer) for name, layer in base.named_modules()
                   if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in TARGETS
                   and name.startswith("layers.") and int(name.split(".")[1]) >= n_layers - last_layers]
        if len(targets) != last_layers * len(TARGETS):
            raise ValueError(f"Expected {last_layers * len(TARGETS)} projection layers, found {len(targets)}")
        for name, layer in targets:
            parent, attr = name.rsplit(".", 1)
            setattr(base.get_submodule(parent), attr, LoRAProjection(layer, rank))
        self.encoder = base
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        width = base.config.hidden_size
        self.norms = nn.ModuleList([nn.LayerNorm(width, device=device) for _ in DEPTHS])
        self.plain = nn.Linear(width * len(DEPTHS), label_count, device=device)
        self.centered = nn.Linear(width * len(DEPTHS), label_count, device=device)
        self.centered.load_state_dict(self.plain.state_dict())

    def forward(self, input_ids, attention_mask, center=False):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True, return_dict=True, use_cache=False)
        rows = torch.arange(len(input_ids), device=input_ids.device)
        last = attention_mask.sum(1) - 1
        layers = [out.hidden_states[d][rows, last].float() for d in DEPTHS[:-1]]
        layers.append(F.normalize(out.last_hidden_state[rows, last].float(), dim=-1))
        features = torch.cat([norm(values) for norm, values in zip(self.norms, layers)], dim=1)
        plain = self.plain(F.dropout(features, 0.1, self.training))
        residual = features - features.mean(dim=0, keepdim=True) if center else features
        return plain.float(), self.centered(F.dropout(residual, 0.1, self.training)).float()


def windows(tokenizer, texts):
    prefix = tokenizer(INSTRUCTION, add_special_tokens=False)["input_ids"]
    terminal = tokenizer("", add_special_tokens=True)["input_ids"]
    terminal = terminal[-1:] or [tokenizer.eos_token_id]
    rows = tokenizer([INSTRUCTION + str(text) for text in texts], add_special_tokens=False)["input_ids"]
    first_limit = MAX_LENGTH - len(terminal)
    capacity = first_limit - len(prefix)
    seqs, owners, weights = [], [], []
    for row, tokens in enumerate(rows):
        seqs.append(tokens[:first_limit] + terminal)
        owners.append(row)
        weights.append(max(1, min(len(tokens), first_limit) - len(prefix)))
        for start in range(first_limit, len(tokens), capacity):
            chunk = tokens[start:start + capacity]
            seqs.append(prefix + chunk + terminal)
            owners.append(row)
            weights.append(len(chunk))
    return seqs, np.asarray(owners), np.asarray(weights)


def padded(tokenizer, tokens, rows, device):
    return tokenizer.pad({"input_ids": [tokens[i] for i in rows]}, padding=True,
                         pad_to_multiple_of=8, return_tensors="pt").to(device)


def trainable_state(model):
    return {name: param.detach().cpu().clone()
            for name, param in model.named_parameters() if param.requires_grad}


def triplets(frame, lengths, rng, world):
    groups = []
    for positions in frame.groupby("prompt_id", sort=False).indices.values():
        order = rng.permutation(positions)
        groups.extend(order[i:i + 3] for i in range(0, len(order), 3))
    order = rng.permutation(len(groups))
    for start in range(0, len(order), 96):
        selected = [groups[i] for i in order[start:start + 96]]
        selected.sort(key=lambda group: lengths[group].max())
        for offset in range(0, len(selected) - world + 1, world):
            yield selected[offset:offset + world]


def fit(model, wrapped, tokenizer, frame, labels, args, device, rank, world, phase, seed):
    tokens, owners, _ = windows(tokenizer, frame["text"])
    first = np.r_[True, owners[1:] != owners[:-1]]
    tokens = [tokens[i] for i in np.flatnonzero(first)]
    label_ids = {label: i for i, label in enumerate(labels)}
    y = torch.tensor(frame["model"].map(label_ids).to_numpy(), device=device)
    length = np.array([len(row) for row in tokens])
    short = torch.tensor(np.where(frame["text"].str.split().str.len().to_numpy() < 100, 1.25, 1.0),
                         dtype=torch.float32, device=device)
    counts = y.bincount(minlength=len(labels)).float().clamp_min(1)
    weight = short * counts[y].rsqrt() / counts.rsqrt().mean()
    local = torch.eye(len(labels), dtype=torch.bool, device=device)
    for group in CONFUSIONS:
        indices = torch.tensor([label_ids[label] for label in group], device=device)
        local[indices[:, None], indices[None, :]] = True
    lr = 6e-5 if phase == "external" else 2.5e-5
    adapters = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    heads = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": adapters, "lr": lr}, {"params": heads, "lr": 8 * lr}],
                                  fused=True, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    epochs = args.external_epochs if phase == "external" else args.official_epochs
    rng = np.random.default_rng(seed)
    wrapped.train()
    started = time.monotonic()
    for epoch in range(epochs):
        packs = list(triplets(frame, length, rng, world))
        for step, pack in enumerate(packs):
            rows = pack[rank]
            active = torch.tensor(rows, device=device)
            multiplier = min((step + 1) / 20, 1.0) * (1.0 - 0.8 * step / max(1, len(packs)))
            for group, factor in zip(optimizer.param_groups, (1.0, 8.0)):
                group["lr"] = lr * factor * multiplier
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                plain, centered = wrapped(**padded(tokenizer, tokens, rows, device), center=True)
                target = y[active]
                per_row = F.cross_entropy(plain, target, reduction="none", label_smoothing=0.03)
                sibling = F.cross_entropy(plain.masked_fill(~local[target], -1e4), target, reduction="none")
                relation = F.cross_entropy(centered, target, reduction="none", label_smoothing=0.03)
                loss = ((per_row + 0.25 * sibling + 0.7 * float(len(rows) > 1) * relation) * weight[active]).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(adapters + heads, 1.0)
            scaler.step(optimizer)
            scaler.update()
            if rank == 0 and (step + 1) % 100 == 0:
                print(f"{phase} epoch {epoch + 1} batch {step + 1}/{len(packs)} loss {loss.item():.3f}, "
                      f"{time.monotonic() - started:.0f}s", flush=True)
    if rank == 0:
        print(f"{phase} finished {len(frame)} passages x {epochs} in {time.monotonic() - started:.0f}s", flush=True)


@torch.inference_mode()
def raw_scores(model, tokenizer, texts, device, batch=8):
    seqs, owners, weights = windows(tokenizer, texts)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    plain = np.zeros((len(texts), model.plain.out_features), dtype=np.float32)
    centered = np.zeros_like(plain)
    model.eval()
    for start in range(0, len(order), batch):
        rows = order[start:start + batch]
        with torch.autocast("cuda", dtype=torch.float16):
            p, c = model(**padded(tokenizer, seqs, rows, device))
        np.add.at(plain, owners[rows], p.cpu().numpy() * weights[rows, None])
        np.add.at(centered, owners[rows], c.cpu().numpy() * weights[rows, None])
    denom = np.bincount(owners, weights, minlength=len(texts))[:, None]
    return plain / denom, centered / denom, model.centered.bias.detach().cpu().numpy().copy()


def native4b_calibrate(plain, centered, bias, groups, calibration):
    _, inverse = np.unique(groups, return_inverse=True)
    count = np.bincount(inverse)
    sums = np.zeros((len(count), plain.shape[1]))
    np.add.at(sums, inverse, centered)
    residual = centered - (sums / count[:, None])[inverse] + bias
    p = log_softmax(plain * calibration["plain_scale"], axis=1)
    c = log_softmax(residual * calibration["centered_scale"], axis=1)
    sizes = count[inverse]
    weight = np.where((sizes >= 2) & (sizes <= 18), calibration["centered_weight"], 0)[:, None]
    return log_softmax((1 - weight) * p + weight * c, axis=1)


def fit_scale(scores, target):
    def nll(scale):
        return -log_softmax(scores * scale, axis=1)[np.arange(len(target)), target].mean()
    return float(minimize_scalar(nll, bounds=(0.1, 4.0), method="bounded").x)


def readout_scales(frame, plain, centered, bias, target):
    rng = np.random.default_rng(14)
    residual, matched = [], []
    for group in frame.groupby("prompt_id").indices.values():
        order = rng.permutation(group)
        for start in range(0, len(order) - 1, 2):
            pair = order[start:start + 2]
            midpoint = centered[pair].mean(axis=0)
            residual.extend(centered[pair] - midpoint + bias[pair])
            matched.extend(target[pair])
    return {"plain_scale": fit_scale(plain, target),
            "centered_scale": fit_scale(np.asarray(residual), np.asarray(matched)),
            "centered_weight": CENTER_WEIGHT}


def local_bases(input_root):
    model = next(path.parent for path in input_root.rglob("config.json")
                 if "qwen-3-embedding" in path.parts and "4b" in path.parts)
    pool = next(input_root.rglob("ntl_open_pool.parquet"))
    native = next(input_root.rglob("ntl-native-q8-multidepth-fivefold-0679"))
    return model, pool, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-prompts", type=int, default=375)
    parser.add_argument("--external-epochs", type=int, default=1)
    parser.add_argument("--official-epochs", type=int, default=1)
    parser.add_argument("--last-layers", type=int, default=12)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    rank = int(os.getenv("LOCAL_RANK", 0))
    world = int(os.getenv("WORLD_SIZE", 1))
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    if world > 1:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        dist.init_process_group("nccl")
    model_path, pool_path, native_path = local_bases(Path("/kaggle/input"))
    train = args.train or next(Path("/kaggle/input").glob("**/name-that-llm/train.csv"))
    official = pd.read_csv(train, dtype={"id": str, "prompt_id": str})
    labels = sorted(official["model"].unique())
    native = native_reference(native_path, official, labels)
    folds = native[0]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Native4B(model_path, device, len(labels), args.last_layers)
    wrapped = DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False) if world > 1 else model
    external = external_panel(pool_path, official, labels, args.external_prompts, seed=42)
    fit(model, wrapped, tokenizer, external, labels, args, device, rank, world, "external", seed=42)
    shared = trainable_state(model)
    plain, centered = np.zeros((len(official), len(labels))), np.zeros((len(official), len(labels)))
    bias = np.zeros_like(plain)
    for fold in [*np.unique(folds), -1]:
        model.load_state_dict(shared, strict=False)
        fit(model, wrapped, tokenizer, official.loc[folds != fold], labels, args, device, rank, world,
            "official", seed=42 + int(fold))
        if rank == 0 and fold >= 0:
            keep = folds == fold
            plain[keep], centered[keep], bias[keep] = raw_scores(model, tokenizer, official.loc[keep, "text"].tolist(), device)
            print(f"fold {fold} plain BA {metrics(plain[keep], official.loc[keep], labels)['balanced_accuracy']:.4f}", flush=True)
        if world > 1:
            dist.barrier()
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        labels_int = np.array([labels.index(label) for label in official["model"]])
        calibration = readout_scales(official, plain, centered, bias, labels_int)
        save_file(trainable_state(model), str(args.output / "native4b.safetensors"))
        overview = {"labels": labels, "last_layers": args.last_layers, "calibration": calibration,
                    "4b_plain": metrics(plain, official, labels)}
        states, biases = {}, {}
        for fold in np.unique(folds):
            fit_rows = folds != fold
            k = readout_scales(official.loc[fit_rows], plain[fit_rows], centered[fit_rows],
                               bias[fit_rows], labels_int[fit_rows])
            states[fold] = {**k, "selected_centered_weight": k["centered_weight"]}
            biases[fold] = bias[folds == fold][0]
        native4 = folds, plain, centered, states, biases
        for seed in (42, 43):
            rng = np.random.default_rng(seed)
            rows = np.sort(np.concatenate([rng.choice(group, min(len(group), 2), replace=False)
                                           for group in official.groupby("prompt_id").indices.values()]))
            ref = replay_native(native, official, rows)
            p4 = replay_native(native4, official, rows)
            frame = official.iloc[rows]
            overview[f"pairs_{seed}"] = {"native8": metrics(ref, frame, labels),
                                         "native4": metrics(p4, frame, labels),
                                         "equal_mix": metrics(0.5 * ref + 0.5 * p4, frame, labels)}
        np.savez_compressed(args.output / "native4b_oof.npz", ids=official["id"].to_numpy(dtype=str),
                            labels=labels, folds=folds, plain=plain, centered=centered, bias=bias)
        (args.output / "native4b.json").write_text(json.dumps(overview, indent=2) + "\n")
    if world > 1:
        dist.destroy_process_group()


def predict_native4b(base, assets, texts, groups, device="cuda:0"):
    """Return label order and one calibrated log-posterior for native_stage."""
    state = json.loads((Path(assets) / "native4b.json").read_text())
    model = Native4B(base, device, len(state["labels"]), state["last_layers"])
    model.load_state_dict(load_file(str(Path(assets) / "native4b.safetensors")), strict=False)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    plain, centered, bias = raw_scores(model, tokenizer, texts, device)
    return state["labels"], native4b_calibrate(plain, centered, bias, groups, state["calibration"])


if __name__ == "__main__":
    main()
