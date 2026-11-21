"""Author attribution from base and post-trained next-token distributions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def reference_path(name):
    return next(Path('/kaggle/input').glob(f'**/qwen-3/[Tt]ransformers/{name}/1/config.json')).parent


@torch.inference_mode()
def token_distribution(path, token_ids, pad_id, device='cuda:0'):
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, torch_dtype=torch.float16,
        device_map={'': device}, attn_implementation='sdpa',
    ).eval()
    lengths = np.array([len(ids) for ids in token_ids])
    order = np.argsort(lengths)
    sequences = [None] * len(token_ids)
    print(f'{path.parent.name}: {len(token_ids)} texts, {lengths.sum():,} tokens', flush=True)
    for start in range(0, len(order), 12):
        rows = order[start:start + 12]
        batch = torch.full((len(rows), int(lengths[rows].max())), pad_id, dtype=torch.long)
        for pos, row in enumerate(rows):
            batch[pos, :lengths[row]] = torch.as_tensor(token_ids[row])
        batch = batch.to(device)
        mask = torch.arange(batch.shape[1], device=device)[None, :] < torch.as_tensor(
            lengths[rows], device=device,
        )[:, None]
        states = model.model(input_ids=batch, attention_mask=mask, use_cache=False).last_hidden_state
        vectors = states[:, :-1][mask[:, 1:]]
        targets = batch[:, 1:][mask[:, 1:]]
        stats = []
        for features, observed in zip(vectors.split(192), targets.split(192)):
            logits = model.lm_head(features).float()
            actual = logits.gather(1, observed[:, None])
            log_rank = (1 + (logits > actual).sum(1)).float().log()
            centered = logits - logits.logsumexp(1, keepdim=True)
            probability = centered.exp()
            entropy = -(probability * centered).sum(1)
            variance = (probability * centered.square()).sum(1) - entropy.square()
            surprisal = -centered.gather(1, observed[:, None]).squeeze(1)
            top = centered.topk(10, dim=1).values
            stats.append(torch.stack((
                surprisal, log_rank, entropy, top[:, 0].exp(), top.exp().sum(1),
                (surprisal - entropy) / variance.clamp_min(0.02).sqrt(),
            ), dim=1).cpu().numpy())
        offset = 0
        chunk = np.concatenate(stats)
        for row in rows:
            stop = offset + lengths[row] - 1
            sequences[row] = chunk[offset:stop]
            offset = stop
        if start // 12 % 60 == 0:
            print(f'{start + len(rows)}/{len(order)} texts', flush=True)
    del model
    torch.cuda.empty_cache()
    return sequences


def likelihood_features(texts, max_tokens=512, device='cuda:0'):
    base_path = reference_path('4b-base')
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    encoded = tokenizer(texts, truncation=True, max_length=max_tokens - 1,
                        add_special_tokens=False, return_offsets_mapping=True)
    bos = tokenizer.eos_token_id
    ids = [[bos] + (row or [bos]) for row in encoded['input_ids']]
    base = token_distribution(base_path, ids, tokenizer.pad_token_id or bos, device)
    aligned = token_distribution(reference_path('4b'), ids, tokenizer.pad_token_id or bos, device)
    sequence = np.zeros((len(ids), max_tokens - 1, 26), dtype=np.float32)
    lengths = np.array([len(row) - 1 for row in ids])
    pooled = []
    for i, (text, offsets, first, second) in enumerate(zip(texts, encoded['offset_mapping'], base, aligned)):
        pieces = [text[lo:hi] for lo, hi in offsets] or ['']
        shape = np.array([
            [bool(p[:1].isspace()), '\n' in p, any(c.isdigit() for c in p), any(c.isupper() for c in p),
             any(c.isalpha() for c in p), any(not c.isalnum() and not c.isspace() for c in p),
             min(len(p), 16) / 16, j / max(1, len(pieces) - 1)]
            for j, p in enumerate(pieces)
        ], dtype=np.float32)
        likelihood = np.concatenate((first, second, second - first), axis=1).clip(-25, 25)
        features = np.concatenate((likelihood, shape), axis=1)
        sequence[i, :len(features)] = features
        position_parts = np.array_split(likelihood, 3)
        masks = (shape[:, 1] == 1, shape[:, 2] == 1, shape[:, 4] == 1, shape[:, 5] == 1)
        conditional = [block.mean(0) if len(block) else np.zeros(likelihood.shape[1])
                       for block in (*position_parts, *(likelihood[mask] for mask in masks))]
        pooled.append(np.concatenate((likelihood.mean(0), likelihood.std(0),
                                      np.quantile(likelihood, [0.1, 0.5, 0.9], axis=0).ravel(),
                                      shape.mean(0), *conditional)))
    return sequence, lengths, np.asarray(pooled, dtype=np.float32)


class LikelihoodSequence(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.project = nn.Conv1d(channels, 64, 1)
        self.local = nn.Conv1d(64, 64, 5, padding=2)
        self.context = nn.Conv1d(64, 64, 5, dilation=3, padding=6)
        self.weight = nn.Conv1d(64, 1, 1)
        self.readout = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.25), nn.Linear(128, classes))

    def forward(self, sequence, lengths):
        valid = torch.arange(sequence.shape[1], device=sequence.device)[None, None, :] < lengths[:, None, None]
        hidden = self.project(sequence.transpose(1, 2)).tanh() * valid
        hidden = (hidden + self.local(hidden).tanh()) * valid
        hidden = (hidden + self.context(hidden).tanh()) * valid
        weights = self.weight(hidden).masked_fill(~valid, -torch.inf).softmax(-1)
        average = (hidden * weights).sum(-1)
        maximum = hidden.masked_fill(~valid, -torch.inf).amax(-1)
        return self.readout(torch.cat((average, maximum), dim=1))


def sequence_logits(sequence, lengths, labels, train, query, device):
    torch.manual_seed(42)
    sequence = torch.as_tensor(sequence, device=device)
    lengths = torch.as_tensor(lengths, device=device)
    labels = torch.as_tensor(labels, device=device)
    train = torch.as_tensor(train, device=device)
    query = torch.as_tensor(query, device=device)
    valid = torch.arange(sequence.shape[1], device=device)[None, :] < lengths[train, None]
    values = sequence[train][valid]
    sequence = ((sequence - values.mean(0)) / values.std(0).clamp_min(0.05)).clamp(-10, 10)
    model = LikelihoodSequence(sequence.shape[2], int(labels.max()) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=24, eta_min=0.00015)
    predictions = []
    for epoch in range(24):
        model.train()
        for batch in train[torch.randperm(len(train), device=device)].split(64):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(sequence[batch], lengths[batch]), labels[batch])
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch in (11, 17, 23):
            model.eval()
            with torch.no_grad():
                predictions.append(torch.cat([
                    model(sequence[batch], lengths[batch]) for batch in query.split(128)
                ]).cpu().numpy())
    return np.mean(predictions, axis=0)


def report(labels, target, weights, predictions):
    result = {}
    for name, values in predictions.items():
        guess = values.argmax(1)
        result[name] = dict(
            balanced_accuracy=float(balanced_accuracy_score(target, guess)),
            recall={label: float((guess[target == j] == j).mean()) for j, label in enumerate(labels)},
            by_words={f'{low}-{high - 1}': float(balanced_accuracy_score(target[keep], guess[keep]))
                      for low, high in [(0, 70), (70, 120), (120, 200), (200, 100_000)]
                      if (keep := (weights >= low) & (weights < high)).any()},
        )
    return result


def public_native_reference():
    from scipy.special import softmax

    candidates = Path('/kaggle/input').glob('**/ntl-native-q8-multidepth-fivefold-0679/oof_reference.npz')
    path = next(candidates, None)
    if path is None:
        print('Public native OOF arrays are unavailable in the attached dataset.', flush=True)
        return None
    labels = json.loads(path.with_name('manifest.json').read_text())['classes']
    with np.load(path.with_name('tokens.npz')) as tokens, np.load(path) as oof:
        folds = tokens['folds']
        plain_scale = np.array([
            json.loads((path.parent / 'calibration' / f'fold{int(fold)}.json').read_text())['models']['qwen8']['plain_scale']
            for fold in np.unique(folds)
        ])
        probability = softmax(oof['plain'] * plain_scale[folds.astype(int), None], axis=1)
        return pd.DataFrame(probability, columns=['prob__' + label for label in labels]).assign(
            fold=folds, id=tokens['ids'].astype(str),
        )


def study(train_csv, output='/kaggle/working', reference=None, device='cuda:0'):
    from catboost import CatBoostClassifier

    frame = pd.read_csv(train_csv, dtype={'id': str, 'prompt_id': str})
    labels, y = np.unique(frame.model, return_inverse=True)
    groups = frame.prompt_id.to_numpy()
    if reference is None:
        splits = list(GroupKFold(5).split(frame, y, groups))
    else:
        reference = reference.set_index('id').loc[frame.id]
        reference_columns = ['prob__' + label for label in labels]
        splits = [(np.flatnonzero(reference.fold.to_numpy() != fold),
                   np.flatnonzero(reference.fold.to_numpy() == fold)) for fold in sorted(reference.fold.unique())]
        reference_logp = np.log(reference[reference_columns].to_numpy().clip(1e-16, 1))
    sequence, lengths, pooled = likelihood_features(frame.text.tolist(), device=device)
    cat = np.empty((len(frame), len(labels)), dtype=np.float32)
    conv = np.empty_like(cat)
    fold_ids = np.empty(len(frame), dtype=np.int8)
    for i, (train, valid) in enumerate(splits):
        model = CatBoostClassifier(iterations=550, depth=5, learning_rate=0.06, thread_count=4,
                                   verbose=False, allow_writing_files=False, random_seed=42)
        model.fit(pooled[train], y[train])
        cat[valid] = model.predict(pooled[valid], prediction_type='RawFormulaVal')
        conv[valid] = sequence_logits(sequence, lengths, y, train, valid, device)
        fold_ids[valid] = i
        print(f'fold {i}: catboost={balanced_accuracy_score(y[valid], cat[valid].argmax(1)):.4f}, '
              f'sequence={balanced_accuracy_score(y[valid], conv[valid].argmax(1)):.4f}', flush=True)
    combined = np.logaddexp(log_softmax(cat, axis=1), log_softmax(conv, axis=1)) - np.log(2)
    predictions = {'pooled': cat, 'sequence': conv, 'likelihood': combined}
    if reference is not None:
        predictions['native_plain'] = reference_logp
        for weight in (0.15, 0.30, 0.50):
            predictions[f'native_plus_likelihood_{weight}'] = (1 - weight) * reference_logp + weight * combined
    metrics = report(labels, y, frame.text.str.split().str.len().to_numpy(), predictions)
    output = Path(output)
    (output / 'likelihood_metrics.json').write_text(json.dumps(metrics, indent=2))
    scores = [pd.DataFrame(logits, columns=[f'{name}__{label}' for label in labels])
              for name, logits in predictions.items()]
    oof = pd.concat([frame[['id', 'prompt_id', 'model']].assign(fold=fold_ids), *scores], axis=1)
    oof.to_parquet(output / 'likelihood_oof.parquet', index=False)
    print({name: scores['balanced_accuracy'] for name, scores in metrics.items()}, flush=True)
