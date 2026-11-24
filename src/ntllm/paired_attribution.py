"""Train single-passage source readouts with and without same-prompt ranking."""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn


HARD_GROUPS = (
    ('claude-fable-5', 'glm-5.3', 'glm-5.3-flash'),
    ('gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'),
    ('gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra'),
    ('gpt-5.5', 'kimi-k3'),
)
HARD_PAIRS = frozenset(tuple(sorted(pair)) for group in HARD_GROUPS for pair in combinations(group, 2))
HOLDOUT = ('drjcfmd/ntl-generated-v2', 'davidpiln/ntl-vertex-gemini-gen')
EXCLUDED_SOURCE_PREFIXES = ('elliotthankinson/', 'prvsiyan/')
EXCLUDED_FLAGS = ('flag_kb_proxy', 'flag_gemini37_routing_risk', 'flag_train_prompt_regen',
                  'flag_near_train_prompt')
POOL_COLUMNS = ('id', 'text', 'model', 'prompt_canon', 'source_dataset', 'source_file',
                'prompt_evidence', 'n_words', *EXCLUDED_FLAGS)


def text_key(texts):
    return texts.str.normalize('NFKC').str.casefold().str.replace(r'\s+', ' ', regex=True).str.strip()


def read_pool(path, official):
    pool = pd.read_parquet(path, columns=list(POOL_COLUMNS))
    pool = pool.dropna(subset=['text', 'prompt_canon', 'source_dataset']).copy()
    pool['source_dataset'] = (pool.source_dataset.str.lower()
                              .str.removeprefix('https://www.kaggle.com/datasets/'))
    valid = (pool.model.isin(official.model) & pool.prompt_canon.ne('') & pool.n_words.between(40, 250)
             & pool.prompt_evidence.isin(('own_text', 'own_bank', 'other_text'))
             & ~pool.source_dataset.str.startswith(EXCLUDED_SOURCE_PREFIXES)
             & ~pool.source_file.fillna('').str.contains(r'train_regen|discriminator|exact_pass', case=False)
             & ~pool[list(EXCLUDED_FLAGS)].fillna(False).any(axis=1))
    pool = pool.loc[valid].copy()
    keys = text_key(pool.text)
    official_keys = text_key(official.text)
    pool = pool.loc[keys.ne('') & ~keys.isin(official_keys) & ~keys.duplicated(keep=False)]
    pool = pool.drop_duplicates(['source_dataset', 'prompt_canon', 'model']).reset_index(drop=True)
    split = pool.source_dataset.isin(HOLDOUT)
    held = pool.loc[split]
    missing = set(HOLDOUT) - set(held.source_dataset)
    if missing:
        raise ValueError(f'Expected both source holdouts, missing: {sorted(missing)}')
    hold_prompts = held.prompt_canon.unique()
    training = pool.loc[~split & ~pool.prompt_canon.isin(hold_prompts)].reset_index(drop=True)
    return training, held.reset_index(drop=True)


def make_pairs(rows):
    columns = ['source_dataset', 'prompt_canon', 'model', 'n_words']
    index = rows[columns].rename_axis('row').reset_index()
    joined = index.merge(index, on=columns[:2], suffixes=('_left', '_right'))
    joined = joined.loc[joined.model_left.ne(joined.model_right)].copy()
    width = (np.log(joined.n_words_left) - np.log(joined.n_words_right)).abs()
    joined = joined.loc[width.le(np.log(1.5))].copy()
    joined['length_weight'] = np.exp(-4 * width.loc[joined.index])
    directed = HARD_PAIRS | {(right, left) for left, right in HARD_PAIRS}
    joined['hard'] = list(zip(joined.model_left, joined.model_right))
    joined['hard'] = joined.hard.isin(directed)
    return joined


def make_schedule(rows, labels, steps, pairs_per_batch, seed):
    pairs = make_pairs(rows)
    rng = np.random.default_rng(seed)
    n_pairs = steps * pairs_per_batch
    anchors = np.tile(labels, (n_pairs + len(labels) - 1) // len(labels))
    rng.shuffle(anchors)
    anchors = anchors[:n_pairs]
    selected = np.empty((n_pairs, 2), dtype=np.int64)
    counts = {}
    for label in labels:
        all_pairs = pairs.loc[pairs.model_left.eq(label)]
        if all_pairs.empty:
            raise ValueError(f'No usable external pairs for {label}')
        positions = np.flatnonzero(anchors == label)
        hard = all_pairs.loc[all_pairs.hard]
        selected_hard = rng.random(positions.size) < 0.5 if not hard.empty else np.zeros(positions.size, bool)
        for choices, places in ((all_pairs, positions[~selected_hard]), (hard, positions[selected_hard])):
            if not len(places):
                continue
            size = choices.groupby('source_dataset').row_left.transform('size').to_numpy()
            weight = choices.length_weight.to_numpy() / np.sqrt(size)
            indices = rng.choice(len(choices), len(places), p=weight / weight.sum())
            selected[places] = choices.iloc[indices][['row_left', 'row_right']].to_numpy()
        counts[label] = {'eligible_directed_pairs': len(all_pairs), 'hard_pairs': len(hard),
                         'sources': all_pairs.source_dataset.nunique()}
    uses = rows.iloc[selected.ravel()]
    report = {'training_rows': len(rows), 'training_sources': rows.source_dataset.nunique(),
              'training_prompts': rows.prompt_canon.nunique(), 'selected_pairs': n_pairs,
              'selected_unique_rows': uses.id.nunique(), 'selected_unique_prompts': uses.prompt_canon.nunique(),
              'selected_rows_by_class': uses.model.value_counts().to_dict(), 'eligible_by_class': counts}
    return selected.reshape(steps, pairs_per_batch * 2), report


class LowRankLinear(nn.Module):
    def __init__(self, original, rank=8):
        super().__init__()
        self.original = original
        self.down = nn.Linear(original.in_features, rank, bias=False)
        self.up = nn.Linear(rank, original.out_features, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden):
        return self.original(hidden) + 2 * self.up(self.down(hidden))


class PassageReadout(nn.Module):
    def __init__(self, backbone, num_labels):
        super().__init__()
        self.backbone = backbone.requires_grad_(False)
        for layer in backbone.encoder.layer[-8:]:
            attention = layer.attention.self
            attention.query = LowRankLinear(attention.query)
            attention.value = LowRankLinear(attention.value)
        self.project = nn.Sequential(nn.LayerNorm(2 * backbone.config.hidden_size),
                                     nn.Linear(2 * backbone.config.hidden_size, 256), nn.Tanh())
        self.classify = nn.Linear(256, num_labels)

    def forward(self, **batch):
        states = self.backbone(**batch).last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1)
        mean = (states.float() * mask).sum(1) / mask.sum(1).clamp_min(1)
        feature = self.project(torch.cat([states[:, 0].float(), mean], dim=1))
        return self.classify(feature).float(), feature.float()


def swap_loss(logits, target):
    # A prompt-specific additive bias for either class cancels between the two rows.
    own = logits.gather(1, target[:, None]).squeeze(1)
    other = logits.gather(1, target.reshape(-1, 2).flip(1).reshape(-1, 1)).squeeze(1)
    swap_advantage = (own - other).reshape(-1, 2).mean(1)
    return F.softplus(0.5 - swap_advantage).mean()


def encode(tokenizer, texts, max_tokens):
    return tokenizer(texts.tolist(), truncation=True, max_length=max_tokens)['input_ids']


def batch_inputs(tokenizer, tokens, rows):
    return tokenizer.pad({'input_ids': [tokens[i] for i in rows]}, pad_to_multiple_of=8,
                         return_tensors='pt').to('cuda')


def fit_arm(base, tokenizer, training, schedule, labels, pair_weight, settings, seed, init_readout=None):
    from transformers import AutoModel

    torch.manual_seed(seed)
    backbone = AutoModel.from_pretrained(base, local_files_only=True)
    model = PassageReadout(backbone, len(labels))
    if init_readout is not None:
        state = torch.load(init_readout, weights_only=True, map_location='cpu')
        expected = {name for name, param in model.named_parameters() if param.requires_grad}
        if set(state) != expected:
            raise ValueError('The continuation checkpoint must contain this readout’s trainable parameters')
        model.load_state_dict(state, strict=False)
    model.to('cuda')
    model.train()
    model.backbone.eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings['learning_rate'], weight_decay=0.01)
    maximum_updates = len(schedule)
    warmup = max(1, maximum_updates // 10)

    def scale(step):
        return min(1.0, (step + 1) / warmup) * (1 - 0.85 * max(0, step - warmup) / maximum_updates)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    scaler = torch.amp.GradScaler('cuda')
    indices = np.unique(schedule)
    tokens = dict(zip(indices, encode(tokenizer, training.iloc[indices].text, settings['max_tokens'])))
    codes = pd.Categorical(training.model, categories=labels).codes
    sums = np.zeros(2)
    start = time.monotonic()
    for step, batch in enumerate(schedule, 1):
        feed = batch_inputs(tokenizer, tokens, batch)
        target = torch.as_tensor(codes[batch], device='cuda', dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits, _ = model(**feed)
            ce = F.cross_entropy(logits, target)
            pair = swap_loss(logits, target)
            loss = ce + pair_weight * pair
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= previous_scale:
            scheduler.step()
        sums += (ce.item(), pair.item())
        if step == 1 or step % 50 == 0 or step == maximum_updates:
            print(f'weight={pair_weight} step={step}/{maximum_updates} '
                  f'ce={sums[0]/step:.4f} swap={sums[1]/step:.4f} '
                  f'elapsed={time.monotonic()-start:.1f}s', flush=True)
    return model


@torch.inference_mode()
def score(model, tokenizer, frame, settings):
    model.eval()
    tokens = encode(tokenizer, frame.text, settings['max_tokens'])
    order = np.argsort([len(x) for x in tokens], kind='stable')
    outputs = np.empty((len(tokens), model.classify.out_features), dtype=np.float32)
    features = np.empty((len(tokens), model.classify.in_features), dtype=np.float32)
    for batch in np.array_split(order, (len(order) + settings['eval_batch'] - 1) // settings['eval_batch']):
        feed = batch_inputs(tokenizer, tokens, batch)
        with torch.autocast('cuda', dtype=torch.float16):
            logits, hidden = model(**feed)
        outputs[batch] = logits.cpu().numpy()
        features[batch] = hidden.cpu().numpy()
    return outputs, features


def log_prob(scores):
    from scipy.special import log_softmax

    return log_softmax(scores, axis=1)


def adapt_oof(logits, features, target, folds):
    adapted = np.empty_like(logits)
    with threadpool_limits(limits=2):
        for fold in np.unique(folds):
            train = folds != fold
            scale = StandardScaler().fit(features[train])
            classifier = LogisticRegression(C=0.03, max_iter=300, class_weight='balanced')
            classifier.fit(scale.transform(features[train]), target[train])
            adapted[~train] = classifier.decision_function(scale.transform(features[~train]))
    return 0.5 * log_prob(logits) + 0.5 * log_prob(adapted)


def metrics(values, truth, words, labels, folds=None):
    pred = np.argsort(-values, axis=1, kind='stable')[:, :3]
    recall = {labels[i]: float((pred[truth == i, 0] == i).mean()) for i in np.unique(truth)}
    result = {'rows': len(truth), 'ba': float(np.mean(list(recall.values()))), 'recall': recall,
              'top3_coverage': float(np.mean([(pred[truth == i] == i).any(1).mean()
                                              for i in np.unique(truth)]))}
    for name, subset in (('short', words < 100), ('long', words >= 100)):
        result[f'{name}_ba'] = float(np.mean([(pred[truth == i, 0][subset[truth == i]] == i).mean()
                                              for i in np.unique(truth[subset])])) if subset.any() else None
    if folds is not None:
        result['by_fold'] = {str(fold): metrics(values[folds == fold], truth[folds == fold],
                                                 words[folds == fold], labels)['ba']
                             for fold in np.unique(folds)}
    return result


def paired_order(values, frame, prompt_column, labels):
    report = {}
    for left, right in sorted(HARD_PAIRS):
        rows = frame[[prompt_column, 'model']].assign(row=np.arange(len(frame)))
        paired = rows.loc[rows.model.isin((left, right))].pivot(index=prompt_column, columns='model', values='row')
        paired = paired.reindex(columns=[left, right]).dropna().to_numpy(dtype=int)
        if len(paired):
            margin = values[:, labels.index(left)] - values[:, labels.index(right)]
            gap = margin[paired[:, 0]] - margin[paired[:, 1]]
            report[f'{left} / {right}'] = {'pairs': len(gap), 'order_accuracy': float(np.mean((gap > 0) + 0.5 * (gap == 0)))}
    return report


def run(args):
    start = time.monotonic()
    settings = json.loads(args.config.read_text())
    official = pd.read_csv(args.train, dtype={'id': str, 'prompt_id': str})
    official['n_words'] = official.text.str.split().str.len()
    labels = sorted(official.model.unique())
    fold_map = json.loads(args.folds.read_text())
    folds = official.prompt_id.map(fold_map)
    if len(labels) != 18 or folds.isna().any() or folds.nunique() != 5:
        raise ValueError('Expected the documented 18 classes and five official prompt folds')
    folds = folds.to_numpy(dtype=int)
    seed = settings['seed']
    training, held = read_pool(args.pool, official)
    schedule, report = make_schedule(training, labels, settings['steps'], settings['pairs_per_batch'], seed)
    prompt_order = held.prompt_canon.drop_duplicates().sample(frac=1, random_state=settings.get('holdout_seed', seed))
    held = (held.assign(_order=held.prompt_canon.map(dict(zip(prompt_order, range(len(prompt_order))))))
            .sort_values('_order', kind='stable').groupby(['source_dataset', 'model'], sort=False)
            .head(settings['holdout_max_per_class']).reset_index(drop=True))
    report['holdout_rows'] = held.source_dataset.value_counts().to_dict()
    report['holdout_models'] = held.groupby('source_dataset').model.nunique().to_dict()
    report['settings'] = settings
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'sample_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True), flush=True)
    if args.prepare_only:
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    print(f'Backbone {args.base.name}, tokenizer lowercase={getattr(tokenizer, "do_lower_case", None)}, '
          f'GPU {torch.cuda.get_device_name(0)}', flush=True)
    arms = [('ce_continued', 0.0)] if args.init_readout else [('ce', 0.0), ('paired', settings['pair_weight'])]
    for name, weight in arms:
        model = fit_arm(args.base, tokenizer, training, schedule, labels, weight, settings, seed, args.init_readout)
        official_logits, official_features = score(model, tokenizer, official, settings)
        target = pd.Categorical(official.model, categories=labels).codes
        oof = adapt_oof(official_logits, official_features, target, folds)
        held_logits, _ = score(model, tokenizer, held, settings)
        held_truth = pd.Categorical(held.model, categories=labels).codes
        results = {'external_only': metrics(official_logits, target, official.n_words.to_numpy(), labels, folds),
                   'adapted_oof': metrics(oof, target, official.n_words.to_numpy(), labels, folds),
                   'official_pair_order': paired_order(oof, official, 'prompt_id', labels),
                   'held_source': {source: metrics(held_logits[mask], held_truth[mask],
                                                   held.n_words.to_numpy()[mask], labels)
                                   for source in HOLDOUT
                                   if (mask := held.source_dataset.eq(source).to_numpy()).any()},
                   'held_pair_order': {source: paired_order(held_logits[mask], held.loc[mask], 'prompt_canon', labels)
                                       for source in HOLDOUT
                                       if (mask := held.source_dataset.eq(source).to_numpy()).any()},
                   'elapsed_seconds': time.monotonic() - start}
        print(name, json.dumps(results, sort_keys=True), flush=True)
        (args.out / f'{name}_metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
        output = official[['id', 'prompt_id', 'model', 'n_words']].assign(fold=folds)
        output = output.join(pd.DataFrame(oof, columns=[f'logp__{label}' for label in labels]))
        output.to_parquet(args.out / f'{name}_oof.parquet', index=False)
        torch.save({key: tensor.detach().cpu() for key, tensor in model.named_parameters() if tensor.requires_grad},
                   args.out / f'{name}_readout.pt')
        del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ('config', 'train', 'pool', 'folds', 'out'):
        parser.add_argument('--' + path, type=Path, required=True)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--init-readout', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not args.prepare_only and args.base is None:
        parser.error('--base is required for GPU training')
    run(args)


if __name__ == '__main__':
    main()
