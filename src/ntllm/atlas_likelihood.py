"""ATLAS-style source-conditioned token likelihood for targeted author review."""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


TRACKS = (
    ('gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'),
    ('gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra'),
)
TEMPERATURES = (1, 2, 4, 8)
METHODS = ('mean', 'clipped')


class SourceProjection(nn.Module):
    def __init__(self, original, count, rank, source_ids):
        super().__init__()
        self.original = original
        self.source_ids = source_ids
        self.down = nn.Parameter(torch.empty(rank, original.in_features, device=original.weight.device))
        self.up = nn.Parameter(torch.zeros(count, original.out_features, rank, device=original.weight.device))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    def forward(self, hidden):
        low = F.linear(hidden, self.down.to(hidden.dtype))
        up = self.up[self.source_ids[0]].transpose(1, 2).to(hidden.dtype)
        return self.original(hidden) + 2 * torch.bmm(low, up)


class AttributionLM(nn.Module):
    def __init__(self, model_path, count, device, rank):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, local_files_only=True, attn_implementation='sdpa',
        ).requires_grad_(False).to(device)
        self.source_ids = [None]
        embedding = self.backbone.get_input_embeddings().weight
        self.control = nn.Parameter(torch.empty(count, 1, embedding.shape[-1], device=device))
        nn.init.normal_(self.control, std=0.02)
        for layer in self.backbone.model.layers:
            for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                attn = layer.self_attn
                setattr(attn, name, SourceProjection(getattr(attn, name), count, rank, self.source_ids))
        self.backbone.config.use_cache = False

    def forward(self, ids, mask, sources):
        self.source_ids[0] = sources
        embeddings = self.backbone.get_input_embeddings()(ids)
        embeddings = torch.cat((self.control[sources].to(embeddings.dtype), embeddings), dim=1)
        attention = F.pad(mask, (1, 0), value=True)
        hidden = self.backbone.model(inputs_embeds=embeddings, attention_mask=attention,
                                     use_cache=False, return_dict=True).last_hidden_state
        positions = mask.nonzero(as_tuple=True)
        hidden = hidden[:, :-1][positions]
        targets = ids[positions]
        head = self.backbone.get_output_embeddings()

        def token_nll(states, target):
            return F.cross_entropy(head(states).float(), target, reduction='none')

        chunks = zip(hidden.split(128), targets.split(128))
        if torch.is_grad_enabled():
            losses = [checkpoint(token_nll, part, target, use_reentrant=False) for part, target in chunks]
        else:
            losses = [token_nll(part, target) for part, target in chunks]
        logp = torch.zeros_like(ids, dtype=torch.float32)
        logp[positions] = -torch.cat(losses)
        return logp


def load_tokenizer(model_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)


def encode(tokenizer, texts, limit):
    tokens = tokenizer(list(texts), add_special_tokens=False, truncation=True, max_length=limit)['input_ids']
    if any(not seq for seq in tokens):
        raise ValueError('Expected a nonempty token sequence for each passage')
    return tokens


def batch_tensors(sequences, sources, pad, device):
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, seq in enumerate(sequences):
        ids[row, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[row, :len(seq)] = True
    return ids.to(device), mask.to(device), torch.as_tensor(sources, dtype=torch.long, device=device)


def normalized_text(frame):
    return (frame.text.str.normalize('NFKC').str.casefold().str.strip()
            .str.replace(r'^\*\*(.*)\*\*$', r'\1', regex=True)
            .str.replace(r'\s+', ' ', regex=True))


def training_rows(pool, benchmark, official, labels, limit, seed):
    flags = ['flag_kb_proxy', 'flag_gemini37_routing_risk', 'flag_train_prompt_regen', 'flag_near_train_prompt']
    fields = ['id', 'text', 'model', 'prompt_canon', 'source_dataset', 'prompt_evidence', 'n_words', *flags]
    rows = pd.read_parquet(pool, columns=fields)
    rows = rows.dropna(subset=['text', 'prompt_canon', 'source_dataset'])
    valid = (rows.model.isin(labels) & rows.n_words.between(40, 250) & rows.prompt_canon.ne('')
             & rows.prompt_evidence.isin(('own_text', 'own_bank', 'other_text'))
             & ~rows.source_dataset.str.startswith(('beroli/', 'markbarney/', 'elliotthankinson/', 'prvsiyan/'))
             & ~rows[flags].fillna(False).any(axis=1))
    rows = rows.loc[valid].copy()
    rows['_key'] = normalized_text(rows)
    excluded = set(normalized_text(benchmark)) | set(normalized_text(official))
    rows = rows.loc[~rows._key.isin(excluded) & ~rows._key.duplicated(keep=False)].copy()
    rows = rows.drop_duplicates(['source_dataset', 'prompt_canon', 'model'])
    rng = np.random.default_rng(seed)
    pieces = []
    for _, family in rows.groupby(rows.model.str.split('-', n=1).str[0]):
        paired = family.groupby(['source_dataset', 'prompt_canon']).model.transform('size').ge(2)
        pool_size = family.groupby(['model', 'source_dataset']).id.transform('size')
        family = family.assign(draw=-np.log(rng.uniform(size=len(family))) * np.sqrt(pool_size) / np.where(paired, 3, 1))
        pieces.append(family.sort_values('draw', kind='stable').groupby('model').head(limit))
    frame = pd.concat(pieces, ignore_index=True)
    counts = frame.model.value_counts()
    if set(counts.index) != set(labels) or counts.min() < min(400, limit):
        raise ValueError(f'Insufficient outside training coverage: {counts.to_dict()}')
    return frame


def fit(model, tokenizer, frame, labels, config, device, output):
    start = time.monotonic()
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(parameters.values(), lr=config['learning_rate'])
    scaler = torch.amp.GradScaler('cuda', init_scale=2048.0)
    tokens = encode(tokenizer, frame.text, config['max_tokens'])
    codes = pd.Categorical(frame.model, categories=labels).codes.astype(np.int64)
    minimal = int(frame.groupby('model').size().min())
    rng = np.random.default_rng(config['seed'])
    per_group = [rng.permutation(np.flatnonzero(codes == i))[:minimal] for i in range(len(labels))]
    order = rng.permutation(np.column_stack(per_group).ravel())
    lengths = np.array([len(seq) for seq in tokens])
    order = np.concatenate([block[np.argsort(lengths[block], kind='stable')] for block in np.array_split(order, 30)])
    model.train()
    batch_size = config['train_batch']
    steps = math.ceil(len(order) / batch_size)
    total = 0.0
    updates = 0
    for step, first in enumerate(range(0, len(order), batch_size), 1):
        idx = order[first:first + batch_size]
        ids, mask, sources = batch_tensors([tokens[i] for i in idx], codes[idx], tokenizer.pad_token_id, device)
        optimizer.zero_grad(set_to_none=True)
        warm = min(1.0, step / max(1, int(steps * 0.05)))
        cosine = 0.5 * (1 + math.cos(math.pi * step / steps))
        optimizer.param_groups[0]['lr'] = config['learning_rate'] * warm * cosine
        logp = model(ids, mask, sources)
        loss = -(logp.sum(1) / mask.sum(1)).mean()
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite token likelihood loss at step {step}')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(parameters.values(), 1.0)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        updates += int(scaler.get_scale() >= previous_scale)
        total += loss.item()
        if step == 1 or step % 100 == 0 or step == steps:
            print(f'{step}/{steps} true-source NLL {total/step:.4f}, {updates} updates, '
                  f'{time.monotonic()-start:.0f}s', flush=True)
    torch.save({'labels': list(labels), 'rank': config['rank'], 'max_tokens': config['max_tokens'],
                'parameters': {name: p.detach().cpu() for name, p in parameters.items()}}, output)
    return {'updates': updates, 'steps': steps, 'training_rows': int(len(order)),
            'mean_nll': total / steps, 'seconds': time.monotonic() - start}


def aggregate_token_likelihood(logp):
    values = np.asarray(logp, dtype=np.float64)
    rivals = np.maximum(values[None] - values[:, None], 0).sum(axis=1)
    clipped = np.mean([np.exp(-rivals / temperature).mean(axis=1) for temperature in TEMPERATURES], axis=0)
    return values.mean(axis=1), np.log(clipped)


@torch.inference_mode()
def score_pairs(model, tokenizer, texts, candidates, config, device, deadline=None):
    raw = {name: np.full((len(texts), 2), np.nan) for name in METHODS}
    model.eval()
    tokens = encode(tokenizer, texts, config['max_tokens'])
    order = np.argsort([len(x) for x in tokens], kind='stable')
    batch_size = config['score_batch']
    for first in range(0, len(order), batch_size):
        if deadline is not None and time.monotonic() >= deadline:
            break
        positions = order[first:first + batch_size]
        inputs = [tokens[i] for i in positions for _ in range(2)]
        ids, mask, sources = batch_tensors(inputs, candidates[positions].ravel(), tokenizer.pad_token_id, device)
        token_lp = model(ids, mask, sources).cpu().numpy().reshape(len(positions), 2, -1)
        for row, values in zip(positions, token_lp):
            mean, clipped = aggregate_token_likelihood(values[:, :len(tokens[row])])
            raw['mean'][row] = mean
            raw['clipped'][row] = clipped
    return raw


def allowed_by_typography(text, label):
    spaced = {'claude-opus-4-6', 'claude-opus-5', 'glm-5.3', 'glm-5.3-flash'}
    return ((' — ' not in text or label in spaced)
            and ('’' not in text or not label.startswith(('claude-', 'glm-')))
            and ('—' not in text or not label.startswith('gemini-')))


def route_candidates(texts, labels, probs, final, config):
    groups = {label: group for group in config.get('tracks', TRACKS) for label in group}
    ids = {name: i for i, name in enumerate(labels)}
    top = np.argsort(-probs, axis=1, kind='stable')[:, :3]
    routes = []
    for row, current in enumerate(final):
        group = groups.get(current)
        if not group:
            continue
        alternate = next((labels[i] for i in top[row] if labels[i] != current and labels[i] in group
                          and probs[row, i] > 0 and allowed_by_typography(texts[row], labels[i])), None)
        if alternate is None or not allowed_by_typography(texts[row], current):
            continue
        gap = max(0.0, math.log(max(probs[row, ids[current]], 1e-15) / probs[row, ids[alternate]]))
        if gap <= config['max_margin']:
            routes.append((row, current, alternate, gap))
    table = pd.DataFrame(routes, columns=['row', 'incumbent', 'challenger', 'gap'])
    cap = max(1, int(len(texts) * config['max_scoring_fraction']))
    return table.nsmallest(cap, 'gap').reset_index(drop=True) if len(table) else table


def apply_scores(final, routes, raw, method, calibration):
    result = np.asarray(final, dtype=str).copy()
    weight = calibration[method]['weight']
    scale = calibration[method]['scale']
    advantage = (raw[method][:, 1] - raw[method][:, 0]) / scale
    change = np.isfinite(advantage) & (weight * advantage > routes.gap.to_numpy())
    result[routes.loc[change, 'row'].to_numpy(dtype=int)] = routes.loc[change, 'challenger'].to_numpy()
    return result


def calibrate(benchmark, routes, raw, final):
    dev = benchmark.cohort.eq('development').to_numpy()
    target = benchmark.model.to_numpy()
    chosen = dev[routes.row.to_numpy(dtype=int)]
    summaries = {}
    for name in METHODS:
        scale = max(1e-5, np.quantile(np.abs(np.diff(raw[name][chosen], axis=1)), 0.75)) if chosen.any() else 1.0
        options = []
        for weight in (0.0, 0.10, 0.25, 0.50):
            setting = {name: {'weight': weight, 'scale': scale}}
            after = apply_scores(final, routes, raw, name, setting)
            macro = (pd.Series(after[dev] == target[dev]).groupby(target[dev]).mean().mean())
            options.append((float(macro), weight))
        best, weight = max(options[1:], key=lambda x: (x[0], -x[1]))
        summaries[name] = {'scale': float(scale), 'weight': weight, 'development_ba': best,
                           'development_choices': options, 'development_routed': int(chosen.sum())}
    return summaries


def report_decisions(benchmark, final, after):
    out = {}
    for cohort, rows in benchmark.groupby('cohort').indices.items():
        truth = benchmark.model.iloc[rows].to_numpy()
        baseline = final[rows] == truth
        correct = after[rows] == truth
        fixed = int((~baseline & correct).sum())
        harmed = int((baseline & ~correct).sum())
        by_model = pd.Series(correct.astype(int) - baseline.astype(int)).groupby(truth).mean()
        short = benchmark.n_words.to_numpy()[rows] < 100
        out[cohort] = {'rows': len(rows), 'before_ba': float(pd.Series(baseline).groupby(truth).mean().mean()),
                       'after_ba': float(pd.Series(correct).groupby(truth).mean().mean()),
                       'fixed': fixed, 'harmed': harmed, 'net': fixed - harmed,
                       'recall_delta': by_model.to_dict(), 'short_rows': int(short.sum()),
                       'short_ba_delta': float(pd.Series(correct[short].astype(int) - baseline[short].astype(int))
                                               .groupby(truth[short]).mean().mean()) if short.any() else None,
                       'changed': int((final[rows] != after[rows]).sum())}
    return out


def report_pairs(benchmark, routes, raw):
    truth = benchmark.iloc[routes.row].model.to_numpy()
    cohort = benchmark.iloc[routes.row].cohort.to_numpy()
    incumbent = truth == routes.incumbent.to_numpy()
    challenger = truth == routes.challenger.to_numpy()
    pairs = {}
    for name in benchmark.cohort.unique():
        in_cohort = cohort == name
        labeled = in_cohort & (incumbent | challenger)
        pairs[name] = {'routed': int(in_cohort.sum()), 'truth_in_pair': int(labeled.sum())}
        if labeled.any():
            for method, scores in raw.items():
                gap = np.diff(scores[labeled], axis=1).ravel()
                orient = np.where(challenger[labeled], 1, -1)
                pairs[name][method] = float(np.mean((orient * gap > 0) + 0.5 * (gap == 0)))
    return pairs


def route_scores(model, tokenizer, benchmark, routes, adapter_labels, config, device, deadline=None):
    codes = {label: i for i, label in enumerate(adapter_labels)}
    pairs = np.array([[codes[first], codes[second]] for first, second in zip(routes.incumbent, routes.challenger)],
                     dtype=np.int64).reshape(-1, 2)
    if not len(pairs):
        return {name: np.empty((0, 2)) for name in METHODS}
    return score_pairs(model, tokenizer, benchmark.text.iloc[routes.row], pairs, config, device, deadline)


def evaluation(args, config, benchmark):
    with np.load(args.trace / 'stage_scores.npz', allow_pickle=False) as trace:
        if trace['ids'].tolist() != benchmark.id.tolist():
            raise ValueError('Baseline trace row order differs from public probe')
        labels = trace['labels'].tolist()
        final = trace['final']
        probs = trace['typography_probs']
    target_labels = sorted({label for group in config.get('tracks', TRACKS) for label in group})
    rows = training_rows(args.pool, benchmark, pd.read_csv(args.official), target_labels,
                         config['per_class'], config['seed'])
    (args.out / 'training_data.json').write_text(json.dumps({'by_model': rows.model.value_counts().to_dict(),
        'by_source': rows.source_dataset.value_counts().to_dict()}, indent=2) + '\n')
    device = 'cuda:0'
    torch.manual_seed(config['seed'])
    tokenizer = load_tokenizer(args.model)
    model = AttributionLM(args.model, len(target_labels), device, config['rank'])
    runtime = fit(model, tokenizer, rows, target_labels, config, device, args.out / 'atlas_readout.pt')
    routes = route_candidates(benchmark.text.tolist(), labels, probs, final, config)
    raw = route_scores(model, tokenizer, benchmark, routes, target_labels, config, device)
    calibration = calibrate(benchmark, routes, raw, final)
    methods = {}
    table = benchmark.drop(columns='text').assign(baseline=final)
    routes = routes.assign(id=benchmark.id.iloc[routes.row].to_numpy())
    for name in METHODS:
        after = apply_scores(final, routes, raw, name, calibration)
        methods[name] = report_decisions(benchmark, final, after)
        table[name] = after
        routes[f'{name}_advantage'] = raw[name][:, 1] - raw[name][:, 0]
    routes.to_parquet(args.out / 'routed_scores.parquet', index=False)
    table.to_parquet(args.out / 'paired_decisions.parquet', index=False)
    report = {'runtime': runtime, 'routed': len(routes), 'config': config, 'calibration': calibration,
              'evaluation': methods, 'pair_attribution': report_pairs(benchmark, routes, raw),
              'labels': target_labels}
    (args.out / 'atlas_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'trace', 'pool', 'official', 'config', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    benchmark = pd.read_parquet(args.trace / 'public_truth.parquet')
    evaluation(args, config, benchmark)


if __name__ == '__main__':
    main()
