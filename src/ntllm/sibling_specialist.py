"""Train Qwen LoRA on error families and replay repairs on a fixed Sibling Odds trace."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from .paired_attribution import LowRankLinear, read_pool
from .sibling_odds_replay import replay_predictions


FAMILIES = ('claude', 'gemini', 'glm', 'gpt', 'kimi', 'qwen')
BLOCKS = (('claude', 'gemini'), ('gpt',))


def family(label):
    if label.startswith('qwen'):
        return 'qwen'
    return label.split('-', 1)[0]


def loss_groups(labels, device):
    families = [family(label) for label in labels]
    family_codes = torch.as_tensor([FAMILIES.index(name) for name in families], device=device)
    block_for_family = {name: i for i, block in enumerate(BLOCKS, 1) for name in block}
    block_codes = torch.as_tensor([block_for_family.get(name, 0) for name in families], device=device)
    family_mask = family_codes[None, :] == torch.arange(len(FAMILIES), device=device)[:, None]
    block_mask = block_codes[None, :] == torch.arange(len(BLOCKS) + 1, device=device)[:, None]
    return family_codes, block_codes, family_mask, block_mask


def classification_loss(scores, target, class_weights, groups=None):
    loss = F.cross_entropy(scores, target, weight=class_weights)
    if groups is None:
        return loss
    family_codes, block_codes, family_mask, block_mask = groups
    family_scores = scores[:, None, :].masked_fill(~family_mask[None], -torch.inf).logsumexp(-1)
    destination = block_codes[target]
    block_scores = scores.masked_fill(~block_mask[destination], -torch.inf)
    within = F.cross_entropy(block_scores, target, reduction='none')
    return (loss + .40 * F.cross_entropy(family_scores, family_codes[target])
            + .60 * (within * destination.ne(0)).mean())


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def batches(tokens, limit, rng=None):
    order = np.argsort([len(row) for row in tokens], kind='stable')
    if rng is not None:
        for start in range(0, len(order), 16 * limit):
            rng.shuffle(order[start:start + 16 * limit])
    result = [order[i:i + limit] for i in range(0, len(order), limit)]
    if rng is not None:
        rng.shuffle(result)
    return result


class Specialist(nn.Module):
    def __init__(self, base, config, n_classes, device):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        arch = AutoConfig.from_pretrained(base, local_files_only=True)
        arch.num_hidden_layers = config['layers']
        if getattr(arch, 'layer_types', None) is not None:
            arch.layer_types = arch.layer_types[:arch.num_hidden_layers]
        self.trunk = AutoModel.from_pretrained(base, config=arch, torch_dtype=torch.float16,
                                               attn_implementation='sdpa', device_map={'': device},
                                               local_files_only=True).requires_grad_(False)
        for layer in self.trunk.layers[-config['adapted_layers']:]:
            for projection in ('q_proj', 'v_proj', 'o_proj'):
                original = getattr(layer.self_attn, projection)
                setattr(layer.self_attn, projection, LowRankLinear(original, rank=8).to(device))
        self.head = nn.Sequential(nn.LayerNorm(2 * arch.hidden_size), nn.Linear(2 * arch.hidden_size, 256),
                                  nn.GELU(), nn.Dropout(.10), nn.Linear(256, n_classes)).to(device)

    def forward(self, feed):
        mask = feed['attention_mask'].bool()
        hidden = self.trunk(**feed, use_cache=False).last_hidden_state.float()
        length = mask.sum(1)
        average = (hidden * mask.unsqueeze(-1)).sum(1) / length[:, None]
        last = hidden[torch.arange(len(hidden), device=hidden.device), length - 1]
        return self.head(torch.cat((average, last), dim=1)).float()


def prepare(pool, official, truth, config):
    known = pd.concat([official[['text', 'model']], truth[['text', 'model']]], ignore_index=True)
    external, held = read_pool(pool, known)
    external = external.loc[~external.source_dataset.str.startswith(('beroli/', 'markbarney/'))].copy()
    rng = np.random.default_rng(config['seed'])
    size = external.groupby(['model', 'source_dataset']).id.transform('size')
    external['priority'] = -np.log(rng.random(len(external))) * np.sqrt(size)
    external = external.sort_values('priority')
    rank = external.groupby('model').cumcount()
    focused = external.model.map(family).isin(('claude', 'gemini', 'gpt'))
    sampled = external[rank < np.where(focused, config['focus_per_class'], config['background_per_class'])]
    held = (held.sample(frac=1, random_state=config['seed']).groupby(['source_dataset', 'model'])
            .head(config['source_holdout_per_class']))
    assert set(sampled.model) == set(official.model)
    assert set(held.prompt_canon).isdisjoint(sampled.prompt_canon)
    return sampled.reset_index(drop=True), held.reset_index(drop=True)


def tokenize(tokenizer, frame, config):
    return tokenizer(frame.text.tolist(), truncation=True, max_length=config['max_tokens'],
                     add_special_tokens=False)['input_ids']


def feed_batch(tokenizer, tokens, rows, device):
    return tokenizer.pad({'input_ids': [tokens[i] for i in rows]}, return_tensors='pt',
                         pad_to_multiple_of=8).to(device)


def train(model, tokenizer, frame, labels, config, device, focused, official=False):
    phase = 'official' if official else 'external'
    epochs = config[phase + '_epochs']
    rate = config['official_learning_rate'] if official else config['learning_rate']
    rng = np.random.default_rng(config['seed'] + int(official))
    tokens = tokenize(tokenizer, frame, config)
    codes = pd.Categorical(frame.model, categories=labels).codes.astype(np.int64)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=rate)
    scaler = torch.amp.GradScaler('cuda', init_scale=256)
    groups = loss_groups(labels, device) if focused else None
    counts = np.bincount(codes, minlength=len(labels))
    class_weights = torch.as_tensor(len(frame) / (len(labels) * counts), device=device, dtype=torch.float32)
    model.train()
    started = time.monotonic()
    for epoch in range(epochs):
        optimizer_steps = 0
        for step, rows in enumerate(batches(tokens, config['batch_size'], rng), 1):
            target = torch.as_tensor(codes[rows], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                scores = model(feed_batch(tokenizer, tokens, rows, device))
                loss = classification_loss(scores, target, class_weights, groups)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters, 1.)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += int(scaler.get_scale() >= scale)
            if step == 1 or step % 100 == 0:
                print(phase, epoch + 1, step, 'loss', round(loss.item(), 4), 'elapsed',
                      round(time.monotonic() - started), flush=True)
        assert optimizer_steps >= step * .95, 'Too many mixed precision steps were skipped'
        print(phase, 'complete', epoch + 1, 'updates', optimizer_steps, flush=True)


@torch.inference_mode()
def predict(model, tokenizer, frame, labels, config, device):
    tokens = tokenize(tokenizer, frame, config)
    output = np.empty((len(frame), len(labels)), dtype=np.float32)
    model.eval()
    for rows in batches(tokens, config['batch_size']):
        with torch.autocast('cuda', dtype=torch.float16):
            output[rows] = model(feed_batch(tokenizer, tokens, rows, device)).cpu().numpy()
    assert np.isfinite(output).all()
    return output


def collect(trace, truth):
    final = replay_predictions(trace, truth.text)
    before_typography = replay_predictions(trace, truth.text, typography=False)
    gemini_mask = ((before_typography != final) & np.char.startswith(before_typography.astype(str), 'gemini-')
                   & truth.text.str.contains('—', regex=False).to_numpy())
    return final, before_typography, gemini_mask


def repair(trace, truth, logits, weight, restore):
    labels = trace['labels']
    final, pre_typography, dash = collect(trace, truth)
    output = np.where(dash & restore, pre_typography, final)
    if not weight:
        return output
    families = np.array([family(label) for label in labels])
    base = trace['sibling_logp']
    top_three = np.argsort(-base, axis=1, kind='stable')[:, :3]
    candidate_mask = np.zeros_like(base, dtype=bool)
    np.put_along_axis(candidate_mask, top_three, True, axis=1)
    incumbent = pd.Index(labels).get_indexer(output)
    rows = np.arange(len(base))
    protected = trace['glm_winner'] | trace['fable_guard']
    blend = base + weight * logits
    for block in BLOCKS:
        relevant = np.isin(families, block)
        mask = candidate_mask & relevant[None, :]
        mask[rows, incumbent] = True
        proposal = np.where(mask, blend, -np.inf).argmax(1)
        close = base[rows, incumbent] - base[rows, proposal] <= 1.5
        # Require the specialist itself to favor every changed label.
        accept = (~protected & relevant[incumbent] & close
                  & (logits[rows, proposal] > logits[rows, incumbent]))
        output[accept] = labels[proposal[accept]]
    return output


def summarize(frame, baseline, predictions):
    result = {}
    for name, selected in [('all', np.ones(len(frame), dtype=bool)), ('short', frame.n_words.lt(100).to_numpy())]:
        part = frame.iloc[selected]
        old = part.model.eq(baseline[selected])
        new = part.model.eq(predictions[selected])
        fixed, harmed = int((~old & new).sum()), int((old & ~new).sum())
        result[name] = {'rows': len(part), 'ba_before': old.groupby(part.model).mean().mean(),
                        'ba_after': new.groupby(part.model).mean().mean(), 'fixed': fixed, 'harmed': harmed,
                        'net': fixed - harmed, 'changed': int((baseline[selected] != predictions[selected]).sum())}
    return result


def evaluate(trace, truth, logits, folder):
    baseline, _, _ = collect(trace, truth)
    settings = [(False, 0.), (True, 0.), (True, .15), (True, .3), (True, .5)]
    base_restore = repair(trace, truth, logits, 0., True)
    evaluation = {}
    output = truth.drop(columns='text').assign(baseline=baseline)
    for restore, weight in settings:
        name = f'dash_{int(restore)}_weight_{weight}'
        pred = repair(trace, truth, logits, weight, restore)
        output[name] = pred
        evaluation[name] = {}
        for cohort in truth.cohort.unique():
            select = truth.cohort.eq(cohort).to_numpy()
            records = summarize(truth.iloc[select], baseline[select], pred[select])
            records['increment_over_dash_only'] = summarize(truth.iloc[select], base_restore[select], pred[select])
            evaluation[name][cohort] = records
    dev = [name for name in evaluation if name.startswith('dash_1')]
    selected = max(dev, key=lambda k: evaluation[k]['development']['all']['ba_after'])
    output.to_parquet(folder / 'paired.parquet')
    result = {'selected_on_development': selected, 'arms': evaluation}
    write_json(folder / 'evaluation.json', result)
    print('paired results', selected, evaluation[selected], flush=True)


def run(args):
    config = json.loads(args.config.read_text())
    official = pd.read_csv(args.official, dtype={'id': str, 'prompt_id': str})
    with np.load(args.trace / 'stage_scores.npz', allow_pickle=False) as src:
        trace = dict(src)
    labels = trace['labels'].tolist()
    truth = pd.read_parquet(args.trace / 'public_truth.parquet').set_index('id').loc[trace['ids']]
    assert np.array_equal(replay_predictions(trace, truth.text, sibling_odds=False), trace['final'])
    external, held = prepare(args.pool, official, truth, config)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = external.drop(columns='text').assign(split='external')
    pd.concat([manifest, held.drop(columns='text').assign(split='held')]).to_parquet(args.out / 'sample.parquet')
    write_json(args.out / 'sample.json', {
        'external': len(external), 'external_short': int(external.n_words.lt(100).sum()),
        'prompts': external.prompt_canon.nunique(), 'sources': external.source_dataset.nunique(),
        'class_counts': external.model.value_counts().to_dict(), 'official': len(official), 'held': len(held),
        'held_sources': held.source_dataset.value_counts().to_dict(), 'config': config})
    if args.prepare_only:
        loss_groups(labels, 'cpu')
        evaluate(trace, truth, np.zeros_like(trace['sibling_logp']), args.out)
        return
    from transformers import AutoTokenizer

    device = 'cuda:0'
    torch.manual_seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, padding_side='right')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Specialist(args.base, config, len(labels), device)
    train(model, tokenizer, external, labels, config, device, args.focused)
    external_held = predict(model, tokenizer, held, labels, config, device)
    train(model, tokenizer, official, labels, config, device, args.focused, official=True)
    public_logits = predict(model, tokenizer, truth, labels, config, device)
    evaluate(trace, truth, public_logits, args.out)
    pd.DataFrame(public_logits, columns=labels, index=truth.index).to_parquet(args.out / 'public_logits.parquet')
    adapted_held = predict(model, tokenizer, held, labels, config, device)
    held_results = held.drop(columns='text').assign(external=np.array(labels)[external_held.argmax(1)],
                                                    adapted=np.array(labels)[adapted_held.argmax(1)])
    held_results.to_parquet(args.out / 'held_predictions.parquet')
    torch.save({name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad},
               args.out / 'specialist.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'official', 'trace', 'pool', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--focused', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    run(parser.parse_args())
