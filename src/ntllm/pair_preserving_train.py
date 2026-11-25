"""Train and evaluate a full reader while changing only the preserved pairs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .peer_context_report import paired_stats, read_ids, read_trace, same_ids, same_meta
from .provenance_readers import curate, read_inputs, text_key
from .sibling_odds_replay import replay_predictions


ARMS = ('plain_ce', 'conditional_pair')
TARGETS = ('gemini-3.7-flash', 'gemini-3.8-flash')
CLASS_LENGTH = ['model', 'length_band']
WEIGHT = .15
PRVSIYAN = 'prvsiyan/prvsiyans-extended-name-that-llm-dataset'


def extra_queries(holdout, marked_audit, marked_raw):
    truth = read_ids(holdout / 'candidate_truth.parquet')
    query = read_ids(holdout / 'candidate_queries.parquet')
    same_ids(truth.index, query.index, 'Independent queries and truth', ordered=True)
    primary = truth.loc[truth.source_dataset.eq(PRVSIYAN + '/13')].join(query[['text']]).assign(cohort='prvsiyan')
    audit = read_ids(marked_audit)
    raw = pd.read_csv(marked_raw, dtype={'id': str, 'prompt_id': str}).set_index('id', verify_integrity=True)
    same_ids(audit.index, raw.index, 'Provider raw and audited IDs', ordered=True)
    same_meta(audit, raw, ['model', 'prompt_id', 'served_model'], 'Verified provider source')
    provider = audit.loc[~audit.any_match, ['model', 'prompt_id', 'n_words']].join(raw[['text']])
    provider = provider.assign(cohort='verified_provider').rename(index=lambda key: 'verified-provider:' + key)
    if len(primary) != 895 or len(provider) != 279:
        raise ValueError('The two frozen external evaluation panels changed scope')
    return pd.concat([primary, provider], verify_integrity=True)


def prepare(official, public, pool, inputs, original_sample, selected_ids, independent, provider_audit,
            provider_raw, work, outputs, config):
    original = read_inputs(official, public, inputs)
    with np.load(public / 'stage_scores.npz', allow_pickle=False) as trace:
        labels = trace['labels'].tolist()
    rows = curate(pool, original, labels).set_index('id', verify_integrity=True)
    original_rows = read_ids(original_sample)
    selected = pd.read_parquet(selected_ids).rename_axis('id')
    if not selected.index.is_unique or selected.index.hasnans:
        raise ValueError('The shared training sample contains missing or repeated IDs')
    if len(original_rows) != 44522 or len(selected) != 44522:
        raise ValueError('The two training arms must share 44,522 passages')
    plan = json.loads(selected_ids.with_name('report.json').read_text())['enriched']
    hash_ids = hashlib.sha256(('\n'.join(selected.index) + '\n').encode()).hexdigest()
    if hash_ids != plan['arms']['pair']['ordered_ids_sha256']:
        raise ValueError('The shared training IDs differ from the pinned sampling report')
    external = extra_queries(independent, provider_audit, provider_raw)
    prior = read_ids(inputs / 'kimi_flash/truth.parquet')
    prior_prompts = prior.loc[prior.source_dataset.eq(PRVSIYAN), 'prompt_id']
    strict = external.loc[external.cohort.eq('prvsiyan') & ~external.prompt_id.isin(prior_prompts)]
    if len(strict) != 316 or strict.prompt_id.nunique() != 23:
        raise ValueError('The strict 18-label topic panel changed')
    evaluation = pd.concat([original, external[original.columns]], verify_integrity=True)
    evaluation['strict_source_topic'] = evaluation.index.isin(strict.index)
    if not evaluation.n_words.eq(evaluation.text.str.split().str.len()).all():
        raise ValueError('A frozen evaluation length disagrees with the actual text')
    excluded = set(text_key(evaluation.text))
    work.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    original_quotas = rows.loc[original_rows.index].groupby(CLASS_LENGTH).size().sort_index()
    sample = rows.loc[selected.index].assign(vertex_pair_id=selected.vertex_pair_id)
    if not sample.groupby(CLASS_LENGTH).size().sort_index().equals(original_quotas):
        raise ValueError('The shared sample changed the original model/length quotas')
    if sample.body_key.isin(excluded).any() or not sample.body_key.is_unique:
        raise ValueError('Evaluation body overlap or duplicated training bodies')
    paired = sample[sample.vertex_pair_id.notna()]
    groups = paired.groupby('vertex_pair_id').model.agg(['size', 'nunique'])
    if not (len(groups) == 941 and groups.eq(2).all().all() and set(paired.model) == set(TARGETS)):
        raise ValueError('There must be 941 verified training pairs with both Gemini labels')
    sample.reset_index().to_parquet(work / 'training.parquet', index=False)
    receipt = dict(rows=len(sample), ordered_ids_sha256=hash_ids, verified_pairs=len(groups),
                   classes=sample.model.value_counts().to_dict())
    evaluation.to_parquet(work / 'evaluation.parquet')
    (work / 'labels.json').write_text(json.dumps(labels))
    (outputs / 'data_report.json').write_text(json.dumps(dict(settings=json.loads(config.read_text()),
                                                               evaluation=evaluation.cohort.value_counts().to_dict(),
                                                               sampling=receipt, verified_pair_design=plan),
                                                     indent=2) + '\n')
    print('Identical training examples and groups for both arms:', receipt, flush=True)


def paired_order(panel, logits):
    margin = logits.loc[panel.index, TARGETS[1]] - logits.loc[panel.index, TARGETS[0]]
    wide = panel.assign(margin=margin).pivot_table(index='prompt_id', columns='model', values='margin', aggfunc='mean')
    if not set(TARGETS).issubset(wide):
        return dict(prompts=0, correct=0, tied=0)
    complete = wide.dropna(subset=list(TARGETS))
    delta = complete[TARGETS[1]] - complete[TARGETS[0]]
    return dict(prompts=len(delta), correct=int(delta.gt(0).sum()), tied=int(delta.eq(0).sum()))


def evaluate(public, work, outputs, draws=3000):
    truth = read_ids(work / 'evaluation.parquet')
    hist = read_ids(public / 'public_truth.parquet')
    trace, baseline = read_trace(public, hist)
    rows = truth.drop(columns='text')
    rows['baseline'] = pd.Series(baseline, index=hist.index)
    scores = {}
    for arm in ARMS:
        scores[arm] = read_ids(outputs / f'{arm}_logits.parquet')
        same_ids(scores[arm].index, rows.index, arm, ordered=True)
        if not scores[arm].columns.equals(pd.Index(trace['labels'])) or not np.isfinite(scores[arm].to_numpy()).all():
            raise ValueError('A training arm produced different output labels or nonfinite scores')
        rows[arm] = scores[arm].idxmax(axis=1)
        rows[f'{arm}_blend'] = pd.Series(replay_predictions(trace, hist.text, external_logits=scores[arm], weight=WEIGHT),
                                        index=hist.index)
    summaries = []
    panels = dict(tuple(rows.groupby('cohort')))
    panels['prvsiyan_strict'] = rows.loc[rows.strict_source_topic]
    for cohort, part in panels.items():
        short = part.n_words.lt(100)
        focus = part.model.isin(TARGETS)
        for name, panel in (('all', part), ('short', part[short]), ('gemini', part[focus]),
                            ('short_gemini', part[focus & short])):
            if panel.empty:
                continue
            entry = dict(cohort=cohort, subset=name, standalone=paired_stats(panel, *ARMS, draws),
                         paired_gemini_ordering={arm: paired_order(panel[panel.model.isin(TARGETS)], logits)
                                                for arm, logits in scores.items()})
            if panel.baseline.notna().all():
                entry['blend'] = {arm: paired_stats(panel, 'baseline', f'{arm}_blend', draws) for arm in ARMS}
            summaries.append(entry)
    rows.to_parquet(outputs / 'predictions.parquet')
    report = dict(weight=WEIGHT, scopes=summaries,
                  limits='Provider stress reuses official reconstructed topics. The independent source has new prompts.')
    (outputs / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('base', 'config', 'work', 'output'):
        parser.add_argument('--' + arg, required=True, type=Path)
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    import torch

    from .exact_pair_train import train_exact
    from .sibling_specialist import Specialist, predict

    config = json.loads(args.config.read_text())
    if torch.cuda.device_count() != 1 or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Each arm must have one T4')
    torch.manual_seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, padding_side='right')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    labels = json.loads((args.work / 'labels.json').read_text())
    sample = pd.read_parquet(args.work / 'training.parquet')
    queries = read_ids(args.work / 'evaluation.parquet')
    model = Specialist(args.base, config, len(labels), 'cuda:0')
    receipt = train_exact(model, tokenizer, sample, labels, config, 'cuda:0', args.arm == 'conditional_pair')
    prediction = predict(model, tokenizer, queries, labels, config, 'cuda:0')
    pd.DataFrame(prediction, index=queries.index, columns=labels).to_parquet(args.output / f'{args.arm}_logits.parquet')
    torch.save({name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad},
               args.output / f'{args.arm}.pt')
    (args.output / f'{args.arm}.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
