"""Compare binary Kimi reviews with the unchanged Sibling Odds decisions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .effort_pair_report import read_holdout
from .kimi_flash import FLASH, KIMI, SHORT_WORDS
from .peer_context_report import paired_stats, read_ids, read_trace, same_ids
from .sibling_specialist import write_json


CAP = 3.


def read_margins(scores, seeds, ids, *, complete):
    inputs = ('training_rows', 'public_rows', 'holdout_rows', 'class_counts', 'training_sha256', 'holdout_sha256')
    receipts, margins = [], {}
    for seed in seeds:
        stem = scores / f'seed_{seed}'
        receipt = json.loads(stem.with_suffix('.json').read_text())
        if receipt['seed'] != seed or (receipts and any(receipt[k] != receipts[0][k] for k in inputs)):
            raise ValueError('The two experts must use the same frozen inputs and distinct seeds')
        receipts.append(receipt)
        logits = read_ids(stem.with_suffix('.parquet'))
        expected = receipt['public_rows'] + receipt['holdout_rows']
        if (len(logits) != expected or set(logits) != {FLASH, KIMI} or not ids.isin(logits.index).all()
                or not np.isfinite(logits.to_numpy()).all()):
            raise ValueError(f'Invalid evaluation coverage or logits for seed {seed}')
        if complete:
            same_ids(logits.index, ids, f'All remote scores for seed {seed}')
        margins[f'seed_{seed}'] = logits.loc[ids, KIMI] - logits.loc[ids, FLASH]
    return pd.DataFrame(margins).assign(expert=lambda x: x.mean(axis=1)), receipts


def predict_rows(truth, trace, baseline, margins, weights, primary):
    if not truth.n_words.eq(truth.text.str.split().str.len()).all():
        raise ValueError('The trace and report disagree about body lengths')
    flash, kimi = pd.Index(trace['labels']).get_indexer([FLASH, KIMI])
    if min(flash, kimi) < 0:
        raise ValueError('Missing labels in original trace')
    glm_stage = trace['glm_winner'] | trace['fable_guard']
    scores = np.where(glm_stage[:, None], trace['glm_logp'], trace['sibling_logp'])
    gap = np.maximum(scores[:, flash] - scores[:, kimi], 0.)
    eligible = truth.n_words.lt(SHORT_WORDS).to_numpy() & (baseline == FLASH)
    rows = truth.drop(columns='text').assign(baseline=baseline, eligible=eligible,
                                           glm_stage=glm_stage, flash_base_gap=gap)
    for arm in margins:
        raw = margins.loc[rows.index, arm].to_numpy()
        rows[f'{arm}_margin'] = raw
        evidence = np.clip(raw, -CAP, CAP)
        for weight in (weights if arm == 'expert' else (primary,)):
            switched = eligible & (weight * evidence > gap)
            rows[f'{arm}_{weight:g}'] = np.where(switched, KIMI, baseline)
    return rows.assign(expert=rows[f'expert_{primary:g}'])


def compare(name, subset, part, prediction, draws):
    summary = paired_stats(part, 'baseline', prediction, draws)
    changes = part[part.baseline.ne(part[prediction])]
    by_model = []
    for model, group in part.groupby('model'):
        old, new = group.baseline.eq(model), group[prediction].eq(model)
        changed = group.baseline.ne(group[prediction])
        by_model.append(dict(model=model, rows=len(group), eligible=int(group.eligible.sum()),
                             changed=int(changed.sum()), correct_before=int(old.sum()), correct_after=int(new.sum())))
    return dict(cohort=name, subset=subset, eligible=int(part.eligible.sum()),
                rescued_kimi=int(changes.model.eq(KIMI).sum()), harmed_flash=int(changes.model.eq(FLASH).sum()),
                other_changed=int((~changes.model.isin([FLASH, KIMI])).sum()), by_true_model=by_model,
                **summary)


def run(scores, config_path, public_trace, output, holdout_panel=None, holdout_trace=None, draws=10000):
    config = json.loads(config_path.read_text())
    weights, primary, seeds = sorted(set(config['fusion_weights'])), config['primary_weight'], config['seeds']
    primary_source = config['primary_source']
    if (not weights or weights[0] != 0 or not np.isfinite(weights).all() or primary not in weights
            or len(set(seeds)) != 2 or draws < 1):
        raise ValueError('Invalid weight grid, seeds or bootstrap count')
    if (holdout_panel is None) != (holdout_trace is None):
        raise ValueError('Provide both holdout paths or neither')
    truth = read_ids(public_trace / 'public_truth.parquet')
    public = (truth, *read_trace(public_trace, truth))
    inputs, exposure = [public], None
    if holdout_panel is not None:
        truth, trace, baseline, exposure = read_holdout(holdout_panel, holdout_trace)
        if set(truth.model) != {FLASH, KIMI} or not truth.n_words.lt(SHORT_WORDS).all():
            raise ValueError('The new panel must have both true short classes')
        exposed = set()
        for field in ('active_pool_ids', 'official_train_ids', 'full_pool_ids', 'stack_extra_ids'):
            exposed.update(exposure[field])
        if not exposed.issubset(truth.index):
            raise ValueError('Unknown IDs in source-holdout exposure audit')
        truth['known_baseline_corpus_overlap'] = truth.index.isin(exposed)
        inputs.append((truth, trace, baseline))
    ids = inputs[0][0].index
    if len(inputs) > 1:
        ids = ids.append(inputs[1][0].index)
    margins, receipts = read_margins(scores, seeds, ids, complete=len(inputs) > 1)
    if len(inputs) > 1:
        query_hash = hashlib.sha256((holdout_panel / 'queries.parquet').read_bytes()).hexdigest()
        if query_hash != receipts[0]['holdout_sha256']:
            raise ValueError('Expert scores were computed on a different holdout file')

    rows = pd.concat([predict_rows(*group, margins, weights, primary) for group in inputs], verify_integrity=True)
    comparisons = []
    for name, group in rows.groupby('cohort'):
        short = group[group.n_words.lt(SHORT_WORDS)]
        slices = {'short': short}
        if name == 'independent_holdout':
            clean = short[~short.known_baseline_corpus_overlap.eq(True)]
            slices['short_without_known_overlap'] = clean
            verified = clean[clean.provenance.eq('requested_and_served_ids')]
            slices['short_with_model_receipts'] = verified
            slices['primary_holdout'] = verified[verified.source_dataset.eq(primary_source)]
            groups = clean.groupby('pair_id').model.transform('nunique').eq(2)
            slices['same_prompt_length_matched'] = clean[groups]
            slices.update({str(source): part for source, part in clean.groupby('source_dataset')})
        for subset, sample in slices.items():
            if sample.empty:
                continue
            predictions = [f'expert_{weight:g}' for weight in weights]
            if subset in ('short', 'short_without_known_overlap', 'primary_holdout'):
                predictions += [f'seed_{seed}_{primary:g}' for seed in seeds]
            comparisons.extend(compare(name, subset, sample, pred, draws) for pred in predictions)

    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'paired_predictions.parquet')
    rows.loc[rows.baseline.ne(rows.expert)].to_parquet(output / 'changed_rows.parquet')
    report = dict(primary_weight=primary, primary_source=primary_source,
                  fusion_weights=weights, expert_score_cap=CAP,
                  routing=dict(original_label=FLASH, max_words_exclusive=SHORT_WORDS, new_label=KIMI),
                  training_receipts=receipts, holdout_exposure=exposure, comparisons=comparisons)
    write_json(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('scores', 'config', 'public-trace', 'output'):
        parser.add_argument('--' + arg, required=True, type=Path)
    parser.add_argument('--holdout-panel', type=Path)
    parser.add_argument('--holdout-trace', type=Path)
    parser.add_argument('--draws', type=int, default=10000)
    args = parser.parse_args()
    run(args.scores, args.config, args.public_trace, args.output, args.holdout_panel, args.holdout_trace, args.draws)
