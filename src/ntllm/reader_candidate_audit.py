"""Use the two trained readers to rescore or expand fixed stage candidates."""

import json

import numpy as np
import pandas as pd

from .counterfactual_native_guard import guard_frame
from .counterfactual_selection import build_actions
from .counterfactual_selection_audit import calibration_ids, canonical_prompt, fit_arm, make_blocks
from .member_evidence_transfer import verify_manifest
from .peer_context_report import paired_stats
from .reader_transfer import PROVENANCE, read_logits


ARMS = ('stage', 'reader_rerank', 'reader_expand')


def reader_actions(truth, trace, baseline, directory, exact):
    labels = pd.Index(trace['labels'])
    matrices = {arm: read_logits(directory / f'{arm}_logits.parquet', truth.index, labels, exact=exact).to_numpy()
                for arm in PROVENANCE}
    both = build_actions(truth, trace, baseline, matrices)
    row = truth.index.get_indexer(both.index)
    candidate = labels.get_indexer(both.candidate)
    incumbent = labels.get_indexer(both.baseline)
    for arm, scores in matrices.items():
        spread = scores.std(axis=1).clip(.10).astype(float)
        margin = (scores[row, candidate] - scores[row, incumbent]) / spread[row]
        both[f'feature_{arm}_margin'] = margin.clip(-8., 8.)
    # These fine-tunes share one backbone. The relative margins preserve
    # differences without presenting their agreement as independent votes.
    member = [col for col in both if col.startswith('feature_member_') or col.startswith('feature_incumbent_member_')]
    expanded = both.drop(columns=member)
    stage_only = expanded.loc[~both.feature_member_only]
    return {
        'stage': build_actions(truth, trace, baseline),
        'reader_rerank': stage_only,
        'reader_expand': expanded,
    }


def run(historical, prior, gen003, kimi, inputs, output, reference, *, independent=None,
        source=None, training=None, heldout=None, draws=2000):
    blocks, held_truth = make_blocks(historical, prior, gen003, kimi, inputs, independent, source)
    folders = [historical, gen003, kimi, *([independent] if independent else [])]
    if independent:
        verify_manifest(heldout / 'manifest.json', held_truth)
    rows, alternatives = [], {arm: [] for arm in ARMS}
    labels = blocks[0][1]['labels']
    for idx, (truth, trace, baseline, _, _) in enumerate(blocks):
        if not np.array_equal(trace['labels'], labels):
            raise ValueError('The reader arms require the same ordered classes across traces')
        fields = ['cohort', 'model', 'prompt_id', 'n_words', 'text']
        rows.append(truth[fields].assign(baseline=baseline, strict=truth.get('strict_topic_holdout', False)))
        for arm, part in reader_actions(truth, trace, baseline, heldout if idx == 3 else training, idx == 3).items():
            alternatives[arm].append(part)
    rows = pd.concat(rows, verify_integrity=True)
    arms = {arm: pd.concat(pieces) for arm, pieces in alternatives.items()}
    dev, extra, separation = calibration_ids(rows, held_truth)
    fit = dev.append(extra)
    c2 = rows.index[rows.cohort.eq('evaluation')]
    stress = rows.index[rows.cohort.eq('reconstructed_stress')]
    held = rows.index[rows.cohort.eq('prvsiyan')]
    strict = held[rows.loc[held, 'strict']]
    target = c2.append([stress, held])
    control = pd.read_parquet(reference)
    control = control.loc[control.arm.eq('stage_multisource')]

    def guard(ids, candidates):
        pieces = []
        for (truth, trace, incumbent, _, _), folder in zip(blocks, folders, strict=True):
            shared = ids.intersection(truth.index, sort=False)
            if len(shared):
                pieces.append(guard_frame(truth, trace, incumbent, candidates.loc[shared], folder))
        return pd.concat(pieces, verify_integrity=True).loc[ids]

    def count_coverage(ids, actual):
        stage_options = arms['stage'].loc[arms['stage'].index.isin(ids)]
        expanded = arms['reader_expand'].loc[arms['reader_expand'].index.isin(ids)]
        wrong = actual.index[actual.baseline.ne(actual.model)]
        stage_present = stage_options.loc[stage_options.outcome.eq(1)].index.unique()
        added_present = expanded.loc[expanded.outcome.eq(1)].index.unique()
        remaining = wrong.difference(stage_present)
        return dict(original_errors=len(wrong), true_in_stage_candidates=len(wrong.intersection(stage_present)),
                    newly_covered=len(remaining.intersection(added_present)))

    def compare(ids, train_ids, verify_reference):
        part = rows.loc[ids].drop(columns='text').copy()
        locked = {}
        for arm in ARMS:
            fitted, audit = fit_arm(arms[arm], rows, train_ids, ids)
            predictions = guard(ids, fitted)
            for mode in ('open', 'native_guard'):
                part[f'{arm}_{mode}'] = predictions[mode]
            locked[arm] = audit['open']['threshold']
        if verify_reference and any(
            not np.array_equal(part.loc[c2, f'stage_{mode}'], control.loc[c2, mode])
            for mode in ('open', 'native_guard')
        ):
            raise ValueError('The reader comparison altered its frozen stage-only control')
        return part, locked

    def summarize(panel):
        metrics = []
        for mode in ('open', 'native_guard'):
            metrics.extend(paired_stats(panel, 'baseline', f'{arm}_{mode}', draws) for arm in ARMS)
            metrics.extend(paired_stats(panel, f'stage_{mode}', f'{arm}_{mode}', draws) for arm in ARMS[1:])
        return dict(paired=metrics, coverage=count_coverage(panel.index, panel))

    evaluated, locked = compare(target, fit, True)
    scopes = {'c2': c2, 'stress': stress, 'prvsiyan_895': held, 'prvsiyan_316': strict}
    panels = {}
    for scope, ids in scopes.items():
        actual = evaluated.loc[ids]
        if actual.empty:
            continue
        panels[scope] = {size: summarize(part) for size, part in
                         [('all', actual), ('short', actual.loc[actual.n_words.lt(100)])] if len(part)}
    leave_source = {}
    prompt = canonical_prompt(rows.prompt_id)
    artifacts = [evaluated.assign(split='frozen_evaluation')]
    for domain, region in rows.loc[fit].groupby('cohort'):
        other = fit.difference(region.index)
        other = other[~prompt.loc[other].isin(prompt.loc[region.index])]
        scored, thresholds = compare(region.index, other, False)
        leave_source[domain] = dict(thresholds=thresholds, **summarize(scored))
        artifacts.append(scored.assign(split='leave_source_' + domain))
    report = dict(calibration=separation, thresholds=locked, panels=panels, leave_source=leave_source)
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(artifacts, verify_integrity=True).to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
