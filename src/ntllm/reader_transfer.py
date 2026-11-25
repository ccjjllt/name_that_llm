"""Fuse existing reader scores into exact historical and held-out Sibling Odds traces."""

import json

import numpy as np
import pandas as pd

from .member_evidence_transfer import EXPOSURE_KEYS, read_capture, read_prvsiyan, verify_manifest
from .peer_context_report import paired_stats, read_ids, same_ids
from .sibling_odds_replay import replay_predictions


STYLE = ('roberta', 'style')
PROVENANCE = ('length_matched', 'provenance_balanced')
ARMS = STYLE + PROVENANCE
WEIGHT = .15


def read_logits(path, ids, labels, *, exact=False):
    scores = read_ids(path)
    if exact:
        same_ids(scores.index, ids, str(path), ordered=True)
    if not scores.columns.is_unique or set(scores.columns) != set(labels) or not set(ids).issubset(scores.index):
        raise ValueError(f'{path} does not cover the required passage IDs and 18 classes')
    values = scores.loc[ids, labels]
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f'Nonfinite reader evidence in {path}')
    return values


def compare_rows(truth, trace, baseline, style, provenance, exact_provenance):
    labels = trace['labels']
    rows = truth[['cohort', 'model', 'prompt_id', 'n_words']].assign(baseline=baseline)
    zero = replay_predictions(trace, truth.text, weight=0)
    if not np.array_equal(zero, baseline):
        raise ValueError('Zero weight changed an original Sibling Odds decision')
    glm = trace['glm_winner'] | trace['fable_guard']
    before_typography = replay_predictions(trace, truth.text, typography=False)
    claude_eligible = rows.baseline.str.startswith('claude-') & ~glm & rows.baseline.eq(before_typography)
    base_scores = np.where(glm[:, None], trace['glm_logp'], trace['sibling_logp'])
    top3 = labels[np.argsort(-base_scores, axis=1, kind='stable')[:, :3]]
    target = rows.model.to_numpy()[:, None]
    rows['base_top3_contains_true'] = (top3 == target).any(1)
    for arm in ARMS if style else PROVENANCE:
        path = style / f'{arm}_logp.parquet' if arm in STYLE else provenance / f'{arm}_logits.parquet'
        logits = read_logits(path, rows.index, labels, exact=exact_provenance and arm in PROVENANCE)
        ranks = np.argsort(-logits.to_numpy(), axis=1, kind='stable')
        rows[arm] = labels[ranks[:, 0]]
        rows[f'{arm}_top3'] = (labels[ranks[:, :3]] == target).any(1)
        rows[f'{arm}_blend'] = replay_predictions(trace, truth.text, external_logits=logits, weight=WEIGHT)
        if arm in PROVENANCE:
            proposed = rows[f'{arm}_blend']
            rows[f'{arm}_claude'] = proposed.where(claude_eligible & proposed.str.startswith('claude-'), rows.baseline)
    return rows


def panel_statistics(panel, draws, arms=ARMS):
    wrong = panel.baseline.ne(panel.model)
    missed = wrong & ~panel.base_top3_contains_true
    measures = {}
    for arm in arms:
        proposal = panel[f'{arm}_blend']
        labels = panel.model
        by_label = panel.assign(correct_before=~wrong, correct_after=proposal.eq(labels),
                                novel_top3=missed & panel[f'{arm}_top3'])
        by_label = by_label.groupby('model').agg(
            rows=('baseline', 'size'), correct_before=('correct_before', 'sum'),
            correct_after=('correct_after', 'sum'), new_top3_on_missed=('novel_top3', 'sum'),
        )
        measures[arm] = dict(
            new_top1_on_missed=int((missed & panel[arm].eq(labels)).sum()),
            new_top3_on_missed=int((missed & panel[f'{arm}_top3']).sum()),
            blend=paired_stats(panel, 'baseline', f'{arm}_blend', draws),
            by_label=by_label.reset_index().to_dict('records'),
        )
    claude = {arm: paired_stats(panel, 'baseline', f'{arm}_claude', draws) for arm in PROVENANCE}
    comparisons = {'provenance_vs_length': paired_stats(panel, *PROVENANCE, draws)}
    if all(arm in arms for arm in STYLE):
        comparisons['style_vs_roberta'] = paired_stats(panel, *STYLE, draws)
    return dict(
        rows=len(panel), baseline_wrong=int(wrong.sum()), baseline_wrong_outside_top3=int(missed.sum()),
        arms=measures, claude_only_replay=claude, standalone_matched_comparisons=comparisons,
    )


def run(historical_trace, original, independent_trace, audit, prior, style, provenance,
        heldout_provenance, output, draws=4000):
    public = read_ids(original / 'public_truth.parquet')
    independent = read_prvsiyan(audit, prior)
    hist_trace, hist_base, *_ = read_capture(historical_trace, public)
    new_trace, new_base, *_ = read_capture(independent_trace, independent)
    verify_manifest(heldout_provenance / 'manifest.json', independent)
    if not np.array_equal(hist_trace['labels'], new_trace['labels']):
        raise ValueError('The two traces use different class order')
    historical_rows = compare_rows(public, hist_trace, hist_base, style, provenance, False)
    held_rows = compare_rows(independent, new_trace, new_base, style, heldout_provenance, True)
    historical_rows['strict_topic_holdout'] = False
    held_rows['strict_topic_holdout'] = independent.strict_topic_holdout
    overlap = json.loads((independent_trace / 'body_overlap.json').read_text())
    exposed = {identity for key in EXPOSURE_KEYS for identity in overlap.get(key, [])}
    if not exposed.issubset(held_rows.index):
        raise ValueError('The exposure check contains unknown held-out passage IDs')
    historical_rows['runtime_corpus_overlap'] = False
    held_rows['runtime_corpus_overlap'] = held_rows.index.isin(exposed)
    rows = pd.concat([historical_rows, held_rows], verify_integrity=True)
    panels = {
        'c2': rows[rows.cohort.eq('evaluation')],
        'prvsiyan_895': held_rows,
        'prvsiyan_316': held_rows[held_rows.strict_topic_holdout],
    }
    if exposed:
        for name in ('prvsiyan_895', 'prvsiyan_316'):
            original_panel = panels[name]
            panels[name + '_no_runtime_match'] = original_panel.loc[~original_panel.runtime_corpus_overlap]
    arms = ARMS if style else PROVENANCE
    summary = {name: {length: panel_statistics(group, draws, arms) for length, group in
                      [('all', panel), ('short', panel[panel.n_words.lt(100)])] if len(group)}
               for name, panel in panels.items() if len(panel)}
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'paired_predictions.parquet')
    report = dict(fixed_weight=WEIGHT, zero_weight_matches_all_rows=True, runtime_exposure=overlap,
                  claude_only_scope='Original and proposed calls must both be Claude; honor original GLM and typography holds',
                  panels=summary)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Reader transfer compared identical base and frozen query IDs', flush=True)
    return report
