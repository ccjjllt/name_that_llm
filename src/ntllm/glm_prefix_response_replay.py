"""Measure matched prefix evidence on the original GLM route and decision rules."""

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import log_softmax

from .peer_context_report import paired_stats, read_ids, read_trace, same_ids
from .sibling_odds_replay import replay_predictions


GLM = ('glm-5.3', 'glm-5.3-flash')
FABLE = 'claude-fable-5'
ARMS = {
    'equal_body': ('task128_neutral32', 'task128_window'),
    'equal_total': ('total_matched_neutral', 'total_matched_window'),
}


def with_glm_scores(trace, updated):
    labels = np.asarray(trace['labels'])
    fable = pd.Index(labels).get_loc(FABLE)
    glm = pd.Index(labels).get_indexer(GLM)
    base = trace['stack_logp']
    base_winner, updated_winner = base.argmax(1), updated.argmax(1)
    before = base[:, fable] - base[:, glm].max(1)
    after = updated[:, fable] - updated[:, glm].max(1)
    protection = (np.isin(base_winner, glm) & (updated_winner == fable)
                  & (after - before >= .30) & (after >= .25))
    return dict(trace, glm_logp=updated, glm=labels[updated_winner],
                glm_winner=np.isin(updated_winner, glm), fable_guard=protection)


def restored_route(trace, texts, reference, receipt):
    post = trace['glm_logp']
    base = trace['stack_logp']
    if receipt.get('status') != 'applied':
        if not np.array_equal(post, base):
            raise ValueError('GLM fallback disagrees with the saved pre-GLM scores')
        return np.zeros(len(post), dtype=bool), 0.
    reproduced = with_glm_scores(trace, post)
    for key in ('glm', 'glm_winner', 'fable_guard'):
        if not np.array_equal(reproduced[key], trace[key]):
            raise ValueError(f'The original decision rule no longer agrees on {key}')
    if not np.array_equal(replay_predictions(reproduced, texts), reference):
        raise ValueError('Zero change must produce the original final predictions')
    relative = post - base
    relative -= relative[:, :1]
    selected = np.abs(relative).max(1) > 1e-8
    if selected.sum() != receipt['n_blended']:
        raise ValueError('Recovered GLM route does not match the actual completed row count')
    return selected, float(receipt['weight'])


def run(trace_path, source_path, panel_path, analysis_path, output, draws=4000):
    ids = read_ids(source_path / 'candidate_truth.parquet')
    ids = ids[ids.source_dataset.eq('prvsiyan/prvsiyans-extended-name-that-llm-dataset/13')]
    truth = ids.join(read_ids(source_path / 'candidate_queries.parquet')[['text']])
    trace, baseline = read_trace(trace_path, truth)
    strict = read_ids(panel_path / 'truth.parquet').query("cohort == 'heldout'")
    parts = read_ids(panel_path / 'queries.parquet')
    if len(truth) != 895 or len(strict) != 316:
        raise ValueError('Expected original trajectory and complete fixed evaluation subset')
    same_ids(strict.index, truth.index[truth.index.isin(strict.index)], 'Independent row subset', ordered=True)
    if not strict.model.equals(truth.loc[strict.index, 'model']) or not strict.prompt_id.equals(
            truth.loc[strict.index, 'prompt_id']) or not parts.loc[strict.index, 'text'].equals(
            truth.loc[strict.index, 'text']):
        raise ValueError('Prefix panels disagree on label, prompt or body')
    frozen = json.loads((trace_path / 'probe_manifest.json').read_text())
    if (frozen['ordered_ids_sha256'] != hashlib.sha256(('\n'.join(truth.index) + '\n').encode()).hexdigest()
            or frozen['input_body_sha256'] != [hashlib.sha256(x.encode()).hexdigest() for x in truth.text]):
        raise ValueError('Archived trajectory does not match the original query bodies')
    saved = read_ids(trace_path / 'frozen_predictions.parquet')
    same_ids(saved.index, truth.index, 'Archived predictions', ordered=True)
    if not np.array_equal(saved.sibling_odds, baseline):
        raise ValueError('The original replay diverges from its saved predictions')

    settings = json.loads((trace_path / 'glm_report.json').read_text())
    routed, weight = restored_route(trace, truth.text, baseline, settings)
    position = truth.index.get_indexer(strict.index)
    on_strict = routed[position]
    with np.load(analysis_path / 'head_logp.npz', allow_pickle=False) as data:
        signals = dict(data)
    head_ids = pd.Index(signals['ids'], name='id')
    head_labels = pd.Index(signals['labels'])
    same_ids(head_ids, parts.index, 'Prefix model predictions', ordered=True)
    if not head_labels.is_unique or set(head_labels) != set(trace['labels']):
        raise ValueError('Prefix score label order cannot be aligned')
    col = head_labels.get_indexer(trace['labels'])
    index = head_ids.get_indexer(strict.index)
    eligible_total = signals['total_budget_eligible'][index]
    full_member = signals['legacy_task160'][index][:, col]
    results = strict[['model', 'prompt_id', 'n_words']].assign(baseline=np.asarray(baseline)[position],
                                                                routed=on_strict, eligible_total=eligible_total)
    for name, (dual, control) in ARMS.items():
        eligible = on_strict if name == 'equal_body' else on_strict & eligible_total
        measured = (signals[dual] - signals[control])[index][:, col]
        truncated = signals[f'legacy_{control}'][index][:, col]
        delta = measured + (truncated - full_member)
        if not np.isfinite(delta).all():
            raise ValueError('Prefix response log odds are not finite')
        if not eligible.any():
            results[name] = results.baseline
            continue
        updated = trace['glm_logp'].copy()
        updated[position[eligible]] += weight * delta[eligible]
        replay = replay_predictions(with_glm_scores(trace, updated), truth.text)
        outside = np.ones(len(truth), dtype=bool)
        outside[position[eligible]] = False
        if not np.array_equal(replay[outside], baseline[outside]):
            raise ValueError('A row outside the original route changed')
        results[name] = replay[position]

    exposure = json.loads((trace_path / 'body_overlap.json').read_text())
    known = {ident for key in ('official_train_ids', 'active_pool_ids', 'full_pool_ids', 'stack_extra_ids')
             for ident in exposure.get(key, [])}
    results['known_baseline_overlap'] = results.index.isin(known)
    comparisons = []
    scopes = [('all', results), ('short', results[results.n_words.lt(100)])]
    if results.known_baseline_overlap.any():
        scopes.append(('without_runtime_overlap', results[~results.known_baseline_overlap]))
    for name, rows in scopes:
        if not len(rows):
            continue
        focal = rows.model.isin((*GLM, FABLE))
        maximum = rows.baseline.mask(rows.routed & focal, rows.model)
        oracle = rows.assign(focal_oracle=maximum)
        comparisons.append(dict(subset=name, rows=len(rows), routed=int(rows.routed.sum()),
                                routed_equal_total=int((rows.routed & rows.eligible_total).sum()),
                                focal_route_ceiling=paired_stats(oracle, 'baseline', 'focal_oracle', draws),
                                classes={key: dict(rows=len(group), routed=int(group.routed.sum()),
                                                   **{arm: int(group[arm].eq(key).sum())
                                                      for arm in ('baseline', *ARMS)})
                                         for key, group in rows.groupby('model')},
                                paired=[paired_stats(rows, 'baseline', arm, draws) for arm in ARMS]))
    output.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output / 'paired_predictions.parquet')
    reproduced = log_softmax(trace['stack_logp'][position], axis=1) + weight * full_member
    saved_post = trace['glm_logp'][position]
    discrepancy = log_softmax(reproduced[on_strict], axis=1) - log_softmax(saved_post[on_strict], axis=1)
    parity = dict(rows=int(on_strict.sum()),
                  argmax_matches=int((reproduced[on_strict].argmax(1) == saved_post[on_strict].argmax(1)).sum()),
                  max_logp_delta=float(np.abs(discrepancy).max()) if discrepancy.size else None)
    (output / 'report.json').write_text(json.dumps(dict(incumbent_weight=weight,
        runtime_overlap=int(results.known_baseline_overlap.sum()), archived_task160_parity=parity,
        panels=comparisons), indent=2, allow_nan=False)+'\n')
