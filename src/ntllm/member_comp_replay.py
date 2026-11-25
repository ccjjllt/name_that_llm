"""Assess the repaired Comp branch with the archived GLM evidence held fixed."""

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import log_softmax

from .glm_prefix_response_replay import GLM, restored_route, with_glm_scores
from .peer_context_report import paired_stats, read_ids, same_ids
from .sibling_odds_replay import replay_predictions


COMP_WEIGHT = .15


def match_temperature(scores, reference):
    def entropy(inverse):
        logits = log_softmax(scores * inverse, axis=1)
        return float(-(np.exp(logits) * logits).sum(1).mean())

    lo, hi = .001, 1000.
    if not entropy(hi) <= reference <= entropy(lo):
        raise ValueError('Archived GLM temperature target is unreachable')
    for _ in range(100):
        middle = np.sqrt(lo * hi)
        if entropy(middle) > reference:
            lo = middle
        else:
            hi = middle
    return float(np.sqrt(lo * hi))


def replay_downstream(trace, texts, baseline, replacement, refiner, receipt, config):
    old = trace['stack_logp']
    expected_style = refiner.predict_log_probability(old, texts)
    style_discrepancy = float(np.abs(expected_style - trace['sibling_logp']).max())
    if not np.allclose(expected_style, trace['sibling_logp'], rtol=0., atol=1e-5):
        raise ValueError(f'Original sibling residual could not be reproduced ({style_discrepancy})')
    if not np.array_equal(replay_predictions(dict(trace, sibling_logp=expected_style), texts), baseline):
        raise ValueError('Recomputed original sibling refinement changed the baseline')
    routed, old_weight = restored_route(trace, texts, baseline, receipt)
    new_pre = log_softmax(replacement, axis=1)
    new_post = new_pre.copy()
    temperature = None
    if routed.any():
        if not np.isclose(config['W'] / receipt['temperature'], old_weight, rtol=0., atol=1e-9):
            raise ValueError('The original GLM coefficient differs from the published settings')
        old_pre = log_softmax(old, axis=1)
        temperature = match_temperature(new_pre, config['entropy_ref'])
        new_weight = config['W'] / temperature
        old_evidence = trace['glm_logp'][routed] - old_pre[routed]
        new_post[routed] += old_evidence * (new_weight / old_weight)
    refined = refiner.predict_log_probability(replacement, texts)
    altered = with_glm_scores(dict(trace, stack_logp=replacement, sibling_logp=refined), new_post)
    final = replay_predictions(altered, texts)
    columns = pd.Index(trace['labels']).get_indexer(GLM)
    mass = log_softmax(replacement, axis=1)[:, columns]
    revised_order = np.argsort(-np.exp(mass).sum(1), kind='stable')
    size = int(routed.sum())
    hypothetical = np.zeros(len(texts), dtype=bool)
    hypothetical[revised_order[:size]] = True
    route_audit = dict(original_rows=size, changed_priority_rows=int((routed != hypothetical).sum()),
                       newly_eligible=int((hypothetical & ~routed).sum()), original_style_max_delta=style_discrepancy,
                       reevaluated_temperature=temperature)
    return final, routed, route_audit


def read_comp(folder, historical, held, labels):
    summary = json.loads((folder / 'report.json').read_text())
    all_rows = pd.concat([historical, held], verify_integrity=True)
    if (not summary['complete'] or summary['rows'] != len(all_rows)
            or summary['ordered_ids_sha256'] != hashlib.sha256(('\n'.join(all_rows.index) + '\n').encode()).hexdigest()
            or summary['input_body_sha256'] != [hashlib.sha256(body.encode()).hexdigest() for body in all_rows.text]):
        raise ValueError('Comp must complete on the exact two benchmark bodies in their archived order')
    with np.load(folder / 'scores.npz', allow_pickle=False) as zip_:
        ids, classes, scores = zip_['ids'], pd.Index(zip_['labels']), zip_['raw_logits']
    same_ids(pd.Index(ids), all_rows.index, 'Comp and archived inputs', ordered=True)
    if set(classes) != set(labels) or scores.shape != (len(all_rows), len(labels)) or not np.isfinite(scores).all():
        raise ValueError('The Comp log probabilities must contain all original classes')
    return pd.DataFrame(scores[:, classes.get_indexer(labels)], index=all_rows.index, columns=labels), summary


def summarize(rows, draws):
    subsets = {'c2': rows[rows.cohort.eq('evaluation')],
               'prvsiyan_895': rows[rows.cohort.eq('prvsiyan')],
               'prvsiyan_316': rows[rows.strict_topic_holdout]}
    report = []
    for name, group in subsets.items():
        for length, part in [('all', group), ('short', group[group.n_words.lt(100)])]:
            if not len(part):
                continue
            report.append(dict(panel=name, length=length,
                               comp_top3_on_baseline_misses=int((part.baseline.ne(part.model) & part.comp_top3).sum()),
                               comp_top3_outside_old_top3=int((part.baseline.ne(part.model)
                                                              & ~part.old_top3 & part.comp_top3).sum()),
                               front_end=paired_stats(part, 'old_stack', 'new_stack', draws),
                               fixed_glm_route=paired_stats(part, 'baseline', 'repaired_comp', draws)))
    return report


def run(historical_folder, original_folder, held_folder, audit_folder, prior, comp_folder, train_path, config, output,
        draws=4000):
    from pngy_residual import Residual
    from .member_comp_frontend import replay_frontend
    from .member_evidence_transfer import read_capture, read_prvsiyan

    historical = read_ids(original_folder / 'public_truth.parquet')
    held = read_prvsiyan(audit_folder, prior)
    groups = [(historical_folder, historical.assign(strict_topic_holdout=False)), (held_folder, held)]
    captured = [read_capture(folder, group) for folder, group in groups]
    labels = np.asarray(captured[0][0]['labels'])
    if not np.array_equal(labels, captured[1][0]['labels']):
        raise ValueError('The trace class orders differ')
    comp, repair_receipt = read_comp(comp_folder, historical, held, labels)
    train = pd.read_csv(train_path, usecols=['text', 'model'])
    counts = train.model.value_counts().reindex(labels, fill_value=0).to_numpy()
    refiner = Residual(labels).fit(train.text.fillna('').astype(str), train.model)
    frames, checks = [], []
    for (folder, truth), (trace, baseline, members, masks, kinds) in zip(groups, captured, strict=True):
        scores = comp.loc[truth.index].to_numpy()
        _, replacement, frontend_audit = replay_frontend(trace, members, masks, kinds, counts, scores)
        if frontend_audit['comp_weight'] != COMP_WEIGHT:
            raise ValueError('The restored Comp did not use the original 0.15 share')
        receipt = json.loads((folder / 'glm_report.json').read_text())
        final, routed, audit = replay_downstream(trace, truth.text, baseline, replacement, refiner, receipt, config)
        old = trace['stack_logp']
        old_rank = labels[np.argsort(-old, axis=1, kind='stable')[:, :3]]
        comp_rank = labels[np.argsort(-scores, axis=1, kind='stable')[:, :3]]
        target = truth.model.to_numpy()
        result = truth[['model', 'prompt_id', 'n_words', 'cohort', 'strict_topic_holdout']].copy()
        frames.append(result.assign(baseline=baseline, repaired_comp=final, routed=routed,
                                    old_stack=labels[old.argmax(1)], new_stack=labels[replacement.argmax(1)],
                                    old_top3=(old_rank == target[:, None]).any(1),
                                    comp_top3=(comp_rank == target[:, None]).any(1)))
        checks.append(dict(rows=len(truth), frontend=frontend_audit, glm=audit))
    rows = pd.concat(frames, verify_integrity=True)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'paired_predictions.parquet')
    report = dict(weight=COMP_WEIGHT, diagnostic_comp_seconds=repair_receipt['seconds'],
                  original_route_held_fixed=True, optional_stages_share_archived_runtime=True,
                  audits=checks, scopes=summarize(rows, draws))
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
