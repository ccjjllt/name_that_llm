"""Replay the full-text GSQ 27B branch at fixed shares on frozen benchmark traces."""

import json

import numpy as np
import pandas as pd

from .member_comp_frontend import GSQ_WEIGHT, replay_gsq_weights
from .member_comp_replay import replay_downstream
from .member_evidence_transfer import read_capture, read_prvsiyan
from .peer_context_report import balanced_accuracy, paired_stats, read_ids


SHARES = (0., GSQ_WEIGHT, .08, .15)


def key(share):
    return f'gsq_{round(share * 100):02}'


def process(folder, truth, counts, refiner, config):
    trace, baseline, members, masks, kinds = read_capture(folder, truth)
    candidates, frontend = replay_gsq_weights(trace, members, masks, kinds, counts, SHARES)
    if not frontend['original']['gsq_included']:
        raise ValueError('This cohort has no observed original full-text GSQ contribution')
    labels = np.asarray(trace['labels'])
    baseline_top = np.argsort(-trace['stack_logp'], axis=1, kind='stable')[:, :3]
    gsq_top = np.argsort(-members['infer__gsq'], axis=1, kind='stable')[:, :3]
    target = truth.model.to_numpy()[:, None]
    frame = truth[['model', 'prompt_id', 'n_words', 'cohort', 'strict_topic_holdout']].copy()
    frame = frame.assign(baseline=baseline, stack=trace['stack'], gsq_top1=labels[gsq_top[:, 0]],
                         stack_top3_has_truth=(labels[baseline_top] == target).any(1),
                         gsq_top3_has_truth=(labels[gsq_top] == target).any(1))
    receipt = json.loads((folder / 'glm_report.json').read_text())
    routes = {}
    for share, stack in candidates.items():
        candidate, routed, audit = replay_downstream(trace, truth.text, baseline, stack, refiner, receipt, config)
        name = key(share)
        frame[name] = candidate
        frame[f'{name}_stack'] = labels[stack.argmax(1)]
        frame['native_scored'] = routed
        routes[name] = audit
    if not frame[key(GSQ_WEIGHT)].eq(frame.baseline).all():
        raise ValueError('The original GSQ share did not reproduce every final prediction')
    return frame, {'front_end': frontend, 'native_route': routes, 'exact_final_parity': len(truth)}


def summarize(part, chosen, draws):
    comparisons = {f'{key(share)}_vs_original': paired_stats(part, 'baseline', key(share), draws)
                   for share in SHARES if share != GSQ_WEIGHT}
    comparisons['original_vs_no_gsq'] = paired_stats(part, key(0.), 'baseline', draws)
    comparisons['chosen_vs_original'] = paired_stats(part, 'baseline', key(chosen), draws)
    misses = part.baseline.ne(part.model)
    return dict(rows=len(part), native_scored=int(part.native_scored.sum()), comparisons=comparisons,
                true_top3_recovered_outside_stack=int(
                    (misses & ~part.stack_top3_has_truth & part.gsq_top3_has_truth).sum()))


def run(historical, original, official, config, output, independent=None, source=None, prior=None, draws=3000):
    from pngy_residual import Residual

    archived = read_ids(original / 'public_truth.parquet').assign(strict_topic_holdout=False)
    if len(archived) != 798 or int(archived.cohort.eq('evaluation').sum()) != 300:
        raise ValueError('Expected the fixed 798-row historical benchmark')
    train = pd.read_csv(official, usecols=['text', 'model'])
    labels = sorted(train.model.unique())
    counts = train.model.value_counts().reindex(labels, fill_value=0).to_numpy()
    refiner = Residual(np.asarray(labels)).fit(train.text.fillna('').astype(str), train.model)
    history, first = process(historical, archived, counts, refiner, config)
    dev = history.loc[history.cohort.eq('development')]
    rule = {share: balanced_accuracy(dev.model, dev[key(share)]) for share in SHARES}
    chosen = max(SHARES, key=lambda share: (rule[share], -abs(share - GSQ_WEIGHT), -share))
    rows = history
    traces = dict(historical=first)
    scopes = {'c2': history.loc[history.cohort.eq('evaluation')],
              'stress': history.loc[history.cohort.eq('reconstructed_stress')]}
    if independent:
        held = read_prvsiyan(source, prior)
        readout, traces['prvsiyan'] = process(independent, held, counts, refiner, config)
        rows = pd.concat([history, readout], verify_integrity=True)
        scopes.update(prvsiyan_895=readout, prvsiyan_316=readout.loc[readout.strict_topic_holdout])
    summaries = {}
    for name, scope in scopes.items():
        for length, part in [('all', scope), ('short', scope.loc[scope.n_words.lt(100)])]:
            if len(part):
                summaries[f'{name}_{length}'] = summarize(part, chosen, draws)
    report = dict(shares=SHARES, original_share=GSQ_WEIGHT, c1_share=chosen, c1_ba=rule, traces=traces,
                  panels=summaries, original_glm_query_set_fixed=True,
                  method='Rebuild Native mixture and prior adjustment; repeat original sibling and final decision.')
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Wrote full-text GSQ contribution and share replay for', len(rows), 'passages', flush=True)
    return report
