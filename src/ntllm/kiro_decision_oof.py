"""Factor the Kiro and Sibling Odds post-processing rules with frozen upstream evidence."""

import hashlib
import json

import numpy as np
import pandas as pd

from .glm_prefix_response_replay import GLM, restored_route
from .peer_context_report import paired_stats, read_ids, read_trace, same_ids
from .sibling_odds_replay import replay_predictions


FOCUS = (*GLM, 'claude-fable-5')
ARMS = {'sibling_odds': (True, False), 'without_sibling_odds': (False, False),
        'all_glm_returns': (True, True), 'kiro_decisions': (False, True)}
PAIRS = (('sibling_odds', 'without_sibling_odds'), ('sibling_odds', 'all_glm_returns'),
         ('sibling_odds', 'kiro_decisions'), ('without_sibling_odds', 'kiro_decisions'),
         ('all_glm_returns', 'kiro_decisions'))


def captured_truth(query_path, truth_path, trace_folder, cohort):
    query = read_ids(query_path)
    truth = read_ids(truth_path)
    same_ids(query.index, truth.index, 'Frozen external IDs', ordered=True)
    if list(query.columns) != ['text'] or not query.text.str.split().str.len().eq(truth.n_words).all():
        raise ValueError('The external passages do not match their frozen metadata')
    frozen = json.loads((trace_folder / 'probe_manifest.json').read_text())
    if (frozen['ordered_ids_sha256'] != hashlib.sha256(('\n'.join(query.index) + '\n').encode()).hexdigest()
            or frozen['input_body_sha256'] != [hashlib.sha256(body.encode()).hexdigest() for body in query.text]):
        raise ValueError('The remote capture used different external passages')
    return truth.join(query).assign(cohort=cohort)


def decisions(folder, truth):
    trace, baseline = read_trace(folder, truth)
    report = json.loads((folder / 'glm_report.json').read_text())
    routed, _ = restored_route(trace, truth.text, baseline, report)
    labels = np.asarray(trace['labels'])
    glm = pd.Index(labels).get_indexer(GLM)
    before, after = [trace[key].argmax(1) for key in ('stack_logp', 'glm_logp')]
    broad = np.isin(before, glm) & ~np.isin(after, glm) if report['status'] == 'applied' else np.zeros(len(truth), bool)
    if (trace['fable_guard'] & ~broad).any() or (broad & ~routed).any():
        raise ValueError('The published protection must be a subset of actual GLM-to-non-GLM decisions')
    columns = [key for key in ('model', 'prompt_id', 'n_words', 'cohort', 'source_dataset', 'strict_topic_holdout')
               if key in truth]
    rows = truth[columns].assign(routed=routed, broad_protection=broad, fable_protection=trace['fable_guard'],
                                original_glm_winner=labels[after])
    for name, (use_sibling_odds, widen) in ARMS.items():
        scores = dict(trace, fable_guard=broad) if widen else trace
        rows[name] = replay_predictions(scores, truth.text, sibling_odds=use_sibling_odds)
        rows[f'{name}_unfiltered'] = replay_predictions(scores, truth.text, sibling_odds=use_sibling_odds,
                                                         typography=False)
    if not np.array_equal(rows.sibling_odds, baseline):
        raise ValueError('Original policy does not reproduce the frozen Sibling Odds predictions')
    return rows


def summarize(name, rows, draws):
    support = rows.model.value_counts()
    new_protection = rows.broad_protection & ~rows.fable_protection
    return dict(panel=name, rows=len(rows), routed=int(rows.routed.sum()), new_protected=int(new_protection.sum()),
                new_protected_by_destination=rows.loc[new_protection, 'original_glm_winner'].value_counts().to_dict(),
                target_truth={key: int(support.get(key, 0)) for key in FOCUS}, labels=support.to_dict(),
                per_model={label: {arm: int(part[arm].eq(label).sum()) for arm in ARMS}
                           for label, part in rows.groupby('model')},
                altered_by_typography={arm: int(rows[arm].ne(rows[f'{arm}_unfiltered']).sum()) for arm in ARMS},
                paired=[paired_stats(rows, before, after, draws) for before, after in PAIRS])


def run(historical_folder, kimi_folder, kimi_data, output, independent=None, draws=4000):
    public = read_ids(historical_folder / 'public_truth.parquet')
    kimi = captured_truth(kimi_data / 'queries.parquet', kimi_data / 'truth.parquet', kimi_folder, 'kimi_flash')
    if len(public) != 798 or len(kimi) != 191:
        raise ValueError('Expected the exact two original comparison cohorts')
    frames = [decisions(historical_folder, public), decisions(kimi_folder, kimi)]
    if independent is not None:
        from .member_evidence_transfer import read_prvsiyan

        audit_folder, trace_folder = independent
        held = read_prvsiyan(audit_folder, kimi_data / 'truth.parquet')
        frozen = json.loads((trace_folder / 'probe_manifest.json').read_text())
        if (frozen['ordered_ids_sha256'] != hashlib.sha256(('\n'.join(held.index) + '\n').encode()).hexdigest()
                or frozen['input_body_sha256'] != [hashlib.sha256(value.encode()).hexdigest() for value in held.text]):
            raise ValueError('The Prvsiyan trace used different input bodies')
        frames.append(decisions(trace_folder, held))
    all_rows = pd.concat(frames, verify_integrity=True)
    groups = dict(tuple(all_rows.groupby('cohort')))
    if 'prvsiyan' in groups:
        groups['prvsiyan_strict'] = groups['prvsiyan'].loc[groups['prvsiyan'].strict_topic_holdout.eq(True)]
    comparisons = []
    for cohort, group in groups.items():
        for subset, panel in [('all', group), ('short', group[group.n_words.lt(100)])]:
            if len(panel):
                comparisons.append(dict(subset=subset, **summarize(cohort, panel, draws)))
    output.mkdir(parents=True, exist_ok=True)
    all_rows.to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(dict(paired=comparisons, total_rows=len(all_rows)),
                                                    indent=2, allow_nan=False) + '\n')
