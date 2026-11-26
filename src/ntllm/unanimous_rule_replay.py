"""Replay the exact final rules and measure narrow consensus overrides."""

import json

import numpy as np
import pandas as pd

from .member_evidence import STAGES
from .member_evidence_transfer import EXPOSURE_KEYS, read_capture, read_prvsiyan
from .peer_context_report import paired_stats, read_ids
from .sibling_odds_replay import replay_predictions


RULES = ('spaced_dash', 'curly_apostrophe', 'em_dash')
NATIVE = ('glm-5.3', 'glm-5.3-flash')
ALLOWED_SPACED = ('claude-opus-4-6', 'claude-opus-5', *NATIVE)


def typography_steps(trace, text, initial, skip):
    labels = np.asarray(trace['labels'])
    scores = np.asarray(trace['sibling_logp'], dtype=np.float64)
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    conditions = (
        (' — ', ~np.isin(labels, ALLOWED_SPACED)),
        ('’', np.char.startswith(labels, 'claude-') | np.char.startswith(labels, 'glm-')),
        ('—', np.char.startswith(labels, 'gemini-')),
    )
    prediction = initial.copy()
    steps = {}
    text = text.fillna('').astype(str)
    for name, (marker, rejected) in zip(RULES, conditions, strict=True):
        rows = text.str.contains(marker, regex=False).to_numpy(dtype=bool)
        rows &= np.isin(prediction, labels[rejected]) & np.logical_not(skip.get(name, False))
        probabilities[np.ix_(rows, rejected)] = 0.
        total = probabilities.sum(axis=1)
        active = rows & (total > 0.)
        probabilities[active] /= total[active, None]
        prediction[active] = labels[probabilities[active].argmax(axis=1)]
        steps[name] = prediction.copy()
    return steps


def replay(truth, trace, incumbent):
    labels = trace['labels']
    raw = {stage: labels[trace[stage].argmax(axis=1)] for stage in STAGES}
    stack, glm, siblings = (raw[stage] for stage in STAGES)
    same = (stack == glm) & (glm == siblings)
    glm_mask = trace['glm_winner']
    fable_mask = trace['fable_guard']
    if (glm_mask & fable_mask).any():
        raise ValueError('Expected distinct GLM and Fable protection outcomes')
    protected = glm_mask | fable_mask
    empty = np.zeros(len(truth), dtype=bool)
    bare = dict(trace, glm_winner=empty, fable_guard=empty)
    odds = replay_predictions(bare, truth.text, typography=False)
    glm_restore = np.where(glm_mask, trace['glm'], odds)
    prior = np.where(fable_mask, trace['glm'], glm_restore)
    if not np.array_equal(prior, replay_predictions(trace, truth.text, typography=False)):
        raise ValueError('The original GLM and Fable restores did not replay')

    original_rules = typography_steps(trace, truth.text, prior, {})
    if not np.array_equal(original_rules[RULES[-1]], incumbent):
        raise ValueError('The three original typography rules did not reproduce every frozen decision')

    unanimous = same & (stack == prior)
    confirmed = unanimous & ~protected
    columns = dict(baseline=incumbent, strict=truth.get('strict_topic_holdout', False),
                   consensus=same, consensus_label=stack, consensus_effective=unanimous,
                   consensus_non_native=confirmed, protected_native=protected,
                   stage_stack=stack, stage_glm=glm, stage_siblings=siblings,
                   after_odds=odds, after_glm_restore=glm_restore, after_fable_restore=prior,
                   **original_rules)
    variants = {
        'global_no_typography': prior,
        'no_typography_non_native': np.where(protected, incumbent, prior),
        'unanimous_no_typography': np.where(same, prior, incumbent),
        'confirmed_unanimous_no_typography': np.where(confirmed, prior, incumbent),
    }
    for name in RULES:
        variants[f'global_drop_{name}'] = typography_steps(trace, truth.text, prior, {name: ~empty})[RULES[-1]]
        variants[f'unanimous_drop_{name}'] = typography_steps(trace, truth.text, prior, {name: confirmed})[RULES[-1]]
    if any(np.any(call[protected] != incumbent[protected]) for name, call in variants.items()
           if name.startswith('unanimous_drop_') or name == 'confirmed_unanimous_no_typography'):
        raise ValueError('A narrow rule changed an originally protected GLM or Fable result')
    return truth[['cohort', 'model', 'prompt_id', 'n_words']].assign(**columns, **variants)


def summarize(rows, draws):
    primary = ('global_no_typography', 'no_typography_non_native', 'unanimous_no_typography',
               'confirmed_unanimous_no_typography', *(f'global_drop_{rule}' for rule in RULES),
               *(f'unanimous_drop_{rule}' for rule in RULES))
    chain = ('stage_siblings', 'after_odds', 'after_glm_restore', 'after_fable_restore', *RULES)
    steps = [paired_stats(rows, before, after, draws) for before, after in zip(chain, chain[1:])]
    true = rows.model
    original_correct = rows.baseline.eq(true)
    unanimous_wrong_final = rows.consensus & rows.consensus_label.eq(true) & ~original_correct
    cause = pd.Series('', index=rows.index)
    for before, after in zip(chain, chain[1:]):
        cause.loc[rows[before].eq(true) & rows[after].ne(true)] = after
    if cause.loc[unanimous_wrong_final].eq('').any():
        raise ValueError('An unanimous correct vote was lost without a recorded rule transition')
    changed_typography = rows.after_fable_restore.ne(rows.baseline)
    eligibility = rows.consensus_non_native & changed_typography
    allowed_labels = rows.loc[eligibility].groupby('consensus_label')
    return dict(rows=len(rows), consensus_rows=int(rows.consensus.sum()),
                changed_by_typography=int(changed_typography.sum()), confirmed_consensus_eligible=int(eligibility.sum()),
                unanimous_true_but_wrong_final=int(unanimous_wrong_final.sum()),
                unanimous_true_lost_by_last_harmful_step=cause.loc[unanimous_wrong_final].value_counts().to_dict(),
                unanimous_true_lost_by_step_and_true_model=[dict(step=step, true_model=model, rows=int(count))
                    for (step, model), count in rows.loc[unanimous_wrong_final].assign(lost_by=cause).groupby(
                        ['lost_by', 'model']).size().items()],
                sequential_steps=steps,
                ablations=[paired_stats(rows, 'baseline', name, draws) for name in primary],
                consensus_by_reinstated_label={label: dict(rows=len(group),
                    fixed=int(group.after_fable_restore.eq(group.model).sum()), harmed=int(group.baseline.eq(group.model).sum()),
                    neutral=int((group.after_fable_restore.ne(group.model) & group.baseline.ne(group.model)).sum()))
                    for label, group in allowed_labels})


def run(historical, original, independent, source, previous, output, draws=3500):
    public = read_ids(original / 'public_truth.parquet')
    fresh = read_prvsiyan(source, previous)
    hist = read_capture(historical, public)
    new = read_capture(independent, fresh)
    if not np.array_equal(hist[0]['labels'], new[0]['labels']):
        raise ValueError('The two frozen traces use different label orders')
    old = replay(public, hist[0], hist[1])
    held = replay(fresh, new[0], new[1])
    overlap = json.loads((independent / 'body_overlap.json').read_text())
    exposure = {item for key in EXPOSURE_KEYS for item in overlap.get(key, [])}
    if not exposure.issubset(held.index):
        raise ValueError('The runtime overlap audit references unknown query IDs')
    panels = {'c2': old.loc[old.cohort.eq('evaluation')], 'prvsiyan_895': held, 'prvsiyan_316': held.loc[held.strict]}
    if list(map(len, panels.values())) != [300, 895, 316]:
        raise ValueError('Unexpected frozen panel sizes')
    if exposure:
        panels.update({name + '_no_runtime_overlap': part.loc[~part.index.isin(exposure)]
                       for name, part in list(panels.items()) if name.startswith('prvsiyan')})
    report = dict(original_identity_rows=len(old) + len(held), sequential_typography=list(RULES), runtime_overlap=overlap,
                  interpretation='Intervention proposed after viewing Prvsiyan error categories. Requires new-source validation.',
                  panels={name: {size: summarize(part, draws) for size, part in
                                 [('all', frame), ('short', frame.loc[frame.n_words.lt(100)])] if len(part)}
                          for name, frame in panels.items() if len(frame)})
    output.mkdir(parents=True, exist_ok=True)
    pd.concat([panels['c2'], held], verify_integrity=True).to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Verified exact sequential decision replay on', report['original_identity_rows'], 'passages', flush=True)
    return report
