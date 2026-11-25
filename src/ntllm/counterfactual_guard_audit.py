"""Compare protection rules on identical candidate calls and acceptance cutoffs."""

import json

import numpy as np
import pandas as pd

from .counterfactual_native_guard import guard_frame
from .counterfactual_selection import build_actions
from .counterfactual_selection_audit import calibration_ids, canonical_prompt, fit_arm
from .counterfactual_selection_audit import make_blocks
from .peer_context_report import paired_stats, read_ids, same_ids


POLICIES = ('open', 'typography_fixed', 'fable_typography_fixed', 'native_guard', 'strict_fixed')
FOCAL = ('glm-5.3', 'glm-5.3-flash', 'claude-fable-5')


def evaluate(rows, draws):
    result = {after: paired_stats(rows, 'baseline', after, draws) for after in POLICIES}
    incremental = {after: paired_stats(rows, 'open', after, draws) for after in POLICIES[1:]}
    counts = {}
    changed = rows.open.ne(rows.baseline)
    for label, part in rows.groupby('model'):
        if label not in FOCAL and not part.open.ne(part.baseline).any():
            continue
        original = part.baseline.eq(label)
        counts[label] = {'rows': len(part), 'baseline_correct': int(original.sum())}
        for after in POLICIES:
            proposed = part[after].eq(label)
            counts[label][after] = dict(fixed=int((proposed & ~original).sum()),
                                         harmed=int((~proposed & original).sum()))
    attribution = {}
    for name in ('route', 'likelihood_support_incumbent', 'explicit_fable', 'incumbent_glm', 'typography_block'):
        subset = rows.loc[changed & rows[name]].copy()
        attribution[name] = dict(calls=len(subset), fixed=int(subset.open.eq(subset.model).sum()),
                                 harmed=int(subset.baseline.eq(subset.model).sum()))
    return dict(paired=result, compared_to_open=incremental, attributed_open_calls=attribution, by_true_class=counts)


def run(historical, prior, gen003, kimi, inputs, output, reference, independent=None, source=None, draws=2000):
    blocks, held_truth = make_blocks(historical, prior, gen003, kimi, inputs, independent, source)
    dirs = [historical, gen003, kimi, *([independent] if independent else [])]
    frames, stages, member_actions = [], [], []
    for truth, trace, baseline, members, masks in blocks:
        cols = ['cohort', 'model', 'prompt_id', 'n_words', 'text']
        frames.append(truth[cols].assign(baseline=baseline, strict=truth.get('strict_topic_holdout', False)))
        stages.append(build_actions(truth, trace, baseline))
        if members:
            member_actions.append(build_actions(truth, trace, baseline, members, masks))
    rows = pd.concat(frames, verify_integrity=True)
    dev, extra, separation = calibration_ids(rows, held_truth)
    c2 = rows.index[rows.cohort.eq('evaluation')]
    stress = rows.index[rows.cohort.eq('reconstructed_stress')]
    held = rows.index[rows.cohort.eq('prvsiyan')]
    strict = held[rows.loc[held, 'strict']]
    target = c2.append([stress, held])
    expected = read_ids(reference)
    same_ids(expected.index, c2, 'Pinned C2 candidate-selection outputs', ordered=True)

    def protected(ids, fitted):
        frames = []
        for block, folder in zip(blocks, dirs, strict=True):
            truth, trace, baseline, _, _ = block
            contained = ids.intersection(truth.index, sort=False)
            if not contained.empty:
                frames.append(guard_frame(truth, trace, baseline, fitted.loc[contained], folder))
        return rows.loc[ids].drop(columns=['text', 'baseline']).join(pd.concat(frames, verify_integrity=True))

    stage = pd.concat(stages)
    member = pd.concat(member_actions)
    groups = {'c2': c2, 'stress': stress, 'prvsiyan_895': held, 'prvsiyan_316': strict}
    arms = {'stage_c1': (stage, dev), 'stage_multisource': (stage, dev.append(extra)),
            'member_c1': (member, dev)}
    report = dict(separation=separation, comparisons={}, leave_source={})
    predictions = []
    for name, (actions, fit_ids) in arms.items():
        fitted, audit = fit_arm(actions, rows, fit_ids, target)
        if not np.array_equal(fitted.loc[c2, 'open'], expected[f'{name}_open']):
            raise ValueError(f'{name} open calls disagree with the frozen C2 calibration')
        pair = protected(target, fitted)
        scopes = {}
        for scope, ids in groups.items():
            for length, part in [('full', pair.loc[ids]), ('short', pair.loc[ids][lambda x: x.n_words.lt(100)])]:
                if not part.empty:
                    scopes[f'{scope}_{length}'] = evaluate(part, draws)
        report['comparisons'][name] = dict(threshold=audit['open']['threshold'], scopes=scopes)
        predictions.append(pair.assign(arm=name))
    all_fit = dev.append(extra)
    prompt = canonical_prompt(rows.prompt_id)
    for name, holdout in rows.loc[all_fit].groupby('cohort'):
        fit = all_fit.difference(holdout.index)
        fit = fit[~prompt.loc[fit].isin(prompt.loc[holdout.index])]
        fitted, audit = fit_arm(stage, rows, fit, holdout.index)
        evaluated = protected(holdout.index, fitted)
        report['leave_source'][name] = dict(threshold=audit['open']['threshold'], **evaluate(evaluated, draws))
        predictions.append(evaluated.assign(arm='leave_source_' + name))
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions).to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
