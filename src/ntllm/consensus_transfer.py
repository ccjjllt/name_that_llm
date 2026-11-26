"""Audit candidate confidence shift and fixed consensus policies across sources."""

import json

import numpy as np
import pandas as pd
from scipy.special import expit, softmax

from .counterfactual_native_guard import guard_frame
from .counterfactual_selection import STAGES, build_actions, fit_scoring_model, member_family, replay_scoring_model
from .counterfactual_selection_audit import calibration_ids, class_weights, make_blocks
from .peer_context_report import paired_stats
from .sibling_odds_replay import replay_predictions
from .support_censored_selector import prompt_folds


BINS = [-1., -.25, -.10, -.025, 0., .025, .05, .10, .20, 1.]
POLICIES = ('frozen_selector', 'consensus_veto', 'consensus_hard_override', 'consensus_hybrid')


def probabilities(actions, fit):
    if fit['features'] != [name for name in actions if name.startswith('feature_')]:
        raise ValueError('Frozen stage features and target action columns differ')
    x = actions[fit['features']].to_numpy(dtype=float)
    z = (x - np.asarray(fit['mean'])) / np.asarray(fit['scale'])
    logits = z @ np.asarray(fit['coefficient']).T + np.asarray(fit['intercept'])
    if len(fit['outcomes']) == 2:
        positive = expit(logits[:, 0])
        p = np.column_stack((1. - positive, positive))
    else:
        p = softmax(logits, axis=1)
    out = pd.DataFrame(p, index=actions.index, columns=fit['outcomes']).reindex(columns=[-1, 0, 1], fill_value=0.)
    return out.set_axis(['p_harm', 'p_neutral', 'p_fix'], axis=1)


def rank(actions, fields):
    ordered = actions.sort_values(fields, ascending=False, kind='stable')
    return ordered.loc[~ordered.index.duplicated()]


def votes(actions, truth, trace, members, masks):
    labels = pd.Index(trace['labels'])
    row = truth.index.get_indexer(actions.index)
    candidate = labels.get_indexer(actions.candidate)
    incumbent = labels.get_indexer(actions.baseline)
    preference = np.column_stack([actions[f'feature_margin_{name}'] for name in STAGES])
    group_support, group_coverage = {}, {}
    for name, scores in members.items():
        mask = masks[name]
        top2 = np.argsort(-np.where(mask, scores, -np.inf), axis=1, kind='stable')[:, :2]
        covered = mask[row, candidate] & mask[row, incumbent]
        support = covered & (scores[row, candidate] > scores[row, incumbent])
        support &= (top2[row] == candidate[:, None]).any(axis=1)
        group = member_family(name)
        group_coverage[group] = group_coverage.get(group, False) | covered
        group_support[group] = group_support.get(group, False) | support
    return actions.assign(stage_prefers=(preference > 0).sum(axis=1), mean_stage_margin=preference.mean(axis=1),
                          member_prefers=sum(group_support.values(), start=np.zeros(len(actions), dtype=int)),
                          member_groups=sum(group_coverage.values(), start=np.zeros(len(actions), dtype=int)))


def oof_probabilities(actions, rows, ids):
    pieces = []
    for train, held in prompt_folds(rows.loc[ids]):
        columns, scale, model = fit_scoring_model(actions, train, class_weights(rows, train))
        if model is None:
            raise ValueError('Historical calibration fold lacks outcomes')
        fit = dict(features=columns, mean=scale.mean_, scale=scale.scale_, coefficient=model.coef_,
                   intercept=model.intercept_, outcomes=model.classes_.tolist())
        valid = actions.loc[actions.index.isin(held)].copy()
        values = probabilities(valid, fit)
        pieces.append(valid[['candidate']].assign(**{name: values[name].to_numpy() for name in values}))
    folded = pd.concat(pieces).set_index('candidate', append=True, verify_integrity=True)
    return folded


def attach_probabilities(actions, frozen, oof):
    scores = probabilities(actions, frozen)
    base = actions[['candidate']].assign(**{name: scores[name].to_numpy() for name in scores})
    query = base.set_index('candidate', append=True, verify_integrity=True)
    replaces = query.index.intersection(oof.index)
    query.loc[replaces, scores.columns] = oof.loc[replaces, scores.columns]
    result = actions.assign(**{name: query[name].to_numpy() for name in scores})
    return result.assign(expected=result.p_fix - result.p_harm)


def pack_block(truth, trace, incumbent, members, masks, folder, row_meta, frozen, oof, shared):
    raw_stage = trace['labels'][trace['sibling_logp'].argmax(1)]
    pre_typography = replay_predictions(trace, truth.text, typography=False)
    typed = pre_typography != incumbent
    prior_restore = typed & pd.Series(pre_typography, index=truth.index).str.startswith('gemini-').to_numpy()
    prior_restore &= truth.text.str.contains('—', regex=False).to_numpy(dtype=bool)
    broad = np.where(prior_restore, pre_typography, incumbent)
    actions = build_actions(truth, trace, incumbent)
    stage = votes(actions, truth, trace, {name: members[name] for name in shared if name in members}, masks)
    fair = attach_probabilities(stage, frozen, oof)
    full = attach_probabilities(stage, frozen, oof.iloc[:0])
    stage_choice = rank(fair, ['feature_stage_votes', 'stage_prefers', 'mean_stage_margin']).loc[truth.index]
    frozen_choice = rank(full, ['expected']).loc[truth.index]
    row = row_meta.assign(hypothesis=stage_choice.candidate, stage_votes=stage_choice.feature_stage_votes,
                          stage_prefers=stage_choice.stage_prefers, member_prefers=stage_choice.member_prefers,
                          member_groups=stage_choice.member_groups, vote_p_fix=stage_choice.p_fix,
                          vote_p_harm=stage_choice.p_harm, vote_expected=stage_choice.expected,
                          selector_hypothesis=frozen_choice.candidate, selector_score=frozen_choice.expected,
                          winner_stage_agree=stage_choice.stage_prefers.eq(3), protected=trace['glm_winner'] | trace['fable_guard'],
                          original_hard_override=incumbent != raw_stage, original_typography=typed,
                          scored_gemini_patch=broad, candidate_has_members=bool(members))
    accepted = frozen_choice.candidate.where(frozen_choice.expected.gt(frozen['threshold']), row.baseline)
    original = replay_scoring_model(actions, frozen, truth.index)
    if not accepted.equals(original.prediction) or not np.allclose(frozen_choice.expected, original.expected_delta,
                                                                     rtol=0., atol=1e-12):
        raise ValueError('Frozen stage head must reproduce the existing selector')
    backed = frozen_choice.member_prefers.ge(2) & frozen_choice.stage_prefers.ge(2)
    open_calls = {'frozen_selector': accepted, 'consensus_veto': accepted.where(backed, row.baseline)}
    if members:
        expanded = build_actions(truth, trace, incumbent, {name: members[name] for name in shared}, masks)
        expanded = votes(expanded, truth, trace, {name: members[name] for name in shared}, masks)
        best = rank(expanded, ['member_prefers', 'stage_prefers', 'feature_stage_votes', 'mean_stage_margin'])
        best = best.loc[truth.index]
        row['member_hypothesis'] = best.candidate
        row['member_vote_support'] = best.member_prefers
        safe = best.member_prefers.ge(2) & best.stage_prefers.ge(2) & row.original_hard_override
        within = best.candidate.where(safe, row.baseline)
        stable = open_calls['consensus_veto']
        open_calls['consensus_hard_override'] = within
        open_calls['consensus_hybrid'] = stable.where(stable.ne(row.baseline), within)
    receipt = json.loads((folder / 'glm_report.json').read_text())
    for name, opened in open_calls.items():
        calls = pd.DataFrame({'baseline': incumbent, 'open': opened}, index=truth.index)
        guarded = guard_frame(truth, trace, incumbent, calls, receipt=receipt)
        after = guarded.native_and_fable if receipt.get('status') == 'applied' else row.baseline
        row[name] = after
        row[f'{name}_after_scored_patch'] = after.where(row.scored_gemini_patch.eq(row.baseline), row.scored_gemini_patch)
    return row


def conflict(part, candidate='hypothesis'):
    y = part.model
    return part[candidate].eq(y).astype(int) - part.baseline.eq(y).astype(int)


def confidence(part, score='vote_expected', candidate='hypothesis'):
    realized = conflict(part, candidate)
    counts = part.groupby(['cohort', 'model']).model.transform('size')
    classes = part.groupby('cohort').model.transform('nunique')
    weight = 1. / (counts * classes)
    result = []
    for interval, rows in part.groupby(pd.cut(part[score], BINS, include_lowest=True), observed=True):
        value = realized.loc[rows.index]
        active = int(value.ne(0).sum())
        current = weight.loc[rows.index].to_numpy()
        item = dict(bin=str(interval), rows=len(rows), fixes=int(value.gt(0).sum()), harms=int(value.lt(0).sum()),
                    neither=int(value.eq(0).sum()), mean_expected=float(rows[score].mean()),
                    actual_signed=float(value.mean()), weighted_actual_signed=float(np.average(value, weights=current)),
                    weighted_mean_expected=float(np.average(rows[score], weights=current)),
                    harm_given_endpoint=float(value.lt(0).sum() / active) if active else None)
        if candidate == 'hypothesis':
            item.update(mean_p_fix=float(rows.vote_p_fix.mean()), mean_p_harm=float(rows.vote_p_harm.mean()))
        result.append(item)
    return result


def summarize(part, draws):
    real = conflict(part)
    report = dict(rows=len(part), prompts=int(part.prompt_id.nunique()),
                  top_voting_competitor=dict(fixed=int(real.eq(1).sum()), harmed=int(real.eq(-1).sum()),
                                             neither=int(real.eq(0).sum()), score_bins=confidence(part)),
                  by_original_family={family: confidence(items) for family, items in part.groupby(part.baseline.str.split('-').str[0])},
                  focal_original_classes={label: confidence(items) for label, items in part.groupby('baseline')
                                          if label in ('claude-fable-5', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3')},
                  original_hard_override_rows=int(part.original_hard_override.sum()),
                  original_typography_rows=int(part.original_typography.sum()))
    if part.candidate_has_members.all():
        report['fixed_conservative_policies'] = [paired_stats(part, 'baseline', arm, draws) for arm in POLICIES]
        report['added_to_scored_dash'] = [paired_stats(part, 'scored_gemini_patch', arm + '_after_scored_patch', draws)
                                           for arm in POLICIES]
        report['compared_to_existing'] = [paired_stats(part, 'frozen_selector', arm, draws) for arm in POLICIES[1:]]
        report['member_voter'] = {scope: dict(rows=len(items), fixed=int(conflict(items, 'member_hypothesis').eq(1).sum()),
                                                 harmed=int(conflict(items, 'member_hypothesis').eq(-1).sum()))
                                  for scope, items in [('all', part), ('supported', part.loc[part.member_vote_support.ge(2)])]}
    return report


def run(historical, prior, gen003, kimi, inputs, independent, source, model, output, draws=3000):
    blocks, held = make_blocks(historical, prior, gen003, kimi, inputs, independent, source)
    folders = (historical, gen003, kimi, independent)
    frames = [truth[['cohort', 'model', 'prompt_id', 'n_words', 'text']].assign(
        baseline=baseline, strict=truth.get('strict_topic_holdout', False)) for truth, _, baseline, _, _ in blocks]
    rows = pd.concat(frames, verify_integrity=True)
    stage_actions = pd.concat([build_actions(truth, trace, baseline) for truth, trace, baseline, _, _ in blocks])
    dev, extra, separation = calibration_ids(rows, held)
    oof = oof_probabilities(stage_actions, rows, dev.append(extra))
    frozen = json.loads(model.read_text())
    labels = blocks[0][1]['labels'].tolist()
    if frozen['arm'] != 'stage_multisource' or frozen['version'] != 1 or frozen['label_order'] != labels:
        raise ValueError('Expected the archived source-balanced stage capsule')
    shared = set(blocks[0][3]) & set(blocks[3][3])
    collected = []
    for (truth, trace, baseline, members, masks), folder, frame in zip(blocks, folders, frames, strict=True):
        if trace['labels'].tolist() != labels:
            raise ValueError('The traces use different score label orders')
        collected.append(pack_block(truth, trace, baseline, members, masks, folder,
                                    frame.drop(columns='text'), frozen, oof, shared))
    predicted = pd.concat(collected, verify_integrity=True)
    domains = {name: part for name, part in predicted.groupby('cohort') if name != 'reconstructed_stress'}
    domains['prvsiyan_316'] = predicted.loc[predicted.strict]
    domains['calibration_oof_pooled'] = predicted.loc[dev.append(extra)]
    for name in ('development', 'gen003', 'kimi_flash'):
        domains[name] = domains[name].loc[domains[name].index.intersection(dev.append(extra), sort=False)]
    result = {name: {length: summarize(frame, draws) for length, frame in
                     [('all', part), ('short', part.loc[part.n_words.lt(100)])] if len(frame)}
              for name, part in domains.items()}
    report = dict(oof_separation=separation, tested_cutoff=frozen['threshold'], member_branches=sorted(shared),
                  coverage_note='Member voting is restricted to matched captured members on C1, C2 and Prvsiyan. Other domains use shared stage evidence.',
                  fitting_note='Existing comparator capsule for all fixed decisions; reproduce original 4-fold heads for OOF score diagnostics only. No independent fit.',
                  policies='Fixed two stage score preferences and two distinct member family preferences. Keep native and Fable guards.', panels=result)
    output.mkdir(parents=True, exist_ok=True)
    predicted.to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
