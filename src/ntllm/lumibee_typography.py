"""Isolate the three final typography changes in the public LumiBee v2."""

import json
from itertools import product

import numpy as np
import pandas as pd
from scipy.special import softmax

from .counterfactual_native_guard import verify_route
from .counterfactual_selection import GLM, SPACED_DASH
from .member_evidence_transfer import read_capture, read_prvsiyan
from .peer_context_report import paired_stats, read_ids, same_ids
from .sibling_odds_replay import replay_predictions


FACTORS = ('posterior', 'four_quotes', 'joint')
ARMS = {''.join(str(int(x)) for x in bits): bits for bits in product((False, True), repeat=3)}


def apply_typography(probabilities, before, labels, presence, joint):
    blocked = np.stack([~np.isin(labels, SPACED_DASH),
                        np.char.startswith(labels, 'claude-') | np.char.startswith(labels, 'glm-'),
                        np.char.startswith(labels, 'gemini-')])
    index = pd.Index(labels)
    prediction = np.asarray(before).copy()
    probabilities = probabilities.copy()
    if joint:
        incumbent = index.get_indexer(prediction)
        violations = presence & blocked[:, incumbent].T
        union = np.any(presence[:, :, None] & blocked[None, :, :], axis=1)
        proposal = np.where(union, 0., probabilities)
        primary = violations.any(1) & (proposal.sum(1) > 0)
        prediction[primary] = labels[proposal[primary].argmax(1)]
        empty = violations.any(1) & ~primary
        for column, forbidden in enumerate(blocked):
            use = empty & (violations.argmax(1) == column)
            fallback = np.where(forbidden, 0., probabilities[use])
            positions = np.flatnonzero(use)[fallback.sum(1) > 0]
            prediction[positions] = labels[fallback[fallback.sum(1) > 0].argmax(1)]
    else:
        for column, forbidden in enumerate(blocked):
            triggered = presence[:, column] & forbidden[index.get_indexer(prediction)]
            probabilities[np.ix_(triggered, forbidden)] = 0.
            changed = triggered & (probabilities.sum(1) > 0)
            prediction[changed] = labels[probabilities[changed].argmax(1)]
    return prediction


def replay(truth, trace, baseline, folder):
    route, _ = verify_route(folder, truth, trace, baseline)
    labels = np.asarray(trace['labels'])
    broad = np.isin(labels[trace['stack_logp'].argmax(1)], GLM) & ~trace['glm_winner']
    if (broad & ~route).any():
        raise ValueError('A GLM-to-other override must come from a completed native score')
    original = replay_predictions(trace, truth.text, typography=False)
    kiro_like = replay_predictions(dict(trace, fable_guard=broad), truth.text, typography=False, sibling_odds=False)
    text = truth.text.fillna('').astype(str)
    separated = text.str.contains(' — ', regex=False).to_numpy()
    any_dash = text.str.contains('—', regex=False).to_numpy()
    curly = text.str.contains('’', regex=False).to_numpy()
    more_curly = text.str.contains('[’‘“”]').to_numpy()
    archived_probability = softmax(trace['sibling_logp'].astype(np.float64), axis=1)
    aligned_probability = archived_probability.copy()
    aligned_probability[route] = softmax(trace['glm_logp'][route].astype(np.float64), axis=1)

    output = truth[['model', 'prompt_id', 'cohort', 'n_words']].assign(
        baseline=baseline, routed=route, strict=truth.get('strict_topic_holdout', False),
        four_quote_extra=more_curly & ~curly)
    for scope, before in (('original', original), ('kiro_like', kiro_like)):
        for arm, (post, quotes, joint) in ARMS.items():
            presence = np.column_stack([separated, more_curly if quotes else curly, any_dash])
            probabilities = aligned_probability if post else archived_probability
            output[f'{scope}__{arm}'] = apply_typography(probabilities, before, labels, presence, joint)
    if not np.array_equal(output.original__000, baseline):
        raise ValueError('The no-change replay did not reproduce original decisions')
    restore = np.char.startswith(np.asarray(original).astype(str), 'gemini-') & any_dash & (original != baseline)
    output['scored_dash_patch'] = np.where(restore, original, baseline)
    return output


def summarize(rows, draws):
    contrasts = [(f'{scope}__000', f'{scope}__{arm}') for scope in ('original', 'kiro_like') for arm in ARMS if arm != '000']
    contrasts.extend([('original__000', 'kiro_like__000'), ('scored_dash_patch', 'original__111')])
    return dict(rows=len(rows), routed=int(rows.routed.sum()), four_quote_additional_rows=int(rows.four_quote_extra.sum()),
                paired=[paired_stats(rows, a, b, draws) for a, b in contrasts])


def run(historical, prior, source, independent, prior_kimi, rank_reference, output, draws=3000):
    public = read_ids(prior / 'public_truth.parquet')
    held = read_prvsiyan(source, prior_kimi)
    groups = []
    for data, capture in ((public, historical), (held, independent)):
        trace, baseline, _, _, _ = read_capture(capture, data)
        groups.append(replay(data, trace, baseline, capture))
    result = pd.concat(groups, verify_integrity=True)
    result = result.loc[result.cohort.isin(['evaluation', 'prvsiyan'])]
    choices = read_ids(rank_reference)
    same_ids(choices.index, result.index, 'Choice-truth controls')
    choices = choices.loc[result.index]
    if not np.array_equal(choices.baseline, result.baseline):
        raise ValueError('The choice comparison must use the same incumbent')
    arms = ('listwise__open_candidates__native_and_fable', 'pairwise__open_candidates__native_and_fable',
            'pairwise__supported_pairs__native_and_fable', 'frozen_supported_selector')
    scored = result.scored_dash_patch.ne(result.baseline)
    for arm in arms:
        result['dash_plus__' + arm] = choices[arm].where(~scored, result.scored_dash_patch)
    scopes = {'c2': result.cohort.eq('evaluation'), 'heldout': result.cohort.eq('prvsiyan'), 'strict': result.strict}
    report = dict(factors=FACTORS, original_parity=len(public) + len(held), panels={}, after_scored_dash={},
                  limitations='Captured Sibling Odds GLM likelihood and routing; no PB115 rescoring for this decision-only ablation.')
    for panel, mask in scopes.items():
        for suffix, rows in (('all', result.loc[mask]), ('short', result.loc[mask & result.n_words.lt(100)])):
            key = f'{panel}__{suffix}'
            report['panels'][key] = summarize(rows, draws)
            report['after_scored_dash'][key] = [paired_stats(rows, 'scored_dash_patch', 'dash_plus__' + arm, draws)
                                                for arm in arms]
    output.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Verified all original typography calls on', len(public) + len(held), 'rows', flush=True)
