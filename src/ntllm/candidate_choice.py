"""Learn relative preferences over the same Sibling Odds candidate sets."""

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_softmax, logsumexp
from sklearn.preprocessing import StandardScaler

from .counterfactual_native_guard import guard_frame
from .counterfactual_selection import FABLE, STAGES
from .counterfactual_selection_audit import calibration_ids, class_weights, make_blocks
from .peer_context_report import balanced_accuracy, paired_stats, read_ids, same_ids
from .support_censored_selector import collect_support, prompt_folds


LOSSES = ('listwise', 'pairwise')
SCOPES = ('open_candidates', 'supported_pairs')
CUTOFFS = (0., .15, .30, .60, 1., 1.75, 1000.)
PENALTY = .05


def choice_features(truth, trace, incumbent):
    labels = pd.Index(trace['labels'])
    if not pd.Index(trace['ids']).equals(pd.Index(truth.index)) or len(labels) != 18:
        raise ValueError('Expected the aligned original 18-way stage scores')
    pred = labels.get_indexer(incumbent)
    if (pred < 0).any():
        raise ValueError('Unknown incumbent label')
    label_groups = np.array(['qwen' if name.startswith('qwen') else name.split('-', 1)[0] for name in labels])
    logps = np.stack([log_softmax(trace[key], axis=1) for key in STAGES], axis=-1)
    ranks = np.argsort(np.argsort(-logps, axis=1, kind='stable'), axis=1, kind='stable')
    is_incumbent = np.arange(len(labels))[None, :] == pred[:, None]
    mask = (ranks < 3).any(-1) | is_incumbent
    margins = np.maximum(logps - logps.max(axis=1, keepdims=True), -12.)
    family = np.empty_like(logps)
    for kind in np.unique(label_groups):
        group = label_groups == kind
        family[:, group] = logsumexp(logps[:, group], axis=1)[:, None, :]
    within = np.maximum(logps - family, -12.)
    short = truth.n_words.lt(100).to_numpy()[:, None]
    existing_override = np.asarray(trace['glm_winner'] | trace['fable_guard'])[:, None]
    same_family = label_groups[None, :] == label_groups[pred, None]
    columns = [f'relative_{name}' for name in STAGES] + [f'within_family_{name}' for name in STAGES]
    columns += ['stage_wins', 'stage_top3', 'incumbent', 'same_family', 'short_incumbent',
                'protected_incumbent', 'short_sibling_margin']
    values = np.concatenate([margins, within, np.stack([(ranks == 0).sum(-1), (ranks < 3).sum(-1),
                            is_incumbent, same_family, short & is_incumbent, existing_override & is_incumbent,
                            short * margins[:, :, -1]], axis=-1)], axis=-1)
    if not np.isfinite(values).all():
        raise ValueError('All stage scores must be finite')
    return values, mask, columns


def train_choice(values, permitted, gold, weights, ids):
    fit = ids[permitted[ids, gold[ids]] & (permitted[ids].sum(1) > 1)]
    if len(fit) == 0:
        raise ValueError('No calibration prompt contains the correct label and an alternative')
    x, mask, y = values[fit], permitted[fit], gold[fit]
    w = np.asarray(weights.loc[fit], dtype=float)
    w /= w.sum()
    scale = StandardScaler().fit(x[mask], sample_weight=np.broadcast_to((w / mask.sum(1))[:, None], mask.shape)[mask])
    x = (x - scale.mean_) / scale.scale_
    gold_x = x[np.arange(len(y)), y]
    diff = gold_x[:, None, :] - x
    rivals = mask.copy()
    rivals[np.arange(len(y)), y] = False
    differences = diff[rivals]
    pair_w = np.broadcast_to((w / (rivals.sum(1) * np.log(2)))[:, None], rivals.shape)[rivals]
    choice_w = w / np.log(mask.sum(1))

    def listwise(beta):
        scores = np.where(mask, x @ beta, -np.inf)
        norm = logsumexp(scores, axis=1)
        posterior = np.exp(scores - norm[:, None])
        cost = choice_w @ (norm - scores[np.arange(len(y)), y])
        gradient = np.einsum('n,nc,ncd->d', choice_w, posterior, x) - choice_w @ gold_x
        return float(cost + .5 * PENALTY * (beta @ beta)), gradient + PENALTY * beta

    def pairwise(beta):
        margin = differences @ beta
        cost = pair_w @ np.logaddexp(0., -margin)
        gradient = -(pair_w * expit(-margin)) @ differences
        return float(cost + .5 * PENALTY * (beta @ beta)), gradient + PENALTY * beta

    model = dict(mean=scale.mean_.tolist(), scale=scale.scale_.tolist(), used_rows=len(fit),
                 comparisons=len(differences), coefficients={})
    for name, loss in zip(LOSSES, (listwise, pairwise), strict=True):
        result = minimize(loss, np.zeros(x.shape[-1]), method='L-BFGS-B', jac=True,
                          options={'maxiter': 300, 'ftol': 1e-12})
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f'{name} fit failed: {result.message}')
        model['coefficients'][name] = result.x.tolist()
    return model


def propose(values, available, baseline, labels, model, supported):
    label_index = pd.Index(labels)
    anchor = label_index.get_indexer(baseline)
    pair_matrix = np.zeros((len(labels), len(labels)), dtype=bool)
    for before, after in supported:
        pair_matrix[label_index.get_loc(before), label_index.get_loc(after)] = True
    masks = {'open_candidates': available, 'supported_pairs': available & pair_matrix[anchor]}
    masks['supported_pairs'][np.arange(len(anchor)), anchor] = True
    standardized = (values - np.asarray(model['mean'])) / np.asarray(model['scale'])
    proposals = {}
    for method, scope in ((method, scope) for method in LOSSES for scope in SCOPES):
        scores = standardized @ np.asarray(model['coefficients'][method])
        preferred = np.where(masks[scope], scores, -np.inf).argmax(axis=1)
        i = np.arange(len(anchor))
        margin = scores[i, preferred] - scores[i, anchor]
        proposals[f'{method}__{scope}'] = (labels[preferred], margin)
    return proposals


def guard_all(captures, proposals):
    frames = []
    for truth, trace, incumbent, receipt in captures:
        ids = proposals.index.intersection(truth.index, sort=False)
        if len(ids):
            # Preserve original Fable and actual incumbent-supporting GLM likelihood.
            guard = guard_frame(truth, trace, incumbent, proposals.loc[ids].rename(columns={'candidate': 'open'}),
                                receipt=receipt)
            guard['native_fable_typography'] = guard.native_guard.where(guard.baseline.ne(FABLE), guard.baseline)
            for scope in ('native_and_fable', 'native_fable_typography'):
                if receipt.get('status') != 'applied':
                    guard[scope] = guard.baseline
            frames.append(guard[['native_and_fable', 'native_fable_typography']])
    return pd.concat(frames, verify_integrity=True).loc[proposals.index]


def guarded_options(captures, ids, keep, predictions):
    arms = {}
    for name, (candidate, margin) in predictions.items():
        proposals = pd.DataFrame(dict(baseline=keep, candidate=candidate, margin=margin), index=ids)
        guard = guard_all(captures, proposals)
        for policy in guard:
            arms[f'{name}__{policy}'] = proposals.assign(candidate=guard[policy])
    return arms


def score_capsule(queries, trace, incumbent, receipt, model):
    if list(queries) != ['id', 'text'] or model['label_order'] != list(trace['labels']):
        raise ValueError('Frozen candidate scoring requires unlabeled IDs and bodies in the original class order')
    text = queries.set_index('id', verify_integrity=True).text.fillna('').astype(str)
    entries = text.to_frame().assign(n_words=text.str.split().str.len())
    values, mask, columns = choice_features(entries, trace, incumbent)
    if model['features'] != columns:
        raise ValueError('The original and deployed candidate feature orders differ')
    keep = pd.Series(incumbent, index=entries.index)
    raw = propose(values, mask, keep, trace['labels'], model, model['supported_pairs'])
    arms = guarded_options([(entries, trace, incumbent, receipt)], entries.index, keep, raw)
    return pd.DataFrame({name: arm.candidate.where(arm.margin.gt(model['thresholds'][name]['cutoff']), keep)
                         for name, arm in arms.items()}, index=entries.index)


def run(historical, prior, gen003, kimi, inputs, output, reference, independent, source, draws=2500):
    blocks, held_truth = make_blocks(historical, prior, gen003, kimi, inputs, independent, source)
    labels = blocks[0][1]['labels']
    frames, variable_blocks, masks, captures = [], [], [], []
    for (truth, trace, incumbent, _, _), directory in zip(blocks, (historical, gen003, kimi, independent), strict=True):
        if not np.array_equal(labels, trace['labels']):
            raise ValueError('The class order varies between frozen traces')
        features, mask, columns = choice_features(truth, trace, incumbent)
        variable_blocks.append(features)
        masks.append(mask)
        frames.append(truth[['cohort', 'model', 'prompt_id', 'n_words', 'text']].assign(
            baseline=incumbent, strict=truth.get('strict_topic_holdout', False)))
        captures.append((truth, trace, incumbent, json.loads((directory / 'glm_report.json').read_text())))
    rows = pd.concat(frames, verify_integrity=True)
    values, available = np.concatenate(variable_blocks), np.concatenate(masks)
    gold = pd.Index(labels).get_indexer(rows.model)
    if (gold < 0).any():
        raise ValueError('Unknown truth class')
    fit_c1, fit_extra, separation = calibration_ids(rows, held_truth)
    fit = fit_c1.append(fit_extra)
    target = rows.index[rows.cohort.isin(['evaluation', 'prvsiyan'])]

    def score(fit_ids, score_ids):
        fit_pos = rows.index.get_indexer(fit_ids)
        score_pos = rows.index.get_indexer(score_ids)
        weights = class_weights(rows, fit_ids)
        weights.index = fit_pos
        capsule = train_choice(values, available, gold, weights, fit_pos)
        _, _, pairs = collect_support(rows.loc[fit_ids])
        keep = rows.loc[score_ids, 'baseline']
        predictions = propose(values[score_pos], available[score_pos], keep, labels, capsule, pairs)
        capsule.update(supported_pairs=pairs)
        return capsule, guarded_options(captures, score_ids, keep, predictions)

    oof = {}
    for fit_fold, valid in prompt_folds(rows.loc[fit]):
        _, arm = score(fit_fold, valid)
        for name, pred in arm.items():
            oof.setdefault(name, []).append(pred)
    by_source = rows.loc[fit].groupby('cohort').groups
    thresholds = {}
    for name, predictions in oof.items():
        seen = pd.concat(predictions, verify_integrity=True).loc[fit]
        metrics = {}
        for cutoff in CUTOFFS:
            pred = seen.candidate.where(seen.margin.gt(cutoff), seen.baseline)
            metrics[cutoff] = float(np.mean([balanced_accuracy(rows.loc[ids, 'model'], pred.loc[ids])
                                             for ids in by_source.values()]))
        thresholds[name] = dict(cutoff=max(CUTOFFS, key=lambda x: (metrics[x], x)), fit_ba=metrics)

    model, arms = score(fit, target)
    rows = rows.loc[target].drop(columns='text')
    legacy = read_ids(reference).loc[target]
    same_ids(legacy.index, rows.index, 'Frozen support selector controls', ordered=True)
    if not np.array_equal(legacy.baseline, rows.baseline):
        raise ValueError('The frozen selector uses a different incumbent')
    rows['frozen_supported_selector'] = legacy.pre_rank_eligibility
    for name, choices in arms.items():
        rows[name] = choices.candidate.where(choices.margin.gt(thresholds[name]['cutoff']), choices.baseline)
        rows[name + '__margin'] = choices.margin
    masks = {'c2': rows.cohort.eq('evaluation'), 'heldout': rows.cohort.eq('prvsiyan'), 'strict': rows.strict}
    report = dict(calibration=separation, thresholds=thresholds, fit_rows=len(fit), fit_covered=model['used_rows'], panels={})
    for scope, mask in masks.items():
        for length, part in [('all', rows.loc[mask]), ('short', rows.loc[mask & rows.n_words.lt(100)])]:
            report['panels'][f'{scope}__{length}'] = dict(
                baseline=[paired_stats(part, 'baseline', name, draws) for name in ('frozen_supported_selector', *arms)],
                pairwise_effect=[paired_stats(part, name, name.replace('listwise', 'pairwise'), draws)
                                 for name in arms if name.startswith('listwise')],
            )
    model.update(thresholds=thresholds, label_order=labels.tolist(), features=columns)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'capsule.json').write_text(json.dumps(model, indent=2, allow_nan=False) + '\n')
    print(f'Fitted {len(fit)} passages and reported {len(target)} fixed evaluation rows', flush=True)
