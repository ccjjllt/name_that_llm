"""Compare family and within-family stage weighting on identical original calls."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import family_factored_stage
from .counterfactual_deployment import score_predictions
from .counterfactual_native_guard import guard_frame
from .counterfactual_selection import STAGES
from .counterfactual_selection_audit import calibration_ids, canonical_prompt, class_weights, make_blocks
from .peer_context_report import balanced_accuracy, paired_stats


ARMS = {'flat': False, 'factored': True}
CUTOFFS = (0., .15, .30, .60, 1.20, None)


def context(blocks, folders, portable):
    labels = tuple(blocks[0][1]['labels'])
    panels, matrices, snapshots = [], [], []
    for (truth, trace, baseline, _, _), folder in zip(blocks, folders, strict=True):
        if tuple(trace['labels']) != labels:
            raise ValueError('Trace class orders differ')
        native = json.loads((folder / 'glm_report.json').read_text())
        frame = truth[['cohort', 'model', 'prompt_id', 'n_words', 'text']].assign(
            baseline=baseline, strict=truth.get('strict_topic_holdout', False))
        queries = truth.text.to_frame().reset_index()[['id', 'text']]
        frame['selector'] = score_predictions(queries, trace, baseline, native, portable, 'native_guard')
        panels.append(frame)
        matrices.append(np.stack([trace[key] for key in STAGES], axis=1))
        snapshots.append((truth, trace, baseline, native))
    return pd.concat(panels, verify_integrity=True), np.concatenate(matrices), labels, snapshots


def proposals(rows, matrix, labels, snapshots, fit_ids, target_ids, *, factored):
    fit_at = rows.index.get_indexer(fit_ids)
    target_at = rows.index.get_indexer(target_ids)
    if (fit_at < 0).any() or (target_at < 0).any():
        raise ValueError('Fit and target IDs must refer to frozen traces')
    fit_rows = rows.loc[fit_ids]
    weights = class_weights(rows, fit_ids)
    params = family_factored_stage.fit(matrix[fit_at], labels, fit_rows.model.to_numpy(), weights.to_numpy(),
                                       factored=factored)
    scores = family_factored_stage.predict(matrix[target_at], labels, params)
    winner = scores.argmax(axis=1)
    incumbents = pd.Index(labels).get_indexer(rows.loc[target_ids, 'baseline'])
    if (incumbents < 0).any():
        raise ValueError('Original calls must appear in the class order')
    result = pd.DataFrame({'open': np.asarray(labels)[winner],
                           'advantage': scores.max(axis=1) - scores[np.arange(len(scores)), incumbents]},
                          index=target_ids)
    result['guarded'] = rows.loc[target_ids, 'baseline']
    for truth, trace, baseline, native in snapshots:
        part = result.loc[result.index.intersection(truth.index, sort=False)]
        if len(part) and native['status'] == 'applied':
            gated = guard_frame(truth, trace, baseline, part, receipt=native)
            result.loc[part.index, 'guarded'] = gated.native_guard
    return result, params


def calls(rows, result, cutoff):
    baseline = rows.loc[result.index, 'baseline']
    if cutoff is None:
        return baseline
    return result.guarded.where(result.advantage.gt(cutoff), baseline)


def fit_one(rows, matrix, labels, snapshots, fit_ids, target_ids, *, factored):
    training = rows.loc[fit_ids]
    prompt = canonical_prompt(training.prompt_id)
    folds = []
    for train, valid in GroupKFold(4).split(training, groups=prompt):
        result, _ = proposals(rows, matrix, labels, snapshots, fit_ids[train], fit_ids[valid], factored=factored)
        folds.append(result)
    oof = pd.concat(folds, verify_integrity=True).loc[fit_ids]
    cutoff_scores = {}
    for cutoff in CUTOFFS:
        predictions = calls(rows, oof, cutoff)
        cutoff_scores[cutoff] = float(np.mean([
            balanced_accuracy(group.model, predictions.loc[group.index])
            for _, group in training.groupby('cohort')
        ]))
    cutoff = max(CUTOFFS, key=lambda value: (cutoff_scores[value], np.inf if value is None else value))
    result, params = proposals(rows, matrix, labels, snapshots, fit_ids, target_ids, factored=factored)
    options = [{'cutoff': value, 'oof_mean_source_ba': score} for value, score in cutoff_scores.items()]
    return (result.assign(locked=calls(rows, result, cutoff)),
            dict(parameters=params, cutoff=cutoff, oof=options, fit_counts=training.cohort.value_counts().to_dict()))


def metrics(panel, draws):
    comparisons = [('baseline', 'selector'), ('baseline', 'flat_open'), ('baseline', 'factored_open'),
                   ('baseline', 'flat_guarded'), ('baseline', 'factored_guarded'),
                   ('baseline', 'flat_locked'), ('baseline', 'factored_locked'),
                   ('flat_guarded', 'factored_guarded'), ('flat_locked', 'factored_locked'),
                   ('selector', 'factored_locked')]
    return {f'{before}_to_{after}': paired_stats(panel, before, after, draws) for before, after in comparisons}


def run(historical, prior, gen003, kimi, inputs, output, reference, independent=None, source=None, draws=2000):
    blocks, held_truth = make_blocks(historical, prior, gen003, kimi, inputs, independent, source)
    folders = [historical, gen003, kimi, *([independent] if independent else [])]
    portable = json.loads(Path(reference).read_text())
    if portable['arm'] != 'stage_multisource':
        raise ValueError('Expected the original frozen multisource selector')
    rows, matrix, labels, snapshots = context(blocks, folders, portable)
    dev, external, separation = calibration_ids(rows, held_truth)
    fit = dev.append(external)
    # No fit from the target cohorts even when the independent trace is available.
    target = rows.index[~rows.cohort.isin(['development', 'gen003', 'kimi_flash'])]
    results = rows.loc[target, ['cohort', 'model', 'prompt_id', 'n_words', 'strict', 'baseline', 'selector']].copy()
    descriptions = {}
    for arm, factored in ARMS.items():
        prediction, descriptions[arm] = fit_one(rows, matrix, labels, snapshots, fit, target, factored=factored)
        results = results.join(prediction.add_prefix(f'{arm}_'))
    panels = {'c2': results.cohort.eq('evaluation'), 'stress': results.cohort.eq('reconstructed_stress'),
              'prvsiyan_895': results.cohort.eq('prvsiyan'),
              'prvsiyan_316': results.cohort.eq('prvsiyan') & results.strict}
    summaries = {}
    for name, mask in panels.items():
        for length, subset in [('all', mask), ('short', mask & results.n_words.lt(100))]:
            if subset.any():
                summaries[f'{name}_{length}'] = metrics(results.loc[subset], draws)
    # Refit on the other sources while excluding every identified prompt in the held source.
    leave_source = {}
    prompt = canonical_prompt(rows.prompt_id)
    for cohort, part in rows.loc[fit].groupby('cohort'):
        other = fit[~prompt.loc[fit].isin(prompt.loc[part.index])]
        details = {}
        readout = part[['model', 'prompt_id', 'n_words', 'baseline']].copy()
        for arm, factored in ARMS.items():
            prediction, spec = fit_one(rows, matrix, labels, snapshots, other, part.index, factored=factored)
            readout = readout.join(prediction.add_prefix(f'{arm}_'))
            details[arm] = spec
        leave_source[cohort] = dict(fitted=details, comparisons={
            f'{before}_to_{after}': paired_stats(readout, before, after, draws)
            for before, after in [('baseline', 'flat_locked'), ('baseline', 'factored_locked'),
                                  ('flat_locked', 'factored_locked')]})
    output.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output / 'predictions.parquet')
    report = dict(stages=STAGES, labels=labels, separation=separation, fitted=descriptions,
                  panels=summaries, leave_source=leave_source, cutoff_options=CUTOFFS)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Wrote paired family-factorized results', len(results), flush=True)
    return report
