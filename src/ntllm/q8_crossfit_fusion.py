"""Prompt-cross-fitted temperature predictions for the fixed fusion family.

This separates each selection row from its temperature-fitting labels. It does
not add independent development prompts or remove weight-selection uncertainty.
"""
from pathlib import Path

import numpy as np
from scipy.special import log_softmax

from .q8_fusion import (_branch_active, _digest, _load_cohorts, _select_recipe,
                       _standalone_recipes, fit_branch_calibration,
                       fit_temperature_records)


def crossfit_temperature_predictions(rows, branches, labels, *, folds=4, seed=42):
    prompts = np.asarray(sorted(rows.prompt_id.astype(str).unique()))
    if not 2 <= folds <= len(prompts):
        raise ValueError('Temperature cross-fitting needs at least two nonempty prompt folds.')
    shuffled = np.random.default_rng(seed).permutation(prompts)
    blocks = np.array_split(shuffled, folds)
    result = {key: np.array(value, copy=True) for key, value in branches.items()}
    assignments = np.full(len(rows), -1, dtype=int)
    evidence = []
    for fold, held_prompts in enumerate(blocks):
        held = rows.prompt_id.astype(str).isin(held_prompts).to_numpy()
        fit = ~held
        if not fit.any() or not held.any():
            raise ValueError('Empty calibration or selection side.')
        train_branches = {key: np.asarray(value)[fit] for key, value in branches.items()}
        records = fit_temperature_records(rows.loc[fit].reset_index(drop=True), train_branches, labels,
                                          fit_scope='other inner prompt folds')
        for name in _standalone_recipes():
            active = _branch_active(branches, name) & held
            result[name + '_lp'][held] = 0.
            result[name + '_lp'][active] = log_softmax(
                np.asarray(branches[name + '_lp'])[active] / records[name]['temperature'], axis=1)
        assignments[held] = fold
        evidence.append(dict(fold=fold, calibration_prompts=sorted(rows.loc[fit, 'prompt_id'].astype(str).unique()),
                             selection_prompts=sorted(held_prompts.tolist()),
                             calibration_ids_sha256=_digest(sorted(rows.loc[fit, 'id'].astype(str))),
                             selection_ids_sha256=_digest(sorted(rows.loc[held, 'id'].astype(str))),
                             calibration_rows=int(fit.sum()), selection_rows=int(held.sum()),
                             temperatures=records))
    if (assignments < 0).any():
        raise ValueError('Cross-fitting did not score every development row.')
    record = dict(policy='prompt_crossfit_temperature_v1', folds=folds, seed=seed,
                  split_rule='Seeded permutation of sorted true prompt IDs, balanced contiguous blocks; no labels or scores choose blocks.',
                  prompts=len(prompts), rows=len(rows), partitions=evidence,
                  assignments_sha256=_digest(assignments.tolist()),
                  inference_scope='Raw branch scores and inferred groups remain fixed at the original full query cohort. Only calibration label membership changes.',
                  final_refit='After choosing the recipe, use scalar temperatures fit on all inner-dev prompts for outer deployment.',
                  limitation='Temperature fits exclude the scored prompt, but weight selection still reuses the same small development set. Outer data must never choose partitions, recipes, or temperatures.')
    record['crossfit_sha256'] = _digest(record)
    return result, assignments, record


def select_crossfit_calibrated_recipe(inner_evaluation_dir, split_manifest, training_evidence, out, *,
                                     allow_exclusion=False, calibration=None):
    """v3: four fixed prompt folds, unchanged candidates and full-dev final refit."""
    cohorts, _ = _load_cohorts(inner_evaluation_dir, split_manifest, 'inner_dev')
    rows, branches = cohorts['full']
    calibrated, assignments, record = crossfit_temperature_predictions(
        rows, branches, split_manifest['labels'], folds=4,
        seed=split_manifest['seed'] + 1009 * split_manifest['outer_fold'])
    if calibration is None:
        calibration = fit_branch_calibration(inner_evaluation_dir, split_manifest, training_evidence)
    locked = _select_recipe(inner_evaluation_dir, split_manifest, training_evidence, out,
                            allow_exclusion=allow_exclusion, calibration=calibration,
                            selection_branches=calibrated, crossfit_evidence=record)
    np.savez_compressed(Path(out) / 'crossfit_selection_branches.npz', **calibrated,
                        temperature_fold=assignments, ids=rows.id.astype(str).to_numpy(dtype=str))
    return locked
