"""Prompt-resampling diagnostics with and without refitting fusion temperatures.

The base model, raw relation scores and query composition remain fixed. This is
conditional method research, not uncertainty over training or future datasets.
"""
import numpy as np
import pandas as pd

from .q8_fusion import (_calibration_spec, _digest, apply_branch_calibration,
                       candidate_recipes, fit_temperature_records, mix_branches)


def _calibration(records):
    result = dict(policy='independent_branch_temperature_v1',
                  specification=_calibration_spec(), branches=records)
    result['calibration_sha256'] = _digest(result)
    return result


def _subset(branches, positions):
    return {key: np.asarray(value)[positions] for key, value in branches.items()}


def _ba(target, prediction):
    return float(np.mean([np.mean(prediction[target == label] == label)
                          for label in np.unique(target)]))


def _decisions(branches, candidates):
    return np.stack([mix_branches(branches, recipe).argmax(1) for recipe in candidates], axis=1)


def summarize_trials(trials, candidates, originals):
    reports = {}
    for policy, table in trials.groupby('policy', sort=False):
        n = len(table)
        counts = table.winner.value_counts()
        frequency = float(table.original_recipe_selected.mean())
        temp = {name.removeprefix('temperature_'): dict(
            median=float(table[name].median()),
            descriptive_percentile_95pct=table[name].quantile([.025, .975]).tolist())
                for name in table if name.startswith('temperature_')}
        oob = table.loc[table.oob_rows.gt(0)]
        reports[policy] = dict(
            draws=n, original_recipe=originals[policy],
            mean_unique_inbag_prompts=float(table.unique_inbag_prompts.mean()),
            unique_inbag_prompts_range=[int(table.unique_inbag_prompts.min()), int(table.unique_inbag_prompts.max())],
            original_recipe_selection_frequency=frequency,
            original_recipe_frequency_monte_carlo_se=float(np.sqrt(frequency * (1 - frequency) / n)),
            maximum_frequency_monte_carlo_se=float(.5 / np.sqrt(n)),
            candidate_frequencies={recipe['name']: float(counts.get(recipe['name'], 0) / n)
                                   for recipe in candidates},
            same_original_dev_decisions_frequency=float(table.same_dev_decisions.mean()),
            same_original_outer_decisions_frequency=float(table.same_outer_decisions.mean()),
            expected_outer_changed_row_fraction=float(table.outer_changed_row_fraction.mean()),
            outer_changed_row_fraction_percentile_95pct=table.outer_changed_row_fraction.quantile([.025, .975]).tolist(),
            temperatures=temp,
            oob=dict(draws=len(oob), mean_unique_prompts=float(oob.oob_prompts.mean()),
                     mean_selected_ba=float(oob.oob_selected_ba.mean()),
                     mean_plain_ba=float(oob.oob_plain_ba.mean()),
                     mean_fixed_gaussian_noninjective_ba=float(oob.oob_gaussian_ba.mean()),
                     mean_selected_minus_plain=float((oob.oob_selected_ba - oob.oob_plain_ba).mean()),
                     mean_selected_minus_fixed_gaussian=float((oob.oob_selected_ba - oob.oob_gaussian_ba).mean()),
                     mean_inbag_selection_ba=float(oob.inbag_selected_ba.mean()),
                     interpretation='Prompt OOB diagnostics conditional on cached scores. OOB sizes differ; repeated draws overlap. Mean inbag/OOB difference is not a pure selection-bias estimate or independent CV score.'))
    return reports


def bootstrap_selection(rows, branches, labels, outer_branches, *, draws=400,
                        seed=20261107, crossfit_draws=200, crossfit_seed=42):
    """Refit temperatures and select weights on each resampled prompt dataset.

    No outer rows/labels enter this API. Outer scores are used only to compare
    decisions to the original procedure. Duplicate sampled prompt IDs stay in
    one cross-fit partition; multiplicities weight both temperature and BA fits.
    """
    from .q8_crossfit_fusion import crossfit_temperature_predictions
    if draws < 1 or not 0 <= crossfit_draws <= draws:
        raise ValueError('Use positive draws and crossfit_draws between zero and draws.')
    rows = rows.reset_index(drop=True)
    prompts = np.asarray(sorted(rows.prompt_id.astype(str).unique()))
    if len(prompts) < 4:
        raise ValueError('This diagnostic requires at least four development prompts.')
    y = rows.model.map({label: i for i, label in enumerate(labels)}).to_numpy()
    if pd.isna(y).any():
        raise ValueError('A development label is outside the recorded model label order.')
    positions = [np.flatnonzero(rows.prompt_id.astype(str).eq(prompt)) for prompt in prompts]
    candidates = candidate_recipes(False)
    records = fit_temperature_records(rows, branches, labels)
    calibration = _calibration(records)
    calibrated = apply_branch_calibration(branches, calibration)
    dev_decisions = _decisions(calibrated, candidates)
    outer_decisions = _decisions(apply_branch_calibration(outer_branches, calibration), candidates)
    winner_v2 = int(np.argmax([_ba(y, dev_decisions[:, i]) for i in range(len(candidates))]))
    baseline_indices = dict(fixed_temperature=winner_v2, refit_temperature=winner_v2)
    if crossfit_draws:
        cross, _, _ = crossfit_temperature_predictions(rows, branches, labels, seed=crossfit_seed)
        cross_decisions = _decisions(cross, candidates)
        baseline_indices['crossfit_refit_temperature'] = int(np.argmax(
            [_ba(y, cross_decisions[:, i]) for i in range(len(candidates))]))
    originals = {name: candidates[value]['name'] for name, value in baseline_indices.items()}
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(len(prompts), np.full(len(prompts), 1 / len(prompts)), size=draws)
    trials, skipped = [], []
    for draw, counts in enumerate(multiplicities):
        sampled = np.concatenate([np.tile(pos, int(count)) for pos, count in zip(positions, counts) if count])
        oob_parts = [pos for pos, count in zip(positions, counts) if not count]
        oob = np.concatenate(oob_parts) if oob_parts else np.empty(0, dtype=int)
        sample_rows, sample_branches = rows.iloc[sampled].reset_index(drop=True), _subset(branches, sampled)
        fitted = fit_temperature_records(sample_rows, sample_branches, labels, fit_scope='bootstrap inbag prompts')
        fitted_calibration = _calibration(fitted)
        fitted_dev = apply_branch_calibration(branches, fitted_calibration)
        fitted_decisions = _decisions(fitted_dev, candidates)
        fitted_outer = _decisions(apply_branch_calibration(outer_branches, fitted_calibration), candidates)
        procedures = [
            ('fixed_temperature', dev_decisions[sampled], dev_decisions, outer_decisions, records),
            ('refit_temperature', fitted_decisions[sampled], fitted_decisions, fitted_outer, fitted)]
        if draw < crossfit_draws:
            if sample_rows.prompt_id.nunique() < 4:
                skipped.append(dict(draw=draw, reason='fewer than four distinct sampled prompts'))
            else:
                cross, assignments, evidence = crossfit_temperature_predictions(
                    sample_rows, sample_branches, labels, seed=crossfit_seed)
                # This verifies the potentially repeated prompt has one fold.
                if pd.DataFrame(dict(prompt=sample_rows.prompt_id, fold=assignments)).groupby('prompt').fold.nunique().max() != 1:
                    raise ValueError('A resampled prompt crossed calibration partitions.')
                procedures.append(('crossfit_refit_temperature', _decisions(cross, candidates),
                                   fitted_decisions, fitted_outer, fitted))
        for policy, selection, applied_dev, applied_outer, temperatures in procedures:
            candidate_ba = [_ba(y[sampled], selection[:, i]) for i in range(len(candidates))]
            chosen = int(np.argmax(candidate_ba))
            original = baseline_indices[policy]
            record = dict(draw=draw, policy=policy, winner=candidates[chosen]['name'],
                          unique_inbag_prompts=int(np.count_nonzero(counts)),
                          oob_prompts=int(np.count_nonzero(counts == 0)), oob_rows=len(oob),
                          inbag_selected_ba=candidate_ba[chosen],
                          original_recipe_selected=chosen == original,
                          same_dev_decisions=bool(np.array_equal(applied_dev[:, chosen], dev_decisions[:, original])),
                          same_outer_decisions=bool(np.array_equal(applied_outer[:, chosen], outer_decisions[:, original])),
                          outer_changed_row_fraction=float(np.mean(applied_outer[:, chosen] != outer_decisions[:, original])),
                          oob_selected_ba=_ba(y[oob], applied_dev[oob, chosen]) if len(oob) else None,
                          oob_plain_ba=_ba(y[oob], applied_dev[oob, 0]) if len(oob) else None,
                          oob_gaussian_ba=_ba(y[oob], applied_dev[oob, 2]) if len(oob) else None,
                          **{'temperature_' + key: val['temperature'] for key, val in temperatures.items()})
            trials.append(record)
    table = pd.DataFrame(trials)
    summary = dict(
        policy='fixed_model_prompt_selection_bootstrap_v1', dev_rows=len(rows), dev_prompts=len(prompts),
        seed=seed, crossfit_seed=crossfit_seed, crossfit_skipped=skipped,
        candidate_family_sha256=_digest(candidates), original_temperatures=records,
        reports=summarize_trials(table, candidates, originals),
        resampling_scope='Whole development prompts with replacement; same draws compare fixed-temperature and refit-temperature selection. Crossfit uses the first crossfit_draws draws.',
        conditional_scope='Base model, Gaussian fit, inferred grouping and full-cohort cached branch scores are fixed. Temperatures and weights are refit only from inbag development labels. Outer labels are not accepted by this function.',
        limitations=['No model retraining, training-seed, grouping-fit, public-source selection, or new-dataset uncertainty.',
                     'Raw full-query score context is fixed rather than recomputed on each bootstrap cohort.',
                     'Recipe changes and decision changes are distinct; neither alone measures predictive harm.',
                     'OOB diagnostics overlap across draws and are not a new independent dataset or a confidence interval for deployment performance.',
                     'Fixed-temperature OOB is conditional on original temperatures fitted using all dev labels; only refit variants exclude OOB labels from calibration.',
                     'Crossfit refits each temperature on other sampled prompt folds, selects weights from crossfit predictions, then uses all-inbag temperature refit for OOB/outer decisions.',
                     'Crossfit partitions are regenerated from distinct inbag prompts using the same fixed label-free seed/rule. All copies of a sampled prompt share one partition.',
                     'Reported temperature distributions describe final all-inbag inference refits, not the several intermediate crossfit temperatures.'])
    return summary, table, multiplicities
