"""Average probabilities from prompt-bootstrap temperature/recipe fits.

This is an explicit research alternative to a single selected recipe. All
members fit development labels only; application requires no query labels.
"""
import numpy as np
import pandas as pd

from .q8_fusion import (apply_branch_calibration, candidate_recipes,
                       fit_temperature_records, mix_branches)
from .q8_fusion_stability import _ba, _calibration, _subset


def apply_bagged_members(branches, members):
    """Fixed arithmetic probability average, not a vote or mean log score."""
    if not len(members):
        raise ValueError('A bagged procedure needs at least one fitted member.')
    if 'policy' in members and not members.policy.eq('refit_temperature').all():
        raise ValueError('Bagging uses complete temperature-and-recipe refit members.')
    recipes = {recipe['name']: recipe for recipe in candidate_recipes(False)}
    probability = np.zeros_like(np.asarray(branches['plain_lp'], dtype=float))
    for _, member in members.iterrows():
        if member.winner not in recipes:
            raise ValueError('A member selected a recipe outside the fixed eleven-candidate family.')
        temperatures = {name: dict(temperature=float(member['temperature_' + name]))
                        for name in ('plain', 'centered', 'gaussian', 'gaussian_noninjective')}
        calibrated = apply_branch_calibration(branches, _calibration(temperatures))
        probability += np.exp(mix_branches(calibrated, recipes[member.winner]))
    probability /= len(members)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability


def fit_bagged_members(rows, branches, labels, *, draws=100, seed=42):
    """Refit four temperatures and the same eleven recipes for every draw."""
    if draws < 1:
        raise ValueError('Use a positive number of bootstrap members.')
    rows = rows.reset_index(drop=True)
    prompts = np.asarray(sorted(rows.prompt_id.astype(str).unique()))
    if not len(prompts):
        raise ValueError('No development prompts.')
    target = rows.model.map({label: i for i, label in enumerate(labels)}).to_numpy()
    if pd.isna(target).any():
        raise ValueError('An input label is outside the recorded class order.')
    positions = [np.flatnonzero(rows.prompt_id.astype(str).eq(prompt)) for prompt in prompts]
    counts = np.random.default_rng(seed).multinomial(
        len(prompts), np.full(len(prompts), 1 / len(prompts)), size=draws)
    candidates = candidate_recipes(False)
    records = []
    for draw, multiplicity in enumerate(counts):
        selected = np.concatenate([np.tile(pos, int(count)) for pos, count in zip(positions, multiplicity) if count])
        sample_rows, sample_branches = rows.iloc[selected].reset_index(drop=True), _subset(branches, selected)
        temperatures = fit_temperature_records(sample_rows, sample_branches, labels,
                                               fit_scope='bagging bootstrap inbag prompts')
        calibrated = apply_branch_calibration(sample_branches, _calibration(temperatures))
        quality = [_ba(target[selected], mix_branches(calibrated, recipe).argmax(1)) for recipe in candidates]
        winner = int(np.argmax(quality))
        records.append(dict(draw=draw, policy='refit_temperature', seed=seed,
                            winner=candidates[winner]['name'], inbag_selected_ba=quality[winner],
                            unique_inbag_prompts=int(np.count_nonzero(multiplicity)),
                            **{'temperature_' + name: record['temperature'] for name, record in temperatures.items()}))
    return pd.DataFrame(records)
