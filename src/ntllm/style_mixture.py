"""Compare a style centroid density with a two-component class density."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, logsumexp
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

from .member_evidence_transfer import verify_manifest
from .peer_context_report import paired_stats, read_ids, read_trace, same_ids
from .sibling_odds_replay import replay_predictions
from .style_distance import HELD_SOURCE, PRIMARY_BLEND, prepare_inputs


ARMS = ('linear', 'single', 'mixture')
PCA_DIMENSIONS = 64
SHRINKAGE = .25
EM_STEPS = 40
SEED = 112525


def read_style(style, frame, labels):
    external = frame.cohort.eq('external')
    sample = frame.loc[external].drop(columns='text')
    if not sample.equals(read_ids(style / 'sample.parquet')):
        raise ValueError('StyleDistance used a different external sample or row order')

    linear = read_ids(style / 'style_logp.parquet')
    same_ids(linear.index, frame.index[~external], 'StyleDistance scores', ordered=True)
    if list(linear.columns) != labels or not np.isfinite(linear.to_numpy()).all():
        raise ValueError('StyleDistance scores must have all 18 label columns in order')

    with np.load(style / 'style_embedding.npz', allow_pickle=False) as data:
        same_ids(pd.Index(data['ids'], name='id'), frame.index, 'StyleDistance embeddings', ordered=True)
        values = data['values']
    if (values.ndim != 2 or values.shape[0] != len(frame) or values.shape[1] < PCA_DIMENSIONS
            or values.dtype != np.float32 or not np.isfinite(values).all()
            or not np.allclose(np.linalg.norm(values, axis=1), 1., rtol=0., atol=5e-4)):
        raise ValueError('Expected unit-normalized float32 StyleDistance embeddings')
    return values, linear


def whiten(training, evaluation, weight):
    mean = np.average(training, axis=0, weights=weight)
    centered = training - mean
    covariance = (centered.T * weight) @ centered / weight.sum()
    variance, axes = np.linalg.eigh(covariance)
    variance = variance[-PCA_DIMENSIONS:]
    scale = np.sqrt(np.maximum(variance, 1e-8))
    transform = axes[:, -PCA_DIMENSIONS:] / scale
    return centered @ transform, (evaluation - mean) @ transform


def component_log_density(values, means, variances):
    residual = values[:, None, :] - means[None, :, :]
    return -.5 * ((residual ** 2 / variances).sum(axis=2) + np.log(variances).sum(axis=1))


def fit_mixture(values, weights, target_variance, single_variance):
    centers = KMeans(n_clusters=2, n_init=3, random_state=SEED).fit(values, sample_weight=weights)
    means = centers.cluster_centers_
    variances = np.broadcast_to(single_variance, means.shape)
    proportions = np.full(2, .5)
    for _ in range(EM_STEPS):
        scores = component_log_density(values, means, variances) + np.log(proportions)
        responsibility = np.exp(log_softmax(scores, axis=1)) * weights[:, None]
        mass = responsibility.sum(axis=0)
        active = mass > 1e-10
        means[active] = (responsibility.T @ values)[active] / mass[active, None]
        residual = values[:, None, :] - means[None, :, :]
        variances = np.einsum('ij,ijk->jk', responsibility, residual ** 2) / np.maximum(mass[:, None], 1e-10)
        variances = (1 - SHRINKAGE) * variances + SHRINKAGE * target_variance
        proportions = np.maximum(mass, 1e-10)
        proportions /= proportions.sum()
    return means, variances, proportions


def fit_predict(values, frame, labels, fit, test):
    samples = frame.iloc[fit]
    counts = samples.groupby(['cohort', 'model']).model.transform('size').to_numpy()
    weights = 1. / counts
    training, evaluation = whiten(values[fit], values[test], weights)
    groups = [samples.model.eq(label).to_numpy() for label in labels]
    means = [np.average(training[mask], axis=0, weights=weights[mask]) for mask in groups]
    class_variance = np.asarray([np.average((training[mask] - mean) ** 2, axis=0, weights=weights[mask])
                                 for mask, mean in zip(groups, means)])
    target = np.maximum(class_variance.mean(axis=0), 1e-6)
    single_scores, mixture_scores = [], []
    for mask, mean, variance in zip(groups, means, class_variance):
        shrunk = (1 - SHRINKAGE) * variance + SHRINKAGE * target
        single_scores.append(component_log_density(evaluation, mean[None, :], shrunk[None, :])[:, 0])
        centers, variances, proportions = fit_mixture(training[mask], weights[mask], target, shrunk)
        joint = component_log_density(evaluation, centers, variances) + np.log(proportions)
        mixture_scores.append(logsumexp(joint, axis=1))
    return np.column_stack(single_scores), np.column_stack(mixture_scores)


def calibrate(raw, target):
    weight = 1. / np.bincount(target)[target]

    def objective(log_temperature):
        logp = log_softmax(raw / np.exp(log_temperature), axis=1)
        return -np.average(logp[np.arange(len(target)), target], weights=weight)

    result = minimize_scalar(objective, bounds=(-3., 5.), method='bounded')
    if not result.success:
        raise RuntimeError('Official OOF temperature calibration failed')
    return float(np.exp(result.x))


def score(values, frame, labels, linear):
    external = np.flatnonzero(frame.cohort.eq('external'))
    official = np.flatnonzero(frame.cohort.eq('official'))
    held = np.flatnonzero(~frame.cohort.isin(['external', 'official']))
    folds = np.full(len(frame), -1)
    raw = np.full((2, len(frame), len(labels)), np.nan)
    groups = frame.prompt_id.iloc[official]
    for fold, (fit, test) in enumerate(GroupKFold(5).split(official, groups=groups)):
        folds[official[test]] = fold
        raw[:, official[test]] = fit_predict(values, frame, labels, np.r_[external, official[fit]], official[test])
        print('Scored official fold', fold, len(test), flush=True)
    raw[:, held] = fit_predict(values, frame, labels, np.r_[external, official], held)
    target = np.searchsorted(labels, frame.model.iloc[official])
    temperatures = {}
    heads = {'linear': linear}
    evaluation = np.flatnonzero(~frame.cohort.eq('external'))
    for arm, scores in zip(ARMS[1:], raw):
        if not np.isfinite(scores[evaluation]).all():
            raise ValueError(f'Gaussian scores are incomplete for {arm}')
        temperature = calibrate(scores[official], target)
        temperatures[arm] = temperature
        scores[held] /= temperature
        heads[arm] = pd.DataFrame(log_softmax(scores[evaluation], axis=1),
                                  index=frame.index[evaluation], columns=labels)
    return heads, pd.Series(folds, index=frame.index).loc[linear.index], temperatures


def add_replay(rows, scores, directory, truth):
    trace, baseline = read_trace(directory, truth)
    rows.loc[truth.index, 'baseline'] = baseline
    protected = trace['glm_winner'] | trace['fable_guard']
    stage = np.where(protected[:, None], trace['glm_logp'], trace['sibling_logp'])
    top3 = trace['labels'][np.argsort(-stage, axis=1, kind='stable')[:, :3]]
    target = truth.model.to_numpy()
    rows.loc[truth.index, 'baseline_top3_miss'] = (baseline != target) & ~(top3 == target[:, None]).any(1)
    for arm, logp in scores.items():
        rows.loc[truth.index, f'{arm}_blend'] = replay_predictions(
            trace, truth.text, external_logits=logp, weight=PRIMARY_BLEND)
    return baseline


def compare(frame, heads, folds, public, held_trace):
    rows = frame.drop(columns='text').loc[folds.index].assign(fold=folds)
    for arm, scores in heads.items():
        rank = np.argsort(-scores.loc[rows.index].to_numpy(), axis=1, kind='stable')
        labels = scores.columns.to_numpy()
        rows[arm] = labels[rank[:, 0]]
        rows[f'{arm}_top3'] = (labels[rank[:, :3]] == rows.model.to_numpy()[:, None]).any(1)
        rows[f'{arm}_blend'] = pd.Series(index=rows.index, dtype=object)
    rows['baseline'] = pd.Series(index=rows.index, dtype=object)
    rows['baseline_top3_miss'] = pd.Series(index=rows.index, dtype='boolean')
    add_replay(rows, heads, public, read_ids(public / 'public_truth.parquet'))
    if held_trace is not None:
        truth = frame.loc[frame.cohort.eq('prvsiyan_clean')]
        verify_manifest(held_trace / 'probe_manifest.json', truth)
        baseline = add_replay(rows, heads, held_trace, truth)
        frozen = read_ids(held_trace / 'frozen_predictions.parquet')
        same_ids(frozen.index, truth.index, 'Frozen Prvsiyan decisions', ordered=True)
        if not np.array_equal(frozen.sibling_odds, baseline):
            raise ValueError('Prvsiyan Sibling Odds replay differs from its frozen decisions')

    report = []
    for cohort, panel in rows.groupby('cohort'):
        for subset, part in (('all', panel), ('short', panel.loc[panel.n_words.lt(100)])):
            if part.empty:
                continue
            entry = dict(cohort=cohort, subset=subset,
                         standalone={arm: paired_stats(part, 'linear', arm, draws=3000) for arm in ARMS[1:]},
                         mixture_vs_single=paired_stats(part, 'single', 'mixture', draws=3000),
                         standalone_top3={arm: float(part[f'{arm}_top3'].mean()) for arm in ARMS})
            if part.baseline.notna().all():
                miss = part.baseline_top3_miss
                entry['baseline_errors_outside_top3'] = int(miss.sum())
                entry['new_true_top3'] = {arm: int((miss & part[f'{arm}_top3']).sum()) for arm in ARMS}
                entry['blend'] = {arm: paired_stats(part, 'baseline', f'{arm}_blend', draws=3000) for arm in ARMS}
                entry['blend_mixture_vs_single'] = paired_stats(part, 'single_blend', 'mixture_blend', draws=3000)
                entry['blend_mixture_vs_linear'] = paired_stats(part, 'linear_blend', 'mixture_blend', draws=3000)
            report.append(entry)
    return rows, report


def run(official, public, audit, pool, style, output, held_trace=None):
    frame, labels = prepare_inputs(official, public, audit, pool)
    values, linear = read_style(style, frame, labels)
    heads, folds, temperatures = score(values, frame, labels, linear)
    rows, cohorts = compare(frame, heads, folds, public, held_trace)
    report = dict(pca_dimensions=PCA_DIMENSIONS, whitened=True, diagonal_shrinkage=SHRINKAGE,
                  components={'single': 1, 'mixture': 2}, fixed_blend=PRIMARY_BLEND,
                  external_sample_rows=int(frame.cohort.eq('external').sum()),
                  temperature_fit='Class-balanced NLL on five official group OOF folds, applied to nonofficial rows',
                  temperatures=temperatures, independent_source=HELD_SOURCE,
                  independent_replay=held_trace is not None, cohorts=cohorts)
    output.mkdir(parents=True, exist_ok=True)
    for arm, logp in heads.items():
        logp.to_parquet(output / f'{arm}_logp.parquet')
    rows.to_parquet(output / 'paired_predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Style mixture diagnostic complete', len(rows), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('official', 'public', 'audit', 'pool', 'style', 'output'):
        cli.add_argument(f'--{name}', type=Path, required=True)
    cli.add_argument('--held-trace', type=Path)
    run(**vars(cli.parse_args()))
