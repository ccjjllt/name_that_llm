"""Compare gradient-selected and randomly selected style training examples."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .member_evidence_transfer import verify_manifest
from .peer_context_report import paired_stats, read_ids, same_ids
from .style_distance import HELD_SOURCE, PRIMARY_BLEND, fit_head, prepare_inputs
from .style_mixture import add_replay, read_style


RANDOM_SEEDS = (13579, 24680, 112525)
RANDOM_ARMS = tuple(f'random_{seed}' for seed in RANDOM_SEEDS)
ARMS = ('all_external', *RANDOM_ARMS, 'relevance')
STRATA = ['model', 'length_band']
SOURCES = STRATA + ['source_dataset']
QUOTA_SEED = 241125
SHORT_WEIGHT = 2.
RIVAL_SHRINK = 20.


def external_metadata(pool, sampled):
    columns = ['id', 'model', 'prompt_canon', 'source_dataset']
    original = pd.read_parquet(pool, columns=columns)
    original = original.loc[original.id.isin(sampled.index)].set_index('id')
    same_ids(original.index, sampled.index, 'External source lookup')
    original = original.loc[sampled.index]
    if (not original.model.eq(sampled.model).all()
            or not original.prompt_canon.eq(sampled.prompt_id).all()):
        raise ValueError('External lookup disagrees with the frozen model and prompt assignments')
    source = original.source_dataset.str.strip().str.lower().str.removeprefix('https://www.kaggle.com/datasets/')
    if source.isna().any() or source.eq('').any():
        raise ValueError('Every sampled external passage needs source provenance')
    return sampled.assign(length_band=np.searchsorted([100, 180], sampled.n_words, side='right'),
                          source_dataset=source)


def source_quotas(rows):
    counts = rows.groupby(SOURCES).size().rename('available').reset_index()
    rng = np.random.default_rng(QUOTA_SEED)
    counts = counts.assign(quota=counts.available // 2, priority=rng.random(len(counts)))
    odd = counts.available.mod(2).eq(1)
    remainder = counts.loc[odd].sort_values('priority')
    take = remainder.groupby(STRATA).cumcount() < (remainder.groupby(STRATA).available.transform('size') + 1) // 2
    counts.loc[remainder.index[take], 'quota'] += 1
    return counts.drop(columns='priority')


def select(rows, priority):
    ordered = rows.assign(priority=priority).sort_values(['priority', 'id'], ascending=[False, True])
    rank = ordered.groupby(SOURCES, sort=False).cumcount()
    return pd.Series(rows.index.isin(ordered.index[rank < ordered.quota]), index=rows.index)


def residual_and_rival(warm, values, codes):
    probabilities = warm.predict_proba(values)
    row = np.arange(len(codes))
    probability_true = probabilities[row, codes].copy()
    probabilities[row, codes] = -1.
    rival = probabilities.argmax(axis=1)
    probabilities[row, codes] = probability_true - 1.
    return probabilities, rival


def direction(embeddings, residuals, weight):
    gradient = residuals.T @ (embeddings * weight[:, None]) / weight.sum()
    return gradient / max(np.linalg.norm(gradient), 1e-12)


def similarities(embeddings, residuals, gradient):
    inner = np.einsum('ij,ij->i', embeddings @ gradient.T, residuals)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(residuals, axis=1)
    return inner / np.maximum(norms, 1e-12)


def relevance(warm, values, frame, labels, official_fit, external):
    candidates, reference = values[external], values[official_fit]
    candidate_labels = np.searchsorted(labels, frame.model.iloc[external])
    reference_labels = np.searchsorted(labels, frame.model.iloc[official_fit])
    residual, rival = residual_and_rival(warm, candidates, candidate_labels)
    ref_residual, ref_rival = residual_and_rival(warm, reference, reference_labels)
    weights = np.where(frame.n_words.iloc[official_fit] < 100, SHORT_WEIGHT, 1.)
    scores = np.full(len(external), np.nan)
    for label in range(len(labels)):
        in_class = np.flatnonzero(reference_labels == label)
        where = np.flatnonzero(candidate_labels == label)
        if not len(in_class):
            raise ValueError(f'Missing fit-only reference for {labels[label]}')
        class_direction = direction(reference[in_class], ref_residual[in_class], weights[in_class])
        scores[where] = similarities(candidates[where], residual[where], class_direction)
        for competitor in np.unique(rival[where]):
            pair_reference = in_class[ref_rival[in_class] == competitor]
            if not len(pair_reference):
                continue
            pair_rows = where[rival[where] == competitor]
            pair_direction = direction(reference[pair_reference], ref_residual[pair_reference], weights[pair_reference])
            alignment = similarities(candidates[pair_rows], residual[pair_rows], pair_direction)
            share = len(pair_reference) / (len(pair_reference) + RIVAL_SHRINK)
            scores[pair_rows] = (1 - share) * scores[pair_rows] + share * alignment
    if not np.isfinite(scores).all():
        raise ValueError('All external rows need a fit-only official gradient direction')
    return scores, np.asarray(labels)[rival]


def score(values, frame, labels, all_scores, metadata):
    external = np.flatnonzero(frame.cohort.eq('external'))
    official = np.flatnonzero(frame.cohort.eq('official'))
    held = np.flatnonzero(~frame.cohort.isin(['external', 'official']))
    eval_pos = np.flatnonzero(~frame.cohort.eq('external'))
    warm = fit_head(values, frame, external)
    if list(warm.classes_) != labels:
        raise ValueError('The shared external warm classifier must have all label columns')

    quotas = source_quotas(metadata)
    sample = metadata.join(quotas.set_index(SOURCES), on=SOURCES)
    random = {arm: select(sample, np.random.default_rng(seed).random(len(sample)))
              for arm, seed in zip(RANDOM_ARMS, RANDOM_SEEDS)}
    folds = np.full(len(frame), -1)
    predictions = {arm: np.full((len(frame), len(labels)), np.nan) for arm in ARMS[1:]}
    ledgers = []
    split = GroupKFold(5).split(official, groups=frame.prompt_id.iloc[official])
    panels = [(fold, official[fit], official[test]) for fold, (fit, test) in enumerate(split)]
    panels.append((-1, official, held))
    for fold, train, test in panels:
        gradient_score, rivals = relevance(warm, values, frame, labels, train, external)
        arms = {**random, 'relevance': select(sample, gradient_score)}
        ledger = sample.drop(columns=['available', 'quota']).assign(
            fold=fold, gradient_cosine=gradient_score, warm_rival=rivals)
        for arm, chosen in arms.items():
            actual = sample.assign(chosen=chosen).groupby(SOURCES).chosen.sum().rename('quota')
            if not actual.equals(quotas.set_index(SOURCES).quota):
                raise ValueError(f'Stratum sample budget mismatch for {arm}')
            model = fit_head(values, frame, np.r_[external[chosen.to_numpy()], train])
            if list(model.classes_) != labels:
                raise ValueError(f'Missing label in {arm} classifier')
            predictions[arm][test] = model.predict_log_proba(values[test])
            ledger[arm] = chosen
        folds[test] = fold
        ledgers.append(ledger)
        print('Completed gradient selection fold', fold, 'fit rows', len(train), 'scored rows', len(test), flush=True)

    heads = {'all_external': all_scores}
    for arm, logp in predictions.items():
        if not np.isfinite(logp[eval_pos]).all():
            raise ValueError(f'Missing evaluation scores for {arm}')
        heads[arm] = pd.DataFrame(logp[eval_pos], index=frame.index[eval_pos], columns=labels)
    return heads, pd.Series(folds[eval_pos], index=all_scores.index), pd.concat(ledgers), quotas


def compare(frame, heads, folds, public, held_trace):
    rows = frame.drop(columns='text').loc[folds.index].assign(fold=folds)
    for arm, logp in heads.items():
        rows[arm] = logp.idxmax(axis=1)
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
            arms = (*RANDOM_ARMS, 'all_external')
            entry = dict(cohort=cohort, subset=subset, relevance_vs={
                arm: paired_stats(part, arm, 'relevance', draws=3000) for arm in arms})
            if part.baseline.notna().all():
                entry['blend_vs_sibling'] = {arm: paired_stats(part, 'baseline', f'{arm}_blend', draws=3000)
                                             for arm in ARMS}
                entry['relevance_blend_vs'] = {arm: paired_stats(part, f'{arm}_blend', 'relevance_blend', draws=3000)
                                               for arm in arms}
            report.append(entry)
    return rows, report


def run(official, public, audit, pool, style, output, held_trace=None):
    frame, labels = prepare_inputs(official, public, audit, pool)
    values, all_scores = read_style(style, frame, labels)
    metadata = external_metadata(pool, frame.loc[frame.cohort.eq('external')].drop(columns=['text', 'cohort']))
    heads, folds, selected, quotas = score(values, frame, labels, all_scores, metadata)
    rows, cohorts = compare(frame, heads, folds, public, held_trace)
    report = dict(random_seeds=RANDOM_SEEDS, quota_seed=QUOTA_SEED, length_edges=[100, 180],
                  short_weight=SHORT_WEIGHT, rival_shrink=RIVAL_SHRINK,
                  external_eligible=len(metadata), external_selected_per_arm=int(quotas.quota.sum()),
                  selection_targets='Official training-side prompts only, same-model and warm-rival gradients',
                  warm_reference='Weighted logistic head fitted only on frozen external rows',
                  fixed_blend=PRIMARY_BLEND, independent_source=HELD_SOURCE,
                  independent_replay=held_trace is not None, cohorts=cohorts)
    output.mkdir(parents=True, exist_ok=True)
    for arm, logp in heads.items():
        logp.to_parquet(output / f'{arm}_logp.parquet')
    selected.reset_index().to_parquet(output / 'external_selection.parquet', index=False)
    quotas.to_parquet(output / 'source_quotas.parquet', index=False)
    rows.to_parquet(output / 'paired_predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('official', 'public', 'audit', 'pool', 'style', 'output'):
        cli.add_argument(f'--{name}', type=Path, required=True)
    cli.add_argument('--held-trace', type=Path)
    run(**vars(cli.parse_args()))
