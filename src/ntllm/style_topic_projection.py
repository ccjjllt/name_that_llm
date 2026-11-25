"""Learn a train-only topic subspace without changing the original label contrasts."""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.utils.extmath import randomized_svd

from .member_evidence_transfer import read_prvsiyan
from .peer_context_report import paired_stats, read_ids
from .style_distance import fit_head, prepare_inputs
from .style_mixture import add_replay, read_style


ARMS = ('original', 'shuffled_prompt_projection', 'real_prompt_projection')
DIMENSIONS = 32
SEED = 112525


def sample_axes(values, rows, shuffle=False):
    train = rows.reset_index(drop=True)
    group = train.groupby(['cohort', 'model']).indices
    residual = values.copy()
    for position in group.values():
        residual[position] -= values[position].mean(axis=0)

    # Preserve the fitted class centroid span. Learn the removed span only
    # from class-centered, genuinely matched prompt groups in this fit fold.
    by_model = np.stack([values[rows.model.eq(model).to_numpy()].mean(axis=0)
                         for model in sorted(rows.model.unique())])
    singular, vectors = np.linalg.svd(by_model[1:] - by_model[:1], full_matrices=False)[1:]
    author = vectors[singular > max(singular[0] * 1e-6, 1e-9)]
    prompt = train.prompt_id.to_numpy(copy=True)
    if shuffle:
        rng = np.random.default_rng(SEED)
        length = np.searchsorted([100, 180], train.n_words, side='right')
        for position in train.groupby(['cohort', 'model', length]).indices.values():
            prompt[position] = rng.permutation(prompt[position])
    key = train.assign(prompt=prompt).groupby(['cohort', 'prompt'], sort=True).indices
    labels = train.model.to_numpy()
    usable = [pos for pos in key.values() if len(np.unique(labels[pos])) >= 2]
    if len(usable) < DIMENSIONS:
        raise ValueError('The matched source does not support the prespecified topic rank')
    centroids = np.stack([residual[pos].mean(axis=0) for pos in usable])
    centroids -= centroids.mean(axis=0)
    centroids -= (centroids @ author.T) @ author
    _, strength, axes = randomized_svd(centroids, n_components=DIMENSIONS, random_state=SEED, n_iter=4)
    if not np.isfinite(axes).all() or np.max(np.abs(axes @ author.T)) > 2e-4:
        raise ValueError('Topic directions must be orthogonal to fitted model contrasts')
    return axes, dict(matched_groups=len(usable), matched_rows=sum(map(len, usable)),
                      author_rank=len(author), weakest_topic_singular=float(strength[-1]))


def run(official, public, audit, pool, style, prior, output, independent=None, draws=3000):
    frame, labels = prepare_inputs(official, public, audit, pool)
    values, scores = read_style(style, frame, labels)
    external = np.flatnonzero(frame.cohort.eq('external'))
    official_rows = np.flatnonzero(frame.cohort.eq('official'))
    transfer = np.flatnonzero(~frame.cohort.isin(['official', 'external']))
    evaluated = np.flatnonzero(~frame.cohort.eq('external'))
    split = GroupKFold(5).split(official_rows, groups=frame.prompt_id.iloc[official_rows])
    folds = [(np.r_[external, official_rows[fit]], official_rows[valid]) for fit, valid in split]
    folds.append((np.r_[external, official_rows], transfer))
    folded = {arm: np.full((len(frame), len(labels)), np.nan) for arm in ARMS[1:]}
    fits = []
    for index, (fit, valid) in enumerate(folds):
        record = dict(split=index, fit_rows=len(fit), eval_rows=len(valid), axes={})
        for name, shuffled in [('shuffled_prompt_projection', True), ('real_prompt_projection', False)]:
            axes, description = sample_axes(values[fit], frame.iloc[fit], shuffled)
            projected = values - (values @ axes.T) @ axes
            model = fit_head(projected, frame, fit)
            if model.classes_.tolist() != labels:
                raise ValueError('Topic fit changed the class order')
            folded[name][valid] = model.predict_log_proba(projected[valid])
            record['axes'][name] = description
        fits.append(record)
        print('Finished projection fold', index, record, flush=True)

    heads = {'original': scores}
    for arm, matrix in folded.items():
        if not np.isfinite(matrix[evaluated]).all():
            raise ValueError('Every evaluation query must have finite projection scores')
        heads[arm] = pd.DataFrame(matrix[evaluated], index=frame.index[evaluated], columns=labels)
    rows = frame.iloc[evaluated].drop(columns='text').copy()
    truth = read_prvsiyan(audit, prior)
    rows['strict'] = rows.index.isin(truth.index[truth.strict_topic_holdout])
    rows['baseline'] = pd.Series(index=rows.index, dtype=object)
    rows['baseline_top3_miss'] = False
    for name, matrix in heads.items():
        rows[name] = matrix.idxmax(1)
        sorted_names = matrix.columns.to_numpy()[np.argsort(-matrix.to_numpy(), axis=1, kind='stable')[:, :3]]
        rows[f'{name}_top3'] = (sorted_names == rows.model.to_numpy()[:, None]).any(1)
        rows[f'{name}_blend'] = pd.Series(index=rows.index, dtype=object)
    add_replay(rows, heads, public, read_ids(public / 'public_truth.parquet'))
    if independent:
        from .member_evidence_transfer import verify_manifest

        verify_manifest(independent / 'probe_manifest.json', truth)
        base = add_replay(rows, heads, independent, truth)
        if not np.array_equal(base, read_ids(independent / 'frozen_predictions.parquet').loc[truth.index].sibling_odds):
            raise ValueError('Independent scores do not reproduce original predictions')
    panels = dict(tuple(rows.groupby('cohort')))
    panels['prvsiyan_strict'] = rows.loc[rows.strict]
    report = {}
    for name, panel in panels.items():
        for scope, part in [('all', panel), ('short', panel.loc[panel.n_words.lt(100)])]:
            if part.empty:
                continue
            result = dict(standalone=[paired_stats(part, 'original', arm, draws) for arm in ARMS[1:]],
                          projected_vs_shuffled=paired_stats(part, ARMS[1], ARMS[2], draws),
                          top3={arm: int(part[f'{arm}_top3'].sum()) for arm in ARMS})
            if part.baseline.notna().all():
                result['blend'] = [paired_stats(part, 'baseline', f'{arm}_blend', draws) for arm in ARMS]
                result['rescued_candidates'] = {arm: int((part.baseline_top3_miss & part[f'{arm}_top3']).sum())
                                                for arm in ARMS}
            report[f'{name}/{scope}'] = result
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'predictions.parquet')
    for arm in ARMS[1:]:
        heads[arm].to_parquet(output / f'{arm}_logp.parquet')
    summary = dict(projection_dimensions=DIMENSIONS, split_fits=fits, panels=report)
    (output / 'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary
