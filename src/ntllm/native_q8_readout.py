"""Readout study on the public 0.679/0.680 Qwen3-Embedding-8B multi-depth LoRA OOF.

The public five-fold expert gives, per official row, held-out `plain` and raw
`centered` head logits (2699 x 18). Its deployed readout (stage19 `expert_logp`)
mixes calibrated plain and group-centered log-probabilities with a constant
weight for every inferred group of size 2..18. This module replays that readout
on prompt-grouped OOF cohorts and compares alternatives:

* fixed weight grid and the original fold-specific inner selection;
* group-size-aware weight  w(k) = w_max * (k - 1) / (k - 1 + tau);
* leave-one-out centering (c_i - mean of the other members);
* shared-prompt Gaussian readout on the 36-d logit vector (exclusive or not).

Every learned scalar or Gaussian fit for outer fold f uses only OOF rows of the
other four folds (prompt-disjoint models, labels never from fold f). Groups of
fold f are inferred from text with TF-IDF fitted on fold f's training texts.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax, logsumexp

from . import gaussian_head
from .q8_evaluation import fit_grouping_vectorizer, make_cohorts, recover_groups

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'runs/public_0680_20261109/native_q8_0679_meta/ntl-native-q8-multidepth-fivefold-0679'
WORD_BINS = ((0, 60, '<60'), (60, 100, '60-99'), (100, 160, '100-159'), (160, 10 ** 9, '>=160'))


# ---------------------------------------------------------------- data ----
def load_study(assets=ASSETS):
    manifest = json.loads((assets / 'manifest.json').read_text())
    labels = manifest['classes']
    frame = pd.read_csv(ROOT / 'data/train.csv')
    tokens = np.load(assets / 'tokens.npz')
    if not (frame.id.to_numpy() == tokens['ids']).all():
        raise ValueError('OOF row order differs from train.csv')
    oof = np.load(assets / 'oof_reference.npz')
    states = [json.loads((assets / f'calibration/fold{f}.json').read_text())['models']['qwen8'] for f in range(5)]
    bias = np.zeros((5, 18))
    bias_source = 'zeros'
    try:
        from safetensors import safe_open
        for f in range(5):
            with safe_open(str(assets / f'fold{f}/trainables.safetensors'), 'np') as archive:
                bias[f] = archive.get_tensor('centered.bias')
        bias_source = 'trainables.safetensors centered.bias'
    except Exception:  # missing or partially downloaded trainables: biases stay zero
        bias[:] = 0.
    y = frame.model.map({m: i for i, m in enumerate(labels)}).to_numpy()
    return dict(frame=frame, labels=labels, y=y, folds=tokens['folds'].astype(int),
                plain=oof['plain'].astype(np.float64), centered=oof['centered'].astype(np.float64),
                states=states, bias=bias, bias_source=bias_source)


# ------------------------------------------------------------ balancing ----
def balance(probabilities, alpha=.25):
    """Public alpha-blended Sinkhorn column balancing (one pass per query set)."""
    result = np.asarray(probabilities, dtype=np.float64).copy()
    if alpha == 0 or len(result) < 18:
        return result
    target = (1 - alpha) * result.sum(0) + alpha * len(result) / 18
    result = result.clip(1e-15)
    for iteration in range(5000):
        result *= target / result.sum(0)
        result /= result.sum(1, keepdims=True)
        if iteration >= 49 and np.abs(result.sum(0) - target).max() <= 1e-7 * len(result):
            break
    return result


def balanced_accuracy(y, prediction):
    present = np.unique(y)
    return float(np.mean([(prediction[y == c] == c).mean() for c in present]))


# ------------------------------------------------------------- readouts ----
def group_stats(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return inverse, counts[inverse]


def centered_logp(centered, bias, groups, scale, loo=False):
    inverse, size = group_stats(groups)
    mean = np.zeros((inverse.max() + 1, centered.shape[1]))
    np.add.at(mean, inverse, centered)
    total = mean[inverse]
    mean = total / size[:, None]
    relative = centered - mean
    if loo:
        relative = relative * np.where(size > 1, size / np.maximum(size - 1, 1), 1.)[:, None]
    return log_softmax((relative + bias) * scale, axis=1), size


def mixed_logp(plain_lp, centered_lp, size, weight_fn):
    weight = np.where((size >= 2) & (size <= 18), weight_fn(size), 0.)[:, None]
    return log_softmax((1 - weight) * plain_lp + weight * centered_lp, axis=1)


def kaware(w_max, tau):
    return lambda k: w_max * (k - 1) / (k - 1 + tau)


def constant(w):
    return lambda k: np.full(np.shape(k), float(w))


KAWARE_GRID = [(w, t) for w in (.35, .5, .65, .8) for t in (0., .5, 1., 2., 4., 8.)]
CONSTANT_GRID = (0., .15, .35, .5, .65, .8)


def gaussian_features(plain, centered, plain_scale):
    return np.hstack([log_softmax(plain * plain_scale, axis=1), centered])


# --------------------------------------------------------- the main loop ----
def fold_queries(study, cohorts, grouping='inferred'):
    """Per (cohort, fold): query positions and their groups (no labels used)."""
    frame, folds = study['frame'], study['folds']
    fitted = {f: fit_grouping_vectorizer(frame.text[folds != f].tolist()) for f in range(5)}
    queries = {}
    for name, positions, _ in cohorts:
        for f in range(5):
            rows = positions[folds[positions] == f]
            if not len(rows):
                continue
            if grouping == 'oracle':
                groups = pd.factorize(frame.prompt_id.iloc[rows])[0]
            else:
                groups, _ = recover_groups(frame.text.iloc[rows].tolist(), fitted=fitted[f])
            queries[(name, f)] = (rows, np.asarray(groups))
    return queries


def base_views(study, rows, groups, fold):
    state = study['states'][fold]
    plain_lp = log_softmax(study['plain'][rows] * state['plain_scale'], axis=1)
    cen_lp, size = centered_logp(study['centered'][rows], study['bias'][fold], groups, state['centered_scale'])
    loo_lp, _ = centered_logp(study['centered'][rows], study['bias'][fold], groups, state['centered_scale'], loo=True)
    return plain_lp, cen_lp, loo_lp, size


def score(study, queries, cohort_names, method, alpha=.25):
    """Pool balanced fold predictions per cohort and return BA per cohort."""
    out = {}
    for name in cohort_names:
        ys, preds = [], []
        for f in range(5):
            if (name, f) not in queries:
                continue
            rows, groups = queries[(name, f)]
            lp = method(rows, groups, f)
            preds.append(balance(np.exp(lp), alpha).argmax(1))
            ys.append(study['y'][rows])
        out[name] = balanced_accuracy(np.concatenate(ys), np.concatenate(preds))
    return out


def predictions(study, queries, name, method, alpha=.25):
    rows_all, pred_all = [], []
    for f in range(5):
        if (name, f) not in queries:
            continue
        rows, groups = queries[(name, f)]
        rows_all.append(rows)
        pred_all.append(balance(np.exp(method(rows, groups, f)), alpha).argmax(1))
    return np.concatenate(rows_all), np.concatenate(pred_all)


def crossfit_select(study, queries, selection, candidates, make_method):
    """For each outer fold choose the candidate maximizing selection BA on the other folds."""
    table = {}
    for c in candidates:
        method = make_method(c)
        per_fold = {}
        for f in range(5):
            # Score this candidate on fold f alone, pooled BA computed below over other folds.
            per_fold[f] = {}
            for name in selection:
                if (name, f) in queries:
                    rows, groups = queries[(name, f)]
                    per_fold[f][name] = (study['y'][rows], balance(np.exp(method(rows, groups, f))).argmax(1))
        table[c] = per_fold
    chosen = {}
    for f in range(5):
        def objective(c):
            values = []
            for name in selection:
                parts = [table[c][g][name] for g in range(5) if g != f and name in table[c][g]]
                values.append(balanced_accuracy(np.concatenate([p[0] for p in parts]),
                                                np.concatenate([p[1] for p in parts])))
            return float(np.mean(values))
        chosen[f] = max(candidates, key=lambda c: (round(objective(c), 10), -candidates.index(c)))
    return chosen


def gaussian_fits(study, cfg):
    """Outer-fold Gaussian fits on the other four folds' OOF logits with true prompts."""
    frame, folds = study['frame'], study['folds']
    fits = {}
    for f in range(5):
        train = folds != f
        X = np.vstack([gaussian_features(study['plain'][i:i + 1], study['centered'][i:i + 1],
                                         study['states'][folds[i]]['plain_scale']) for i in np.flatnonzero(train)])
        fits[f] = gaussian_head.fit(X, study['y'][train], frame.prompt_id[train].to_numpy(), cfg)
    return fits


def paired_bootstrap(study, rows, first, second, draws=2000, seed=0):
    prompts = study['frame'].prompt_id.to_numpy()[rows]
    y = study['y'][rows]
    unique, inverse = np.unique(prompts, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        take = np.concatenate([members[g] for g in rng.integers(0, len(unique), len(unique))])
        deltas.append(balanced_accuracy(y[take], second[take]) - balanced_accuracy(y[take], first[take]))
    deltas = np.asarray(deltas)
    return dict(delta=balanced_accuracy(y, second) - balanced_accuracy(y, first),
                ci95=[float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))],
                p_le_zero=float((deltas <= 0).mean()))


def run(out, grouping='inferred'):
    out.mkdir(parents=True, exist_ok=True)
    study = load_study()
    frame, labels = study['frame'], study['labels']
    cohorts = make_cohorts(frame, labels, seeds=(42, 43), imbalance_suite='rotating_v1')
    rng = np.random.default_rng(42)
    sparse12 = []
    for positions in frame.groupby('prompt_id', sort=True).indices.values():
        sparse12.extend(rng.choice(positions, min(len(positions), int(rng.integers(1, 3))), replace=False).tolist())
    cohorts.append(('sparse1_2_seed42', np.sort(sparse12), dict(kind='sparse1_2', seed=42)))
    names = [c[0] for c in cohorts]
    main = ['full', 'one_per_prompt_seed42', 'sparse1_2_seed42', 'sparse1_3_seed42', 'sparse1_6_seed42',
            'one_per_prompt_seed43', 'sparse1_3_seed43', 'sparse1_6_seed43',
            'class_imbalanced_seed42', 'class_imbalanced_seed43']
    imbalance = [n for n in names if n.startswith('imbalance_')]
    selection = ['full', 'sparse1_3_seed42', 'sparse1_6_seed42', 'one_per_prompt_seed42']
    queries = fold_queries(study, cohorts, grouping)

    sizes = {}
    for name in main:
        counts = np.concatenate([np.bincount(group_stats(queries[(name, f)][1])[0]) for f in range(5)
                                 if (name, f) in queries])
        sizes[name] = {str(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))}

    cache = {}

    def views(rows, groups, f):
        key = (rows.tobytes(), groups.tobytes(), f)
        if key not in cache:
            cache[key] = base_views(study, rows, groups, f)
        return cache[key]

    def step(w):
        return lambda rows, groups, f: mixed_logp(views(rows, groups, f)[0], views(rows, groups, f)[1],
                                                  views(rows, groups, f)[3], constant(w))

    def step_loo(w):
        return lambda rows, groups, f: mixed_logp(views(rows, groups, f)[0], views(rows, groups, f)[2],
                                                  views(rows, groups, f)[3], constant(w))

    def kaw(params):
        return lambda rows, groups, f: mixed_logp(views(rows, groups, f)[0], views(rows, groups, f)[1],
                                                  views(rows, groups, f)[3], kaware(*params))

    def original(rows, groups, f):
        return step(study['states'][f]['selected_centered_weight'])(rows, groups, f)

    methods = {'plain_calibrated': step(0.), 'original_inner_selected': original}
    for w in CONSTANT_GRID[1:]:
        methods[f'constant_w{w:g}'] = step(w)
        methods[f'loo_w{w:g}'] = step_loo(w)
    for params in [(.5, 1.), (.65, 1.), (.8, 2.), (.65, 2.)]:
        methods[f'kaware_w{params[0]:g}_tau{params[1]:g}'] = kaw(params)

    chosen_const = crossfit_select(study, queries, selection, list(CONSTANT_GRID), step)
    methods['crossfit_constant'] = lambda rows, groups, f: step(chosen_const[f])(rows, groups, f)
    chosen_k = crossfit_select(study, queries, selection, KAWARE_GRID, kaw)
    methods['crossfit_kaware'] = lambda rows, groups, f: kaw(chosen_k[f])(rows, groups, f)

    # Gaussian readout on the logit vector, fitted on other folds' OOF rows.
    gcfg = dict(red='none', d=36, n_exact=3, iters=5, repel=True)
    fits = gaussian_fits(study, gcfg)

    def gauss(T, inj):
        def method(rows, groups, f):
            X = gaussian_features(study['plain'][rows], study['centered'][rows], study['states'][f]['plain_scale'])
            dense = pd.factorize(groups)[0]
            return gaussian_head.scores(X, dense, fits[f], T1=T, Tg=T, cfg=dict(inj=inj))
        return method

    gauss_grid = [(T, inj) for T in (1., 2., 3., 5.) for inj in (False, True)]
    chosen_g = crossfit_select(study, queries, selection, gauss_grid, lambda c: gauss(*c))
    methods['crossfit_gaussian'] = lambda rows, groups, f: gauss(*chosen_g[f])(rows, groups, f)
    for T in (2., 3.):
        methods[f'gaussian_T{T:g}_nonexclusive'] = gauss(T, False)
        methods[f'gaussian_T{T:g}_exclusive'] = gauss(T, True)

    def blend(a, base, other):
        return lambda rows, groups, f: log_softmax((1 - a) * base(rows, groups, f) + a * other(rows, groups, f), axis=1)

    blend_grid = (0., .25, .5, .75)
    chosen_b = crossfit_select(study, queries, selection, list(blend_grid),
                               lambda a: blend(a, methods['crossfit_kaware'], methods['crossfit_gaussian']))
    methods['crossfit_kaware_plus_gaussian'] = lambda rows, groups, f: blend(
        chosen_b[f], methods['crossfit_kaware'], methods['crossfit_gaussian'])(rows, groups, f)

    results = {}
    for key, method in methods.items():
        values = score(study, queries, main + imbalance, method)
        imb = [values[n] for n in imbalance]
        results[key] = dict(main={n: values[n] for n in main},
                            imbalance=dict(mean=float(np.mean(imb)), min=float(np.min(imb)),
                                           severe_mean=float(np.mean([values[n] for n in imbalance if '_severe_' in n])),
                                           mild_mean=float(np.mean([values[n] for n in imbalance if '_mild_' in n]))))
        results[key]['selection_mean'] = float(np.mean([values[n] for n in selection]))

    # Paired prompt bootstrap vs the deployed readout on key cohorts.
    bootstrap = {}
    for candidate in ['crossfit_kaware', 'crossfit_gaussian', 'crossfit_kaware_plus_gaussian', 'crossfit_constant']:
        bootstrap[candidate] = {}
        for name in ['full', 'sparse1_2_seed42', 'sparse1_3_seed42', 'sparse1_6_seed42', 'one_per_prompt_seed42']:
            rows, first = predictions(study, queries, name, methods['original_inner_selected'])
            rows2, second = predictions(study, queries, name, methods[candidate])
            assert (rows == rows2).all()
            bootstrap[candidate][name] = paired_bootstrap(study, rows, first, second)

    # Length and per-class view on the full cohort.
    words = frame.text.str.split().str.len().to_numpy()
    detail = {}
    for key in ['plain_calibrated', 'original_inner_selected', 'crossfit_kaware', 'crossfit_gaussian',
                'crossfit_kaware_plus_gaussian']:
        rows, pred = predictions(study, queries, 'full', methods[key])
        y = study['y'][rows]
        detail[key] = dict(
            length={label: balanced_accuracy(y[m], pred[m]) for lo, hi, label in WORD_BINS
                    for m in [(words[rows] >= lo) & (words[rows] < hi)] if m.any()},
            per_class={labels[c]: float((pred[y == c] == c).mean()) for c in range(18)})

    report = dict(
        source='Public Qwen3-Embedding-8B multi-depth LoRA five-fold OOF (ntl-native-q8-multidepth-fivefold-0679)',
        grouping=grouping, bias_source=study['bias_source'], alpha=.25,
        selection_cohorts=selection, group_size_counts=sizes,
        crossfit_choices=dict(constant=chosen_const, kaware=chosen_k, gaussian=chosen_g, blend=chosen_b),
        gaussian_config=gcfg, results=results, paired_bootstrap_vs_original=bootstrap, full_detail=detail)
    (out / f'readout_{grouping}.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    rows = [dict(method=k, selection_mean=v['selection_mean'], **v['main'], imbalance_mean=v['imbalance']['mean'],
                 imbalance_severe=v['imbalance']['severe_mean'], imbalance_min=v['imbalance']['min'])
            for k, v in results.items()]
    table = pd.DataFrame(rows).set_index('method')
    table.to_csv(out / f'readout_{grouping}.csv')
    return report, table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/public_0680_20261109/research')
    parser.add_argument('--grouping', choices=['inferred', 'oracle'], default='inferred')
    args = parser.parse_args()
    _, table = run(args.out, args.grouping)
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.precision', 4):
        print(table)


if __name__ == '__main__':
    main()
