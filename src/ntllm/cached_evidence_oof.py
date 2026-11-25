"""Cross-fit broad readers from frozen external evidence and raw likelihood features."""

import json

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .kiro_pb115_oof import GLM, load_features, select_original_route
from .member_comp_replay import match_temperature
from .peer_context_report import paired_stats, read_ids, read_trace, same_ids
from .sibling_odds_replay import replay_predictions


WEIGHT = .15
METHODS = ('linear', 'boosted')
ARMS = ('external_linear', 'external_boosted', 'pb54_linear', 'pb115_linear', 'pb115_boosted')


def evidence_table(fingerprint, roberta_file, required, labels):
    manifest = pd.read_parquet(fingerprint / 'sample_manifest.parquet')
    if not manifest.id.is_unique:
        raise ValueError('Fingerprint manifest IDs must be unique')
    expected = np.flatnonzero(manifest.partition.ne('external'))
    roberta = read_ids(roberta_file)
    if not roberta.columns.is_unique or set(roberta.columns) != set(labels):
        raise ValueError('RoBERTa is missing or duplicating official model columns')
    matrices, audits = {}, {}
    for name in ('states_only', 'states_forecasts'):
        path = fingerprint / f'{name}_external_only.npz'
        with np.load(path, allow_pickle=False) as bundle:
            positions, columns, logits = (bundle[k] for k in ('rows', 'labels', 'logits'))
        if (not np.issubdtype(positions.dtype, np.integer) or len(np.unique(positions)) != len(positions)
                or not np.array_equal(np.sort(positions), expected)):
            raise ValueError(f'{name}: raw score positions differ from the original query manifest')
        if len(columns) != len(labels) or set(columns) != set(labels) or logits.shape != (len(positions), len(labels)):
            raise ValueError(f'{name}: invalid logits shape or labels')
        supplied = manifest.iloc[positions].set_index('id')
        common = supplied.loc[required.index, ['model', 'prompt_id']]
        if any(not common[key].astype(str).equals(required[key].astype(str)) for key in common):
            raise ValueError(f'{name}: cached labels or prompts disagree with official or historical truth')
        frame = pd.DataFrame(logits, index=supplied.index, columns=columns)
        matrices[name] = frame.loc[required.index, labels].to_numpy(dtype=np.float64)
        audits[name] = dict(total_rows=len(frame), aligned_rows=len(required))
    if not required.index.isin(roberta.index).all():
        raise ValueError('RoBERTa external-only scores omit evaluation rows')
    matrices['roberta'] = roberta.loc[required.index, labels].to_numpy(dtype=np.float64)
    audits['roberta'] = dict(total_rows=len(roberta), aligned_rows=len(required))
    blocks = [np.log1p(required.n_words.to_numpy())[:, None]]
    for scores in matrices.values():
        if not np.isfinite(scores).all():
            raise ValueError('Frozen external logits contain nonfinite values')
        logits = log_softmax(scores, axis=1)
        best = np.sort(logits, axis=1)[:, -2:]
        entropy = -(np.exp(logits) * logits).sum(axis=1)
        blocks.extend((logits, np.column_stack((entropy, best[:, 1] - best[:, 0]))))
    return np.column_stack(blocks), audits


def score_head(x, y, validation, method, labels):
    if method == 'linear':
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=.08, class_weight='balanced', max_iter=1500)
        model.fit(scaler.transform(x), y)
        logp = model.predict_log_proba(scaler.transform(validation))
    else:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(objective='multiclass', class_weight='balanced', n_estimators=90, learning_rate=.04,
                              num_leaves=11, min_child_samples=28, reg_alpha=.7, reg_lambda=4.,
                              colsample_bytree=.75, max_bin=63, random_state=112525, n_jobs=4, verbosity=-1)
        model.fit(x, y)
        logp = np.log(np.clip(model.predict_proba(validation), 1e-12, 1.))
    if not np.array_equal(model.classes_, labels) or not np.isfinite(logp).all():
        raise ValueError(f'{method} head lost a class or returned nonfinite scores')
    return logp


def crossfit(blocks, targets, folds, labels, route):
    scores = {arm: np.empty((len(targets), len(labels))) for arm in ARMS}
    for fold in sorted(set(folds)):
        fit = folds != fold
        for arm in ARMS:
            x = blocks[arm.rsplit('_', 1)[0]]
            model = arm.rsplit('_', 1)[1]
            validation = ~fit
            if arm.startswith('pb'):
                scores[arm][validation] = scores['external_' + model][validation]
                validation = validation & route
            if validation.any():
                scores[arm][validation] = score_head(x[fit], targets[fit], x[validation], model, labels)
        print('Finished original outer fold', int(fold), flush=True)
    return scores


def restore_legacy_anchor(features, targets, labels, folds, native, route, expected):
    scores = native.copy()
    for fold in sorted(set(folds)):
        held = folds == fold
        selected = held & route
        scaler = StandardScaler().fit(features[~held])
        model = LogisticRegression(C=.05, max_iter=1000).fit(scaler.transform(features[~held]), targets[~held])
        if not np.array_equal(model.classes_, labels):
            raise ValueError('A Legacy54 fitting fold lost an official class')
        weight = 1.5 / match_temperature(native[held], .5692304418233374)
        scores[selected] += weight * model.predict_log_proba(scaler.transform(features[selected]))
    if not np.array_equal(labels[scores.argmax(1)], expected):
        raise ValueError('Reconstructed Legacy54 scores differ from the original 2,699 decisions')
    return log_softmax(scores, axis=1)


def add_evidence(rows, labels, baseline, candidate_scores, native=None, trace=None, texts=None):
    rows = rows.assign(baseline=baseline)
    target = rows.model.to_numpy()
    if native is not None:
        base_stage = native
    else:
        glm = trace['glm_winner'] | trace['fable_guard']
        base_stage = np.where(glm[:, None], trace['glm_logp'], trace['sibling_logp'])
        if not np.array_equal(replay_predictions(trace, texts, weight=0), baseline):
            raise ValueError('Frozen historical baseline does not reproduce')
    base_top3 = labels[np.argsort(-base_stage, axis=1, kind='stable')[:, :3]]
    rows['base_top3_contains_true'] = (base_top3 == target[:, None]).any(1)
    for arm, scores in candidate_scores.items():
        ranks = np.argsort(-scores, axis=1, kind='stable')
        rows[arm + '_standalone'] = labels[ranks[:, 0]]
        rows[arm + '_top3'] = (labels[ranks[:, :3]] == target[:, None]).any(1)
        if native is not None:
            rows[arm] = labels[((1. - WEIGHT) * native + WEIGHT * scores).argmax(1)]
        else:
            logits = pd.DataFrame(scores, index=rows.index, columns=labels)
            rows[arm] = replay_predictions(trace, texts, external_logits=logits, weight=WEIGHT)
    return rows


def describe(rows, arms, draws):
    missed = rows.baseline.ne(rows.model) & ~rows.base_top3_contains_true
    report = dict(rows=len(rows), baseline_wrong_outside_top3=int(missed.sum()), comparisons={})
    reference = ['baseline'] + (['legacy54'] if 'legacy54' in rows else [])
    if 'legacy54' in rows:
        report['legacy54_vs_native'] = paired_stats(rows, 'baseline', 'legacy54', draws)
    for arm in arms:
        correct = rows[arm].eq(rows.model)
        report['comparisons'][arm] = dict(
            true_class_top3_among_base_misses=int((missed & rows[arm + '_top3']).sum()),
            true_class_top1_among_base_misses=int((missed & rows[arm + '_standalone'].eq(rows.model)).sum()),
            fixed_by_label=rows.model[correct & rows.baseline.ne(rows.model)].value_counts().to_dict(),
            harmed_by_label=rows.model[~correct & rows.baseline.eq(rows.model)].value_counts().to_dict(),
            paired=[paired_stats(rows, first, arm, draws) for first in reference],
        )
    return report


def run(official, native_file, legacy_file, fingerprint, roberta, features, historical, output, draws=3000):
    train = pd.read_csv(official, dtype={'id': str, 'prompt_id': str}).set_index('id', verify_integrity=True)
    train = train.assign(n_words=train.text.str.split().str.len())
    labels = np.sort(train.model.unique())
    if len(train) != 2699 or len(labels) != 18:
        raise ValueError('Expected the original complete 18-class official train set')
    public = read_ids(historical / 'public_truth.parquet')
    needed = pd.concat([train, public], verify_integrity=True)
    observed, sources = evidence_table(fingerprint, roberta, needed, labels)
    ext, public_features = observed[:len(train)], observed[len(train):]
    base_table, legacy = read_ids(native_file), read_ids(legacy_file)
    for name, table in [('Native OOF', base_table), ('Legacy54 OOF', legacy)]:
        same_ids(table.index, train.index, name)
        if any(not table.loc[train.index, key].astype(str).equals(train[key].astype(str))
               for key in ('prompt_id', 'model')):
            raise ValueError(f'{name}: label or prompt identity differs')
    base_table, legacy = base_table.loc[train.index], legacy.loc[train.index]
    folds = base_table.fold.to_numpy()
    if (len(set(folds)) != 5 or not np.array_equal(folds, legacy.fold.to_numpy())
            or not base_table.groupby('prompt_id').fold.nunique().eq(1).all()):
        raise ValueError('Native and Legacy54 must share the original five prompt-disjoint folds')
    native = log_softmax(base_table[[f'reference__{label}' for label in labels]].to_numpy(), axis=1)
    if not np.isfinite(native).all():
        raise ValueError('Native OOF inputs contain nonfinite values')
    pb54, pb115, tokens = load_features(features, train.index)
    costs = tokens + 30
    glm = pd.Index(labels).get_indexer(GLM)
    route = select_original_route(native, folds, glm, costs)
    if route.sum() != 587 or costs[route].sum() != 87409 or not np.array_equal(route, legacy.routed.to_numpy()):
        raise ValueError('The unchanged GLM route must match the audited 587-row token budget')
    branches = {'external': ext, 'pb54': np.column_stack([ext, pb54]), 'pb115': np.column_stack([ext, pb115])}
    predictions = crossfit(branches, train.model.to_numpy(), folds, labels, route)
    old = labels[native.argmax(1)]
    rows = train[['model', 'prompt_id', 'n_words']].assign(legacy54=legacy.legacy54, fold=folds, routed=route)
    rows = add_evidence(rows, labels, old, predictions, native=native)
    legacy_anchor = restore_legacy_anchor(pb54, train.model.to_numpy(), labels, folds, native, route, rows.legacy54)
    legacy_top3 = labels[np.argsort(-legacy_anchor, axis=1, kind='stable')[:, :3]]
    rows['legacy_top3_contains_true'] = (legacy_top3 == rows.model.to_numpy()[:, None]).any(1)
    for method in METHODS:
        arm = 'external_' + method
        rows[f'legacy_{arm}'] = labels[((1. - WEIGHT) * legacy_anchor + WEIGHT * predictions[arm]).argmax(1)]
    official_results = {}
    for subset, mask in [('all', np.ones(len(rows), dtype=bool)), ('short', rows.n_words.lt(100)),
                         ('routed', route), ('unrouted', ~route)]:
        panel = rows.loc[mask]
        entry = describe(panel, ARMS, draws)
        missed = panel.legacy54.ne(panel.model) & ~panel.legacy_top3_contains_true
        entry['legacy54_extension'] = dict(missed=int(missed.sum()), arms={
            arm: dict(new_top3=int((missed & panel[f'{arm}_top3']).sum()),
                      paired=paired_stats(panel, 'legacy54', f'legacy_{arm}', draws))
            for arm in ('external_linear', 'external_boosted')})
        entry['incremental'] = [paired_stats(panel, a, b, draws) for a, b in (
            ('external_linear', 'pb54_linear'), ('pb54_linear', 'pb115_linear'),
            ('external_boosted', 'pb115_boosted'), ('external_linear', 'external_boosted'),
        )]
        official_results[subset] = entry
    trace, baseline = read_trace(historical, public)
    public_heads = {
        'external_' + method: score_head(ext, train.model.to_numpy(), public_features, method, labels)
        for method in METHODS
    }
    public_rows = add_evidence(public[['model', 'prompt_id', 'n_words', 'cohort']], labels, baseline, public_heads,
                               trace=trace, texts=public.text)
    historical_results = {}
    for cohort, group in public_rows.groupby('cohort'):
        historical_results[cohort] = {'all': describe(group, public_heads, draws)}
        if group.n_words.lt(100).any():
            historical_results[cohort]['short'] = describe(group[group.n_words.lt(100)], public_heads, draws)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'official_predictions.parquet')
    public_rows.to_parquet(output / 'historical_predictions.parquet')
    changes = (rows[list(ARMS)].ne(rows.baseline, axis=0).any(axis=1)
               | rows[['legacy_external_linear', 'legacy_external_boosted']].ne(rows.legacy54, axis=0).any(axis=1))
    rows.loc[changes].to_parquet(output / 'official_changed_rows.parquet')
    report = dict(weight=WEIGHT, external_archives=sources, routed=int(route.sum()),
                  estimated_positions=int(costs[route].sum()), official=official_results, historical=historical_results,
                  fit_scope='Fit all official complement rows with frozen external/raw PB; ignore Native and route when fitting',
                  deployment_scope='Evaluate PB heads only for the fixed original route; use external counterpart elsewhere',
                  limitations=['Native is an evaluation-time blend only', 'PB is available on official bodies only',
                               'Historical evaluation uses actual Sibling Odds and the external-only arms'])
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
