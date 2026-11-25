"""Compare task and neutral response likelihood views without fitting the holdout."""

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from prefix_response_features import one
from .peer_context_report import paired_stats, read_ids, same_ids


MODELS = ('flash', 'glm53')
FOCUS = ('claude-fable-5', 'glm-5.3', 'glm-5.3-flash')
ARMS = {
    'task160': (160, False, False),
    'task128_window': (128, False, False),
    'task128_neutral32': (128, True, False),
    'total_matched_task160': (160, False, False),
    'total_matched_window': (160, False, True),
    'total_matched_neutral': (160, True, True),
}
COMPARISONS = (
    ('task160', 'task128_neutral32'),
    ('task128_window', 'task128_neutral32'),
    ('total_matched_task160', 'total_matched_neutral'),
    ('total_matched_window', 'total_matched_neutral'),
)


def observations(folder, tag, ids):
    table = pd.read_csv(folder / f'{tag}_features.csv', usecols=['id', 'n_tok'], dtype={'id': str})
    expected = [f'{mode}:{row}' for mode in ('task', 'neutral') for row in ids]
    if table.id.tolist() != expected:
        raise ValueError(f'{tag}: scored row IDs differ from the two frozen views')
    with np.load(folder / f'{tag}_scores.npz', allow_pickle=False) as raw:
        offsets, scores, prefixes, tokens = (raw[k] for k in ('offsets', 'scores', 'n_prefix', 'tokens'))
    count = len(ids)
    lengths = np.diff(offsets)
    if (offsets.shape != (2 * count + 1,) or not np.issubdtype(offsets.dtype, np.integer)
            or offsets[0] != 0 or offsets[-1] != len(scores) or scores.shape != (len(tokens), 4)
            or prefixes.shape != (2 * count,) or (prefixes < 1).any()
            or not np.array_equal(lengths, table.n_tok) or (lengths < 1).any()
            or (lengths[:count] > 160).any() or (lengths[count:] > 32).any() or not np.isfinite(scores).all()):
        raise ValueError(f'{tag}: raw scores or token offsets do not match the claimed budget')
    parts = list(zip(np.split(scores, offsets[1:-1]), np.split(tokens, offsets[1:-1]), prefixes, strict=True))
    views = (parts[:count], parts[count:])
    for task, neutral in zip(*views, strict=True):
        if not np.array_equal(task[1][:len(neutral[1])], neutral[1]):
            raise ValueError(f'{tag}: score views changed the actual answer tokens')
    return views


def row_features(blocks, width, chars, diffs):
    tables = {model: one(blocks[model][:width[model]], chars) for model in MODELS}
    return [value for model in MODELS for value in tables[model].values()] + [
        tables[MODELS[0]][key] - tables[MODELS[1]][key] for key in diffs]


def feature_views(traces, chars, diffs):
    usable = np.array([all(len(task[0]) >= 2 * len(neutral[0]) + neutral[2]
                           for task, neutral in bundle.values()) for bundle in traces])
    vectors, costs = {name: [] for name in ARMS}, {name: [] for name in ARMS}
    for row, (bundle, size) in enumerate(zip(traces, chars, strict=True)):
        primary = {model: pair[0][0] for model, pair in bundle.items()}
        neutral = {model: pair[1][0] for model, pair in bundle.items()}
        secondary = {model: len(values) for model, values in neutral.items()}
        for name, (cap, dual, total_match) in ARMS.items():
            use_dual = dual and (not total_match or usable[row])
            prefix = {model: pair[0][2] for model, pair in bundle.items()}
            main = {model: min(cap, len(values)) for model, values in primary.items()}
            if total_match and usable[row]:
                main = {model: length - secondary[model] - bundle[model][1][2]
                        for model, length in main.items()}
            vectors[name].append(row_features(primary, main, size, diffs)
                                 + row_features(neutral if use_dual else primary, secondary, size, diffs))
            body = sum(main.values()) + (sum(secondary.values()) if use_dual else 0)
            pre = sum(prefix.values()) + (sum(pair[1][2] for pair in bundle.values()) if use_dual else 0)
            costs[name].append((body, pre, body + pre))
    costs = {key: np.asarray(values) for key, values in costs.items()}
    if not np.array_equal(costs['task160'][usable, 2], costs['total_matched_neutral'][usable, 2]):
        raise ValueError('The matched-cost dual view spent a different number of token positions')
    return {key: np.asarray(values) for key, values in vectors.items()}, costs, usable


def predict(values, targets, train, evaluation, labels):
    scale = StandardScaler().fit(values[train])
    head = LogisticRegression(C=.05, max_iter=1200).fit(scale.transform(values[train]), targets[train])
    if not np.array_equal(head.classes_, labels):
        raise ValueError('All 18 classes must remain in every fitting fold')
    return head.predict_log_proba(scale.transform(values[evaluation]))


def fit_arms(values, panel, labels, available):
    train = np.flatnonzero(panel.cohort.eq('calibration'))
    heldout = np.flatnonzero(panel.cohort.eq('heldout'))
    groups = panel.prompt_id.iloc[train]
    folds = list(GroupKFold(5).split(train, groups=groups))
    targets = panel.model.to_numpy()
    targets = np.where(panel.cohort.eq('calibration'), targets, '')
    logits = {}
    for name, features in values.items():
        conditional = name.startswith('total_matched')
        eligible_train = available if conditional else np.ones(len(panel), dtype=bool)
        out = logits['task160'].copy() if conditional else np.empty((len(panel), len(labels)))
        for fit, valid in folds:
            a, b = train[fit], train[valid]
            b = b[eligible_train[b]]
            if len(b):
                out[b] = predict(features, targets, a[eligible_train[a]], b, labels)
        b = heldout[eligible_train[heldout]]
        if len(b):
            out[b] = predict(features, targets, train[eligible_train[train]], b, labels)
        logits[name] = out
    return logits


def archived_head(values, config, view, labels, feature_names):
    width = len(feature_names)
    if (values[view].shape[1] != width * 2 or feature_names != config['features']
            or list(labels) != config['classes']):
        raise ValueError('The shortened task window does not match the published 54-feature head')
    features = values[view][:, :width]
    scale = np.asarray(config['scale'])
    logits = ((features - np.asarray(config['mean'])) / scale) @ np.asarray(config['coef']).T
    return log_softmax(logits + np.asarray(config['intercept']), axis=1)


def summarize(rows, costs, available, draws):
    output = []
    for cohort, group in rows.groupby('cohort'):
        for length, frame in [('all', group), ('short', group[group.n_words.lt(100)])]:
            if frame.empty:
                continue
            selected = rows.index.isin(frame.index)
            trio = frame[frame.model.isin(FOCUS)]
            sections = {
                'full': [paired_stats(frame, before, after, draws) for before, after in COMPARISONS],
                'trio_conditional': [paired_stats(trio, 'trio_' + before, 'trio_' + after, draws)
                                     for before, after in COMPARISONS] if len(trio) else [],
            }
            per_class = pd.DataFrame({key: frame[key].eq(frame.model) for key in ARMS}).groupby(frame.model).sum()
            present = frame.model.value_counts()
            output.append(dict(cohort=cohort, length=length, rows=len(frame), predictions=sections,
                               total_budget_eligible=int((available & selected).sum()),
                               by_label={label: {'rows': int(present.get(label, 0)),
                                                 'correct': per_class.loc[label].to_dict() if label in per_class else {}}
                                         for label in sorted(rows.model.unique())},
                               estimated_positions={key: dict(zip(('body', 'prefix', 'total'),
                                                                  map(int, value[selected].sum(0))))
                                                    for key, value in costs.items()}))
    return output


def run(panel_dir, score_dir, model_json, output, draws=4000):
    truth = read_ids(panel_dir / 'truth.parquet')
    query = read_ids(panel_dir / 'queries.parquet')
    same_ids(query.index, truth.index, 'Frozen prefix rows', ordered=True)
    ordered = hashlib.sha256(('\n'.join(query.index) + '\n').encode()).hexdigest()
    manifest = json.loads((panel_dir / 'manifest.json').read_text())
    scored = json.loads((score_dir / 'manifest.json').read_text())
    if (len(query) != 892 or any(x['rows'] != 892 or x['ordered_ids_sha256'] != ordered for x in (manifest, scored))
            or manifest['input_body_sha256'] != [hashlib.sha256(x.encode()).hexdigest() for x in query.text]):
        raise ValueError('The score and input packages do not share the frozen IDs and bodies')
    by_model = {tag: list(zip(*observations(score_dir, tag, query.index), strict=True)) for tag in MODELS}
    traces = [dict(zip(MODELS, pairs, strict=True)) for pairs in zip(*(by_model[tag] for tag in MODELS), strict=True)]
    config = json.loads(model_json.read_text())
    values, costs, available = feature_views(traces, query.text.str.len().to_numpy(), config['diff_features'])
    labels = np.sort(truth.model.unique())
    if len(labels) != 18 or truth.cohort.value_counts().to_dict() != {'calibration': 576, 'heldout': 316}:
        raise ValueError('Expected the original calibration and blind holdout sizes')
    logits = fit_arms(values, truth, labels, available)
    rows = truth.assign(total_budget_eligible=available)
    indices = pd.Index(labels).get_indexer(FOCUS)
    for name, scores in logits.items():
        rows[name] = labels[scores.argmax(1)]
        rows['trio_' + name] = labels[indices[scores[:, indices].argmax(1)]]
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'paired_predictions.parquet')
    feature_names = [f'{tag}:{name}' for tag in MODELS
                     for name in one(traces[0][tag][0][0], len(query.text.iloc[0]))]
    feature_names += [f'd:{name}' for name in config['diff_features']]
    for view in ('task160', 'task128_window', 'total_matched_window'):
        logits[f'legacy_{view}'] = archived_head(values, config, view, labels, feature_names)
    np.savez_compressed(output / 'head_logp.npz', ids=rows.index.to_numpy(dtype=str), labels=labels.astype(str),
                        total_budget_eligible=available, **logits)
    report = dict(actual_scoring=scored, holdout_training_rows=0, calibration_prompt_folds=5,
                  features_by_arm={name: array.shape[1] for name, array in values.items()},
                  comparisons=summarize(rows, costs, available, draws))
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
