"""Isolate Kiro's paired features and three-class correction on original prompt folds."""

import json

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .member_comp_replay import match_temperature
from .peer_context_report import paired_stats, read_ids


MODELS = ('flash', 'glm53')
GLM = ('glm-5.3', 'glm-5.3-flash')
FOCAL = (*GLM, 'claude-fable-5')
SEEDS = (42, 1042, 2042)
ARMS = ('pb54_18', 'pb54_hybrid', 'pb115_18', 'pb115_hybrid')
COMPARISONS = (
    ('baseline', 'legacy54'),
    ('legacy54', 'pb54_18'),
    ('pb54_18', 'pb115_18'),
    ('pb54_hybrid', 'pb115_hybrid'),
    ('pb54_18', 'pb54_hybrid'),
    ('pb115_18', 'pb115_hybrid'),
    ('baseline', 'pb115_hybrid'),
)


def load_features(directory, ids):
    spec = json.loads((directory / 'spec.json').read_text())
    if spec['format'] != 'ntl-prefixb-115-v1' or len(spec['features_54']) != 54 or len(spec['features_61']) != 61:
        raise ValueError('The published PB115 feature recipe does not match its fixed version')
    frames = {tag: pd.read_csv(directory / f'{tag}_features.csv', dtype={'id': str}).set_index('id')
              for tag in MODELS}
    pair = read_ids(directory / 'paired_token_features.parquet')
    pair.index = pair.index.astype(str)
    for name, frame in {**frames, 'paired': pair}.items():
        if not frame.index.is_unique or set(frame.index) != set(ids):
            raise ValueError(f'{name} must cover all official IDs exactly once')
    block = pd.concat([frames[tag].loc[ids].add_prefix(f'{tag}:') for tag in MODELS], axis=1)
    for key in spec['diff_features']:
        block[f'd:{key}'] = frames['flash'].loc[ids, key] - frames['glm53'].loc[ids, key]
    base = block[spec['features_54']].to_numpy(dtype=np.float64)
    extra = pair.loc[ids, spec['features_61']].to_numpy(dtype=np.float64)
    counts = np.column_stack([frames[tag].loc[ids, 'n_tok'] for tag in MODELS])
    if not np.isfinite(base).all() or not np.isfinite(extra).all() or not ((1 <= counts) & (counts <= 160)).all():
        raise ValueError('PB115 data must contain finite features within the published 160-token window')
    return base, np.column_stack([base, extra]), counts.max(axis=1)


def fit_component(x, target, test, classes):
    from lightgbm import LGBMClassifier

    three = classes == 3
    lr = LogisticRegression(C=.03 if three else .025, max_iter=1000).fit(x, target)
    weight = .50 if three else .55
    probabilities = weight * lr.predict_proba(test)
    if not np.array_equal(lr.classes_, np.arange(classes)):
        raise ValueError('A fitting fold is missing a class')
    for seed in SEEDS:
        tree = LGBMClassifier(objective='multiclass', num_class=classes, n_estimators=240 if three else 220,
                              learning_rate=.03, num_leaves=15, min_child_samples=20 if three else 25,
                              subsample=.8, colsample_bytree=.7, reg_alpha=1., reg_lambda=5., random_state=seed,
                              verbose=-1, n_jobs=4).fit(x, target)
        probabilities += ((1 - weight) / len(SEEDS)) * tree.predict_proba(test)
    return np.log(np.clip(probabilities, 1e-8, 1.))


def select_original_route(base, folds, glm, costs):
    confidence = np.exp(base[:, glm]).sum(1)
    chosen = np.zeros(len(base), dtype=bool)
    for fold in sorted(set(folds)):
        rows = np.flatnonzero(folds == fold)
        head = rows[np.argsort(-confidence[rows], kind='stable')][:int(len(rows)*.30)]
        chosen[head[np.cumsum(costs[head]) <= 130000 * len(rows) / 4000]] = True
    return chosen


def fit_predictions(blocks, target, folds, glm):
    names = ('legacy54', *ARMS)
    result = {name: np.zeros((len(target), 18)) for name in names}
    corrections = {name: np.zeros((len(target), 18)) for name in ('pb54', 'pb115')}
    y3 = np.where(target == glm[0], 1, np.where(target == glm[1], 2, 0))
    for fold in sorted(set(folds)):
        held = folds == fold
        for name, x in zip(('pb54', 'pb115'), blocks, strict=True):
            scale = StandardScaler().fit(x[~held])
            learn, test = scale.transform(x[~held]), scale.transform(x[held])
            likelihood = fit_component(learn, target[~held], test, 18)
            lg3 = fit_component(learn, y3[~held], test, 3)
            result[name + '_18'][held] = likelihood
            result[name + '_hybrid'][held] = likelihood
            marginal = np.zeros_like(likelihood)
            marginal[:, glm] = lg3[:, 1:3] - lg3[:, [0]] + np.log(16.)
            corrections[name][held] = marginal
            if name == 'pb54':
                model = LogisticRegression(C=.05, max_iter=1000).fit(learn, target[~held])
                result['legacy54'][held] = model.predict_log_proba(test)
        print('Finished prompt fold', int(fold), flush=True)
    return result, corrections


def load_native_oof(official, native):
    train = pd.read_csv(official, dtype={'id': str, 'prompt_id': str}).set_index('id')
    labels = np.sort(train.model.unique())
    if len(train) != 2699 or len(labels) != 18 or not train.index.is_unique:
        raise ValueError('Expected the official data shape')
    columns = [f'reference__{label}' for label in labels]
    ref = pd.read_parquet(native, columns=['id', 'fold', *columns]).set_index('id')
    ref.index = ref.index.astype(str)
    if not ref.index.is_unique or set(ref.index) != set(train.index):
        raise ValueError('Stored Native OOF and training IDs are not identical')
    ref = ref.loc[train.index]
    folds = ref.fold.to_numpy()
    if len(set(folds)) != 5 or not pd.Series(folds, index=train.index).groupby(train.prompt_id).nunique().eq(1).all():
        raise ValueError('Each official prompt must belong to precisely one of the original five folds')
    base = log_softmax(ref[columns].to_numpy(), axis=1)
    return train, labels, folds, base


def combine_scores(base, folds, route, likelihood, marginal=None):
    boost = np.zeros_like(base)
    for fold in sorted(set(folds)):
        whole = folds == fold
        chosen = whole & route
        w18 = 1.5 / match_temperature(base[whole], .5692304418233374)
        if marginal is None:
            boost[chosen] = w18 * likelihood[chosen]
        else:
            w3 = 1.2 / match_temperature(base[whole], .90)
            boost[chosen] = .75*w18*likelihood[chosen] + .25*w3*marginal[chosen]
    return base + boost


def run(official, native, features, output, draws=4000):
    train, labels, folds, base = load_native_oof(official, native)
    x54, x115, token_count = load_features(features, train.index)
    glm = pd.Index(labels).get_indexer(GLM)
    target = pd.Index(labels).get_indexer(train.model)
    costs = token_count + 30
    route = select_original_route(base, folds, glm, costs)
    ll, marginal = fit_predictions((x54, x115), target, folds, glm)

    rows = train[['model', 'prompt_id']].assign(n_words=train.text.str.split().str.len(), fold=folds,
                                               routed=route, baseline=labels[base.argmax(1)])
    for arm, values in ll.items():
        correction = marginal[arm.split('_', 1)[0]] if arm.endswith('_hybrid') else None
        rows[arm] = labels[combine_scores(base, folds, route, values, correction).argmax(1)]
        if not rows.loc[~route, arm].equals(rows.loc[~route, 'baseline']):
            raise ValueError('An unscored row changed')
        if not arm.endswith('_hybrid'):
            rows[f'{arm}_head'] = labels[values.argmax(1)]

    standalone = [('legacy54_head', 'pb54_18_head'), ('pb54_18_head', 'pb115_18_head')]
    report = dict(rows=len(rows), selected=int(route.sum()), estimated_positions=int(costs[route].sum()),
                  short_selected=int((route & rows.n_words.lt(100)).sum()), scopes=[])
    for name, piece in [('all', rows), ('short', rows[rows.n_words.lt(100)]), ('routed', rows[rows.routed]),
                        ('glm_fable', rows[rows.model.isin(FOCAL)])]:
        counts = {head: piece.assign(correct=piece[head].eq(piece.model)).groupby('model').correct.sum().to_dict()
                  for head in ('baseline', 'legacy54', *ARMS)}
        report['scopes'].append(dict(scope=name, rows=len(piece), counts=piece.model.value_counts().to_dict(),
                                     correct_by_label=counts,
                                     standalone=[paired_stats(piece, a, b, draws) for a,b in standalone],
                                     fused=[paired_stats(piece, a, b, draws) for a,b in COMPARISONS]))
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Routed rows', report['selected'], 'estimated token positions', report['estimated_positions'], flush=True)
