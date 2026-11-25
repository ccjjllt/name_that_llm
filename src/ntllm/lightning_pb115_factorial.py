"""Cross-fit the Lightning and Kiro scoring differences on one Native OOF trace."""

import json

import numpy as np
import pandas as pd

from .glm_prefix_response_replay import FABLE, with_glm_scores
from .kiro_pb115_oof import (GLM, combine_scores, fit_predictions, load_features,
                            load_native_oof, select_original_route)
from .peer_context_report import paired_stats, read_ids, same_ids, same_meta
from .sibling_odds_replay import replay_predictions


FINAL = ('kiro', 'kiro_odds', 'lightning_without_odds', 'lightning')
CONTRASTS = (
    ('kiro', 'kiro_odds'),                       # sibling odds on pure PB115
    ('lightning_without_odds', 'lightning'),     # sibling odds on blended evidence
    ('kiro', 'lightning_without_odds'),          # Legacy54 mix without sibling odds
    ('kiro_odds', 'lightning'),                  # Legacy54 mix with sibling odds
    ('kiro', 'lightning'),
    ('kiro_bare_glm', 'lightning_bare_glm'),
)


def assess(rows, draws):
    counts = {name: {label: int(group[name].eq(label).sum()) for label, group in rows.groupby('model')}
              for name in FINAL}
    return dict(rows=len(rows), prompts=int(rows.prompt_id.nunique()), routed=int(rows.routed.sum()),
                recall_numerators=counts, class_counts=rows.model.value_counts().to_dict(),
                paired=[paired_stats(rows, before, after, draws) for before, after in CONTRASTS])


def run(official, native, features, reference, output, draws=4000):
    from pngy_residual import Residual

    train, labels, folds, base = load_native_oof(official, native)
    texts = train.text.fillna('').astype(str)
    x54, x115, tokens = load_features(features, train.index)
    glm = pd.Index(labels).get_indexer(GLM)
    target = pd.Index(labels).get_indexer(train.model)
    costs = tokens + 30
    routed = select_original_route(base, folds, glm, costs)
    likelihood, calibration = fit_predictions((x54, x115), target, folds, glm)
    pb = likelihood['pb115_hybrid']
    legacy = likelihood['legacy54']
    likelihood_mix = np.logaddexp(np.log(.75) + pb, np.log(.25) + legacy)
    posts = {
        'kiro': combine_scores(base, folds, routed, pb, calibration['pb115']),
        'lightning': combine_scores(base, folds, routed, likelihood_mix, calibration['pb115']),
    }
    rows = train[['model', 'prompt_id']].assign(n_words=texts.str.split().str.len(), fold=folds, routed=routed,
                                               baseline=labels[base.argmax(1)])
    prior = read_ids(reference)
    prior.index = prior.index.astype(str)
    same_ids(prior.index, rows.index, 'Archived PB115 OOF', ordered=True)
    same_meta(prior, rows, ['model', 'prompt_id', 'n_words', 'fold', 'routed', 'baseline'], 'Archived PB115 OOF')
    rows['kiro_bare_glm'] = labels[posts['kiro'].argmax(1)]
    if not rows.kiro_bare_glm.eq(prior.pb115_hybrid).all():
        raise ValueError('Pure PB115 must reproduce the archived headline predictions on every row')
    if int(routed.sum()) != 587 or int(costs[routed].sum()) != 87409:
        raise ValueError('The frozen PB115 route or position budget changed')

    residual = np.empty_like(base)
    for fold in np.unique(folds):
        held = folds == fold
        model = Residual(labels).fit(texts.loc[~held], train.loc[~held, 'model'])
        residual[held] = model.predict_log_probability(base[held], texts.loc[held])
    start = dict(ids=train.index.to_numpy(dtype=str), labels=labels, stack_logp=base, sibling_logp=residual)
    base_glm = np.isin(base.argmax(1), glm)
    for name, post in posts.items():
        if not np.array_equal(post[~routed], base[~routed]):
            raise ValueError('The treatment changed GLM evidence outside the common route')
        trace = with_glm_scores(start, post)
        broad = base_glm & ~trace['glm_winner']
        if (broad & ~routed).any():
            raise ValueError('A GLM-to-other protection fell outside the scored rows')
        trace['fable_guard'] = broad
        rows[f'{name}_broad_glm_protection'] = broad
        rows[f'{name}_bare_glm'] = trace['glm']
        with_odds = 'kiro_odds' if name == 'kiro' else 'lightning'
        without_odds = 'kiro' if name == 'kiro' else 'lightning_without_odds'
        rows[with_odds] = replay_predictions(trace, texts, sibling_odds=True)
        rows[without_odds] = replay_predictions(trace, texts, sibling_odds=False)
    plain = rows.loc[~routed, list(FINAL)]
    if not plain.eq(plain.kiro, axis=0).all().all():
        raise ValueError('A decision changed outside the fixed GLM likelihood route')

    panels = {
        'all': rows,
        'short': rows.loc[rows.n_words.lt(100)],
        'routed': rows.loc[rows.routed],
        'routed_short': rows.loc[rows.routed & rows.n_words.lt(100)],
        'glm_fable': rows.loc[rows.model.isin((*GLM, FABLE))],
        'short_glm_fable': rows.loc[rows.model.isin((*GLM, FABLE)) & rows.n_words.lt(100)],
    }
    report = dict(archived_pb115_parity=len(rows), routed=int(routed.sum()),
                  estimated_positions=int(costs[routed].sum()),
                  legacy_18_class_probability_share=.25, three_way_share=.25,
                  protocol='Fixed Native OOF, common 587-row route, broad GLM protection, prompt-fold residuals.',
                  panels={name: assess(part, draws) for name, part in panels.items() if len(part)})
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Verified frozen PB115 parity and scored four decisions on', len(rows), 'passages', flush=True)
    return report
