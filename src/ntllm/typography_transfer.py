"""Measure typography rules and release only fixed, evidence-supported alternatives."""

import json

import numpy as np
import pandas as pd

from .counterfactual_selection import GLM, SPACED_DASH
from .counterfactual_selection_audit import canonical_prompt, make_blocks
from .heldout_source_audit import text_key
from .peer_context_report import paired_stats, read_ids, same_ids
from .sibling_odds_replay import replay_predictions


MARKERS = {'spaced_dash': ' — ', 'curly_apostrophe': '’', 'em_dash': '—'}
EXCLUDED_PUBLISHERS = ('beroli', 'markbarney', 'tadeuhayashida', 'prvsiyan', 'drjcfmd',
                       'coltondowning', 'enaimaan', 'elliotthankinson')
ARMS = ('native_guard', 'compatible_history', 'multisource_exceptions')


def marker_frame(body, calls):
    calls = pd.Series(np.asarray(calls), index=body.index)
    return pd.DataFrame({
        'spaced_dash': body.str.contains(MARKERS['spaced_dash'], regex=False) & ~calls.isin(SPACED_DASH),
        'curly_apostrophe': body.str.contains(MARKERS['curly_apostrophe'], regex=False) &
                            calls.str.startswith(('claude-', 'glm-')),
        'em_dash': body.str.contains(MARKERS['em_dash'], regex=False) & calls.str.startswith('gemini-'),
    })


def count_markers(rows):
    keep = rows[['source', 'prompt_id', 'model', 'n_words']].copy()
    markers = marker_frame(rows.text, rows.model)
    segments = []
    for size, mask in [('all', rows.n_words.ge(40)), ('short', rows.n_words.between(40, 99))]:
        subset = keep.loc[mask]
        denominators = subset.groupby(['source', 'model']).prompt_id.nunique().rename('sample_prompts')
        shifts = []
        for rule in MARKERS:
            exposed = subset.loc[markers.loc[mask, rule]]
            numerator = exposed.groupby(['source', 'model']).prompt_id.nunique().rename('violating_prompts')
            shifts.append(pd.concat([denominators, numerator], axis=1).fillna(0).reset_index().assign(rule=rule))
        segments.extend(piece.assign(length=size) for piece in shifts)
    return pd.concat(segments, ignore_index=True).astype({'violating_prompts': int, 'sample_prompts': int})


def support_table(pool_path, training, reserved, label_order):
    columns = ['source_dataset', 'prompt_canon', 'model', 'text', 'gen_served_model', 'flag_kb_proxy',
               'flag_kb_proxy_after_ruling', 'flag_train_prompt_regen', 'flag_near_train_prompt',
               'flag_gemini37_routing_risk']
    pool = pd.read_parquet(pool_path, columns=columns).dropna(subset=['source_dataset', 'prompt_canon', 'text'])
    publisher = (pool.source_dataset.str.lower().str.removeprefix('https://www.kaggle.com/datasets/')
                 .str.split('/').str[0])
    served = (pool.gen_served_model.astype('string').fillna('').str.lower().str.rsplit('/', n=1).str[-1]
              .str.replace(r'-\d{4}-\d{2}-\d{2}$', '', regex=True))
    body = text_key(pool.text)
    safe = ~pool[[key for key in columns if key.startswith('flag_')]].fillna(False).any(axis=1)
    forbidden_prompt = canonical_prompt(reserved.prompt_id)
    allowed = (safe & pool.model.isin(label_order) & served.eq(pool.model.astype('string'))
               & ~publisher.isin(EXCLUDED_PUBLISHERS) & ~body.isin(text_key(reserved.text))
               & ~canonical_prompt(pool.prompt_canon).isin(forbidden_prompt))
    clean = pool.loc[allowed, ['text', 'model', 'prompt_canon']].rename(columns={'prompt_canon': 'prompt_id'})
    clean = clean.assign(source=publisher.loc[allowed], body=body.loc[allowed]).drop_duplicates('body')
    clean['n_words'] = clean.text.str.split().str.len()
    clean = clean.loc[clean.n_words.between(40, 250), training.columns]
    combined = pd.concat([training, clean], ignore_index=True)
    return count_markers(combined), combined.source.nunique()


def compatible(frame, proposed, support, omitted=None):
    direct = marker_frame(frame.text, proposed)
    eligible = support.loc[support.source.ne(omitted) & support.violating_prompts.ge(5)]
    counts = eligible.groupby(['length', 'model', 'rule']).source.nunique()
    scope = np.where(frame.n_words.lt(100), 'short', 'all')
    allowed = pd.DataFrame(index=frame.index)
    for rule in MARKERS:
        keys = pd.MultiIndex.from_arrays([scope, np.asarray(proposed), np.repeat(rule, len(frame))])
        supported = counts.reindex(keys, fill_value=0).to_numpy() >= 2
        allowed[rule] = ~direct[rule] | supported
    return allowed.all(axis=1)


def replay(frame, trace, guard, support, omit):
    order = pd.Index(trace['ids'])
    selected = order.get_indexer(frame.index)
    labels = pd.Index(trace['labels'])
    before = labels.get_indexer(frame.baseline)
    after = labels.get_indexer(guard.open)
    if (selected < 0).any() or (before < 0).any() or (after < 0).any():
        raise ValueError('Typography actions do not align to the original labels and rows')
    corroborated = np.ones(len(frame), dtype=bool)
    for key in ('stack_logp', 'sibling_logp'):
        corroborated &= trace[key][selected, after] > trace[key][selected, before]
    protected = guard.native_veto | guard.fable_veto
    literal = marker_frame(frame.text, guard.open)
    history_only = guard.original_typography_change & ~literal.any(axis=1)
    if not guard.typography_block.equals(guard.original_typography_change | literal.any(axis=1)):
        raise ValueError('Literal typography rules do not reproduce the original recorded vetoes')
    exception = literal.any(axis=1) & compatible(frame, guard.open, support, omit)
    release_a = ~protected & corroborated & history_only
    release_b = ~protected & corroborated & exception
    return guard.assign(compatible_history=guard.open.where(release_a, guard.native_guard),
                        multisource_exceptions=guard.open.where(release_a | release_b, guard.native_guard),
                        released_history=release_a, released_exception=release_b, corroborated=corroborated)


def run(historical, prior, gen003, kimi, inputs, output, reference, *, independent=None, source=None,
        pool=None, focal=None, draws=2000):
    blocks, held_truth = make_blocks(historical, prior, gen003, kimi, inputs, independent, source)
    query = read_ids(source / 'candidate_queries.parquet')
    truth = read_ids(source / 'candidate_truth.parquet')
    same_ids(query.index, truth.index, 'Source punctuation truth', ordered=True)
    sources = truth.join(query).assign(source=truth.source_dataset.str.split('/').str[0])
    raw = [sources]
    if focal:
        extra_truth = read_ids(focal / 'truth.parquet')
        extra_queries = read_ids(focal / 'queries.parquet')
        same_ids(extra_truth.index, extra_queries.index, 'DRJ punctuation truth', ordered=True)
        raw.append(extra_truth.join(extra_queries).assign(source='drjcfmd'))
    panel_columns = ['text', 'source', 'model', 'prompt_id', 'n_words']
    external = sources.loc[~sources.source.eq('prvsiyan'), panel_columns]
    held_for_counts = held_truth.assign(source='prvsiyan_strict').loc[held_truth.strict_topic_holdout, panel_columns]
    reserved = pd.concat([part for part, *_ in blocks] + [held_truth, *raw])
    support, calibration_sources = support_table(pool, external, reserved, blocks[0][1]['labels'])
    raw_counts = count_markers(pd.concat([*(x[panel_columns] for x in raw), held_for_counts]))
    reference = pd.read_parquet(reference)
    panels, artifacts = {}, []
    for name, guard in reference.groupby('arm'):
        if name not in ('stage_multisource', 'leave_source_gen003', 'leave_source_kimi_flash'):
            continue
        for truth, trace, baseline, _, _ in blocks:
            block_ids = guard.index.intersection(truth.index, sort=False)
            if block_ids.empty:
                continue
            frame = truth.loc[block_ids].assign(baseline=np.asarray(baseline)[truth.index.get_indexer(block_ids)])
            frozen = guard.loc[block_ids]
            if not np.array_equal(frame.baseline, frozen.baseline):
                raise ValueError('Typography comparison uses a different incumbent')
            omit = 'tadeuhayashida' if name == 'leave_source_gen003' else None
            pair = replay(frame, trace, frozen, support, omit)
            pair['scope'] = frame.cohort
            diagnostics = [('all', pair), ('short', pair.loc[pair.n_words.lt(100)])]
            if 'strict_topic_holdout' in frame:
                strict = pair.loc[frame.strict_topic_holdout]
                diagnostics.extend([('strict', strict), ('strict_short', strict.loc[strict.n_words.lt(100)])])
            for subset, part in diagnostics:
                for cohort, rows in part.groupby('scope'):
                    panels[f'{name}/{cohort}/{subset}'] = {
                        'metrics': [paired_stats(rows, 'baseline', arm, draws) for arm in ARMS],
                        'incremental': [paired_stats(rows, 'native_guard', arm, draws) for arm in ARMS[1:]],
                        'history_released': int(rows.released_history.sum()),
                        'exceptions_released': int(rows.released_exception.sum()),
                    }
            artifacts.append(pair)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(calibration_publishers=calibration_sources, support_min_prompts_per_publisher=5,
                  support_min_publishers=2, fixed_candidate=True, panels=panels,
                  literal_source_marker_examples=raw_counts.loc[raw_counts.violating_prompts.gt(0)].to_dict('records'))
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    pd.concat(artifacts).to_parquet(output / 'predictions.parquet')
    support.to_parquet(output / 'marker_support.parquet')
    return report
