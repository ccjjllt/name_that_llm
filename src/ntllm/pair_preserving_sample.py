"""Maximize verified Gemini request pairs inside the original sample strata."""

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

from .paired_attribution import text_key
from .peer_context_report import read_ids, same_ids, same_meta
from .provenance_readers import curate, draw, read_inputs


ARM = 'provenance_balanced'
SOURCE = 'davidpiln/ntl-vertex-gemini-gen'
MODELS = ('gemini-3.7-flash', 'gemini-3.8-flash')
ID_COLS = ['curated_id_left', 'curated_id_right']
STRATA = ['model', 'length_band', 'source_dataset', 'gen_route', 'gen_reasoning']
METADATA = [*STRATA, 'prompt_canon', 'n_words', 'returned_id_exact']


def complete_pairs(pairs, ids):
    return pairs[ID_COLS].isin(ids).all(axis=1)


def select_pairs(pairs, available, original):
    endpoints = pairs[ID_COLS].to_numpy().ravel()
    strata = pd.MultiIndex.from_frame(available.loc[endpoints, STRATA])
    quotas = original.groupby(STRATA).size()
    destinations = strata.unique().sort_values()
    capacities = quotas.reindex(destinations, fill_value=0).to_numpy()
    matrix = csc_matrix((np.ones(len(endpoints)),
                         (destinations.get_indexer(strata), np.arange(len(pairs)).repeat(2))),
                        shape=(len(destinations), len(pairs)))
    additions = (~pairs[ID_COLS].isin(original.index)).sum(axis=1).to_numpy()
    # One more pair dominates any possible change in the number of replacements.
    solve = milp(additions - (2 * len(pairs) + 1), integrality=np.ones(len(pairs)),
                 bounds=Bounds(0., 1.), constraints=LinearConstraint(matrix, 0., capacities),
                 options={'mip_rel_gap': 0.})
    if not solve.success or solve.status != 0 or not np.allclose(solve.x, np.round(solve.x), atol=1e-7):
        raise RuntimeError(f'Could not certify the maximum paired sample: {solve.message}')
    chosen = solve.x > .5
    return chosen, int(additions[chosen].sum())


def replace(original, available, pairs):
    pair_ids = pd.Series(np.repeat(pairs.index.to_numpy(), 2), index=pairs[ID_COLS].to_numpy().ravel(),
                         name='vertex_pair_id')
    added = available.loc[pair_ids.index.difference(original.index)]
    unused = original.loc[~original.index.isin(pair_ids.index)].sort_values('priority', ascending=False)
    entries = []
    for key, group in added.groupby(STRATA, sort=True):
        donors = unused.loc[unused[STRATA].eq(key).all(axis=1)].head(len(group))
        if len(donors) != len(group):
            raise ValueError(f'Pair selection exceeded the sample quota in {key}')
        entry = group[STRATA].copy()
        entry['added_id'] = group.index
        entry['removed_id'] = donors.index
        entry['added_prompt_canon'] = group.prompt_canon.to_numpy()
        entry['removed_prompt_canon'] = donors.prompt_canon.to_numpy()
        entry['vertex_pair_id'] = pair_ids.loc[group.index].to_numpy()
        entries.append(entry.reset_index(drop=True))
    ledger_fields = [*STRATA, 'added_id', 'removed_id', 'added_prompt_canon', 'removed_prompt_canon', 'vertex_pair_id']
    ledger = pd.concat(entries, ignore_index=True) if entries else pd.DataFrame(columns=ledger_fields)
    keep = original.index.difference(ledger.removed_id)
    sample = available.loc[keep.append(added.index).sort_values(), METADATA].copy()
    if not sample.groupby(STRATA).size().equals(original.groupby(STRATA).size()):
        raise ValueError('The paired sample changed an original joint stratum quota')
    sample['original_selected'] = sample.index.isin(original.index)
    sample['vertex_pair_id'] = pair_ids.reindex(sample.index)
    return sample, ledger


def summarize(original, sample, pairs, ledger, availability, queries):
    before_prompts, after_prompts = set(original.prompt_canon), set(sample.prompt_canon)
    held_prompts = {name: set(group.prompt_id.astype(str)) for name, group in queries.groupby('cohort')}
    return dict(
        rows=len(sample), unchanged_quotas=True, quota_keys=STRATA, quota_strata=int(len(original.groupby(STRATA))),
        eligible_exact_pairs=len(availability), original_exact_pairs=int(complete_pairs(availability, original.index).sum()),
        selected_exact_pairs=len(pairs), selected_pairs_by_source={SOURCE: len(pairs)},
        selected_pairs_by_vertex_arm=pairs.arm.value_counts().to_dict(),
        selected_pairs_with_same_curated_prompt=int(pairs.prompt_canon_left.eq(pairs.prompt_canon_right).sum()),
        selected_unique_ledger_requests=int(pairs[['arm', 'prompt_id']].drop_duplicates().shape[0]),
        selected_ids_sha256=hashlib.sha256(('\n'.join(sample.index) + '\n').encode()).hexdigest(),
        retained_rows=int(sample.original_selected.sum()), replacements=len(ledger),
        symmetric_difference=2 * len(ledger), overlap_fraction=float(sample.original_selected.mean()),
        original_sources=original.source_dataset.value_counts().to_dict(),
        paired_sample_sources=sample.source_dataset.value_counts().to_dict(),
        replacements_by_source=ledger.source_dataset.value_counts().to_dict(),
        prompt_exposure=dict(original_unique=len(before_prompts), paired_unique=len(after_prompts),
                             introduced=len(after_prompts - before_prompts), removed=len(before_prompts - after_prompts),
                             newly_added_row_prompts=int(ledger.added_prompt_canon.nunique()),
                             exact_held_prompt_id_overlap={name: len(after_prompts & ids) for name, ids in held_prompts.items()},
                             exact_held_body_overlap=0),
    )


def enriched_draws(original, available, pairs, ledger, queries, seed, output):
    # Every source-ledger row already passed the published provider identity and
    # completion checks on both sides. An individual endpoint can be curated
    # and short even when its partner is missing or has a different length.
    verified = ledger.loc[ledger.identical_message.eq(True)]
    endpoints = []
    duplicate_candidates = 0
    for model, side in zip(MODELS, ('left', 'right'), strict=True):
        identity = 'curated_id_' + side
        short = verified.loc[verified[identity].notna() & verified['n_words_' + side].lt(100)].copy()
        duplicate_candidates += int(short[identity].duplicated().sum())
        short['required_pair'] = short.both_curated & short.matched_short
        short = short.sort_values(['required_pair', 'arm', 'prompt_id'], ascending=[False, True, True])
        short = short.drop_duplicates(identity)
        view = available.loc[short[identity], METADATA].assign(vertex_arm=short.arm.to_numpy())
        if (not view.index.is_unique or not view.model.eq(model).all() or not view.source_dataset.eq(SOURCE).all()
                or not np.array_equal(view.n_words.to_numpy(), short['n_words_' + side].to_numpy())):
            raise ValueError(f'The Vertex control endpoints disagree with curated rows for {model}')
        endpoints.append(view)
    stock = pd.concat(endpoints, verify_integrity=True).sort_index()
    stock['word_bin_5'] = stock.n_words // 5
    pair_ids = pd.Series(np.repeat(pairs.index.to_numpy(), 2), index=pairs[ID_COLS].to_numpy().ravel(),
                         name='vertex_pair_id')
    treatment = stock.loc[pair_ids.index].copy()
    if len(pairs) != 941 or len(treatment) != 1882 or not treatment.model.value_counts().eq(941).all():
        raise ValueError('Expected all 941 verified short pairs')
    keys = [*STRATA, 'word_bin_5', 'vertex_arm']
    quota = treatment.groupby(keys).size().rename('quota')
    random = np.random.default_rng(seed)
    ordered = stock.assign(draw=random.random(len(stock))).join(quota, on=keys).sort_values('draw')
    control = ordered.loc[ordered.groupby(keys).cumcount().lt(ordered.quota), stock.columns]
    if not control.groupby(keys).size().equals(quota):
        raise ValueError('The independent control could not reproduce the exact enriched quota')

    short = original.model.isin(MODELS) & original.length_band.eq(0)
    prior_david = original.loc[short & original.source_dataset.eq(SOURCE)]
    non_david = original.loc[short & original.source_dataset.ne(SOURCE)].sort_values('priority', ascending=False)
    removed = []
    for model in MODELS:
        if int((short & original.model.eq(model)).sum()) != 1250:
            raise ValueError(f'Unexpected original short quota for {model}')
        n_remove = 941 - int(prior_david.model.eq(model).sum())
        candidates = non_david.loc[non_david.model.eq(model)]
        if n_remove < 0 or len(candidates) < n_remove:
            raise ValueError(f'Cannot widen the shared David quota for {model}')
        removed.append(candidates.head(n_remove))
    donor = pd.concat(removed, verify_integrity=True)
    common = original.drop(index=prior_david.index.append(donor.index))[METADATA]
    original_size = original.groupby(['model', 'length_band']).size()
    held_bodies = text_key(queries.text)
    held_prompts = {name: set(group.prompt_id.astype(str)) for name, group in queries.groupby('cohort')}
    report = dict(seed=seed, quota_keys=keys, joint_vertex_quota=quota.reset_index().to_dict('records'),
                  donor_rows=len(donor), donors_by_source_model=donor.groupby(['source_dataset', 'model']).size()
                  .rename('rows').reset_index().to_dict('records'),
                  short_total_by_model={model: int(original_size[model, 0]) for model in MODELS},
                  original_david_short_by_model=prior_david.model.value_counts().to_dict(),
                  enriched_david_short_by_model=treatment.model.value_counts().to_dict(),
                  verified_individual_stock_by_model=stock.model.value_counts().to_dict(),
                  duplicate_ledger_endpoint_records=duplicate_candidates,
                  shared_non_enriched_rows=len(common), arms={})
    frozen = {}
    original_prompts = set(original.prompt_canon)
    for name, arm in [('control', control), ('pair', treatment)]:
        selected = pd.concat([common, arm], verify_integrity=True).sort_index()
        if not selected.groupby(['model', 'length_band']).size().equals(original_size):
            raise ValueError(f'The enriched {name} draw changed an original model and length quota')
        prompt_set = set(selected.prompt_canon)
        prompt_overlap = {name: len(prompt_set & ids) for name, ids in held_prompts.items()}
        body_overlap = int(available.loc[selected.index, 'body_key'].isin(held_bodies).sum())
        if body_overlap or any(prompt_overlap.values()):
            raise ValueError(f'The enriched {name} draw overlaps a frozen evaluation input')
        complete = complete_pairs(pairs, arm.index)
        report['arms'][name] = dict(rows=len(selected), vertex_rows=len(arm), verified_pairs=int(complete.sum()),
                                    verified_pairs_by_vertex_arm=pairs.loc[complete, 'arm'].value_counts().to_dict(),
                                    overlap_with_archived_ids=int(selected.index.isin(original.index).sum()),
                                    unique_canonical_prompts=len(prompt_set), exact_held_body_overlap=body_overlap,
                                    introduced_canonical_prompts=len(prompt_set - original_prompts),
                                    removed_canonical_prompts=len(original_prompts - prompt_set),
                                    exact_held_prompt_id_overlap=prompt_overlap,
                                    ordered_ids_sha256=hashlib.sha256(('\n'.join(selected.index) + '\n').encode()).hexdigest())
        selected['original_selected'] = selected.index.isin(original.index)
        retained_pairs = pairs.loc[complete]
        selected['vertex_pair_id'] = pd.Series(np.repeat(retained_pairs.index.to_numpy(), 2),
                                                index=retained_pairs[ID_COLS].to_numpy().ravel()).reindex(selected.index)
        selected['enriched_vertex'] = selected.index.isin(arm.index)
        frozen[name] = selected
    if not frozen['control'].groupby(STRATA).size().equals(frozen['pair'].groupby(STRATA).size()):
        raise ValueError('The two enriched draws disagree on a class, length, source, route or effort quota')
    report['shared_ids_between_enriched_arms'] = len(frozen['control'].index.intersection(frozen['pair'].index))
    report['pair_retention_difference'] = report['arms']['pair']['verified_pairs'] - report['arms']['control']['verified_pairs']
    for name, selected in frozen.items():
        selected.rename_axis('id').to_parquet(output / f'enriched_{name}_ids.parquet')
    donor[METADATA].to_parquet(output / 'enriched_shared_donor_ids.parquet')
    return report


def run(official, historical, pool, panels, settings, archived_sample, supply, output, enriched=False):
    queries = read_inputs(official, historical, panels)
    labels = sorted(queries.model.unique())
    rows = curate(pool, queries, labels)
    available = rows.set_index('id', verify_integrity=True)
    config = json.loads(settings.read_text())
    original = draw(rows, config)[ARM].set_index('id', verify_integrity=True)
    archived = read_ids(archived_sample)
    same_ids(original.index, archived.index, 'Source-balanced reader sample', ordered=True)
    same_meta(original, archived, METADATA, 'Source-balanced reader sample')

    table = pd.read_parquet(supply / 'vertex_pairs.parquet')
    verified = table[['both_curated', 'identical_message', 'matched_short']].eq(True).all(axis=1)
    fields = ['arm', 'prompt_id', *ID_COLS, 'target_words_left', 'n_words_left', 'n_words_right']
    pairs = table.loc[verified, fields].sort_values(['arm', 'prompt_id']).copy()
    pairs.index = pd.Index(pairs.arm.astype(str) + ':' + pairs.prompt_id.astype(str), name='vertex_pair_id')
    endpoints = pd.Index(pairs[ID_COLS].to_numpy().ravel())
    if (pairs.empty or not pairs.index.is_unique or not endpoints.is_unique or not endpoints.isin(available.index).all()
            or not available.loc[endpoints, 'source_dataset'].eq(SOURCE).all()):
        raise ValueError('Verified Vertex pairs must have disjoint, available endpoints in the same published source')
    for label, side in zip(MODELS, ('left', 'right'), strict=True):
        endpoint = available.loc[pairs['curated_id_' + side]]
        if (not endpoint.model.eq(label).all() or not endpoint.n_words.lt(100).all()
                or not np.array_equal(endpoint.n_words.to_numpy(), pairs['n_words_' + side].to_numpy())):
            raise ValueError(f'The audit endpoint no longer matches its curated label and length: {side}')
        pairs['prompt_canon_' + side] = endpoint.prompt_canon.to_numpy()
    audit = json.loads((supply / 'report.json').read_text())['vertex_exact_user_message']
    if (len(pairs) != audit['matched_short_in_curated_pool']['pairs']
            or complete_pairs(pairs, original.index).sum() != audit['selected'][ARM]['pairs']):
        raise ValueError('The verified pair inventory no longer reproduces the pinned audit')

    chosen, minimum_replacements = select_pairs(pairs, available, original)
    selected_pairs = pairs.loc[chosen]
    sample, ledger = replace(original, available, selected_pairs)
    if not np.array_equal(complete_pairs(pairs, sample.index), chosen) or len(ledger) != minimum_replacements:
        raise ValueError('The realized sample differs from the certified pair and replacement optimum')
    if available.loc[sample.index, 'body_key'].isin(text_key(queries.text)).any():
        raise ValueError('A selected training body overlaps the frozen evaluation')
    report = summarize(original, sample, selected_pairs, ledger, pairs, queries)
    if any(report['prompt_exposure']['exact_held_prompt_id_overlap'].values()):
        raise ValueError('A selected canonical prompt ID overlaps a frozen held prompt ID')
    output.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(output / 'selected_ids.parquet')
    ledger.to_parquet(output / 'replacements.parquet', index=False)
    selected_pairs.to_parquet(output / 'selected_pairs.parquet')
    if enriched:
        report['enriched'] = enriched_draws(original, available, pairs, table, queries, config['seed'], output)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Verified pairs:', report['original_exact_pairs'], '→', len(selected_pairs),
          'with', len(ledger), 'same-stratum replacements', flush=True)
