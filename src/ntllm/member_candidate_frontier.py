"""Measure fixed candidate coverage against exact historical and independent stacks."""

import json

import numpy as np
import pandas as pd

from .counterfactual_selection import member_family
from .glm_prefix_response_replay import restored_route
from .member_evidence import STAGES
from .member_evidence_transfer import EXPOSURE_KEYS, read_capture, read_prvsiyan, verify_manifest
from .peer_context_report import read_ids
from .reader_transfer import PROVENANCE, STYLE, read_logits
from .sibling_odds_replay import replay_predictions


def top_mask(scores, depth, observed=None):
    if observed is None:
        observed = np.ones(scores.shape, dtype=bool)
    order = np.argsort(-np.where(observed, scores, -np.inf), axis=1, kind='stable')[:, :depth]
    result = np.zeros_like(observed, dtype=bool)
    np.put_along_axis(result, order, True, axis=1)
    return result & observed


def union(masks, shape):
    return np.logical_or.reduce(list(masks)) if masks else np.zeros(shape, dtype=bool)


def build_pools(trace, incumbent, members, masks, shared, readers):
    scores = trace[STAGES[0]]
    shape = scores.shape
    anchor = np.zeros(shape, dtype=bool)
    anchor[np.arange(len(anchor)), pd.Index(trace['labels']).get_indexer(incumbent)] = True
    stages1 = {name: top_mask(trace[name], 1) for name in STAGES}
    stages3 = {name: top_mask(trace[name], 3) for name in STAGES}
    staged = union(stages3.values(), shape)
    branch = {name: top_mask(matrix, 2, masks[name]) for name, matrix in members.items()}
    groups = {}
    for name, candidates in branch.items():
        group = member_family(name)
        groups[group] = groups.get(group, np.zeros(shape, dtype=bool)) | candidates
    votes = np.asarray(list(groups.values()), dtype=np.uint8).sum(axis=0)
    shared_members = union([branch[name] for name in sorted(shared)], shape)
    all_members = union(branch.values(), shape)
    top1 = {name: top_mask(matrix, 1) for name, matrix in readers.items()}
    top2 = {name: top_mask(matrix, 2) for name, matrix in readers.items()}
    top3 = {name: top_mask(matrix, 3) for name, matrix in readers.items()}
    reader2, reader3 = union(top2.values(), shape), union(top3.values(), shape)
    full = staged | all_members
    series = {
        'stage_top1': union(stages1.values(), shape), 'stages_top3': staged,
        'stages_plus_two_member_groups': staged | (votes >= 2),
        'stages_plus_shared_members': staged | shared_members, 'stages_plus_all_members': full,
        'stages_plus_reader_top2': staged | reader2, 'stages_plus_reader_top3': staged | reader3,
        'all_members_and_reader_top2': full | reader2, 'all_members_and_reader_top3': full | reader3,
    }
    series = {name: matrix | anchor for name, matrix in series.items()}
    sources = {**{f'stage:{name}': value for name, value in stages3.items()},
               **{f'member_group:{name}': value for name, value in groups.items()},
               **{f'reader:{name}': value for name, value in top2.items()}}
    shared_coverage = dict(stages=staged, shared_members=shared_members, all_members=all_members,
                           reader_top1=union(top1.values(), shape), reader_top2=reader2, reader_top3=reader3)
    return series, sources, shared_coverage, list(stages1.values()), votes


def evaluate_panel(truth, trace, incumbent, folder, members, masks, shared, readers):
    pools, sources, block, winners, votes = build_pools(trace, incumbent, members, masks, shared, readers)
    labels = trace['labels']
    target = pd.Index(labels).get_indexer(truth.model)
    stage_winner = labels[np.column_stack([matrix.argmax(axis=1) for matrix in winners])]
    row = np.arange(len(truth))
    agree = (stage_winner == stage_winner[:, :1]).all(1)
    wrong = incumbent != truth.model.to_numpy()
    prior = replay_predictions(trace, truth.text, typography=False)
    receipt = json.loads((folder / 'glm_report.json').read_text())
    routed, _ = restored_route(trace, truth.text, incumbent, receipt)
    rows = truth[['cohort', 'model', 'prompt_id', 'n_words']].assign(
        baseline=incumbent, error=wrong, short=truth.n_words.lt(100),
        strict=truth.get('strict_topic_holdout', False), routed=routed,
        error_stage=np.where(agree, np.where(stage_winner[:, 0] == truth.model, 'unanimous_true', 'unanimous_wrong'),
                             'stage_disagreement'),
        error_family=np.where(truth.model.str.split('-').str[0] == pd.Series(incumbent, index=truth.index).str.split('-').str[0],
                              'sibling', 'cross_family'),
        typography_changed=prior != incumbent, native_protected=trace['glm_winner'] | trace['fable_guard'],
        true_member_groups=votes[row, target],
    )
    details = {f'covers__{name}': mask[row, target] for name, mask in pools.items()}
    details.update({f'width__{name}': mask.sum(1) for name, mask in pools.items()})
    details.update({f'evidence__{name}': matrix[row, target] for name, matrix in sources.items()})
    details.update({f'block__{name}': matrix[row, target] for name, matrix in block.items()})
    for index, name in enumerate(STAGES):
        details[f'winner__{name}'] = stage_winner[:, index]
    return rows.assign(**details)


def summarize(panel):
    counts = panel.model.value_counts()
    weights = 100. / (len(counts) * panel.model.map(counts))
    names = [name.removeprefix('covers__') for name in panel if name.startswith('covers__')]
    source_columns = [name for name in panel if name.startswith('evidence__') and not name.startswith('evidence__stage:')]
    available = panel[source_columns].sum(axis=1)
    wrong = panel.error
    missed_stages = wrong & ~panel.block__stages
    missed_both = missed_stages & ~panel.block__all_members
    stage_members = panel.block__all_members
    readers = panel.block__reader_top2

    def measure(rows):
        mistakes = rows.error
        uncovered = mistakes & ~rows.block__stages
        new_after_members = uncovered & ~rows.block__all_members
        result = {}
        for name in names:
            found = mistakes & rows[f'covers__{name}']
            widths = rows[f'width__{name}']
            result[name] = dict(recoverable=int(found.sum()), panel_ba_gain_ceiling_pp=float(weights.loc[found.index[found]].sum()),
                                mean_candidates=float(widths.mean()), p95_candidates=float(widths.quantile(.95)))
        return dict(rows=len(rows), errors=int(mistakes.sum()), candidate_oracles=result,
                    stage_misses=int(uncovered.sum()), misses_after_members=int(new_after_members.sum()),
                    misses_after_readers=int((new_after_members & ~rows.block__reader_top2).sum()),
                    native_routed_errors=int((mistakes & rows.routed).sum()),
                    native_protected_errors=int((mistakes & rows.native_protected).sum()),
                    typography_changed_errors=int((mistakes & rows.typography_changed).sum()))

    categories = {'all': measure(panel)}
    for name, group in panel.groupby('error_stage'):
        categories[name] = measure(group)
    for name, group in panel.groupby('error_family'):
        categories[name] = measure(group)
    for (stage, family), group in panel.groupby(['error_stage', 'error_family']):
        categories[f'{stage}/{family}'] = measure(group)
    sources = {}
    for name in source_columns:
        candidate = panel[name]
        marginal = missed_stages & candidate
        sources[name.removeprefix('evidence__')] = dict(error_coverage=int((wrong & candidate).sum()),
                                                       marginal_after_stages=int(marginal.sum()),
                                                       exclusive_after_stages=int((marginal & available.eq(1)).sum()),
                                                       marginal_after_members=int((missed_both & candidate).sum()),
                                                       marginal_short=int((marginal & panel.short).sum()),
                                                       marginal_unanimous_wrong=int((marginal & panel.error_stage.eq('unanimous_wrong')).sum()))
    residual = panel.loc[wrong & ~panel.covers__all_members_and_reader_top3]
    return dict(rows=len(panel), prompts=int(panel.prompt_id.nunique()), true_class_counts=counts.to_dict(),
                baseline_ba=1. - float(weights.loc[panel.index[wrong]].sum() / 100.), categories=categories,
                branch_marginal_coverage=sources, stage_miss_overlap=dict(
                    members_and_readers=int((missed_stages & stage_members & readers).sum()),
                    members_only=int((missed_stages & stage_members & ~readers).sum()),
                    readers_only=int((missed_stages & ~stage_members & readers).sum()),
                    neither=int((missed_stages & ~stage_members & ~readers).sum())),
                residual_confusions=[dict(true_model=true, baseline=pred, errors=int(count))
                                     for (true, pred), count in residual.groupby(['model', 'baseline']).size().sort_values(ascending=False).items()],
                by_true_class={str(label): measure(part) for label, part in panel.groupby('model')})


def run(historical, original, independent, source, previous, style, trained, heldout, output):
    public = read_ids(original / 'public_truth.parquet')
    fresh = read_prvsiyan(source, previous)
    hist = read_capture(historical, public)
    ext = read_capture(independent, fresh)
    if not np.array_equal(hist[0]['labels'], ext[0]['labels']):
        raise ValueError('The baseline captures use different label orders')
    shared = set(hist[2]) & set(ext[2])
    if any(hist[4][name]['kind'] != ext[4][name]['kind'] for name in shared):
        raise ValueError('A branch changed score kinds between the two traces')
    verify_manifest(heldout / 'manifest.json', fresh)

    def scores(panel, trace, provenance, exact):
        labels = trace['labels']
        result = {name: read_logits(provenance / f'{name}_logits.parquet', panel.index, labels, exact=exact).to_numpy()
                  for name in PROVENANCE}
        result.update({name: read_logits(style / f'{name}_logp.parquet', panel.index, labels).to_numpy()
                       for name in STYLE})
        return result

    frozen_public = evaluate_panel(public, hist[0], hist[1], historical, hist[2], hist[3], shared,
                                   scores(public, hist[0], trained, False))
    frozen_held = evaluate_panel(fresh, ext[0], ext[1], independent, ext[2], ext[3], shared,
                                 scores(fresh, ext[0], heldout, True))
    overlap = json.loads((independent / 'body_overlap.json').read_text())
    exposures = {item for key in EXPOSURE_KEYS for item in overlap.get(key, [])}
    if not exposures.issubset(fresh.index):
        raise ValueError('Runtime exposure audit has unknown independent IDs')
    frozen_held['runtime_overlap'] = frozen_held.index.isin(exposures)
    frozen_public['runtime_overlap'] = False
    panels = dict(c2=frozen_public.loc[frozen_public.cohort.eq('evaluation')], prvsiyan_895=frozen_held,
                  prvsiyan_316=frozen_held.loc[frozen_held.strict])
    if exposures:
        panels.update({name + '_no_runtime_overlap': part.loc[~part.runtime_overlap]
                       for name, part in list(panels.items()) if name.startswith('prvsiyan')})
    if list(map(len, list(panels.values())[:3])) != [300, 895, 316]:
        raise ValueError('Expected the frozen panels only')
    report = dict(protocol='No fit. Truth is used only to score fixed stage, member and reader candidate lists.',
                  reader_arms=list(PROVENANCE + STYLE), common_members=sorted(shared),
                  historical_only=sorted(set(hist[2]) - shared), independent_only=sorted(set(ext[2]) - shared),
                  runtime_overlap=overlap, cohorts={name: {size: summarize(part) for size, part in
                      [('all', frame), ('short', frame.loc[frame.short])] if len(part)}
                      for name, frame in panels.items() if len(frame)})
    output.mkdir(parents=True, exist_ok=True)
    pd.concat([panels['c2'], frozen_held], verify_integrity=True).to_parquet(output / 'coverage.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Audited 300 historical and 895 independent rows, including the strict 316', flush=True)
    return report
