"""Summarize a complete Dennis v1 replay on labeled public passages."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd


STAGES = ('stack', 'glm', 'siblings', 'final')


def report(trace_path, output):
    truth = pd.read_parquet(trace_path / 'public_truth.parquet')
    with np.load(trace_path / 'stage_scores.npz', allow_pickle=False) as raw:
        if raw['ids'].tolist() != truth.id.tolist():
            raise ValueError('The public query labels and score rows are misaligned')
        labels = raw['labels']
        table = truth.drop(columns='text').assign(**{name: raw[name] for name in STAGES},
                                                   fable_guard=raw['fable_guard'], glm_winner=raw['glm_winner'])
        for name, key in zip(STAGES, ('stack_logp', 'glm_logp', 'sibling_logp', 'typography_probs')):
            scores = raw[key]
            if scores.shape != (len(table), len(labels)) or not np.isfinite(scores).all():
                raise ValueError(f'Invalid scores for {name}')
            top = labels[np.argsort(-scores, axis=1, kind='stable')[:, :3]]
            for i in range(3):
                table[f'{name}_candidate_{i + 1}'] = top[:, i]
            if name == 'final':
                ordered = np.argsort(-scores, axis=1, kind='stable')
                alternatives = [labels[order[labels[order] != selected][:2]]
                                for order, selected in zip(ordered, table.final)]
                table['decision_candidate_1'] = table.final
                table[['decision_candidate_2', 'decision_candidate_3']] = np.asarray(alternatives)
                table['final_margin'] = np.log(np.maximum(np.sort(scores, axis=1)[:, -1], 1e-15)) - np.log(
                    np.maximum(np.sort(scores, axis=1)[:, -2], 1e-15))

    summaries = {}
    for cohort, rows in table.groupby('cohort'):
        previous = None
        stages = {}
        for name in STAGES:
            correct = rows[name].eq(rows.model)
            stages[name] = {'ba': float(correct.groupby(rows.model).mean().mean()), 'correct': int(correct.sum())}
            if previous is not None:
                before = rows[previous].eq(rows.model)
                stages[name].update(changed=int(rows[name].ne(rows[previous]).sum()),
                                    fixed=int((~before & correct).sum()), harmed=int((before & ~correct).sum()))
            previous = name
        wrong = rows.loc[rows.model.ne(rows.final)]
        confusion = wrong.groupby(['model', 'final']).size().nlargest(16).rename('rows').reset_index()
        alternatives = wrong[[f'decision_candidate_{i + 1}' for i in range(3)]].to_numpy()
        recoverable = wrong.loc[(alternatives == wrong.model.to_numpy()[:, None]).any(axis=1)].copy()
        recoverable['pair'] = [' / '.join(sorted(pair)) for pair in zip(recoverable.model, recoverable.final)]
        summaries[cohort] = {'rows': len(rows), 'prompts': rows.prompt_id.nunique(), 'stages': stages,
                             'guarded_fable': int(rows.fable_guard.sum()),
                             'confusion_pairs': confusion.to_dict(orient='records'),
                             'top3_recoverable_errors': recoverable.pair.value_counts().to_dict()}
    output.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output / 'baseline_decisions.parquet', index=False)
    (output / 'baseline_summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summaries, indent=2), flush=True)


def select_tracks(trace_path, base_config, output):
    from .atlas_likelihood import TRACKS, route_candidates

    config = json.loads(base_config.read_text())
    public = pd.read_parquet(trace_path / 'public_truth.parquet')
    development = public.cohort.eq('development').to_numpy()
    truth = public.model.to_numpy()[development]
    texts = public.text.to_numpy()[development]
    supported = public.loc[development, 'model'].value_counts()
    with np.load(trace_path / 'stage_scores.npz', allow_pickle=False) as raw:
        if raw['ids'].tolist() != public.id.tolist():
            raise ValueError('Development data does not match the baseline trace')
        final = raw['final'][development]
        probs = raw['typography_probs'][development]
        labels = raw['labels'].tolist()
    diagnostics = []
    for a, b in combinations(sorted(supported[supported >= 8].index), 2):
        routes = route_candidates(texts, labels, probs, final, {**config, 'tracks': [(a, b)]})
        positions = routes.row.to_numpy(dtype=int)
        fixes = truth[positions] == routes.challenger.to_numpy()
        if fixes.any():
            diagnostics.append({'pair': [a, b], 'reachable_misses': int(fixes.sum()),
                                'baseline_correct': int((truth[positions] == final[positions]).sum()),
                                'routed': len(positions)})
    diagnostics.sort(key=lambda item: (-item['reachable_misses'], item['baseline_correct'], item['pair']))
    used = set()
    selected = []
    for item in diagnostics:
        a, b = item['pair']
        if a in used or b in used:
            continue
        selected.append([a, b])
        used.update((a, b))
        if len(selected) == 3:
            break
    config['tracks'] = selected or TRACKS
    output.mkdir(parents=True, exist_ok=True)
    (output / 'atlas_selected_config.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'track_selection.json').write_text(json.dumps({'tracks': config['tracks'], 'development_pairs': diagnostics},
                                                           indent=2) + '\n')
    print(json.dumps(config['tracks'], ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--select-from-config', type=Path)
    args = parser.parse_args()
    report(args.trace, args.out)
    if args.select_from_config is not None:
        select_tracks(args.trace, args.select_from_config, args.out)


if __name__ == '__main__':
    main()
