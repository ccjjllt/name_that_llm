"""Replay frozen PB115 decisions with gates that do not use the true labels."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .peer_context_report import paired_stats, read_ids, same_ids


GLM = ('glm-5.3', 'glm-5.3-flash')
FABLE = 'claude-fable-5'
FOCAL = (*GLM, FABLE)
OLD = 'dedicated25'
ACTIONS = {
    'original_marginal': ('marginal35', 'original', 'prespecified'),
    'late_marginal': ('marginal35_more_positions', 'late', 'prespecified'),
    'original_and_late_marginal': ('marginal35_more_positions', 'expanded', 'prespecified'),
    'late_dedicated_control': ('dedicated25_more_positions', 'late', 'control'),
    'original_marginal25': ('marginal25', 'original', 'diagnostic'),
}
PRIMARY = ('unconditional', 'into_focal', 'within_focal', 'native_focal', 'native_focal_into_focal')


def eligible_gates(native, old, new):
    native_focal = native.isin(FOCAL)
    new_focal = new.isin(FOCAL)
    new_glm = new.isin(GLM)
    return {
        'unconditional': pd.Series(True, index=native.index),
        'into_focal': new_focal,
        'within_focal': old.isin(FOCAL) & new_focal,
        'native_focal': native_focal,
        'native_focal_into_focal': native_focal & new_focal,
        'native_nonfocal': ~native_focal,
        'into_glm': new_glm,
        'within_focal_into_glm': old.isin(FOCAL) & new_glm,
        'native_focal_into_glm': native_focal & new_glm,
        'into_fable': new.eq(FABLE),
    }


def read_costs(features, ids):
    counts = {}
    for model in ('flash', 'glm53'):
        table = pd.read_csv(features / f'{model}_features.csv', usecols=['id', 'n_tok'], dtype={'id': str})
        table = table.set_index('id')
        same_ids(table.index, ids, model)
        counts[model] = table.loc[ids, 'n_tok']
    return pd.DataFrame(counts).max(axis=1).add(30)


def by_label(rows, arm):
    old_correct = rows[OLD].eq(rows.model)
    new_correct = rows[arm].eq(rows.model)
    return (rows.assign(fixed=~old_correct & new_correct, harmed=old_correct & ~new_correct,
                        correct_before=old_correct, correct_after=new_correct)
            .groupby('model')[['fixed', 'harmed', 'correct_before', 'correct_after']].sum().to_dict('index'))


def run(reference, features, output, draws=4000):
    rows = read_ids(reference / 'predictions.parquet')
    prior = json.loads((reference / 'report.json').read_text())
    if (len(rows) != 2699 or rows.fold.nunique() != 5 or rows.prompt_id.nunique() != 150
            or not rows.groupby('prompt_id').fold.nunique().eq(1).all()
            or not prior['archive_exact_prediction_match']):
        raise ValueError('Expected the archived official five-fold PB115 control')
    costs = read_costs(features, rows.index)
    original = rows.routed_original
    expanded = rows.routed_more_positions
    routes = {'original': original, 'late': expanded & ~original, 'expanded': expanded}
    if (int(original.sum()), int(costs[original].sum()), int(expanded.sum()), int(costs[expanded].sum())) != (587, 87409, 891, 130586):
        raise ValueError('Frozen original and expanded PB115 routing budgets changed')
    if (original & ~expanded).any() or not original.equals(rows.routed_same_positions):
        raise ValueError('The same-budget route or expanded route differs from the archived experiment')
    if not rows.loc[~original, OLD].equals(rows.loc[~original, 'baseline']):
        raise ValueError('The archived original control differs from Native before the verifier')

    native_focal = rows.baseline.isin(FOCAL)
    short = rows.n_words.lt(100)
    route_costs = {}
    for route, requests in routes.items():
        route_costs[route] = {
            'selected': int(requests.sum()), 'positions': int(costs[requests].sum()),
            'short_selected': int((requests & short).sum()),
            'native_focal_selected': int((requests & native_focal).sum()),
            'native_focal_positions': int(costs[requests & native_focal].sum()),
            'native_focal_short_selected': int((requests & native_focal & short).sum()),
        }

    predicted = {}
    arms = {}
    for action, (candidate, route, purpose) in ACTIONS.items():
        requests = routes[route]
        proposed_change = requests & rows[candidate].ne(rows[OLD])
        for gate, accepted in eligible_gates(rows.baseline, rows[OLD], rows[candidate]).items():
            name = f'{action}__{gate}'
            arm_purpose = purpose if gate in PRIMARY else 'diagnostic'
            # Only the Native predicate is observable before extra likelihood scoring.
            prefilter = (native_focal if gate.startswith('native_focal') else
                         ~native_focal if gate == 'native_nonfocal' else pd.Series(True, index=rows.index))
            inference_requests = requests & prefilter
            if route == 'original':
                # The existing 18/3-way decision already uses both likelihood inputs here.
                positions = int(costs[original].sum())
            else:
                positions = int(costs[original | (inference_requests & ~original)].sum())
            actual_change = proposed_change & accepted
            predicted[name] = rows[OLD].where(~actual_change, rows[candidate])
            arms[name] = dict(action=action, gate=gate, purpose=arm_purpose, candidate=candidate, route=route,
                              nominal_positions=int(costs[original | requests].sum()),
                              projected_positions_if_native_prefilter=int(positions),
                              proposed_changes=int(proposed_change.sum()),
                              accepted_changes=int(actual_change.sum()),
                              accepted_short_changes=int((actual_change & short).sum()))
    rows = pd.concat([rows, pd.DataFrame(predicted)], axis=1)
    if (not rows.original_marginal__unconditional.equals(rows.marginal35)
            or not rows.original_and_late_marginal__unconditional.equals(rows.marginal35_more_positions)
            or not rows.late_dedicated_control__unconditional.equals(rows.dedicated25_more_positions)):
        raise ValueError('Ungated replay differs from the archived PB115 predictions')

    truth_glm = rows.model.isin(GLM)
    truth_fable = rows.model.eq(FABLE)
    truth_focal = truth_glm | truth_fable
    report = dict(rows=len(rows), prompts=int(rows.prompt_id.nunique()), folds=int(rows.fold.nunique()),
                  original_archive_reproduced=True, route_costs=route_costs, arms=arms, scopes={})
    scopes = {
        'all': rows, 'short': rows[short], 'under_50_words_diagnostic': rows[rows.n_words.lt(50)],
        'true_focal': rows[truth_focal], 'short_true_focal': rows[short & truth_focal],
        'true_glm': rows[truth_glm], 'short_true_glm': rows[short & truth_glm],
        'true_fable': rows[truth_fable], 'short_true_fable': rows[short & truth_fable],
        'outsiders': rows[~truth_focal], 'short_outsiders': rows[short & ~truth_focal],
    }
    for name, scope in scopes.items():
        report['scopes'][name] = {
            'rows': len(scope),
            'versus_original': {arm: paired_stats(scope, OLD, arm, draws) for arm in arms},
        }
        if name in ('all', 'short'):
            report['scopes'][name]['by_label'] = {arm: by_label(scope, arm) for arm in arms}

    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / 'predictions.parquet')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Replayed frozen actions', len(arms), 'rows', len(rows), 'scope splits', len(scopes), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.features, args.output)
