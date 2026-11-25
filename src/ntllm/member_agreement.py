"""Bound and evaluate narrow within-family decisions on frozen ensemble scores."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .member_evidence import STAGES
from .peer_context_report import balanced_accuracy, paired_stats
from .sibling_odds_replay import replay_predictions


MIN_PROMPTS = 5
MARGIN = .8
STAGE_ACTIONS = ('pick_stack', 'pick_glm', 'pick_sibling')
ARMS = {'stages': STAGE_ACTIONS, 'stages_and_members': (*STAGE_ACTIONS, 'pick_members')}
MEMBER_GROUPS = {
    'qwen3': ('infer__q14v5', 'infer__q8v5', 'infer__n14b_clean'),
    'qwen25': ('infer__q25_14',),
    'llama': ('infer__ll8v5',),
    'deberta': ('infer__deb',),
    'gsq': ('infer__gsq',),
    'native': tuple(f'native__fold_{fold}' for fold in range(5)),
}


def unique_winner(votes, incumbent, minimum):
    winner = votes.argmax(1)
    largest = votes.max(1)
    qualified = (largest >= minimum) & ((votes == largest[:, None]).sum(1) == 1)
    return np.where(qualified, winner, incumbent), qualified


def build_actions(truth, trace, baseline, members, coverage):
    labels = np.asarray(trace['labels'])
    classes = pd.Index(labels)
    family = np.array([label.split('-', 1)[0] for label in labels])
    incumbent = classes.get_indexer(baseline)
    if (incumbent < 0).any():
        raise ValueError('Agreement actions require known incumbent labels')
    row = np.arange(len(truth))
    incumbent_family = family[incumbent]
    within_family = incumbent_family[:, None] == family[None, :]
    before_typography = replay_predictions(trace, truth.text, typography=False)
    typography = (before_typography != baseline)
    typography |= (incumbent_family == 'gpt') & truth.text.str.contains(' — ', regex=False).to_numpy()
    typography |= (incumbent_family == 'gemini') & truth.text.str.contains('—', regex=False).to_numpy()
    protected = trace['glm_winner'] | trace['fable_guard'] | typography
    eligible = np.isin(incumbent_family, ['gpt', 'gemini']) & ~protected

    alternatives = within_family.copy()
    alternatives[row, incumbent] = False
    result = pd.DataFrame(index=truth.index)
    for stage, action in zip(STAGES, STAGE_ACTIONS, strict=True):
        winner = np.where(alternatives, trace[stage], -np.inf).argmax(1)
        result[action] = labels[np.where(eligible, winner, incumbent)]
    agreement = np.logical_and.reduce([trace[stage].argmax(1) == incumbent for stage in STAGES])
    runner_up = classes.get_indexer(result['pick_sibling'])
    margin = trace['sibling_logp'][row, incumbent] - trace['sibling_logp'][row, runner_up]
    result['agreement_cell'] = [
        f'{name}:{"agree" if agree else "disagree"}:{"close" if gap < MARGIN else "clear"}' if permitted else ''
        for name, agree, gap, permitted in zip(incumbent_family, agreement, margin, eligible, strict=True)
    ]
    result['agreement_protected'] = protected

    # Checkpoints of one backbone have one vote. Each voter must cover every
    # variant in the incumbent family, and ties abstain at both voting levels.
    ballot = np.eye(len(labels), dtype=int)
    total_votes = np.zeros_like(within_family, dtype=int)
    for keys in MEMBER_GROUPS.values():
        votes = np.zeros_like(total_votes)
        for key in keys:
            if key in members:
                complete = (~within_family | coverage[key]).all(1)
                top = np.where(within_family, members[key], -np.inf).argmax(1)
                votes += ballot[top] * complete[:, None]
        winner, qualified = unique_winner(votes, incumbent, minimum=1)
        total_votes += ballot[winner] * qualified[:, None]
    consensus, _ = unique_winner(total_votes, incumbent, minimum=2)
    result['pick_members'] = labels[np.where(eligible, consensus, incumbent)]
    return result


def fit_policy(panel, actions):
    """Choose one action per observable cell, including the original call."""
    weights = 1. / (panel.model.map(panel.model.value_counts()) * panel.model.nunique())
    original = panel.baseline.eq(panel.model)
    selected, report = {}, {}
    for cell, group in panel.loc[panel.agreement_cell.ne('')].groupby('agreement_cell'):
        options = {}
        best, best_delta = 'baseline', 0.
        for action in actions:
            changed = group[action].ne(group.baseline)
            prompts = int(group.loc[changed, 'prompt_id'].nunique())
            new = group[action].eq(group.model)
            delta = float(weights.loc[group.index] @ (new.astype(int) - original.loc[group.index].astype(int)))
            options[action] = {'changed_prompts': prompts, 'ba_delta_pp': 100. * delta}
            if prompts >= MIN_PROMPTS and delta > best_delta + 1e-12:
                best, best_delta = action, delta
        selected[cell] = best
        report[cell] = dict(action=best, rows=len(group), prompts=int(group.prompt_id.nunique()), options=options)
    return selected, report


def apply_policy(panel, selected):
    predictions = panel.baseline.copy()
    for cell, action in selected.items():
        if action != 'baseline':
            mask = panel.agreement_cell.eq(cell)
            predictions.loc[mask] = panel.loc[mask, action]
    return predictions


def ceiling(panel, actions):
    selected, details = fit_policy(panel, actions)
    retrospective = apply_policy(panel, selected)
    includes_true = panel[list(actions)].eq(panel.model, axis=0).any(axis=1)
    union = panel.baseline.where(~includes_true, panel.model)
    baseline_ba = balanced_accuracy(panel.model, panel.baseline)
    report = {}
    for name, pred in [('cell_choice_upper', retrospective), ('row_union_upper', union)]:
        correct = pred.eq(panel.model)
        wrong = panel.baseline.ne(panel.model)
        fixed = panel.model.loc[correct & wrong].value_counts()
        harmed = panel.model.loc[~correct & ~wrong].value_counts()
        report[name] = dict(ba_delta_pp=100. * (balanced_accuracy(panel.model, pred) - baseline_ba),
                            fixed=int(fixed.sum()), harmed=int(harmed.sum()),
                            fixed_by_true_class=fixed.to_dict(), harmed_by_true_class=harmed.to_dict())
    report['retrospective_cells'] = details
    return report


def analyze_agreement(rows, panels, draws):
    dev = rows.loc[rows.cohort.eq('development')]
    folds = list(GroupKFold(4).split(dev, groups=dev.prompt_id))
    result = dict(min_distinct_changed_prompts=MIN_PROMPTS, close_margin=MARGIN,
                  cell='incumbent family, whether all three stage winners match incumbent, same-family sibling margin',
                  member_vote='At least two covered backbone groups must agree on a unique variant',
                  member_groups=MEMBER_GROUPS,
                  bound='Cell choice maximizes measured BA contribution among panel-supported actions. Union chooses per row with truth. Both are retrospective.',
                  arms={})
    for arm, actions in ARMS.items():
        name = 'agreement_' + arm
        fit, c1_cells = fit_policy(dev, actions)
        rows[name] = apply_policy(rows, fit)
        for train, val in folds:
            selected, _ = fit_policy(dev.iloc[train], actions)
            pred = apply_policy(dev.iloc[val], selected)
            rows.loc[pred.index, name] = pred
        scope = []
        for panel_name, ids in {'c1_oof': dev.index, **panels}.items():
            found = rows.loc[ids]
            for length, panel in [('all', found), ('short', found.loc[found.n_words.lt(100)])]:
                if panel.empty:
                    continue
                eligible = panel.agreement_cell.ne('')
                scope.append(dict(panel=panel_name, length=length, eligible=int(eligible.sum()),
                                  eligible_errors=int((eligible & panel.baseline.ne(panel.model)).sum()),
                                  paired=paired_stats(panel, 'baseline', name, draws),
                                  **ceiling(panel, actions)))
        result['arms'][arm] = dict(c1_cells=c1_cells, scopes=scope)
    for column in (*STAGE_ACTIONS, 'pick_members', *('agreement_' + arm for arm in ARMS)):
        if not rows.loc[rows.agreement_cell.eq(''), column].equals(rows.loc[rows.agreement_cell.eq(''), 'baseline']):
            raise ValueError('Agreement analysis changed an ineligible or protected decision')
    return result
