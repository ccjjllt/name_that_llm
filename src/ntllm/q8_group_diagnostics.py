"""Truth-based evaluation of recovered groups; never an inference routing rule."""
import numpy as np
import pandas as pd


def _pairs(counts):
    counts = np.asarray(counts, dtype=np.int64)
    return int(np.sum(counts * (counts - 1) // 2))


def _quality(rows, labels, first, second, mask):
    """Compare exclusion to noninjective with all label columns retained."""
    if not np.any(mask):
        return dict(rows=0, represented_classes=0, balanced_accuracy_exclusion=None,
                    balanced_accuracy_noninjective=None, delta_ba=None,
                    mean_nll_exclusion=None, mean_nll_noninjective=None,
                    delta_mean_nll=None, delta_balanced_nll=None,
                    delta_ba_contribution_to_cohort=0., delta_mean_nll_contribution_to_cohort=0.)
    selected = rows.loc[mask]
    truth = selected.model.to_numpy()
    target = pd.Series(truth).map({name: i for i, name in enumerate(labels)}).to_numpy(int)
    a, b = first[mask], second[mask]
    pa, pb = a.argmax(1), b.argmax(1)
    ca, cb = pa == target, pb == target
    na, nb = -a[np.arange(len(a)), target], -b[np.arange(len(b)), target]
    ba_a = float(pd.Series(ca).groupby(truth).mean().mean())
    ba_b = float(pd.Series(cb).groupby(truth).mean().mean())
    full_counts = rows.model.value_counts()
    cohort_weights = 1. / (pd.Series(truth).map(full_counts).to_numpy() * len(full_counts))
    return dict(rows=len(selected), prompts=int(selected.prompt_id.nunique()),
                represented_classes=int(len(set(truth))),
                missing_classes=sorted(set(labels) - set(truth)),
                balanced_accuracy_exclusion=ba_a, balanced_accuracy_noninjective=ba_b,
                delta_ba=ba_a - ba_b,
                changed_predictions=int(np.sum(pa != pb)),
                exclusion_rescued=int(np.sum(ca & ~cb)), exclusion_lost=int(np.sum(~ca & cb)),
                mean_nll_exclusion=float(na.mean()), mean_nll_noninjective=float(nb.mean()),
                delta_mean_nll=float((na - nb).mean()),
                delta_balanced_nll=float(pd.Series(na - nb).groupby(truth).mean().mean()),
                delta_ba_contribution_to_cohort=float(np.dot(cohort_weights, ca.astype(int)-cb.astype(int))),
                delta_mean_nll_contribution_to_cohort=float((na - nb).sum()/len(rows)))


def grouping_conflict_diagnostics(rows, labels, branches=None, *, n_exact=3, max_cases=20):
    """Separate observed group errors and their conditional prediction effects.

    Truth is used solely after inference to describe errors. Cross-fold group
    IDs are namespaced by fold. Subgroup BA averages only represented labels,
    while predictions always retain all columns. No truth-driven correction,
    rejection, threshold, or recipe is returned.
    """
    rows = rows.reset_index(drop=True).copy()
    if 'model' not in rows and 'true_model' in rows:
        rows['model'] = rows.true_model
    required = {'id', 'prompt_id', 'model', 'inferred_group'}
    if not required <= set(rows) or not rows.model.isin(labels).all():
        raise ValueError('Group diagnostics require aligned IDs, true prompts/classes and inferred groups.')
    scope = ['fold'] if 'fold' in rows else []
    group_columns = scope + ['inferred_group']
    grouped = rows.groupby(group_columns, sort=False, dropna=False)
    mask_names = ('singleton', 'pure_prompt_active', 'mixed_prompt_active',
                  'repeated_true_label_active', 'cross_prompt_same_label_collision',
                  'hard_exclusion_true_tuple_conflict', 'soft_repulsion_true_label_collision')
    masks = {name: np.zeros(len(rows), dtype=bool) for name in mask_names}
    inferred_pairs = correct_pairs = repeated_label_pairs = cross_label_pairs = 0
    conflict_groups = repeated_groups = hard_conflicts = soft_conflicts = 0
    details = []
    for key, positions in grouped.indices.items():
        positions = np.asarray(positions, dtype=int)
        group = rows.iloc[positions]
        size = len(group)
        mixed = group.prompt_id.nunique() > 1
        repeated = group.model.nunique() < size
        same_label = _pairs(group.model.value_counts())
        cross = same_label - _pairs(group.groupby(['prompt_id', 'model']).size())
        inferred_pairs += size * (size - 1) // 2
        correct_pairs += _pairs(group.prompt_id.value_counts())
        repeated_label_pairs += same_label
        cross_label_pairs += cross
        masks['singleton'][positions] = size == 1
        masks['pure_prompt_active'][positions] = size > 1 and not mixed
        masks['mixed_prompt_active'][positions] = mixed
        masks['repeated_true_label_active'][positions] = repeated
        masks['cross_prompt_same_label_collision'][positions] = cross > 0
        masks['hard_exclusion_true_tuple_conflict'][positions] = repeated and size <= n_exact
        masks['soft_repulsion_true_label_collision'][positions] = repeated and size > n_exact
        conflict_groups += int(mixed)
        repeated_groups += int(repeated)
        hard_conflicts += int(repeated and size <= n_exact)
        soft_conflicts += int(repeated and size > n_exact)
        if (mixed or repeated) and len(details) < max_cases:
            details.append(dict(fold=int(group.fold.iloc[0]) if 'fold' in group else None,
                                inferred_group=str(group.inferred_group.iloc[0]),
                                rows=size, true_prompts=int(group.prompt_id.nunique()),
                                repeated_true_label=bool(repeated), cross_prompt_same_label_pairs=cross,
                                exclusion_mechanism='hard_joint_tuple_mask' if size <= n_exact else 'soft_cavity_repulsion',
                                row_positions=positions.tolist(), ids=group.id.astype(str).tolist(),
                                prompt_ids=group.prompt_id.astype(str).tolist(), true_models=group.model.tolist()))
    true_pairs = _pairs(rows.groupby(scope + ['prompt_id'], dropna=False).size())
    report = dict(policy='truth_group_conflict_diagnostics_v1', rows=len(rows),
                  inferred_groups=int(grouped.ngroups), inferred_pairs=inferred_pairs,
                  true_pairs=true_pairs, correct_pairs=correct_pairs,
                  false_positive_pairs=inferred_pairs-correct_pairs,
                  missed_true_pairs=true_pairs-correct_pairs,
                  pair_precision=correct_pairs/inferred_pairs if inferred_pairs else None,
                  pair_recall=correct_pairs/true_pairs if true_pairs else None,
                  mixed_prompt_groups=conflict_groups, repeated_true_label_groups=repeated_groups,
                  repeated_true_label_pairs=repeated_label_pairs,
                  cross_prompt_same_label_pairs=cross_label_pairs,
                  hard_exclusion_true_tuple_conflict_groups=hard_conflicts,
                  soft_repulsion_true_label_collision_groups=soft_conflicts,
                  row_counts={name: int(mask.sum()) for name, mask in masks.items()},
                  conflict_cases=details, reported_conflict_cases=len(details), case_limit=max_cases,
                  interpretation='Evaluation-only truth slices, not deployable routing. Exact groups mask repeated-class joint tuples; larger groups only discourage repeats. Marginal argmax predictions need not form a unique-class joint assignment. Slice BA uses represented labels with all prediction columns retained. Contribution fields use the full cohort denominators and add only over disjoint slices; collision-slice differences are descriptive, not independently randomized effects.')
    if branches is None or not {'gaussian_lp', 'gaussian_noninjective_lp', 'plain_lp', 'gaussian_active'} <= set(branches):
        report['exclusion_comparison'] = dict(available=False, reason='Both Gaussian variants and plain fallback must be available.')
        return report
    active = np.asarray(branches['gaussian_active'], dtype=bool)
    plain = np.asarray(branches['plain_lp'], dtype=float)
    if active.shape != (len(rows),) or plain.shape != (len(rows), len(labels)):
        raise ValueError('Gaussian branches must align with all diagnostic rows and label columns.')
    values = []
    for name in ('gaussian_lp', 'gaussian_noninjective_lp'):
        lp = np.asarray(branches[name], dtype=float)
        if lp.shape != plain.shape or not np.isfinite(lp).all() or not np.isfinite(plain).all():
            raise ValueError('Finite aligned Gaussian branch arrays are required.')
        lp = np.where(active[:, None], lp, plain)
        values.append(lp - np.logaddexp.reduce(lp, axis=1, keepdims=True))
    first, second = values
    report['exclusion_comparison'] = dict(available=True, scale='raw cached Gaussian log probabilities',
        sign='exclusion minus noninjective; positive delta BA is better, positive delta NLL is worse',
        active_rows=int(active.sum()), fallback_rows=int((~active).sum()),
        overall=_quality(rows, labels, first, second, np.ones(len(rows), dtype=bool)),
        slices={name: _quality(rows, labels, first, second, mask) for name, mask in masks.items()})
    for case in details:
        selected = np.asarray(case['row_positions'])
        targets = rows.iloc[selected].model.map({label:i for i,label in enumerate(labels)}).to_numpy(int)
        case['predicted_exclusion'] = np.asarray(labels)[first[selected].argmax(1)].tolist()
        case['predicted_noninjective'] = np.asarray(labels)[second[selected].argmax(1)].tolist()
        case['true_probability_exclusion'] = np.exp(first[selected, targets]).tolist()
        case['true_probability_noninjective'] = np.exp(second[selected, targets]).tolist()
    return report
