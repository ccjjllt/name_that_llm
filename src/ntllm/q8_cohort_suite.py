"""Prespecified, direction-balanced class-prior stress cohorts.

Labels determine evaluation membership only. No class prior, label or true
prompt is supplied to grouping or prediction. Existing cohorts stay unchanged.
"""
import hashlib
import json

import numpy as np


SUITE = 'rotating_v1'
DIRECTION_SEED = 20261107
STRENGTHS = {'mild': .5, 'severe': .05}


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def imbalance_directions(labels):
    """Each label occupies every retention rank once in the rotating family."""
    labels = list(labels)
    if len(labels) != 18 or len(set(labels)) != 18:
        raise ValueError('The stress suite requires the recorded 18-label order.')
    random_order = np.random.default_rng(DIRECTION_SEED).permutation(labels).tolist()
    return [('ascending', labels), ('descending', list(reversed(labels))),
            *[(f'rotation{shift:02d}', np.roll(random_order, shift).tolist())
              for shift in range(len(labels))]]


def make_imbalance_cohorts(frame, labels, seeds=(42, 43)):
    """Return (name, positions, fold-independent scenario) for 80 default layouts.

    The same per-ID uniform is reused across directions/strengths within a seed,
    giving paired membership and nested severity. Membership is row-order stable.
    No class is artificially replenished when severe sampling removes it.
    """
    labels = list(labels)
    directions = imbalance_directions(labels)
    if not frame.id.is_unique or not frame.model.isin(labels).all():
        raise ValueError('Stress sampling needs unique text IDs and known labels.')
    seeds = [int(seed) for seed in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Membership seeds must be distinct.')
    cohorts = []
    for seed in seeds:
        uniforms = np.asarray([
            (int(_digest([SUITE, seed, str(text_id)])[:16], 16) >> 11) / (2 ** 53)
            for text_id in frame.id])
        for strength, minimum in STRENGTHS.items():
            for direction, order in directions:
                rates = dict(zip(order, np.linspace(minimum, 1., len(labels)).tolist()))
                selected = np.flatnonzero(uniforms < frame.model.map(rates).to_numpy())
                name = f'imbalance_{strength}_{direction}_seed{seed}'
                scenario = dict(kind='class_imbalance_multidirection', suite=SUITE,
                                strength=strength, direction=direction,
                                direction_family=('antithetic_original' if direction in ('ascending', 'descending')
                                                  else 'balanced_rotations'),
                                direction_seed=DIRECTION_SEED, seed=seed,
                                label_order=labels, class_retention_probability=rates,
                                uniform_rule='First 53 bits of SHA256([suite, seed, string text ID]); paired across directions and strengths.',
                                interpretation='Prespecified label-dependent stress membership; not hidden-test prior estimates. Each subset requires newly inferred text groups and relation scores; all predictions compete over 18 labels.')
                cohorts.append((name, selected, scenario))
    return cohorts


def membership_summary(frame, positions, labels):
    """Fold-specific support belongs beside membership, not in scenario identity."""
    chosen = frame.iloc[positions]
    counts = chosen.model.value_counts()
    source_counts = frame.model.value_counts()
    observed = [label for label in labels if int(counts.get(label, 0))]
    ids = chosen.id.astype(str).tolist()
    return dict(rows=len(chosen), source_rows=len(frame),
                ids=ids, ids_sha256=_digest(sorted(ids)),
                class_counts={label: int(counts.get(label, 0)) for label in labels},
                source_class_counts={label: int(source_counts.get(label, 0)) for label in labels},
                represented_classes=observed,
                missing_classes=[label for label in labels if label not in observed],
                all_18_classes_present=len(observed) == 18,
                comparison_rule='Compare methods only on identical cohort rows. When classes are absent, BA averages represented classes and is not a complete 18-class BA. Do not pool overlapping stress layouts as independent observations.')
