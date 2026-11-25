"""Fit portable stage-only weights and verify unlabeled inference against Kaggle results."""

import json

import numpy as np
import pandas as pd

from .counterfactual_deployment import POLICIES, score_predictions
from .counterfactual_selection import build_actions, export_scoring_model, fit_predict, replay_scoring_model
from .counterfactual_selection_audit import calibration_ids, class_weights, fit_arm, make_blocks


def run(historical, prior, gen003, kimi, inputs, output, reference, independent=None, source=None):
    blocks, held_truth = make_blocks(historical, prior, gen003, kimi, inputs, independent, source)
    dirs = [historical, gen003, kimi, *([independent] if independent else [])]
    panels, actions = [], []
    for truth, trace, baseline, _, _ in blocks:
        panels.append(truth[['cohort', 'model', 'prompt_id', 'n_words', 'text']].assign(baseline=baseline))
        actions.append(build_actions(truth, trace, baseline))
    rows = pd.concat(panels, verify_integrity=True)
    actions = pd.concat(actions)
    dev, external, separation = calibration_ids(rows, held_truth)
    frozen = pd.read_parquet(reference)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(separation=separation, arms={})
    labels = blocks[0][1]['labels'].tolist()
    for name, fit in [('stage_c1', dev), ('stage_multisource', dev.append(external))]:
        live, selected = fit_arm(actions, rows, fit, rows.index)
        weights = class_weights(rows, fit)
        threshold = selected['open']['threshold']
        exported = export_scoring_model(actions, fit, weights, threshold)
        exported.update(label_order=labels, arm=name, version=1)
        weight_file = output / f'{name}.json'
        weight_file.write_text(json.dumps(exported, indent=2, allow_nan=False) + '\n')
        # Reload bytes to test exactly the artifact shipped with the Notebook.
        exported = json.loads(weight_file.read_text())
        portable = replay_scoring_model(actions, exported, rows.index)
        sklearn = fit_predict(actions, fit, rows.index, weights, threshold=threshold)
        error = float(np.abs(portable.expected_delta - sklearn.expected_delta).max())
        if error > 1e-12 or not portable.prediction.equals(sklearn.prediction):
            raise ValueError(f'Portable scores disagree with the training classifier for {name}')
        if not portable.prediction.equals(live.open):
            raise ValueError(f'The archived threshold does not reproduce original candidate calls for {name}')
        expected = frozen.loc[frozen.arm.eq(name)]
        if not expected.index.is_unique:
            raise ValueError('The frozen approval replay does not have unique IDs')
        parity = {}
        for (truth, trace, baseline, _, _), folder in zip(blocks, dirs, strict=True):
            queries = truth.text.to_frame().reset_index()[['id', 'text']]
            receipt = json.loads((folder / 'glm_report.json').read_text())
            shared = truth.index.intersection(expected.index, sort=False)
            if shared.empty:
                continue
            positions = truth.index.get_indexer(shared)
            for policy in POLICIES:
                predicted = score_predictions(queries, trace, baseline, receipt, exported, policy)
                matches = int((predicted[positions] == expected.loc[shared, policy]).sum())
                parity.setdefault(policy, {'matched': 0, 'rows': 0})
                parity[policy]['matched'] += matches
                parity[policy]['rows'] += len(positions)
                if matches != len(positions):
                    raise ValueError(f'Unlabeled runtime differs from frozen {name}/{policy} calls')
        report['arms'][name] = dict(max_score_delta=error, threshold=threshold, comparisons=parity,
                                    artifact_bytes=weight_file.stat().st_size)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
