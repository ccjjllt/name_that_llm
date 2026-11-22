"""Replay Mirza v7's post-Apex GLM-to-Fable override on the fixed Low/user OOF."""

import json
from pathlib import Path

import numpy as np

from glm_fable_typography import GLM, compare, inputs
from padilla_apex import apex_decisions
from pngy_residual import Residual


def main():
    frame, labels, base, post, selected, budget_positions, weight = inputs()
    names = np.asarray(labels)
    folds = frame.fold.to_numpy()
    target = frame.model.to_numpy()
    texts = frame.text.fillna('').astype(str).to_numpy()
    base_labels = names[base.argmax(axis=1)]
    likelihood_labels = names[post.argmax(axis=1)]
    original = likelihood_labels.copy()
    for fold in np.unique(folds):
        hold = folds == fold
        residual = Residual(labels).fit(texts[~hold], target[~hold])
        refined = residual.predict_log_probability(base[hold], texts[hold])
        original[hold] = apex_decisions(refined, labels, texts[hold], likelihood_labels[hold])

    preserve = np.isin(base_labels, GLM) & (likelihood_labels == 'claude-fable-5')
    restored = original.copy()
    restored[preserve] = likelihood_labels[preserve]

    def metrics(mask):
        row = compare(target, original, restored, mask)
        row['with_v7_restore'] = row.pop('selective')
        row['override_candidates'] = int((mask & preserve).sum())
        return row

    report = {
        'selected_rows': len(selected), 'budget_positions': budget_positions, 'likelihood_weight': weight,
        'typography_multiplier': 0.0,
        'scope': 'Padilla Apex over fixed Native/Low-user 160 held-out scores. The style residual fits other prompt folds.',
        'all': metrics(np.ones(len(frame), dtype=bool)),
        'by_fold': {str(fold): metrics(folds == fold) for fold in np.unique(folds)},
        'fable_recall': metrics(target == 'claude-fable-5'),
    }
    output = json.dumps(report, indent=2, allow_nan=False)
    Path('/kaggle/working/glm_fable_restore_metrics.json').write_text(output + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
