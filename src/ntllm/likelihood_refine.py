"""Apply a GLM likelihood decision experiment to sanpier's public v2 notebook."""

import argparse
import inspect
import json
from copy import deepcopy
from pathlib import Path


def select_glm_rows(logp, glm_columns, token_rows, max_cover):
    """Prioritize GLM recall and the rows where model likelihood could change the decision."""
    import numpy as np

    probability = np.exp(logp)
    glm = np.sort(probability[:, glm_columns], axis=1)
    other = np.ones(probability.shape[1], dtype=bool)
    other[glm_columns] = False
    best_other = probability[:, other].max(axis=1)
    glm_mass = glm.sum(axis=1)
    decision_value = glm_mass - np.maximum(glm[:, -1] - best_other, 0.0)
    priority = (glm_mass + 0.65 * decision_value) / np.maximum(token_rows, 160) ** 0.25

    selected = []
    used_tokens = 0.0
    limit = int(max_cover * len(logp))
    for i in np.argsort(-priority, kind="stable"):
        if len(selected) >= limit:
            break
        if used_tokens + token_rows[i] <= 130_000:
            selected.append(i)
            used_tokens += token_rows[i]
    return np.array(sorted(selected), dtype=np.intp), int(used_tokens)


def transfer_glm_posterior(styled, combined, labels, modeled_labels=("glm-5.3", "glm-5.3-flash")):
    """Keep modeled classes' posterior and preserve the other models' style ratios."""
    import numpy as np
    from scipy.special import logsumexp

    modeled = np.isin(labels, modeled_labels)
    other = ~modeled
    update = styled.copy()
    update[:, modeled] = combined[:, modeled]
    update[:, other] += logsumexp(combined[:, other], axis=1, keepdims=True)
    update[:, other] -= logsumexp(styled[:, other], axis=1, keepdims=True)
    return update


def _replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Unexpected sanpier v2 source around {old.splitlines()[0]!r}")
    return source.replace(old, new, 1)


def patch_notebook(notebook):
    """Return a notebook with the same 320-token model, maximum coverage and token budget."""
    notebook = deepcopy(notebook)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("%%writefile /kaggle/temp/v1/v1_stage.py"):
            stage = cell
        elif source.startswith("%%writefile apex_postprocess.py"):
            apex = cell
    stage_source = "".join(stage["source"])
    stage_source = _replace(
        stage_source,
        "def main():",
        inspect.getsource(select_glm_rows) + "\n\ndef main():",
    )
    first = stage_source.index("    mass = np.exp(zl")
    last = stage_source.index("    if n_sel < a.min_texts:", first)
    stage_source = stage_source[:first] + (
        "    glm_columns = [labels.index(c) for c in sel_cls]\n"
        "    tpc = model.get('tok_per_char', 0.235)\n"
        "    rows_i = np.minimum(frame.text.str.len().to_numpy() * tpc * 1.05, a.max_text_tokens) + 13\n"
        "    sel, token_count = select_glm_rows(zl, glm_columns, rows_i, a.max_cover)\n"
        "    n_sel = len(sel)\n"
        "    save(status='selected', n_texts=len(ids), n_selected=n_sel, estimated_token_rows=token_count)\n"
    ) + stage_source[last:]
    stage_source = _replace(stage_source, "    sel = np.sort(order[:n_sel])\n", "")
    stage_source = _replace(
        stage_source,
        "    after = post_ll.argmax(1)\n",
        "    after = post_ll.argmax(1)\n"
        "    scored = np.zeros(len(ids), dtype=bool)\n"
        "    scored[sel] = True\n",
    )
    stage["source"] = _replace(
        stage_source,
        "labels=np.asarray(labels), logits=post_ll)",
        "labels=np.asarray(labels), logits=post_ll, scored=scored)",
    )

    apex_source = "".join(apex["source"])
    apex_source = _replace(
        apex_source,
        "def apply_arrays(",
        inspect.getsource(transfer_glm_posterior) + "\n\ndef apply_arrays(",
    )
    apex_source = _replace(
        apex_source,
        "current,strength=.10,corrected=False,penalty=0.):",
        "current,strength=.10,corrected=False,penalty=0.,likelihood=None,scored=None):",
    )
    apex_source = _replace(
        apex_source,
        "    probs=softmax(refined,axis=1)",
        "    if likelihood is not None and scored.any():\n"
        "        refined[scored] = transfer_glm_posterior(refined[scored], likelihood[scored], labels)\n"
        "        out[scored] = np.asarray(labels)[refined[scored].argmax(1)]\n"
        "    probs=softmax(refined,axis=1)",
    )
    apex_source = _replace(
        apex_source,
        "    model=Residual(labels).fit(",
        "    with np.load(glm_file, allow_pickle=False) as saved:\n"
        "        scored = saved['scored'] if 'scored' in saved else None\n"
        "    likelihood = aligned_logits(glm_file, ids, labels) if scored is not None else None\n"
        "    model=Residual(labels).fit(",
    )
    apex["source"] = _replace(
        apex_source,
        "current.model.to_numpy(),strength,corrected,penalty)",
        "current.model.to_numpy(),strength,corrected,penalty,likelihood,scored)",
    )
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    notebook = patch_notebook(json.loads(args.source.read_text()))
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
