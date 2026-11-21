"""Study token likelihood dynamics for the GLM family and Claude Fable."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax, logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

if __package__:
    from .likelihood_refine import patch_notebook, select_glm_rows, transfer_glm_posterior
else:
    from likelihood_refine import patch_notebook, select_glm_rows, transfer_glm_posterior


GLM = ("glm-5.3", "glm-5.3-flash")
FABLE = "claude-fable-5"
DIFFERENCES = ("lp_mean", "lp_q50", "ent_mean", "top1", "logrank_mean", "fdg")


def public_native_reference(directory):
    directory = Path(directory)
    labels = json.loads((directory / "manifest.json").read_text())["classes"]
    with np.load(directory / "tokens.npz") as tokens, np.load(directory / "oof_reference.npz") as oof:
        ids = tokens["ids"].astype(str)
        folds = tokens["folds"].astype(int)
        plain = oof["plain"]
    scales = {
        k: json.loads((directory / "calibration" / f"fold{k}.json").read_text())["models"]["qwen8"]["plain_scale"]
        for k in np.unique(folds)
    }
    probability = softmax(plain * np.array([scales[k] for k in folds])[:, None], axis=1)
    return pd.DataFrame(probability, columns=[f"prob__{name}" for name in labels]).assign(id=ids, fold=folds)


def download_public_outputs(directory, urls):
    from urllib.request import urlretrieve

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for tag in ("flash", "glm53"):
        for field in ("features.csv", "scores.npz"):
            name = f"{tag}_{field}"
            path = directory / name
            if not path.is_file():
                urlretrieve(urls[name], path)


def token_dynamics(scores):
    logp, entropy, second_moment, rank = np.asarray(scores, dtype=np.float64).T
    residual = -(logp + entropy)
    variance = np.maximum(second_moment - entropy ** 2, 0.02)
    curvature = np.clip(residual / np.sqrt(variance), -8.0, 8.0)
    logrank = np.log1p(rank)
    result = {}
    for name, values in (("curvature", curvature), ("logrank", logrank), ("entropy", entropy)):
        middle = len(values) // 2
        centered = values - values.mean()
        total_variance = max(np.dot(centered, centered), 1e-9)
        result[f"{name}_lag1"] = float(np.dot(centered[:-1], centered[1:]) / total_variance)
        result[f"{name}_step"] = float(np.abs(np.diff(values)).mean()) if len(values) > 1 else 0.0
        result[f"{name}_head_tail"] = float(values[:middle].mean() - values[middle:].mean()) if middle else 0.0
        result[f"{name}_spread"] = float(np.subtract(*np.quantile(values, [0.90, 0.10])))
    surprises = curvature > 2.0
    result["surprise_clusters"] = float((surprises[1:] & surprises[:-1]).mean()) if len(scores) > 1 else 0.0
    result["surprise_fraction"] = float(surprises.mean())
    return result


def likelihood_frame(directory, ids):
    directory = Path(directory)
    frames = {}
    sequence = {}
    for name in ("flash", "glm53"):
        pooled = pd.read_csv(directory / f"{name}_features.csv", dtype={"id": str}).set_index("id")
        with np.load(directory / f"{name}_scores.npz", allow_pickle=False) as raw:
            offsets = raw["offsets"]
            scores = raw["scores"]
        if len(offsets) != len(pooled) + 1 or not np.array_equal(np.diff(offsets), pooled["n_tok"]):
            raise ValueError(f"{name} score rows and features.csv do not align")
        dynamics = [token_dynamics(scores[first:last]) for first, last in zip(offsets[:-1], offsets[1:])]
        sequence[name] = pd.DataFrame(dynamics, index=pooled.index).loc[ids]
        frames[name] = pooled.loc[ids]
    flat = pd.concat([frames[name].add_prefix(f"{name}:") for name in ("flash", "glm53")], axis=1)
    for name in DIFFERENCES:
        flat[f"d:{name}"] = frames["flash"][name] - frames["glm53"][name]
    augmented = pd.concat(
        [flat, sequence["flash"].add_prefix("flash:"), sequence["glm53"].add_prefix("glm53:"),
         (sequence["flash"] - sequence["glm53"]).add_prefix("d:")],
        axis=1,
    )
    return flat, augmented


def fit_head(features, target):
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(C=0.05, max_iter=1000).fit(scaler.transform(features), target)
    return {
        "features": features.columns.tolist(),
        "classes": model.classes_.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "log_prior": np.log(np.asarray([(target == c).mean() for c in model.classes_])).tolist(),
    }


def head_logp(head, features):
    x = features[head["features"]].to_numpy()
    z = ((x - head["mean"]) / head["scale"]) @ np.asarray(head["coef"]).T + head["intercept"]
    if z.shape[1] == 1:
        z = np.concatenate((np.zeros_like(z), z), axis=1)
    return log_softmax(z, axis=1)


def family_targets(target):
    return np.where(np.isin(target, GLM), "glm", np.where(target == FABLE, FABLE, "other"))


def fit_hierarchy(features, target):
    glm_rows = np.flatnonzero(np.isin(target, GLM))
    return {
        "family": fit_head(features, family_targets(target)),
        "glm_variant": fit_head(features.iloc[glm_rows], target[glm_rows]),
    }


def hierarchy_evidence(model, features, labels):
    group_head = model["family"]
    family = head_logp(group_head, features) - group_head["log_prior"]
    family_indices = [group_head["classes"].index(c) for c in family_targets(np.asarray(labels))]
    combined = family[:, family_indices]
    variant_head = model["glm_variant"]
    variant = head_logp(variant_head, features) - variant_head["log_prior"]
    for name in GLM:
        combined[:, labels.index(name)] += variant[:, variant_head["classes"].index(name)]
    return combined


def normalize_temperature(logp, entropy_target=2.3677449936232726):
    low, high = 0.001, 1_000.0
    for _ in range(45):
        mid = np.sqrt(low * high)
        p = log_softmax(logp * mid, axis=1)
        if -(np.exp(p) * p).sum(1).mean() > entropy_target:
            low = mid
        else:
            high = mid
    return float(np.sqrt(low * high))


def evaluate(target, labels, words, decisions):
    metrics = {}
    for name, logits in decisions.items():
        winner = np.asarray(labels)[logits.argmax(axis=1)]
        metrics[name] = {
            "balanced_accuracy": float(balanced_accuracy_score(target, winner)),
            "recall": {label: float((winner[target == label] == label).mean()) for label in labels},
            "length": {
                f"{low}-{high - 1}": float(balanced_accuracy_score(target[rows], winner[rows]))
                for low, high in ((0, 70), (70, 120), (120, 200), (200, 1000))
                if len(rows := np.flatnonzero((words >= low) & (words < high)))
            },
        }
    return metrics


def study(train_csv, cache, output, reference=None, use_paired=False):
    train = pd.read_csv(train_csv, dtype={"id": str, "prompt_id": str})
    labels = sorted(train.model.unique())
    target = train.model.to_numpy()
    words = train.text.str.split().str.len().to_numpy()
    flat, sequence = likelihood_frame(cache, train.id)
    inputs = {"likelihood_pooled": flat, "likelihood_sequence": sequence}
    if use_paired:
        if __package__:
            from .glm_paired_features import paired_features
        else:
            from glm_paired_features import paired_features
        paired, pairing = paired_features(cache, train.id)
        print(f"GLM token alignment: {pairing}", flush=True)
        if paired is not None:
            inputs["likelihood_paired"] = pd.concat([flat, paired], axis=1)
    variants = {name: np.empty((len(train), len(labels))) for name in (*inputs, "likelihood_hierarchy")}
    if reference is None:
        splits = list(GroupKFold(5).split(train, target, train.prompt_id))
    else:
        reader = reference.assign(id=reference.id.astype(str)).set_index("id").loc[train.id]
        folds = reader.fold.to_numpy()
        groups = pd.DataFrame({"fold": folds, "prompt": train.prompt_id})
        if groups.groupby("prompt").fold.nunique().max() != 1:
            raise ValueError("Reference probabilities split a prompt between folds")
        splits = [(np.flatnonzero(folds != k), np.flatnonzero(folds == k)) for k in sorted(set(folds))]
        base = np.log(np.maximum(reader[[f"prob__{c}" for c in labels]].to_numpy(), 1e-15))

    fold_ids = np.empty(len(train), dtype=np.int8)
    for fold, (fit, valid) in enumerate(splits):
        fold_ids[valid] = fold
        for name, features in inputs.items():
            head = fit_head(features.iloc[fit], target[fit])
            variants[name][valid] = head_logp(head, features.iloc[valid])
        hierarchy = fit_hierarchy(sequence.iloc[fit], target[fit])
        variants["likelihood_hierarchy"][valid] = hierarchy_evidence(hierarchy, sequence.iloc[valid], labels)

    decisions = dict(variants)
    glm_columns = [labels.index(name) for name in GLM]
    if reference is not None:
        decisions["reference"] = base
        token_rows = np.minimum(train.text.str.len().to_numpy() * 0.1952 * 1.05, 320) + 13
        weight = 1.0 / normalize_temperature(base)
        allocation = {"family_mass": [], "adaptive": []}
        for _, valid in splits:
            reference_logp = log_softmax(base[valid], axis=1)
            mass = np.exp(reference_logp[:, glm_columns]).sum(axis=1)
            order = np.argsort(-mass, kind="stable")
            limit = min(int(0.15 * len(valid)), int((np.cumsum(token_rows[valid][order]) <= 130_000).sum()))
            allocation["family_mass"].extend(valid[order[:limit]])
            chosen, _ = select_glm_rows(reference_logp, glm_columns, token_rows[valid], 0.15)
            allocation["adaptive"].extend(valid[chosen])
        for method, selected_rows in allocation.items():
            corrected = base[selected_rows] + weight * variants["likelihood_pooled"][selected_rows]
            public_rule = base.copy()
            keep = np.isin(corrected.argmax(1), glm_columns)
            public_rule[np.asarray(selected_rows)[keep]] = corrected[keep]
            decisions[f"{method}_glm_winners"] = public_rule
            posterior = base.copy()
            posterior[selected_rows] = transfer_glm_posterior(base[selected_rows], corrected, labels)
            decisions[f"{method}_glm_posterior"] = posterior
            if "likelihood_paired" in variants:
                paired_corrected = base[selected_rows] + weight * variants["likelihood_paired"][selected_rows]
                paired_posterior = base.copy()
                paired_posterior[selected_rows] = transfer_glm_posterior(base[selected_rows], paired_corrected, labels)
                decisions[f"{method}_paired_posterior"] = paired_posterior
                # Pairwise token evidence can improve the GLM split without moving its total mass.
                split = posterior.copy()
                positions = np.ix_(selected_rows, glm_columns)
                split[positions] = (logsumexp(posterior[positions], axis=1, keepdims=True)
                                    + log_softmax(paired_corrected[:, glm_columns], axis=1))
                decisions[f"{method}_paired_glm_split"] = split
        gated_rows = allocation["adaptive"]
        for name, evidence in variants.items():
            blended = base.copy()
            selected = base[gated_rows] + weight * evidence[gated_rows]
            blended[gated_rows] = transfer_glm_posterior(base[gated_rows], selected, labels)
            decisions[f"reference_plus_{name}"] = blended
            if name == "likelihood_hierarchy":
                with_fable = base.copy()
                with_fable[gated_rows] = transfer_glm_posterior(base[gated_rows], selected, labels, (*GLM, FABLE))
                decisions["reference_plus_hierarchy_with_fable"] = with_fable

    observed_glm = np.isin(target, GLM)
    metrics = evaluate(target, labels, words, decisions)
    for name, logits in variants.items():
        logp = log_softmax(logits, axis=1)
        metrics[name]["glm_membership_auc"] = float(roc_auc_score(observed_glm, logsumexp(logp[:, glm_columns], axis=1)))
        glm_difference = logits[observed_glm, glm_columns[1]] - logits[observed_glm, glm_columns[0]]
        metrics[name]["glm_variant_auc"] = float(roc_auc_score(target[observed_glm] == GLM[1], glm_difference))
        fable_or_glm = observed_glm | (target == FABLE)
        fable_margin = logp[:, labels.index(FABLE)] - logsumexp(logp[:, glm_columns], axis=1)
        metrics[name]["fable_vs_glm_auc"] = float(roc_auc_score(target[fable_or_glm] == FABLE, fable_margin[fable_or_glm]))

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if use_paired:
        (output / "glm_paired_alignment.json").write_text(json.dumps(pairing, indent=2) + "\n")
    (output / "glm_dynamics_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    model = {
        "max_text_tokens": 320,
        "classes": labels,
        "pooled": fit_head(flat, target),
        "sequence": fit_head(sequence, target),
        "hierarchy": fit_hierarchy(sequence, target),
    }
    if "likelihood_paired" in inputs:
        model["paired"] = fit_head(inputs["likelihood_paired"], target)
    (output / "glm_dynamics_model.json").write_text(json.dumps(model) + "\n")
    oof = train[["id", "prompt_id", "model"]].assign(fold=fold_ids)
    for name, matrix in decisions.items():
        oof = oof.join(pd.DataFrame(matrix, columns=[f"{name}__{c}" for c in labels]))
    oof.to_parquet(output / "glm_dynamics_oof.parquet", index=False)
    print(json.dumps(metrics, indent=2), flush=True)


def patch_fitted_notebook(notebook, model, head):
    """Use the chosen full-train token dynamics head in the same notebook."""
    if head not in ("sequence", "hierarchy"):
        raise ValueError("The fitted head must be 'sequence' or 'hierarchy'")
    notebook = patch_notebook(notebook)
    functions = [token_dynamics, likelihood_frame, head_logp]
    if head == "hierarchy":
        functions.extend((family_targets, hierarchy_evidence))
        score = "ll = hierarchy_evidence(FITTED_HEAD, dynamics, labels)"
    else:
        score = (
            "ll = head_logp(FITTED_HEAD, dynamics)"
            "[:, [FITTED_HEAD['classes'].index(c) for c in labels]]"
        )
    config = json.dumps(model[head])
    source = (
        f"GLM = {GLM!r}\nFABLE = {FABLE!r}\nDIFFERENCES = {DIFFERENCES!r}\n"
        f"FITTED_HEAD = json.loads({config!r})\n\n"
        + "\n\n".join(inspect.getsource(fn) for fn in functions)
    )
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if text.startswith("%%writefile /kaggle/temp/v1/v1_stage.py"):
            first = text.index("    feats = {t: pd.read_csv(")
            last = text.index("\n    tp = match_temperature", first)
            text = text[:first] + f"    _, dynamics = likelihood_frame(a.outdir, sel_ids)\n    {score}\n" + text[last:]
            cell["source"] = text.replace("def main():", source + "\n\ndef main():", 1)
        if head == "hierarchy" and text.startswith("%%writefile apex_postprocess.py"):
            old = "transfer_glm_posterior(refined[scored], likelihood[scored], labels)"
            new = f"transfer_glm_posterior(refined[scored], likelihood[scored], labels, {(*GLM, FABLE)!r})"
            cell["source"] = text.replace(old, new, 1)
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", default="/kaggle/working")
    parser.add_argument("--public-output-links", type=Path)
    parser.add_argument("--paired", action="store_true")
    reader = parser.add_mutually_exclusive_group()
    reader.add_argument("--reference", type=Path)
    reader.add_argument("--native-reference", type=Path)
    args = parser.parse_args()
    if args.public_output_links is not None:
        download_public_outputs(args.cache, json.loads(args.public_output_links.read_text()))
    if args.native_reference is not None:
        probabilities = public_native_reference(args.native_reference)
    else:
        probabilities = pd.read_parquet(args.reference) if args.reference is not None else None
    study(args.train, args.cache, args.output, reference=probabilities, use_paired=args.paired)
