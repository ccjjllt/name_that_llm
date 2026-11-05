"""Real CPU text-classifier probe of the multi-branch evaluation path.

This fits TF-IDF/SVD and two linear heads on each official training fold. Its
scores describe this CPU model only; they are not Qwen or the 0.668 ensemble.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .q8_data import file_sha
from .q8_supervision import common_relation_rows, shared_triplet_batches, centered_loss_weights
from .train_lora import update_windows


MODEL_KIND = "cpu_tfidf_svd_two_heads"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def condition_identity(config, input_sha256, availability=None):
    """Identity of a declared CPU recipe, excluding fold-specific fit outcomes."""
    representation_keys = ("word_ngram", "char_ngram", "max_features_each", "min_df",
                           "lowercase", "sublinear_tf", "svd_dimensions", "svd_iterations",
                           "train_only_standard_scaler")
    head_keys = ("C", "solver", "max_iter", "tolerance", "plain_weights",
                 "centered_weights", "relation_partitions")
    definition = dict(
        signature_version=1, model_kind=config["model_kind"], input_sha256=input_sha256,
        labels=config["labels"], seed=config["seed"], feature_fit_scope="outer_training_fold_only",
        representation={key: config["representation"][key] for key in representation_keys},
        heads={key: config["heads"][key] for key in head_keys},
        relation_recipe=dict(plan="shared_triplets", batch_size=3, window_rows_target=18,
                             balance="class_mass", seed_rule="experiment seed + fold"),
        exported_branches=availability or dict(plain=True, centered=True, gaussian=True),
        interpretation="Same training recipe identity, not proof of encoder weights or cross-model OOF independence.")
    signature = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(",", ":"),
                                          ensure_ascii=False).encode()).hexdigest()
    return signature, definition


def fit_fold(official, labels, fold, directory, dimensions=128, head_c=2.0, seed=42):
    start = time.perf_counter()
    fit = official.loc[official.fold.ne(fold)].reset_index(drop=True)
    query = official.loc[official.fold.eq(fold)].reset_index(drop=True)
    if set(fit.prompt_id) & set(query.prompt_id):
        raise ValueError("CPU probe training and validation prompts overlap.")
    targets = fit.model.map({label: index for index, label in enumerate(labels)}).to_numpy()
    weights = len(fit) / (len(labels) * np.bincount(targets, minlength=len(labels)))[targets]

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000,
                          lowercase=False, sublinear_tf=True, dtype=np.float32)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                          max_features=12000, lowercase=False, sublinear_tf=True,
                          dtype=np.float32)
    vectorizer = FeatureUnion([("word", word), ("char", char)])
    sparse_fit = vectorizer.fit_transform(fit.text)
    sparse_query = vectorizer.transform(query.text)
    svd = TruncatedSVD(n_components=dimensions, n_iter=7, random_state=seed)
    fit_features = svd.fit_transform(sparse_fit)
    query_features = svd.transform(sparse_query)
    scaler = StandardScaler()
    fit_features = scaler.fit_transform(fit_features).astype(np.float64)
    query_features = scaler.transform(query_features).astype(np.float64)

    plain = LogisticRegression(C=head_c, max_iter=600, tol=1e-6, random_state=seed)
    plain.fit(fit_features, targets, sample_weight=weights)
    common, relation_report = common_relation_rows(fit, fit)
    batches = shared_triplet_batches(fit, common, seed + fold)
    windows = update_windows(batches, 18)
    auxiliary, supervision = centered_loss_weights(
        fit, labels, weights, [windows], balance="class_mass", enabled=True)
    centered_features, centered_y, centered_w, centered_rows = [], [], [], []
    for window in windows:
        denominator = sum(len(rows) for rows, _ in window)
        for rows, same in window:
            if same:
                z = fit_features[rows]
                centered_features.append(z - z.mean(axis=0, keepdims=True))
                centered_y.extend(targets[rows])
                centered_w.extend(auxiliary[0, rows] / denominator)
                centered_rows.extend(rows)
    centered_features = np.concatenate(centered_features)
    centered_w = np.asarray(centered_w, dtype=np.float64)
    centered_w /= centered_w.mean()
    centered = LogisticRegression(C=head_c, max_iter=600, tol=1e-6, random_state=seed)
    centered.fit(centered_features, centered_y, sample_weight=centered_w)
    expected = np.arange(len(labels))
    if not np.array_equal(plain.classes_, expected) or not np.array_equal(centered.classes_, expected):
        raise ValueError("CPU head classes do not match the requested labels.")

    # An independently fitted real sparse model supplies an optional other LP
    # branch for replay diagnostics. It is not an ATLAS or complementary cache.
    char_only = LogisticRegression(C=2.0, max_iter=600, tol=1e-6, random_state=seed)
    char_fit = vectorizer.transformer_list[1][1].transform(fit.text)
    char_query = vectorizer.transformer_list[1][1].transform(query.text)
    char_only.fit(char_fit, targets, sample_weight=weights)
    other_lp = log_softmax(char_only.decision_function(char_query), axis=1)

    directory.mkdir(parents=True, exist_ok=True)
    fit.to_parquet(directory / "fit_rows.parquet", index=False)
    query.to_parquet(directory / "query_rows.parquet", index=False)
    np.save(directory / "fit_features.npy", fit_features)
    np.save(directory / "query_features.npy", query_features)
    np.savez_compressed(directory / "heads.npz", plain_weight=plain.coef_,
                        plain_bias=plain.intercept_, centered_weight=centered.coef_,
                        centered_bias=centered.intercept_,
                        plain_logits=plain.decision_function(query_features),
                        raw_centered_logits=centered.decision_function(query_features))
    np.save(directory / "other_char_logprob.npy", other_lp)
    np.savez_compressed(directory / "centered_training.npz", rows=np.asarray(centered_rows),
                        weights=centered_w, coefficients_before_mean_normalization=auxiliary,
                        plain_weights=weights)
    metadata = dict(
        model_kind=MODEL_KIND, fold=fold, labels=labels, seed=seed,
        fit_rows=len(fit), query_rows=len(query), fit_prompts=int(fit.prompt_id.nunique()),
        query_prompts=int(query.prompt_id.nunique()), prompt_overlap=0,
        representation=dict(word_ngram=[1, 2], char_ngram=[3, 5], max_features_each=12000,
                            min_df=2, lowercase=False, sublinear_tf=True,
                            svd_dimensions=dimensions, svd_iterations=7,
                            train_only_standard_scaler=True,
                            fitted_sparse_features=int(sparse_fit.shape[1]),
                            svd_explained_variance=float(svd.explained_variance_ratio_.sum())),
        heads=dict(C=head_c, solver="lbfgs", max_iter=600, tolerance=1e-6,
                   plain_iterations=plain.n_iter_.tolist(), centered_iterations=centered.n_iter_.tolist(),
                   plain_weights="N/(18*n_class)",
                   centered_weights="Actual triplet-window class_mass coefficients w/n_window, then mean-one across centered examples for the declared LR C.",
                   relation_partitions=1, relation_seed=seed + fold),
        other_branch=dict(name="cpu_char_lr", C=2.0, iterations=char_only.n_iter_.tolist(),
                          scope="Independent char-only TF-IDF LR trained on this same training fold; not an ATLAS/comp surrogate."),
        relations=relation_report, supervision=supervision,
        fit_seconds=time.perf_counter() - start,
        scope="Actual CPU text model, fold-fitted features and two heads. No Qwen model or 0.668 system score.",
    )
    write_json(directory / "config.json", metadata)
    print(f"fold {fold}: fitted {len(fit)} rows, {len(centered_y)} centered examples in {metadata['fit_seconds']:.1f}s", flush=True)


def evaluate_fold(directory, full_only=False, grouping_protocol='train_fold'):
    from .q8_evaluation import evaluate_cohorts
    metadata = json.loads((directory / "config.json").read_text())
    experiment = json.loads((directory.parent / "experiment.json").read_text())
    signature, condition = condition_identity(metadata, experiment["data_sha256"])
    head = np.load(directory / "heads.npz")
    provenance = dict(model_kind=MODEL_KIND, fold=metadata["fold"],
                      source_run=str(directory.resolve()),
                      training_scope="Same fold's training-side texts only; TF-IDF, SVD, scaler, both LR heads and Gaussian fit on that side.",
                      source_config_sha256=file_sha(directory / "config.json"),
                      condition_signature=signature, condition_definition=condition,
                      checkpoint_scope="CPU probe with no pretrained encoder; not a Qwen checkpoint or full ensemble.")
    return evaluate_cohorts(
        pd.read_parquet(directory / "fit_rows.parquet"),
        pd.read_parquet(directory / "query_rows.parquet"), metadata["labels"],
        head["plain_logits"], head["raw_centered_logits"], head["centered_bias"],
        directory / "evaluation", fit_features=np.load(directory / "fit_features.npy"),
        query_features=np.load(directory / "query_features.npy"), centered_trained=True,
        provenance=provenance, config={"grouping_protocol": grouping_protocol,
                                       **({"eval_cohorts": ["full"]} if full_only else {})})


def candidate_heads(output, head_c, grouping_protocol='train_fold'):
    """Fit a real regularization candidate on exactly the same fold features."""
    candidate_root = output / "candidate"
    candidate_root.mkdir(parents=True, exist_ok=True)
    experiment = json.loads((output / "experiment.json").read_text())
    experiment.update(head_C=head_c, code_sha256=file_sha(Path(__file__)),
                      feature_source=str(output.resolve()),
                      scope="CPU C-change candidate: exact same fold features and centered partition; not a Qwen or complete-ensemble claim.")
    write_json(candidate_root / "experiment.json", experiment)
    labels = experiment["labels"]
    for fold in range(5):
        source, destination = output / f"fold_{fold}", candidate_root / f"fold_{fold}"
        destination.mkdir(parents=True, exist_ok=True)
        config = json.loads((source / "config.json").read_text())
        fit = pd.read_parquet(source / "fit_rows.parquet")
        features = np.load(source / "fit_features.npy")
        query_features = np.load(source / "query_features.npy")
        training = np.load(source / "centered_training.npz")
        y = fit.model.map({label: index for index, label in enumerate(labels)}).to_numpy()
        common, _ = common_relation_rows(fit, fit)
        batches = shared_triplet_batches(fit, common, config["heads"]["relation_seed"])
        windows = update_windows(batches, 18)
        zc, yc, positions = [], [], []
        for window in windows:
            for rows, same in window:
                if same:
                    z = features[rows]
                    zc.append(z - z.mean(axis=0, keepdims=True))
                    yc.extend(y[rows]); positions.extend(rows)
        if not np.array_equal(np.asarray(positions), training["rows"]):
            raise ValueError("Candidate relation partition differs from the baseline partition.")
        plain = LogisticRegression(C=head_c, max_iter=600, tol=1e-6, random_state=config["seed"])
        centered = LogisticRegression(C=head_c, max_iter=600, tol=1e-6, random_state=config["seed"])
        plain.fit(features, y, sample_weight=training["plain_weights"])
        centered.fit(np.concatenate(zc), yc, sample_weight=training["weights"])
        for filename in ("fit_rows.parquet", "query_rows.parquet", "fit_features.npy", "query_features.npy",
                         "other_char_logprob.npy", "centered_training.npz"):
            shutil.copy2(source / filename, destination / filename)
        np.savez_compressed(destination / "heads.npz", plain_weight=plain.coef_, plain_bias=plain.intercept_,
                            centered_weight=centered.coef_, centered_bias=centered.intercept_,
                            plain_logits=plain.decision_function(query_features),
                            raw_centered_logits=centered.decision_function(query_features))
        config["heads"].update(C=head_c, plain_iterations=plain.n_iter_.tolist(),
                               centered_iterations=centered.n_iter_.tolist())
        config["feature_source_fit_seconds"] = config.pop("fit_seconds", None)
        config.update(feature_source_run=str(source.resolve()),
                      feature_source_config_sha256=file_sha(source / "config.json"),
                      scope="Real CPU regularization candidate on the same fold-fitted features and weighted centered examples.")
        write_json(destination / "config.json", config)
        evaluate_fold(destination, full_only=True, grouping_protocol=grouping_protocol)
        print(f"candidate fold {fold}: C={head_c}, evaluated full cohort", flush=True)
    summarize(candidate_root, labels)


def summarize(output, labels):
    from .compare_lora import metrics, outcomes, paired_interval, infer_grouping_protocol
    paths, protocol_evidence = [], []
    for directory in sorted(output.glob('fold_*')):
        index = json.loads((directory / 'evaluation/evaluation.json').read_text())
        for scenario, entry in index['cohorts'].items():
            if not entry.get('available'):
                continue
            cohort = directory / 'evaluation' / scenario
            manifest = json.loads((cohort / 'manifest.json').read_text())
            protocol_evidence.append(infer_grouping_protocol(index, manifest))
            paths.extend(cohort / f'{view}_predictions.parquet' for view in entry['views'])
    protocols = {item['protocol'] for item in protocol_evidence}
    if len(protocols) != 1:
        raise ValueError('CPU summaries must keep grouping protocols separate.')
    groups = {}
    for path in paths:
        scenario, view = path.parent.name, path.stem.removesuffix("_predictions")
        groups.setdefault((scenario, view), []).append(path)
    result = dict(model_kind=MODEL_KIND,
                  scope="Five-fold pooled OOF of the CPU probe, not averages of sparse fold BA; seeds are sensitivity analyses, not independent added observations.",
                  evaluation_code_sha256=file_sha(Path(__file__)),
                  grouping_protocol=next(iter(protocols)),
                  grouping_protocol_inferred=any(item['inferred'] for item in protocol_evidence),
                  scenarios={})
    for (scenario, view), files in sorted(groups.items()):
        if len(files) != 5:
            raise ValueError(f"Need all five fold files for {scenario}/{view}, got {len(files)}.")
        rows = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True).sort_values("id").reset_index(drop=True)
        if not rows.id.is_unique or rows.groupby("prompt_id").fold.nunique().max() != 1:
            raise ValueError("Pooled CPU OOF rows or prompt folds are inconsistent.")
        probability = rows[[f"prob__{label}" for label in labels]].to_numpy()
        score = metrics(rows, labels, probability)
        counts = rows.true_model.value_counts().reindex(labels, fill_value=0)
        score.update(represented_classes=int((counts > 0).sum()),
                     balanced_accuracy_18=score["balanced_accuracy"] if (counts > 0).all() else None,
                     class_support=counts.to_dict(),
                     per_class_recall={label: float((rows.loc[rows.true_model.eq(label), "pred_model"] == label).mean()) if counts[label] else None for label in labels})
        score["length_slices"] = {}
        for name, mask in {"<60": rows.n_words.lt(60), "60-99": rows.n_words.between(60, 99),
                           "100-159": rows.n_words.between(100, 159), ">=160": rows.n_words.ge(160)}.items():
            if mask.any():
                score["length_slices"][name] = metrics(rows.loc[mask], labels, probability[mask])
        for column in ("group_size", "cluster_size"):
            if column in rows:
                score["group_slices"] = {name: metrics(rows.loc[mask], labels, probability[mask])
                                         for name, mask in {"singleton": rows[column].eq(1), "grouped": rows[column].ge(2)}.items() if mask.any()}
                break
        destination = output / "oof" / scenario
        destination.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(destination / f"{view}_predictions.parquet", index=False)
        result["scenarios"].setdefault(scenario, {})[view] = score
    for scenario, views in result["scenarios"].items():
        base = pd.read_parquet(output / "oof" / scenario / "plain_predictions.parquet")
        for view, score in views.items():
            if view == "plain":
                continue
            candidate = pd.read_parquet(output / "oof" / scenario / f"{view}_predictions.parquet")
            if not base.id.equals(candidate.id):
                raise ValueError("A CPU branch changed evaluation row identities.")
            score["paired_vs_plain"] = outcomes(base, base.pred_model.to_numpy(), candidate.pred_model.to_numpy())
            score["descriptive_prompt_bootstrap_95pct"] = paired_interval(
                base, base.pred_model.to_numpy(), candidate.pred_model.to_numpy())
        active_parts = []
        for directory in sorted(output.glob("fold_*")):
            cohort = directory / "evaluation" / scenario
            branch = np.load(cohort / "branches.npz")
            mask = branch["centered_active"].astype(bool)
            if not mask.any():
                continue
            rows = pd.read_parquet(cohort / "rows.parquet").loc[mask].copy()
            rows = rows.rename(columns={"model": "true_model"})
            probability = softmax(branch["centered_lp"][mask], axis=1)
            rows["pred_model"] = np.asarray(labels)[probability.argmax(axis=1)]
            for label, values in zip(labels, probability.T):
                rows[f"prob__{label}"] = values
            active_parts.append(rows)
        if active_parts:
            active = pd.concat(active_parts, ignore_index=True).sort_values("id").reset_index(drop=True)
            probability = active[[f"prob__{label}" for label in labels]].to_numpy()
            score = metrics(active, labels, probability)
            support = active.true_model.value_counts().reindex(labels, fill_value=0)
            score.update(represented_classes=int((support > 0).sum()), class_support=support.to_dict(),
                         balanced_accuracy_18=score["balanced_accuracy"] if (support > 0).all() else None,
                         scope="Centered head on active recovered-group rows only; pooled across folds.")
            plain = base.set_index("id").loc[active.id]
            score["paired_vs_plain_on_same_active_rows"] = outcomes(
                active, plain.pred_model.to_numpy(), active.pred_model.to_numpy())
            active.to_parquet(output / "oof" / scenario / "centered_active_only_predictions.parquet", index=False)
        else:
            score = dict(rows=0, balanced_accuracy=None, balanced_accuracy_18=None,
                         represented_classes=0, class_support={label: 0 for label in labels},
                         scope="No active recovered groups; centered-only accuracy is undefined.")
        views["centered_active_only"] = score
    write_json(output / "summary.json", result)
    print(json.dumps({scenario: {view: values["balanced_accuracy"] for view, values in views.items()}
                      for scenario, views in result["scenarios"].items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/q8/official.parquet"))
    parser.add_argument("--output", type=Path, default=Path("runs/q8_evaluation_20261105/cpu"))
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--head-c", type=float, default=2.0)
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--grouping-protocol", choices=['train_fold', 'query_transductive'],
                        default='train_fold', help='TF-IDF fit scope for relation grouping, not head training')
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--candidate-c", type=float, help="Also fit a same-feature CPU head-C candidate and evaluate full cohorts")
    parser.add_argument("--candidate-only", action="store_true", help="Reuse existing baseline feature artifacts and fit only --candidate-c")
    args = parser.parse_args()
    if args.fit_only and args.evaluate_only:
        parser.error("Choose either fit-only or evaluate-only.")
    if args.candidate_only and (args.candidate_c is None or args.fit_only or args.evaluate_only):
        parser.error("candidate-only needs candidate-c and no fit-only/evaluate-only flag.")
    official = pd.read_parquet(args.data)
    labels = sorted(official.model.unique())
    if set(official.fold) != set(range(5)) or len(labels) != 18:
        raise ValueError("Expected the prepared official five-fold, 18-class dataset.")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(model_kind=MODEL_KIND, labels=labels, data_path=str(args.data.resolve()),
                    data_sha256=file_sha(args.data), dimensions=args.dimensions, head_C=args.head_c,
                    seed=42, code_sha256=file_sha(Path(__file__)), scope="Actual CPU-only two-head model probe.")
    path = args.output / "experiment.json"
    if args.evaluate_only or args.candidate_only:
        saved = json.loads(path.read_text())
        if any(saved[key] != metadata[key] for key in ("data_sha256", "dimensions", "head_C", "labels")):
            raise ValueError("Saved CPU features do not match this experiment configuration.")
    else:
        write_json(path, metadata)
    with threadpool_limits(limits=args.threads):
        if args.candidate_only:
            candidate_heads(args.output, args.candidate_c, args.grouping_protocol)
            return
        for fold in range(5):
            directory = args.output / f"fold_{fold}"
            if not args.evaluate_only:
                fit_fold(official, labels, fold, directory, args.dimensions, args.head_c)
            if not args.fit_only:
                evaluate_fold(directory, grouping_protocol=args.grouping_protocol)
        if not args.fit_only:
            summarize(args.output, labels)
            if args.candidate_c is not None:
                candidate_heads(args.output, args.candidate_c, args.grouping_protocol)


if __name__ == "__main__":
    main()
