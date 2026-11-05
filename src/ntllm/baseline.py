"""Run a prompt-grouped TF-IDF baseline and retain its research artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import time
import tomllib
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline


DEFAULTS = {
    "name": "tfidf_lr",
    "data_dir": "data",
    "output_dir": "runs",
    "folds": 5,
    "seed": 42,
    "word_ngram_range": [1, 2],
    "char_ngram_range": [3, 5],
    "word_max_features": 25_000,
    "char_max_features": 35_000,
    "min_df": 2,
    "lowercase": False,
    "sublinear_tf": True,
    "C": 2.0,
    "max_iter": 500,
}


def make_model(config: dict) -> Pipeline:
    """Keep fitted vocabulary/IDF inside each fold's pipeline."""
    common = {
        "min_df": config["min_df"],
        "lowercase": config["lowercase"],
        "sublinear_tf": config["sublinear_tf"],
        "dtype": np.float32,
    }
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=tuple(config["word_ngram_range"]),
            max_features=config["word_max_features"],
            **common,
        )),
        ("char", TfidfVectorizer(
            analyzer="char",
            ngram_range=tuple(config["char_ngram_range"]),
            max_features=config["char_max_features"],
            **common,
        )),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            C=config["C"],
            max_iter=config["max_iter"],
            random_state=config["seed"],
        )),
    ])


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(config: dict, fit_full: bool = False) -> Path:
    started = time.perf_counter()
    created_at = datetime.now(timezone.utc)
    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", config["name"])
    run_dir = Path(config["output_dir"]) / (
        f"{created_at:%Y%m%dT%H%M%SZ}_{run_name}_{uuid4().hex[:6]}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot = dict(config)
    snapshot.update({
        "created_at_utc": created_at.isoformat(),
        "fit_full": fit_full,
        "splitter": "StratifiedGroupKFold(shuffle=True, groups=prompt_id)",
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    })
    write_json(run_dir / "config.json", snapshot)

    train = pd.read_csv(Path(config["data_dir"]) / "train.csv", dtype=str)
    text = train["text"]
    target = train["model"].to_numpy()
    groups = train["prompt_id"].to_numpy()
    labels = np.unique(target)
    probabilities = np.zeros((len(train), len(labels)), dtype=np.float32)
    fold_ids = np.full(len(train), -1, dtype=np.int16)
    fold_metrics = []
    splitter = StratifiedGroupKFold(
        n_splits=config["folds"], shuffle=True, random_state=config["seed"],
    )

    print(f"Run: {run_dir}", flush=True)
    print(f"{len(train)} rows | {len(np.unique(groups))} prompts | {len(labels)} classes", flush=True)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(text, target, groups)):
        fold_started = time.perf_counter()
        model = make_model(config)
        model.fit(text.iloc[train_idx], target[train_idx])
        fold_probabilities = model.predict_proba(text.iloc[valid_idx])
        class_positions = np.searchsorted(labels, model.classes_)
        probabilities[np.ix_(valid_idx, class_positions)] = fold_probabilities
        fold_ids[valid_idx] = fold
        predictions = labels[probabilities[valid_idx].argmax(axis=1)]
        score = float(balanced_accuracy_score(target[valid_idx], predictions))
        elapsed = time.perf_counter() - fold_started
        fold_metrics.append({
            "fold": fold,
            "train_rows": len(train_idx),
            "valid_rows": len(valid_idx),
            "train_prompts": len(np.unique(groups[train_idx])),
            "valid_prompts": len(np.unique(groups[valid_idx])),
            "balanced_accuracy": score,
            "elapsed_seconds": round(elapsed, 3),
        })
        print(f"Fold {fold + 1}/{config['folds']}: BA={score:.5f} ({elapsed:.1f}s)", flush=True)

    predictions = labels[probabilities.argmax(axis=1)]
    word_counts = text.str.split().str.len().to_numpy()
    recall = recall_score(target, predictions, labels=labels, average=None, zero_division=0)
    length_metrics = []
    for bucket, lower, upper in [("<60", 0, 60), ("60-99", 60, 100),
                                  ("100-159", 100, 160), (">=160", 160, np.inf)]:
        mask = (word_counts >= lower) & (word_counts < upper)
        if mask.any():
            length_metrics.append({
                "bucket": bucket,
                "rows": int(mask.sum()),
                "classes_present": len(np.unique(target[mask])),
                "balanced_accuracy": float(balanced_accuracy_score(target[mask], predictions[mask])),
            })

    scores = [fold["balanced_accuracy"] for fold in fold_metrics]
    metrics = {
        "oof_balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "fold_mean_balanced_accuracy": float(np.mean(scores)),
        "fold_std_balanced_accuracy": float(np.std(scores)),
        "rows": len(train),
        "prompts": len(np.unique(groups)),
        "folds": fold_metrics,
        "per_class": [
            {"model": label, "rows": int((target == label).sum()), "recall": float(value)}
            for label, value in zip(labels, recall)
        ],
        "by_word_count": length_metrics,
        "word_count_definition": "Python whitespace split",
        "cv_elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(run_dir / "metrics.json", metrics)
    oof = train[["id", "prompt_id"]].copy()
    oof["true_model"] = target
    oof["pred_model"] = predictions
    oof["fold"] = fold_ids
    oof["n_words"] = word_counts
    for index, label in enumerate(labels):
        oof[f"prob__{label}"] = probabilities[:, index]
    oof.to_parquet(run_dir / "oof.parquet", index=False)
    confusion = pd.DataFrame(confusion_matrix(target, predictions, labels=labels), index=labels, columns=labels)
    confusion.index.name = "true_model"
    confusion.to_csv(run_dir / "confusion_matrix.csv")
    print(f"OOF BA={metrics['oof_balanced_accuracy']:.5f} | folds={np.mean(scores):.5f} ± {np.std(scores):.5f}", flush=True)

    if fit_full:
        full_started = time.perf_counter()
        print("Fitting full training data...", flush=True)
        model = make_model(config)
        model.fit(text, target)
        joblib.dump(model, run_dir / "model.joblib", compress=3)
        test = pd.read_csv(Path(config["data_dir"]) / "test.csv", dtype=str)
        submission = pd.DataFrame({"id": test["id"], "model": model.predict(test["text"])})
        submission.to_parquet(run_dir / "submission.parquet", index=False)
        metrics["full_fit_and_prediction_seconds"] = round(time.perf_counter() - full_started, 3)
        metrics["submission_rows"] = len(submission)
        metrics["total_elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_json(run_dir / "metrics.json", metrics)
        print(f"Saved model.joblib and submission.parquet ({len(submission)} rows)", flush=True)
    print(f"Artifacts: {run_dir.resolve()}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tfidf.toml"))
    parser.add_argument("--data-dir", help="Directory containing train.csv and test.csv")
    parser.add_argument("--output-dir", help="Parent directory for timestamped experiment runs")
    parser.add_argument("--folds", type=int, help="Override config fold count (default: 5)")
    parser.add_argument("--seed", type=int, help="Override config seed (default: 42)")
    parser.add_argument("--fit-full", action="store_true", help="After CV, fit all data and save model + predictions")
    args = parser.parse_args()
    with args.config.open("rb") as handle:
        config = {**DEFAULTS, **tomllib.load(handle)}
    for name in ("data_dir", "output_dir", "folds", "seed"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    config["config_path"] = str(args.config)
    run(config, fit_full=args.fit_full)


if __name__ == "__main__":
    main()
