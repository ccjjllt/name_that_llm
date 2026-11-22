"""Score real competition passages with the public segmented Kimi-K3 checkpoint."""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


KIMI = "kimi-k3"
REFERENCE = "family_mass_glm_posterior"
LLAMA_REPO = "https://github.com/unslothai/llama.cpp.git"
LLAMA_BRANCH = "dsv41-mxfp8-pin"
LLAMA_COMMIT = "ce078ec1223f01937df7e1391daafdcba6058065"
SEGMENT_NAME = re.compile(r"kimik3-q1-seg\d+-of-23-L(\d+)-(\d+)\.gguf")
STRENGTHS = (0.25, 0.50)


def find_segments(inputs):
    segments = {}
    for path in inputs.rglob("kimik3-q1-seg*-of-23-*.gguf"):
        match = SEGMENT_NAME.fullmatch(path.name)
        if match:
            key = tuple(map(int, match.groups()))
            if key in segments:
                raise ValueError(f"Duplicate segment for layers {key}")
            segments[key] = path
    ordered = sorted(segments.items())
    expected = [(start, min(start + 4, 88)) for start in range(0, 88, 4)] + [(88, 93)]
    if [key for key, _ in ordered] != expected:
        raise ValueError("Kimi segments must cover layers 0–93 in the published order")
    return ordered


def build_runner(work, patch):
    source = work / "llama.cpp"
    build = work / "build"
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", LLAMA_BRANCH, LLAMA_REPO, str(source)],
        check=True,
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != LLAMA_COMMIT:
        subprocess.run(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], cwd=source, check=True)
        subprocess.run(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=source, check=True)
    # The published patch predates the additional Falcon-style hybrid cases in
    # this revision. Preserve the existing cases next to Kimi's new filter.
    adjusted_patch = work / "ntl-kimi-k3-segrun.patch"
    adjusted_patch.write_text(patch.read_text().replace(
        " } else if (arch == LLM_ARCH_FALCON_H1) {\n",
        " } else if (arch == LLM_ARCH_FALCON_H1 || arch == LLM_ARCH_INKLING2 || arch == LLM_ARCH_ZAYA1) {\n",
    ))
    subprocess.run(["git", "apply", str(adjusted_patch)], cwd=source, check=True)
    subprocess.run(
        [
            "cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CUDA_ARCHITECTURES=75", "-DBUILD_SHARED_LIBS=OFF", "-DGGML_CUDA=ON",
            "-DGGML_CUDA_NO_VMM=ON",
            "-DGGML_NATIVE=OFF", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TOOLS=ON",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "--target", "ntl-segrun", "-j", "4"], check=True)
    return build / "bin" / "ntl-segrun"


def read_reference(train_path, reference_path):
    train = pd.read_csv(train_path, dtype={"id": str, "prompt_id": str})
    labels = sorted(train.model.unique())
    columns = [f"{REFERENCE}__{label}" for label in labels]
    reference = pd.read_parquet(reference_path, columns=["id", "fold", *columns])
    reference["id"] = reference.id.astype(str)
    frame = train.merge(reference, on="id", validate="one_to_one")
    if len(frame) != len(train) or frame.groupby("prompt_id").fold.nunique().max() != 1:
        raise ValueError("Reference rows must cover the official training prompts with one fold per prompt")
    logits = frame[columns].to_numpy()
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite reference scores")
    return frame, labels, logits


def choose_rows(logits, folds, kimi, max_rows):
    other = np.delete(logits, kimi, axis=1)
    rank = np.sum(other > logits[:, kimi, None], axis=1)
    gaps = logits[:, kimi] - other.max(axis=1)
    rows = []
    per_fold = max_rows // len(np.unique(folds))
    for fold in np.unique(folds):
        eligible = np.flatnonzero((folds == fold) & (rank < 3))
        order = np.argsort(np.abs(gaps[eligible]), kind="stable")
        rows.extend(eligible[order[:per_fold]])
    return np.asarray(rows, dtype=np.int64), gaps


def write_texts(path, texts):
    with path.open("wb") as handle:
        handle.write(struct.pack("<4sI", b"NTLT", len(texts)))
        for text in texts:
            encoded = text.encode("utf-8")
            handle.write(struct.pack("<I", len(encoded)))
            handle.write(encoded)


def read_scores(path, count):
    values = []
    with path.open("rb") as handle:
        magic, rows, width = struct.unpack("<4sII", handle.read(12))
        if (magic, rows, width) != (b"NTLP", count, 4):
            raise ValueError("Unexpected Kimi score format or row count")
        for _ in range(rows):
            tokens, = struct.unpack("<I", handle.read(4))
            value = np.frombuffer(handle.read(tokens * width * 4), dtype="<f4").reshape(tokens, width)
            if tokens == 0 or not np.isfinite(value).all():
                raise ValueError("An official passage has no finite Kimi token scores")
            values.append(value)
        if handle.read(1):
            raise ValueError("Unexpected trailing data in Kimi token scores")
    return values


def runner_call(argv, log):
    with log.open("w") as handle:
        result = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 11500))
            print(handle.read().decode(errors="replace"), flush=True)
        result.check_returncode()
    with log.open("rb") as handle:
        handle.seek(max(0, log.stat().st_size - 4000))
        tail = handle.read().decode(errors="replace")
    return [line for line in tail.splitlines() if "ntl-segrun: done." in line or "state max |x|" in line]


def tokenize_passages(runner, model, texts, work, max_tokens, prefix):
    input_path = work / "texts.bin"
    tokens = work / "tokens.bin"
    write_texts(input_path, texts)
    runner_call(
        [
            str(runner), "tokenize", "--model", str(model), "--texts", str(input_path),
            "--out", str(tokens), "--prefix", prefix, "--max-text-tokens", str(max_tokens),
        ],
        work / "tokenize.log",
    )
    return tokens


def score_passages(runner, segments, texts, work, max_tokens, prefix, sequence_batch, *, token_path=None):
    tokens = token_path if token_path is not None else tokenize_passages(
        runner, segments[0][1], texts, work, max_tokens, prefix,
    )
    scores = work / "kimi_scores.bin"
    states = (work / "state_a.bin", work / "state_b.bin")
    timing = []
    position = None
    for index, ((start, end), segment) in enumerate(segments):
        before = time.monotonic()
        output = scores if index == len(segments) - 1 else states[index % 2]
        batch = 1 if end == 93 else sequence_batch
        command = [
            str(runner), "run", "--model", str(segment), "--tokens", str(tokens), "--out", str(output),
            "--state-dtype", "bf16", "--ngl", "999", "--split-mid", "--threads", "4",
            "--seq-batch", str(batch),
        ]
        if position:
            command.extend(("--state-in", str(position)))
        log = work / f"layers_{start}_{end}.log"
        lines = runner_call(command, log)
        if position:
            position.unlink()
        position = output
        elapsed = round(time.monotonic() - before, 1)
        print(f"Kimi layers {start}–{end}: {elapsed:.1f}s {'; '.join(lines)}", flush=True)
        timing.append({"start": start, "end": end, "seconds": elapsed, "runner": lines})
    return read_scores(scores, len(texts)), timing


def features(value):
    lp, entropy, _, rank = value.T
    return [
        lp.mean(), (lp + entropy).mean(), lp.std(), entropy.mean(),
        np.log1p(rank).mean(), (rank < 10).mean(),
        lp[-min(len(lp), 32):].mean() - lp[:32].mean(),
    ]


def fit_calibrator(values, truths):
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=0.4, class_weight="balanced", max_iter=300)
    ).fit(values, truths)


def crossfit(values, truths, folds, prompts):
    model_log_odds = np.empty(len(values), dtype=float)
    for fold in np.unique(folds):
        rows = np.flatnonzero(folds == fold)
        for left, right in GroupKFold(2).split(rows, groups=prompts[rows]):
            fit = rows[left]
            holdout = rows[right]
            if len(np.unique(truths[fit])) < 2:
                model_log_odds[holdout] = 0.0
                continue
            model = fit_calibrator(values[fit], truths[fit])
            model_log_odds[holdout] = model.decision_function(values[holdout])
    return model_log_odds


def evaluate(frame, labels, logits, chosen, values, margin, *, calibrate=crossfit):
    kimi = labels.index(KIMI)
    truth = pd.Categorical(frame.model, categories=labels).codes
    original = logits.argmax(axis=1)
    chosen_truth = truth[chosen] == kimi
    fold = frame.fold.to_numpy()
    odds = calibrate(values, chosen_truth, fold[chosen], frame.prompt_id.to_numpy()[chosen])
    result = {
        "selected_rows": len(chosen), "kimi_rows": int(chosen_truth.sum()),
        "auc_reference": roc_auc_score(chosen_truth, margin[chosen]),
        "auc_kimi_features": roc_auc_score(chosen_truth, odds),
        "reference_ba": balanced_accuracy_score(truth, original),
        "reference_kimi_recall": float((original[truth == kimi] == kimi).mean()),
        "weights": {},
    }
    for weight in STRENGTHS:
        updated = logits.copy()
        updated[chosen, kimi] += weight * odds
        prediction = updated.argmax(axis=1)
        result["weights"][str(weight)] = {
            "auc": roc_auc_score(chosen_truth, margin[chosen] + weight * odds),
            "balanced_accuracy": balanced_accuracy_score(truth, prediction),
            "kimi_recall": float((prediction[truth == kimi] == kimi).mean()),
            "fixed": int(((original != truth) & (prediction == truth)).sum()),
            "lost": int(((original == truth) & (prediction != truth)).sum()),
            "by_fold": {
                str(f): balanced_accuracy_score(truth[fold == f], prediction[fold == f])
                - balanced_accuracy_score(truth[fold == f], original[fold == f])
                for f in np.unique(fold)
            },
        }
    return result, odds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--work", type=Path, default=Path("/kaggle/temp/kimi-real-likelihood"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--max-rows", type=int, default=360)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seq-batch", type=int, default=4)
    prefix_options = parser.add_mutually_exclusive_group()
    prefix_options.add_argument("--prefix", default=" ")
    prefix_options.add_argument("--response-prefix", action="store_true",
                                help="use a K3 user request, explicit empty think and open response channel")
    args = parser.parse_args()
    os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    segments = find_segments(args.input)
    train = next(args.input.glob("**/name-that-llm/train.csv"))
    reference = next(args.input.rglob("glm_dynamics_oof.parquet"))
    frame, labels, logits = read_reference(train, reference)
    chosen, margin = choose_rows(logits, frame.fold.to_numpy(), labels.index(KIMI), args.max_rows)
    print(f"Scoring {len(chosen)} official passages at up to {args.max_tokens} tokens", flush=True)
    runner = args.runner or build_runner(args.work, segments[0][1].with_name("ntl-kimi-k3-segrun.patch"))
    if not args.runner:
        binary = args.output / runner.name
        shutil.copy2(runner, binary)
    texts = frame.text.iloc[chosen].fillna("").tolist()
    config = {"model": "Kimi-K3 UD-Q1_0", "max_tokens": args.max_tokens, "prefix": args.prefix}
    token_path = None
    if args.response_prefix:
        if __package__:
            from .kimi_response_prefix import MARKER_PROBE, ORDINARY_SEGMENTS, write_response_tokens
        else:
            from kimi_response_prefix import MARKER_PROBE, ORDINARY_SEGMENTS, write_response_tokens

        model = segments[0][1]
        baseline = tokenize_passages(runner, model, texts, args.work, args.max_tokens, args.prefix)
        probe_dir = args.work / "response-prefix"
        probe_dir.mkdir(exist_ok=True)
        probe = tokenize_passages(runner, model, ORDINARY_SEGMENTS, probe_dir, 128, MARKER_PROBE)
        token_path = probe_dir / "response-tokens.bin"
        config.update(write_response_tokens(baseline, probe, token_path, len(texts)))
        print(f"Kimi XTML prefix: {config['prefix_tokens']} tokens, "
              f"single-space baseline: {config['baseline_prefix_tokens']}", flush=True)
    values, timing = score_passages(
        runner, segments, texts, args.work, args.max_tokens, args.prefix, args.seq_batch, token_path=token_path,
    )
    matrix = np.asarray([features(value) for value in values])
    report, odds = evaluate(frame, labels, logits, chosen, matrix, margin)
    report.update(configuration=config, segments=timing)
    (args.output / "kimi_real_likelihood_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output / "kimi_real_likelihood_oof.npz",
        ids=frame.id.iloc[chosen].to_numpy(dtype=str), fold=frame.fold.to_numpy()[chosen],
        likelihood_features=matrix, corrected_log_odds=odds,
        offsets=np.r_[0, np.cumsum([len(value) for value in values])], token_scores=np.concatenate(values),
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
