"""Run source-conditioned likelihood on final, close candidate decisions."""

import multiprocessing as mp
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

if __package__:
    from .source_routing import select_candidate_rows
else:
    from source_routing import select_candidate_rows


def _score_worker(pipe, gpu, texts, candidates, labels, model_dir, checkpoints):
    import torch
    from transformers import AutoTokenizer

    if __package__:
        from .source_likelihood import SourceLikelihood, encode, score_candidates
    else:
        from source_likelihood import SourceLikelihood, encode, score_candidates

    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(gpu)
        device = torch.device("cuda", gpu)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        model = SourceLikelihood(model_dir, len(labels), device)
        parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
        ratios = np.zeros(candidates.shape, dtype=np.float64)
        settings = None
        for path in checkpoints:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            current = (checkpoint["max_tokens"], checkpoint["score_tokens"])
            if sorted(checkpoint["labels"]) != sorted(labels) or (settings is not None and current != settings):
                raise ValueError("Source checkpoint labels or token settings disagree")
            if settings is None:
                settings = current
                encoded = encode(tokenizer, texts, *settings)
            values = checkpoint["weights"]
            if parameters.keys() != values.keys():
                raise ValueError("Source checkpoint does not match the routed parameters")
            with torch.no_grad():
                for name, value in parameters.items():
                    value.copy_(values[name])
            order = np.array([checkpoint["labels"].index(label) for label in labels])
            ratios += score_candidates(model, encoded, np.arange(len(texts)), order[candidates], pad, device, batch_size=6)
        pipe.send((ratios / len(checkpoints), None))
    except Exception as error:
        pipe.send((None, f"{type(error).__name__}: {error}"))
    finally:
        pipe.close()


def candidate_posterior(logp, rows, candidates, ratios, scale):
    """Reallocate each candidate set's existing total mass."""
    updated = logp.copy()
    indices = (rows[:, None], candidates)
    old = logp[indices]
    new = old + scale * ratios
    updated[indices] = new - logsumexp(new, axis=1, keepdims=True) + logsumexp(old, axis=1, keepdims=True)
    return updated


def rerank_sources(texts, probs, labels, current, model_dir, checkpoints, scale, deadline, max_rows=480):
    """Return a complete prediction array if all selected rows finish before the deadline."""
    import torch

    original = np.asarray(current)
    if not scale or len(checkpoints) != 5 or deadline - time.time() < 90:
        return original
    labels = list(labels)
    label_values = np.asarray(labels)
    text_values = np.asarray(texts, dtype=str)
    with np.errstate(divide="ignore"):
        logp = np.log(probs)
    eligible = np.flatnonzero(label_values[logp.argmax(axis=1)] == original)
    if not len(eligible):
        return original
    words = [len(text.split()) for text in text_values[eligible]]
    coverage = min(0.25, max_rows / len(eligible))
    selected, candidates = select_candidate_rows(logp[eligible], words, coverage, candidate_count=3)
    rows = eligible[selected]
    workers = min(torch.cuda.device_count(), len(rows), 2)
    if not workers:
        return original

    context = mp.get_context("spawn")
    active = []
    ratios = np.empty(candidates.shape, dtype=np.float64)
    try:
        for gpu in range(workers):
            subset = np.arange(gpu, len(rows), workers)
            receive, send = context.Pipe(duplex=False)
            process = context.Process(
                target=_score_worker,
                args=(send, gpu, text_values[rows[subset]].tolist(), candidates[subset], labels, model_dir, checkpoints),
            )
            process.start()
            send.close()
            active.append((receive, process, subset))
        for receive, _, subset in active:
            if not receive.poll(max(0.0, deadline - time.time())):
                raise TimeoutError("Source likelihood exceeded its notebook time allowance")
            values, error = receive.recv()
            if error is not None:
                raise RuntimeError(error)
            ratios[subset] = values
        if time.time() >= deadline or not np.isfinite(ratios).all():
            return original
    finally:
        for receive, process, _ in active:
            receive.close()
            process.join(timeout=min(0.5, max(0.0, deadline - time.time())))
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)

    if time.time() >= deadline:
        return original
    updated = candidate_posterior(logp, rows, candidates, ratios, scale)
    winner = updated[rows].argmax(axis=1)
    in_candidates = (winner[:, None] == candidates).any(axis=1)
    final = original.copy()
    final[rows[in_candidates]] = label_values[winner[in_candidates]]
    print(f"Source likelihood: scored {len(rows)} passages, changed {int((final != original).sum())}", flush=True)
    return final


def patch_source_notebook(notebook, checkpoint_kernel, scale, max_rows=480, hours=8 + 55 / 60):
    """Add a final-pass source reranker to original or GLM-posterior sanpier v2."""
    notebook = deepcopy(notebook)
    module_dir = Path(__file__).parent
    inference_cell = None
    apex_found = False
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        if code.startswith("%%writefile apex_postprocess.py"):
            if "return out,refined" not in code or "predictions,_=apply_arrays(" not in code:
                raise ValueError("Unexpected Apex postprocessing source")
            code = code.replace("return out,refined", "return out,probs", 1)
            code = code.replace("corrected=False,penalty=0.):\n    labels=",
                                "corrected=False,penalty=0.,return_probs=False):\n    labels=", 1)
            code = code.replace("    predictions,_=apply_arrays(", "    predictions,probs=apply_arrays(", 1)
            code = code.replace("    return result\n", "    return (result, probs, labels) if return_probs else result\n", 1)
            if "def apply_postprocessing" not in code or code.count("return_probs") != 2:
                raise ValueError("Unexpected Apex postprocessing source")
            cell["source"] = code
            apex_found = True
        elif "    final=apply_postprocessing(" in code:
            code = code.replace("    final=apply_postprocessing(", "    final,source_probs,source_labels=apply_postprocessing(", 1)
            code = code.replace("strength=0.1, corrected=False, penalty=0.05)",
                                "strength=0.1, corrected=False, penalty=0.05, return_probs=True)", 1)
            if "return_probs=True" not in code:
                raise ValueError("Unexpected Apex invocation")
            inference_cell = cell
            inference_code = code
    if not apex_found or inference_cell is None:
        raise ValueError("The notebook does not contain sanpier v2 Apex inference")

    module_sources = {name: (module_dir / f"{name}.py").read_text()
                      for name in ("source_routing", "source_likelihood", "source_inference")}
    rendered = f'''
if not FAST_COMMIT:
    from pathlib import Path
    import os
    import time
    modules = {module_sources!r}
    for name, module_source in modules.items():
        Path(f"{{name}}.py").write_text(module_source)
    from source_inference import rerank_sources
    deadline = NB_T0 + {hours!r} * 3600
    if deadline - time.time() >= 90:
        adjacent = Path(OUT).with_name("submission_sources.parquet")
        try:
            input_root = Path("/kaggle/input")
            base = next(input_root.glob("**/qwen-3/[Tt]ransformers/1.7b-base/1/config.json")).parent
            owner, slug = {checkpoint_kernel!r}.split("/", 1)
            roots = (input_root / "notebooks" / owner / slug, input_root / slug)
            weights_root = next((root for root in roots if root.exists()), None)
            folds = [next(weights_root.rglob(f"source_likelihood_fold{{k}}.pt")) for k in range(5)] if weights_root else []
            prediction = rerank_sources(
                query_df.text.fillna("").tolist(), source_probs, source_labels, final.model.to_numpy(),
                base, folds, {scale!r}, deadline, max_rows={max_rows!r},
            )
            if time.time() < deadline and (prediction != final.model.to_numpy()).any():
                replaced = final.assign(model=prediction)
                replaced.to_parquet(adjacent, index=False)
                if time.time() < deadline:
                    os.replace(adjacent, OUT)
                    final = replaced
        except (OSError, ValueError, RuntimeError, TimeoutError, StopIteration, EOFError) as error:
            print(f"Source likelihood did not produce a complete reranking: {{error}}", flush=True)
        finally:
            adjacent.unlink(missing_ok=True)
'''
    last = inference_code.index("assert final.id.astype(str)")
    inference_cell["source"] = inference_code[:last] + rendered + "\n" + inference_code[last:]
    return notebook
