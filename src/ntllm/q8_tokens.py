"""Measure the actual tokenizer inputs and physical forward token budgets.

Counts come from the tokenizer's returned IDs, including its default special
token behavior. No EOS, chat template or text normalization is added here.
"""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .q8_compute import packed_forwards


TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def tokenizer_identity(directory, tokenizer=None):
    """Bind file bytes separately from effective tokenizer runtime settings."""
    directory = Path(directory)
    files = {name: _sha((directory / name).read_bytes()) for name in TOKENIZER_FILES
             if (directory / name).is_file()}
    if not files:
        raise ValueError("No tokenizer assets were found in the supplied directory.")
    result = {"path": str(directory.resolve()), "files_sha256": files,
              "files_identity_sha256": _sha(json.dumps(files, sort_keys=True).encode())}
    if tokenizer is not None:
        result["runtime"] = dict(
            tokenizer_class=type(tokenizer).__name__, is_fast=bool(tokenizer.is_fast),
            padding_side=tokenizer.padding_side, truncation_side=tokenizer.truncation_side,
            pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            add_bos_token=getattr(tokenizer, "add_bos_token", None),
            add_eos_token=getattr(tokenizer, "add_eos_token", None),
            transformers_version=version("transformers"), tokenizers_version=version("tokenizers"),
        )
        try:
            backend = json.loads(tokenizer.backend_tokenizer.to_str())
            # Fast tokenizers mutate these transient fields on each call. Their
            # actual values are supplied by max_len and recorded batch layout.
            backend.pop("truncation", None)
            backend.pop("padding", None)
            result["backend_tokenizer_sha256"] = _sha(json.dumps(backend, sort_keys=True, separators=(",", ":")).encode())
            result["backend_identity_scope"] = "Canonical backend JSON excluding transient padding/truncation call state"
        except AttributeError:
            result["backend_tokenizer_sha256"] = None
    return result


def encode_with_statistics(tokenizer, texts, max_len, batch_size=2048, *, retain_tokens=True):
    """Return production IDs and measured before/after truncation row metadata.

    Two real tokenizer calls preserve special-token and truncation semantics.
    The returned IDs are the truncated call itself, not manually sliced IDs.
    Measurement-only callers can discard each batch's IDs after recording its
    statistics; in that case the first return value is None, not a full pool.
    """
    if max_len < 2 or batch_size < 1:
        raise ValueError("max_len must be >=2 and batch_size must be positive.")
    texts = list(texts)
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("Token accounting requires unmodified string texts.")
    encoded, records = ([] if retain_tokens else None), []
    for start in range(0, len(texts), batch_size):
        part = texts[start:start + batch_size]
        raw = tokenizer(part, truncation=False, return_attention_mask=False)["input_ids"]
        kept = tokenizer(part, truncation=True, max_length=max_len,
                         return_attention_mask=False)["input_ids"]
        if len(raw) != len(part) or len(kept) != len(part):
            raise ValueError("Tokenizer output is not aligned with input texts.")
        for index, (text, before, after) in enumerate(zip(part, raw, kept), start=start):
            removed = len(before) - len(after)
            if removed < 0 or len(after) > max_len:
                raise ValueError("Unexpected tokenizer truncation behavior.")
            records.append(dict(
                position=index, raw_text_sha256=_sha(text.encode()),
                input_ids_sha256=_sha(np.asarray(after, dtype="<i8").tobytes()),
                raw_tokens=len(before), input_tokens=len(after), removed_tokens=removed,
                truncated=removed > 0,
            ))
        if retain_tokens:
            encoded.extend(kept)
    columns = ["position", "raw_text_sha256", "input_ids_sha256", "raw_tokens",
               "input_tokens", "removed_tokens", "truncated"]
    return encoded, pd.DataFrame(records, columns=columns)


def summarize_exposure(row_stats, successful_rows):
    """Count successful exposures, including repetition after later epochs.

    A repeated source row counts again toward computation, once toward unique
    rows/texts/inputs. Distinct full texts may become the same truncated input.
    """
    positions = np.asarray(successful_rows, dtype=np.int64)
    if positions.ndim != 1 or ((positions < 0) | (positions >= len(row_stats))).any():
        raise ValueError("Token exposure positions are outside the row table.")
    part = row_stats.iloc[positions]
    count = len(part)
    return dict(
        row_exposures=count, unique_rows=int(len(np.unique(positions))),
        unique_raw_texts=int(part.raw_text_sha256.nunique()),
        unique_input_sequences=int(part.input_ids_sha256.nunique()),
        raw_tokens=int(part.raw_tokens.sum()), input_tokens=int(part.input_tokens.sum()),
        removed_tokens=int(part.removed_tokens.sum()), truncated_rows=int(part.truncated.sum()),
        truncation_rate=float(part.truncated.mean()) if count else 0.0,
    )


def physical_token_counts(tokens_or_lengths, chunk, maximum_rows, relation_tokens_or_lengths=None):
    """Count actual unpadded tokens and longest-sequence batch padding slots."""
    def length(values, index):
        value = values[index]
        return int(value) if np.isscalar(value) else len(value)

    records = []
    for pack, groups in enumerate(packed_forwards(chunk, maximum_rows)):
        plain = [row for rows, _ in groups for row in rows]
        streams = [("plain", plain, tokens_or_lengths)]
        if relation_tokens_or_lengths is not None:
            relation = [row for rows, same in groups if same for row in rows]
            if relation:
                streams.append(("anchor", relation, relation_tokens_or_lengths))
        for stream, rows, values in streams:
            lengths = [length(values, row) for row in rows]
            nonpadding = sum(lengths)
            padded = len(rows) * max(lengths)
            records.append(dict(pack=pack, stream=stream, rows=len(rows),
                                input_tokens=nonpadding, padded_tokens=padded,
                                padding_tokens=padded - nonpadding, maximum_tokens=max(lengths)))
    return records


def _slice_summary(frame, row_stats, positions):
    result = {}
    positions = np.asarray(positions, dtype=np.int64)
    metadata = frame.iloc[positions]
    for column, name in (("model", "by_model"), ("word_bin", "by_word_bin"),
                         ("source_dataset", "by_source")):
        if column not in metadata:
            continue
        result[name] = {str(key): summarize_exposure(row_stats, positions[indices])
                        for key, indices in metadata.groupby(column, observed=True, sort=True).indices.items()}
    result["by_model_word_bin"] = [dict(model=str(model), word_bin=str(word_bin),
                                       **summarize_exposure(row_stats, positions[indices]))
                                    for (model, word_bin), indices in metadata.groupby(
                                        ["model", "word_bin"], observed=True, sort=True).indices.items()]
    return result


def profile_plan(frame, row_stats, windows_by_epoch, max_updates, config, out, relation_stats=None,
                 relation_frame=None):
    """Measure the declared update prefix before training, not its full pool.

    `step_metrics` can be joined to executed train logs by successful update.
    Failed/retried physical work must be reported separately from this declared
    success budget. Shared relation rows are not counted as extra forwards.
    """
    if len(row_stats) != len(frame) or not np.array_equal(row_stats.position, np.arange(len(frame))):
        raise ValueError("Token row statistics must align with the prepared frame.")
    if relation_stats is not None and len(relation_stats) != len(frame):
        raise ValueError("Separate anchor statistics must align with plain slots.")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = [name for name in ("id", "training_slot_id", "model", "word_bin", "source_dataset",
                                  "research_condition", "protocol_evidence_level", "evidence_record_ids")
                if name in frame]
    pd.concat([frame[metadata].reset_index(drop=True), row_stats.reset_index(drop=True)], axis=1).to_parquet(
        out / "token_rows.parquet", index=False)
    if relation_stats is not None:
        relation_stats.to_parquet(out / "relation_token_rows.parquet", index=False)
    remaining = sum(map(len, windows_by_epoch)) if max_updates is None else max_updates
    if remaining < 0:
        raise ValueError("max_updates must be nonnegative.")
    lengths = row_stats.input_tokens.to_numpy()
    relation_lengths = relation_stats.input_tokens.to_numpy() if relation_stats is not None else None
    relation_effective_stats = relation_stats if relation_stats is not None else row_stats
    relation_metadata = frame if relation_stats is None else relation_frame
    if relation_metadata is None:
        # Labels and bins belong to aligned slots, but plain-source metadata
        # must never be attributed to an independent anchor input.
        relation_metadata = frame.drop(columns="source_dataset", errors="ignore")
    if len(relation_metadata) != len(frame):
        raise ValueError("Relation metadata must align with the token row table.")
    actual_rows, centered_rows, plain_only_rows, records, physical = [], [], [], [], []
    update = 0
    for epoch, windows in enumerate(windows_by_epoch):
        active = min(len(windows), remaining)
        remaining -= active
        for window, chunk in enumerate(windows[:active]):
            update += 1
            rows = [row for batch, _ in chunk for row in batch]
            centered = [row for batch, same in chunk if same for row in batch]
            plain_only = [row for batch, same in chunk if not same for row in batch]
            actual_rows.extend(rows)
            centered_rows.extend(centered)
            plain_only_rows.extend(plain_only)
            plain = summarize_exposure(row_stats, rows)
            auxiliary = summarize_exposure(relation_effective_stats, centered)
            batches = physical_token_counts(lengths, chunk, config.get("forward_batch_rows", config["batch_size"]),
                                            relation_lengths)
            for batch in batches:
                physical.append(dict(update=update, epoch=epoch + 1, window=window, **batch))
            record = dict(update=update, epoch=epoch + 1, window=window,
                          **{"plain_" + key: value for key, value in plain.items()},
                          **{"centered_" + key: value for key, value in auxiliary.items()},
                          **{"plain_only_" + key: value for key, value in summarize_exposure(row_stats, plain_only).items()},
                          physical_forward_rows=sum(item["rows"] for item in batches),
                          physical_input_tokens=sum(item["input_tokens"] for item in batches),
                          physical_padded_tokens=sum(item["padded_tokens"] for item in batches),
                          physical_padding_tokens=sum(item["padding_tokens"] for item in batches))
            records.append(record)
    step_metrics = pd.DataFrame(records)
    physical_table = pd.DataFrame(physical)
    step_metrics.to_parquet(out / "step_token_metrics.parquet", index=False)
    physical_table.to_parquet(out / "physical_token_batches.parquet", index=False)
    forward = {key: int(physical_table[key].sum()) if len(physical_table) else 0
               for key in ("rows", "input_tokens", "padded_tokens", "padding_tokens")}
    summary = dict(
        scope="Actually requested successful-update prefix; no model execution",
        updates=update, max_len=int(config["max_len"]),
        truncation_definition="token count before production truncation exceeds returned input count",
        input_identity_definition="SHA256 of little-endian int64 production input IDs; includes default special tokens",
        padding_definition="Rows times longest returned input in each actual physical forward; right padding, no pad-to-multiple",
        plain=summarize_exposure(row_stats, actual_rows),
        centered=summarize_exposure(relation_effective_stats, centered_rows),
        plain_only=summarize_exposure(row_stats, plain_only_rows),
        separate_anchor=relation_stats is not None, forward=forward,
        plain_slices=_slice_summary(frame, row_stats, actual_rows),
        centered_slices=_slice_summary(relation_metadata, relation_effective_stats, centered_rows),
        plain_only_slices=_slice_summary(frame, row_stats, plain_only_rows),
        centered_source_slices_available=relation_stats is None or relation_frame is not None,
        limitation="Padded token slots measure batch shape, not FLOPs or elapsed GPU time. Failed/retried work is not included in this success plan.",
    )
    (out / "token_profile.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return dict(summary=summary, step_metrics=step_metrics)
