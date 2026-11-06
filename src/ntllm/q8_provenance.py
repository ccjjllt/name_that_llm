"""Preserve separate identity, prompt, protocol and transformation evidence.

Evidence is not an admission score. A normalized text ID links historical
variants, but only byte-identical text and label may supply row-level protocol
claims. Source descriptions are explicitly weaker than a recorded request.
"""

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .q8_data import STRONG_EVIDENCE, file_sha, text_key

POLICY = "multidimensional_protocol_evidence_v2"
UNKNOWN = "unknown"
DATASETS = {
    "original_v3": "dhruvtibarewal/nlm-matched-clean",
    "open_pool_v15": "crtproduction/name-that-llm-open-pool",
    "extended_v8": "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
    "exact_route_v2": "zeminglii/ntl-own-exact-route-generations",
}
SUMMARY_COLUMNS = [
    "evidence_record_ids", "variant_evidence_record_ids", "evidence_conflicts",
    "model_identity_status", "prompt_identity_status", "generation_status",
    "transformation_status", "research_condition", "source_files", "origin_hints",
    "source_history_status", "protocol_evidence_level",
]


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if value is None or (not isinstance(value, (str, dict, list)) and pd.isna(value)):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _pick(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() not in ("", "nan", "None"):
            return value
    return None


def _sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def _canonical_model(value):
    value = str(value or "").rsplit("/", 1)[-1].removesuffix("@default")
    return {"gpt-5.5-2026-04-23": "gpt-5.5", "qwen3.8-max-0902": "qwen3.8-max"}.get(value, value)


def _read(path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path, keep_default_na=False)


def _unique_mapping(frame, key, value):
    if frame is None or key not in frame or value not in frame:
        return {}
    distinct = frame[[key, value]].dropna().drop_duplicates()
    valid = distinct.groupby(key)[value].nunique()
    return distinct[distinct[key].isin(valid[valid.eq(1)].index)].set_index(key)[value].to_dict()


def _source_identity(name, path, specification):
    if "source_dataset" in specification:
        return specification["source_dataset"]
    if "elliot" in name or "elliot" in str(path):
        return "elliotthankinson/ntl-exact-gens-v3"
    return DATASETS.get(name, name)


def _source_context(dataset, cards):
    """Source-wide claims, never asserted as independently observed API calls."""
    if dataset not in cards:
        return {}
    if dataset == DATASETS["extended_v8"]:
        return {
            "scope": "source_author_report",
            "request": "one user message; no system or temperature; provider defaults except recorded reasoning overrides",
            "transform": "exported generated passage; raw-envelope comparison required to establish unchanged response",
            "selection": "generation length targets drawn from official class lengths; new2 prompts informed by reconstructed official prompts; exported funnel flags retained, not applied here",
        }
    if dataset == DATASETS["exact_route_v2"]:
        return {
            "scope": "source_author_report", "request": "one user message; no system; provider defaults; max_tokens=16000",
            "transform": "surrounding whitespace stripped; author reports no other changes",
            "selection": "prompt groups chosen for label completion; target length sampled from official per-label lengths",
        }
    if dataset == "elliotthankinson/ntl-exact-gens-v3":
        return {
            "scope": "source_author_report", "request": "mixed host/calm/mt2; row setting needed to resolve",
            "transform": "host raw; host_join paragraphs joined; mt2 second-turn rewrite; filename alone does not identify a row setting",
            "selection": "exact-id/format/length filters; similarity against official passages; discriminator trained on official data and reference pool; selection history retained, not reapplied",
        }
    return {"scope": "source_description_available", "selection": "see frozen source description; not inferred per row"}


def _redact(value):
    if isinstance(value, dict):
        return {k: ("<redacted>" if str(k).lower() in ("authorization", "x-api-key", "x-goog-api-key", "api_key", "access_token") else _redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _response_text(response):
    """Extract visible text only; keep the full envelope in raw metadata."""
    if not isinstance(response, dict):
        return None, None
    if isinstance(response.get("content"), list):
        parts = [part["text"] for part in response["content"] if part.get("type") == "text" and isinstance(part.get("text"), str)]
        return ("".join(parts), "ordered Anthropic text blocks; thinking excluded") if parts else (None, None)
    choices = response.get("choices") or []
    if choices:
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, str):
            return content, "first choice assistant content"
    candidates = response.get("candidates") or []
    if candidates:
        parts = [part["text"] for part in candidates[0].get("content", {}).get("parts", []) if isinstance(part.get("text"), str) and not part.get("thought", False)]
        return ("".join(parts), "first Google candidate visible text parts; thought parts excluded") if parts else (None, None)
    return None, None


def _envelope_fields(row):
    request = row.get("request")
    if not isinstance(request, dict):
        return row
    result = dict(row)
    body = request.get("body", request)
    response = row.get("response") or {}
    response = response.get("json", response) if isinstance(response, dict) else {}
    response = response or {}
    result["request_payload"] = body
    if not isinstance(body, dict):
        return result
    config = body.get("config") or {}
    messages = body.get("messages") or []
    users = [m.get("content") for m in messages if m.get("role") == "user"]
    if not result.get("prompt_sent"):
        if len(users) == 1 and isinstance(users[0], str):
            result["prompt_sent"] = users[0]
        elif isinstance(body.get("contents"), str):
            result["prompt_sent"] = body["contents"]
    systems = [m.get("content") for m in messages if m.get("role") == "system"]
    result["system_prompt"] = body.get("system", systems if systems else config.get("system_instruction", "not_sent"))
    result["temperature"] = body.get("temperature", config.get("temperature", "not_sent"))
    result["requested_model_id"] = result.get("requested_model_id") or body.get("model")
    result["served_model_id"] = result.get("served_model_id") or response.get("model") or response.get("model_version")
    result["served_provider"] = result.get("served_provider") or response.get("provider")
    result["model_version"] = result.get("model_version") or response.get("model_version")
    result["generated_at_utc"] = result.get("generated_at_utc") or row.get("started_at_utc")
    result["response_id"] = result.get("response_id") or response.get("id") or response.get("response_id")
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        if key in body:
            result[key] = body[key]
        elif key in config:
            result[key] = config[key]
    result["reasoning_request"] = {k: body[k] for k in ("reasoning", "reasoning_effort", "thinking") if k in body} or {"parameter": "not_sent"}
    raw, method = _response_text(response)
    if raw is not None:
        result["raw_response_text"] = raw
        result["raw_response_extraction"] = method
    return result


def _record(raw, name, path, digest, number, specification, cards, bank, bodies, source_prompts=None):
    original_metadata = _redact({str(k): _clean(v) for k, v in raw.items()})
    row = _envelope_fields(original_metadata)
    text = _pick(row, "text", "final_text", "passage")
    label = _pick(row, "model", "label")
    if not isinstance(text, str) or not label:
        return None
    dataset = _pick(row, "source_dataset", "src") or _source_identity(name, path, specification)
    record_id = f"{name}:{digest[:16]}:{number}"
    identity = "text:" + _sha(text_key(text))
    requested = _pick(row, "requested_model_id", "requested_model", "request_model")
    returned = _pick(row, "served_model_id", "model_seen", "returned_model", "response_model", "openrouter_model")
    provider = _pick(row, "served_provider", "provider", "served_by")
    route = _pick(row, "provider_route", "route")
    date = _pick(row, "generated_at_utc", "generated_at", "timestamp")
    version = _pick(row, "model_version", "returned_model_version")
    has_request = isinstance(original_metadata.get("request"), dict)
    source_response = original_metadata.get("response") or {}
    source_response = source_response.get("json", source_response) if isinstance(source_response, dict) else {}
    source_response = source_response if isinstance(source_response, dict) else {}
    model_basis = {
        "label": "source_label",
        "requested_model": "recorded_request_or_envelope_identity_metadata" if has_request else "exported_row_field" if requested else UNKNOWN,
        "returned_model": "recorded_response_or_envelope_identity_metadata" if source_response else "exported_row_field" if returned else UNKNOWN,
        "provider": "recorded_response_field" if provider and provider == source_response.get("provider") else "exported_route_metadata" if provider else UNKNOWN,
        "route": "exported_route_metadata" if route else UNKNOWN,
        "generation_date": "exported_call_timestamp" if date else UNKNOWN,
        "model_version": "recorded_provider_response_field" if version and version == source_response.get("model_version") else "exported_row_field" if version else UNKNOWN,
    }
    if returned and _canonical_model(returned) != _canonical_model(label):
        model_status = "returned_label_conflict"
    elif requested and _canonical_model(requested) != _canonical_model(label):
        model_status = "requested_label_conflict"
    elif requested and returned:
        model_status = "request_and_return_recorded"
    elif returned:
        model_status = "return_recorded_request_unknown"
    elif requested:
        model_status = "request_recorded_return_unknown"
    else:
        model_status = "label_only"
    model_evidence = {"label": label, "requested_model": requested, "returned_model": returned,
                      "provider": provider, "route": route, "generation_date": date,
                      "model_version": version, "scope": "mixed_exported_evidence_not_independent_api_verification", "field_basis": model_basis}
    canon = _pick(row, "prompt_canon", "canonical_prompt_id")
    prompt_original = _pick(row, "prompt_sent", "original_prompt", "prompt_text", "prompt")
    prompt_id = _pick(row, "prompt_id", "prompt_key", "prompt_group")
    body = bodies.get(prompt_id) if dataset == DATASETS["extended_v8"] else None
    source_body = (source_prompts or {}).get(prompt_id)
    mapped = bank.get(canon)
    prompt_status = (("request_prompt_recorded" if has_request or row.get("prompt_sent") else "source_row_prompt_recorded") if prompt_original else
                     "body_and_source_id_recorded" if body else
                     "source_prompt_file_mapped" if source_body else
                     "canonical_bank_mapped" if mapped else
                     "identifier_only" if canon or prompt_id else UNKNOWN)
    prompt_evidence = {"request_prompt": prompt_original, "source_prompt_id": prompt_id,
                       "scope": "mixed_evidence_see_field_basis",
                       "field_basis": {
                           "request_prompt": "recorded_request" if has_request and prompt_original else "exported_request_prompt_field" if row.get("prompt_sent") else "source_row_prompt_not_api_log" if prompt_original else UNKNOWN,
                           "source_prompt_body": "source_prompt_id_lookup" if body else UNKNOWN,
                           "supplemental_prompt_body": "documented_prompt_bank_id_lookup_not_sent_message" if source_body else UNKNOWN,
                           "canonical_bank_text": "canonical_key_lookup_not_sent_message" if mapped else UNKNOWN,
                           "canonical_identity": "source_canonical_key" if canon else UNKNOWN,
                       },
                       "source_prompt_body": body, "supplemental_prompt_body": source_body,
                       "supplemental_prompt_file": str(specification.get("prompt_file", "")),
                       "supplemental_prompt_scope": specification.get("prompt_file_scope", "unique exact prompt-id lookup; not an API request log"),
                       "canonical_identity": canon,
                       "canonical_bank_text": mapped, "source_mapping_evidence": row.get("prompt_evidence"),
                       "mapping_basis": "exact source canonical-key lookup; bank text is not asserted to be the sent prompt" if mapped else
                                        "exact source prompt-id lookup" if body else "source row only"}
    setting = _pick(row, "setting", "generation_setting")
    system = _pick(row, "system_prompt", "system")
    temperature = _pick(row, "temperature", "temp")
    reasoning = _pick(row, "reasoning_setting", "reasoning", "reasoning_effort")
    limits = {k: row[k] for k in ("max_tokens", "max_completion_tokens", "max_output_tokens", "word_min", "word_max", "target_words") if row.get(k) not in (None, "")}
    request = _pick(row, "request_payload", "request")
    generation = {"system_prompt": system, "temperature": temperature, "reasoning": reasoning,
                  "output_limits": limits, "setting": setting, "tier": row.get("tier"), "template": row.get("template"),
                  "request_payload": request, "reasoning_request": row.get("reasoning_request"), "sampling": row.get("sampling"),
                  "scope": "mixed_evidence_see_field_basis",
                  "field_basis": {
                      "system_prompt": "recorded_request" if has_request else "exported_row_field" if system is not None else UNKNOWN,
                      "temperature": "recorded_request" if has_request else "exported_row_field" if temperature is not None else UNKNOWN,
                      "reasoning": "exported_reasoning_annotation_not_necessarily_request_parameter" if reasoning else UNKNOWN,
                      "reasoning_request": "recorded_request" if has_request else UNKNOWN,
                      "output_limits": {k: ("recorded_request" if has_request and k.startswith("max_") else "exported_row_field") for k in limits},
                      "setting": "exported_row_field" if setting else UNKNOWN,
                      "request_payload": "recorded_request" if has_request else UNKNOWN,
                      "sampling": "exported_sampling_annotation" if row.get("sampling") else UNKNOWN,
                      "documented_setting": "row_setting_plus_source_doc",
                      "documented_defaults": "source_author_report",
                      "documented_source_protocol": "source_author_report",
                  }}
    generation_status = "partial_row_record" if any(x is not None for x in (system, temperature, reasoning, request, setting)) or limits else UNKNOWN
    if isinstance(request, dict):
        generation_status = "request_payload_recorded"
    # Only a row setting plus its specific source description binds a protocol.
    if not isinstance(request, dict) and dataset in cards and dataset == "elliotthankinson/ntl-exact-gens-v3" and setting in ("host", "host_join", "calm", "mt2"):
        generation_status = "row_setting_source_documented"
        generation["documented_setting"] = {
            "host": "user only; system and temperature not sent; provider defaults with paid reasoning overrides",
            "host_join": "host generation followed by paragraph join",
            "calm": "system=You are a helpful assistant.; temperature=0.3 where accepted",
            "mt2": "second-turn request to rewrite passage more concisely in one paragraph",
        }[setting]
    elif not isinstance(request, dict) and dataset in cards and dataset == DATASETS["extended_v8"]:
        generation_status = "partial_row_and_source_documented"
        generation["documented_defaults"] = "system and temperature not sent; source-level claim; row reasoning retained"
    elif not isinstance(request, dict) and dataset in cards and dataset == DATASETS["exact_route_v2"]:
        generation_status = "source_documented_row_limits"
        generation["documented_defaults"] = "system not sent; provider defaults; max_tokens=16000; source-level claim"
    claims = specification.get("protocol_claims") or {}
    if claims.get("generation"):
        generation["documented_source_protocol"] = claims["generation"]
        generation["documented_source_protocol_ref"] = claims.get("evidence_ref")
        if not isinstance(request, dict) and generation_status in (UNKNOWN, "partial_row_record"):
            generation_status = "source_documented_protocol" if claims.get("generation_complete") is True else "source_documented_partial"
    original_response = _pick(row, "raw_response_text", "original_response", "original_text", "raw_text")
    response_id = _pick(row, "response_id", "call_id")
    parent_response = _pick(row, "parent_response_id", "original_response_id")
    transform = _pick(row, "transformation", "generation_transformation", "transform_type")
    operation_basis = "exported_row_field" if transform else UNKNOWN
    if dataset == "elliotthankinson/ntl-exact-gens-v3" and dataset in cards:
        if setting == "host_join":
            transform = "paragraph_join"
            operation_basis = "row_setting_plus_source_doc"
        elif setting == "mt2":
            transform = "second_turn_rewrite"
            operation_basis = "row_setting_plus_source_doc"
    declared_transform = claims.get("transformation")
    declared_transform = {"strip_surrounding_whitespace": "boundary_whitespace_trim", "none_reported": "unchanged"}.get(declared_transform, declared_transform)
    if not declared_transform and claims.get("unchanged_claim"):
        declared_transform = "unchanged"
    if not transform and declared_transform:
        transform = declared_transform
        operation_basis = "source_author_report"
    changed = bool(transform and str(transform).lower() not in ("none", "identity", "raw", "unchanged", "boundary_whitespace_trim"))
    transformation_status = "explicit_transformation_recorded" if changed else UNKNOWN
    if declared_transform and changed:
        transformation_status = "transformation_source_reported"
    elif declared_transform in ("none", "identity", "raw", "unchanged"):
        transformation_status = "unmodified_source_reported"
    elif declared_transform == "boundary_whitespace_trim":
        transformation_status = "whitespace_trim_source_reported"
    if original_response is not None:
        if original_response == text:
            transformation_status = "raw_equals_final"
            if not transform:
                operation_basis = "raw_response_comparison"
        elif isinstance(original_response, str) and original_response.strip() == text:
            transformation_status = "whitespace_trim_observed"
            transform = transform or "boundary_whitespace_trim"
            operation_basis = "raw_response_comparison"
        else:
            transformation_status = "raw_differs_final_operation_unknown" if not changed else "raw_differs_final_explicit_operation"
    elif dataset == "elliotthankinson/ntl-exact-gens-v3" and dataset in cards and setting == "host":
        transformation_status = "unmodified_source_reported"
        operation_basis = "row_setting_plus_source_doc"
    elif dataset == DATASETS["exact_route_v2"] and dataset in cards:
        transformation_status = "whitespace_trim_source_reported"
        operation_basis = "source_author_report"
    transformation = {"operation": transform, "raw_response_text": original_response,
                      "raw_response_status": "recorded" if original_response is not None else UNKNOWN,
                      "raw_response_extraction": row.get("raw_response_extraction"),
                      "response_id": response_id, "parent_response_id": parent_response,
                      "final_text_sha256": _sha(text), "final_text_locator": record_id,
                      "scope": "mixed_evidence_see_field_basis", "operation_basis": operation_basis,
                      "field_basis": {"operation": operation_basis,
                                      "raw_response_text": "recorded_response_visible_text_extraction" if row.get("raw_response_extraction") else "explicit_source_raw_response_field" if original_response is not None else UNKNOWN,
                                      "final_text": "source_export_text",
                                      "response_id": "exported_response_identity" if response_id else UNKNOWN,
                                      "parent_response_id": "explicit_exported_parent_identity" if parent_response else UNKNOWN,
                                      "comparison": "raw_response_comparison" if original_response is not None else UNKNOWN,
                                      "source_transformation_claim": "source_author_report" if declared_transform else UNKNOWN},
                      "is_truncated_flag": row.get("is_truncated"),
                      "finish_reason": _pick(row, "finish_reason_norm", "finish_reason"),
                      "source_transformation_ref": claims.get("evidence_ref") if declared_transform else None,
                      "source_transformation_claim": declared_transform}
    if generation_status == "request_payload_recorded":
        protocol_evidence_level = "request_logged"
    elif generation_status in ("row_setting_source_documented", "source_documented_row_limits", "source_documented_protocol"):
        protocol_evidence_level = "source_documented"
    elif generation_status == UNKNOWN:
        protocol_evidence_level = UNKNOWN
    else:
        protocol_evidence_level = "partial"
    if changed and operation_basis == "source_author_report" and transform == "whitespace_collapse":
        condition = "normalization_reported"
    elif changed or transformation_status == "raw_differs_final_operation_unknown":
        condition = "transformed"
    elif generation_status in ("request_payload_recorded", "row_setting_source_documented", "source_documented_row_limits", "source_documented_protocol") and transformation_status in ("raw_equals_final", "unmodified_source_reported", "whitespace_trim_source_reported", "whitespace_trim_observed"):
        condition = "protocol_documented"
    else:
        condition = "protocol_unknown"
    source_file = _pick(row, "source_file") or path.name
    field_claims = {"model.requested": _canonical_model(requested) if requested else None,
                    "model.returned": _canonical_model(returned) if returned else None,
                    "prompt.request_text": prompt_original, "generation.system": system,
                    "generation.temperature": temperature,
                    "generation.reasoning_request": row.get("reasoning_request"),
                    "transformation.operation": transform if changed else None}
    field_claims = {k: v for k,v in field_claims.items() if v is not None}
    history = {"source_dataset": dataset, "source_file": source_file, "origin_hint": row.get("origin_hint"),
               "source_record_id": row.get("id"), "also_in": row.get("also_in"),
               "source_file_basis": "row source_file" if row.get("source_file") else "import container; upstream file not separately known",
               "selection_fields": {k: v for k, v in row.items() if any(t in k for t in ("funnel", "discriminator", "match_weight", "style_score"))},
               "source_description_ref": cards.get(dataset, {}).get("path"),
               "source_claims": _source_context(dataset, cards), "additional_protocol_claims": claims}
    return {
        "evidence_record_id": record_id, "id": identity, "model": str(label), "exact_text_sha256": _sha(text),
        "final_text": text, "source_pool": name, "source_path": str(path.resolve()), "source_sha256": digest,
        "source_row": number, "source_version": str(specification.get("version", UNKNOWN)),
        "source_dataset": dataset, "source_file": str(source_file), "origin_hint": str(row.get("origin_hint") or ""),
        "source_record_id": str(row.get("id") or ""), "model_identity_status": model_status,
        "prompt_identity_status": prompt_status, "generation_status": generation_status,
        "transformation_status": transformation_status, "research_condition": condition,
        "protocol_evidence_level": protocol_evidence_level,
        "model_evidence_json": _json(model_evidence), "prompt_evidence_json": _json(prompt_evidence),
        "generation_evidence_json": _json(generation), "transformation_evidence_json": _json(transformation),
        "source_history_json": _json(history), "field_claims_json": _json(field_claims),
        "source_history_status": "row_locator_and_source_description" if dataset in cards else "row_locator_only",
        "raw_metadata_json": _json({k: v for k, v in original_metadata.items() if k not in ("text", "final_text", "passage")}),
    }


def build_evidence(sources, *, prompt_bank=None, extended_prompts=None, source_cards=None):
    """Read frozen files into a lossless evidence ledger without choosing rows.

    sources maps names to paths, or to {path, source_dataset, version}. All raw
    metadata and exact source-row locators survive. Missing fields stay null;
    source-document claims do not overwrite row fields. Returned text is never
    silently treated as the raw response.
    """
    cards = {}
    for dataset, value in (source_cards or {}).items():
        path = Path(value)
        cards[dataset] = {"path": str(path.resolve()), "sha256": file_sha(path)}
    bank_frame = _read(Path(prompt_bank)) if prompt_bank else None
    prompt_frame = _read(Path(extended_prompts)) if extended_prompts else None
    bank = _unique_mapping(bank_frame, "prompt_canon", "prompt_text")
    bodies = _unique_mapping(prompt_frame, "prompt_id", "prompt_body")
    records, reports = [], {}
    for name, spec in sources.items():
        spec = dict(spec) if isinstance(spec, dict) else {"path": spec}
        path = Path(spec["path"])
        digest = file_sha(path)
        raw = _read(path)
        source_prompts, prompt_reference = {}, None
        if spec.get("prompt_file"):
            prompt_path = Path(spec["prompt_file"])
            prompt_rows = _read(prompt_path)
            source_prompts = _unique_mapping(prompt_rows, "prompt_id", "prompt")
            prompt_reference = {"path": str(prompt_path.resolve()), "sha256": file_sha(prompt_path), "unambiguous_ids": len(source_prompts)}
        before = len(records)
        for number, row in enumerate(raw.to_dict(orient="records")):
            record = _record(row, name, path, digest, number, spec, cards, bank, bodies, source_prompts)
            if record is not None:
                records.append(record)
        reports[name] = {"path": str(path.resolve()), "sha256": digest, "raw_rows": len(raw),
                         "evidence_rows": len(records) - before, "columns": list(raw.columns), "prompt_reference": prompt_reference,
                         "protocol_claims": spec.get("protocol_claims", {})}
    ledger = pd.DataFrame.from_records(records)
    report = {"policy": POLICY, "sources": reports, "source_cards": cards,
              "records": len(ledger), "condition_counts": ledger.research_condition.value_counts().to_dict(),
              "model_status_counts": ledger.model_identity_status.value_counts().to_dict(),
              "prompt_status_counts": ledger.prompt_identity_status.value_counts().to_dict(),
              "generation_status_counts": ledger.generation_status.value_counts().to_dict(),
              "protocol_evidence_level_counts": ledger.protocol_evidence_level.value_counts().to_dict(),
              "transformation_status_counts": ledger.transformation_status.value_counts().to_dict(),
              "interpretation": "Research conditions describe evidence availability, not correctness or quality. Source-author claims remain distinct from request envelopes; no rows excluded here."}
    return ledger, report


def attach_evidence(frame, evidence):
    """Attach conservative summaries; preserve every conflicting source record.

    Exact equality protects paragraph-joined/case/Unicode variants from receiving
    each other's row-level protocol evidence despite the legacy normalized ID.
    """
    result = frame.copy()
    groups = {}
    columns = ["id", "model", "exact_text_sha256", "evidence_record_id", "research_condition",
               "model_identity_status", "prompt_identity_status", "generation_status",
               "transformation_status", "source_file", "origin_hint", "field_claims_json", "source_history_status", "protocol_evidence_level"]
    for row in evidence[columns].itertuples(index=False, name=None):
        groups.setdefault((row[0], row[1]), []).append(row[2:])
    summaries = []
    for ident, label, text in zip(result.id, result.model, result.text):
        group = groups.get((ident, label), [])
        digest = _sha(text)
        exact = [r for r in group if r[0] == digest]
        variant = [r for r in group if r[0] != digest]
        conflicts = []
        conditions = {r[2] for r in exact}
        if "transformed" in conditions and "protocol_documented" in conditions:
            conflicts.append("transformed_and_documented_original_claims")
        claims = {}
        for row in exact:
            for field, value in json.loads(row[9]).items():
                claims.setdefault(field, set()).add(_json(value))
            if row[3] == "returned_label_conflict":
                conflicts.append("returned_model_differs_from_label")
            if row[3] == "requested_label_conflict":
                conflicts.append("requested_model_differs_from_label")
        conflicts.extend("multiple_recorded_values:" + key for key, values in claims.items() if len(values) > 1)
        conflicts = sorted(set(conflicts))
        statuses = {}
        for index, column in enumerate(("model_identity_status", "prompt_identity_status", "generation_status", "transformation_status"), 3):
            values = sorted({r[index] for r in exact} - {UNKNOWN, "label_only", "identifier_only"})
            statuses[column] = "|".join(values) if values else ("label_only" if column == "model_identity_status" else UNKNOWN)
        condition = ("protocol_unknown" if conflicts else "transformed" if "transformed" in conditions else
                     "normalization_reported" if "normalization_reported" in conditions else
                     "protocol_documented" if "protocol_documented" in conditions else "protocol_unknown")
        levels = {r[11] for r in exact}
        evidence_level = ("partial" if conflicts else "request_logged" if "request_logged" in levels else
                          "source_documented" if "source_documented" in levels else "partial" if "partial" in levels else UNKNOWN)
        summaries.append({"evidence_record_ids": _json([r[1] for r in exact]),
                          "variant_evidence_record_ids": _json([r[1] for r in variant]),
                          "evidence_conflicts": _json(conflicts), **statuses, "research_condition": condition,
                          "source_files": _json(sorted({r[7] for r in exact})),
                          "origin_hints": _json(sorted({r[8] for r in exact} - {""})),
                          "source_history_status": "|".join(sorted({r[10] for r in exact})) or UNKNOWN,
                          "protocol_evidence_level": evidence_level})
    for column in SUMMARY_COLUMNS:
        result[column] = [row[column] for row in summaries]
    return result


def confirmed_transformation_pairs(evidence):
    """Extract only explicit before/after responses, never fuzzy or prompt pairs.

    An exported raw response beside its final text is a direct pair. Otherwise an
    explicit parent_response_id must resolve uniquely to an unmodified response
    from the same source dataset and label; a shared prompt or row ID is not enough.
    """
    parsed = [(row, json.loads(row.transformation_evidence_json)) for row in evidence.itertuples(index=False)]
    by_response = {}
    for row, transform in parsed:
        if transform.get("response_id"):
            by_response.setdefault((row.source_dataset, row.model, transform["response_id"]), []).append((row, transform))
    pairs, unresolved = [], 0
    for row, transform in parsed:
        if row.research_condition != "transformed":
            continue
        raw = transform.get("raw_response_text")
        if isinstance(raw, str) and _sha(raw) != row.exact_text_sha256:
            pairs.append({"after_evidence_record_id": row.evidence_record_id,
                          "before_evidence_record_id": None, "model": row.model,
                          "before_text": raw, "before_text_sha256": _sha(raw),
                          "after_text_sha256": row.exact_text_sha256, "pair_basis": "explicit_raw_response_with_final_text",
                          "operation": transform.get("operation") or "unspecified_change"})
            continue
        parent = transform.get("parent_response_id")
        candidates = by_response.get((row.source_dataset, row.model, parent), []) if parent else []
        candidates = [(r,t) for r,t in candidates if r.transformation_status in ("raw_equals_final", "unmodified_source_reported")]
        hashes = {r.exact_text_sha256 for r,t in candidates}
        if len(hashes) == 1 and row.exact_text_sha256 not in hashes:
            pairs.append({"after_evidence_record_id": row.evidence_record_id,
                          "before_evidence_record_id": candidates[0][0].evidence_record_id, "model": row.model,
                          "before_text": None, "before_text_sha256": next(iter(hashes)),
                          "after_text_sha256": row.exact_text_sha256, "pair_basis": "explicit_parent_response_id",
                          "operation": transform.get("operation") or "unspecified_change"})
        else:
            unresolved += 1
    columns = ["after_evidence_record_id", "before_evidence_record_id", "model", "before_text", "before_text_sha256", "after_text_sha256", "pair_basis", "operation"]
    result = pd.DataFrame(pairs, columns=columns)
    return result, {"confirmed_pairs": len(result), "unresolved_transformed_records": unresolved,
                    "interpretation": "Only explicit same-response links permit transformation attribution. Different-response source swaps estimate a source/protocol combination effect."}
