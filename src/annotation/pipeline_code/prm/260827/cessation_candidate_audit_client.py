#!/usr/bin/env python3
"""Recall and audit cessation candidates in materialized PRM traces.

The script runs after ``prm_step_materializer.py``.  It deterministically
recalls every C-labelled step after the first error (plus every existing
cessation), asks an OpenAI-compatible model for one binary judgment per
candidate, and conservatively promotes only accepted high-confidence results.

Successful audits are append-only and resumable.  Refined and review JSONL
files are rebuilt deterministically from the input and all successful audits.
``--max-retries 3`` means one initial attempt plus at most three retries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "cessation_candidate_audit_client_v1.1"
INPUT_SCHEMA = "cessation_candidate_input_v1"
OUTPUT_SCHEMA = "cessation_binary_output_v1"
AUDIT_SCHEMA = "cessation_candidate_audit_v1"
ERROR_SCHEMA = "cessation_candidate_error_v1"
MATERIALIZED_SCHEMA = "prm_materialized_trace_v1"
DEFAULT_MODEL = "Qwen3.8-27B"
DEFAULT_ENDPOINT = "http://127.0.0.1:4000/v1/chat/completions"
STEP_RE = re.compile(r"^s\d+$")
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.I)
RECOVERY_CUE_RE = re.compile(
    r"\b(correct(?:ion|ed|ing)?|recalculat(?:e|ed|ing)|recomput(?:e|ed|ing)|"
    r"instead|alternatively|however|actually|another\s+(?:way|method)|"
    r"re-?evaluat(?:e|ed|ing)|previous|earlier|wrong|error|mistake|revise)\b|"
    r"纠正|改正|重新计算|重新推导|错误|有误|换一种|另一种方法|实际上|应为",
    re.I,
)
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
LOG_LOCK = threading.Lock()


class OutputValidationError(ValueError):
    """A model response violates the binary audit contract."""


@dataclass(frozen=True)
class SourceRecord:
    line_number: int
    raw: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class Candidate:
    order: int
    key: str
    record_id: str
    step_id: str
    source: SourceRecord
    prepared: dict[str, Any]
    prepared_sha256: str
    audit_context: dict[str, Any]


@dataclass
class WorkResult:
    candidate: Candidate
    audit: dict[str, Any] | None
    error: dict[str, Any] | None


def log(message: str) -> None:
    with LOG_LOCK:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}",
            file=sys.stderr,
            flush=True,
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(stream: Any, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    stream.flush()


def output_sibling(path: Path, suffix: str) -> Path:
    name = path.name[:-6] if path.name.endswith(".jsonl") else path.stem
    return path.with_name(name + suffix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recall and binary-audit cessation candidates in PRM traces."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--candidate-output", type=Path, default=None)
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--review-output", type=Path, default=None)
    parser.add_argument("--error-output", type=Path, default=None)
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--max-in-flight", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--response-format", choices=["json-object", "none"], default="json-object"
    )
    parser.add_argument(
        "--min-confidence",
        choices=["high", "medium", "low"],
        default="high",
        help="Minimum confidence for automatic correct->cessation promotion.",
    )
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_in_flight is None:
        args.max_in_flight = args.workers * 2
    if args.max_in_flight < args.workers:
        parser.error("--max-in-flight must be at least --workers")
    if args.max_retries < 0 or args.retry_backoff < 0:
        parser.error("retry values cannot be negative")
    if args.timeout <= 0 or args.max_tokens < 1:
        parser.error("--timeout and --max-tokens must be positive")
    if args.heartbeat_seconds <= 0 or args.log_every < 1:
        parser.error("heartbeat/log values must be positive")
    if args.start_line < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--start-line and --limit must be positive")
    if not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1:
        parser.error("invalid sampling parameters")
    defaults = {
        "candidate_output": ".cessation_candidates.jsonl",
        "audit_output": ".cessation_audits.jsonl",
        "review_output": ".cessation_review.jsonl",
        "error_output": ".cessation_errors.jsonl",
        "stats_output": ".cessation_stats.json",
    }
    for field, suffix in defaults.items():
        if getattr(args, field) is None:
            setattr(args, field, output_sibling(args.output, suffix))
    return args


def read_materialized(path: Path) -> list[SourceRecord]:
    if path.suffix.lower() != ".jsonl":
        raise ValueError("--input must be a JSONL file")
    records: list[SourceRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            validate_materialized(value, line_number)
            record_id = value["id"]
            if record_id in seen:
                raise ValueError(f"duplicate input id: {record_id}")
            seen.add(record_id)
            records.append(SourceRecord(line_number, value, sha256_bytes(canonical_bytes(value))))
    return records


def validate_materialized(record: Any, line_number: int) -> None:
    if not isinstance(record, dict) or record.get("schema") != MATERIALIZED_SCHEMA:
        raise ValueError(f"line {line_number}: expected {MATERIALIZED_SCHEMA}")
    if not isinstance(record.get("id"), str) or not record["id"]:
        raise ValueError(f"line {line_number}: invalid id")
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{record['id']}: steps must be non-empty")
    ids = [step.get("id") for step in steps if isinstance(step, dict)]
    if len(ids) != len(steps) or len(ids) != len(set(ids)) or any(
        not isinstance(item, str) or not STEP_RE.fullmatch(item) for item in ids
    ):
        raise ValueError(f"{record['id']}: invalid/duplicate step ids")
    first_error = record.get("first_error")
    if first_error is not None and first_error not in ids:
        raise ValueError(f"{record['id']}: first_error is not a step")
    if first_error is not None:
        first = steps[ids.index(first_error)]
        if first.get("role") != "new_error":
            raise ValueError(f"{record['id']}: first_error must have role new_error")


def candidate_reasons(
    record: dict[str, Any], step: dict[str, Any], index: int, error_indices: set[int]
) -> list[str]:
    reasons: list[str] = []
    if step.get("role") == "cessation":
        reasons.append("existing_cessation")
    else:
        reasons.append("correct_after_first_error")
    if index - 1 in error_indices:
        reasons.append("immediately_after_error")
    if RECOVERY_CUE_RE.search(str(step.get("text", ""))):
        reasons.append("lexical_recovery_cue")
    supporting = {
        support
        for answer in record.get("answers", [])
        if isinstance(answer, list) and len(answer) == 3 and isinstance(answer[2], list)
        for support in answer[2]
    }
    if step.get("id") in supporting:
        reasons.append("direct_answer_support")
    return reasons


def evidence_step(step: dict[str, Any]) -> dict[str, Any]:
    """Return only observable reasoning evidence for the binary auditor.

    Teacher-B judgments such as role, PRM label, caused_by, error type, and note
    are intentionally excluded.  In particular, notes often restate the prior
    classification and would let the binary auditor confirm a label instead of
    independently evaluating the step text.
    """
    return {
        key: step.get(key)
        for key in (
            "id", "text", "op", "in", "assume", "out",
            "audited_dependency_incoming",
        )
    }


def existing_annotation(step: dict[str, Any]) -> dict[str, Any]:
    """Preserve upstream annotation for local audit logs, never model input."""
    return {
        key: step.get(key)
        for key in (
            "prm_label", "role", "error_type", "caused_by", "note",
            "mixed_error", "relation_to_first_error",
        )
    }


def build_candidates(records: list[SourceRecord]) -> list[Candidate]:
    candidates: list[Candidate] = []
    order = 0
    for source in records:
        record = source.raw
        steps = record["steps"]
        ids = [step["id"] for step in steps]
        first_error = record.get("first_error")
        if first_error is None:
            continue
        first_index = ids.index(first_error)
        error_indices = {
            index
            for index, step in enumerate(steps)
            if step.get("role") in {"new_error", "propagated_error"}
        }
        for index, step in enumerate(steps):
            if index <= first_index:
                continue
            if step.get("prm_label") != "C" or step.get("role") not in {
                "correct", "cessation"
            }:
                continue
            eligible_errors = [
                {
                    "id": prior["id"],
                    "text": prior.get("text"),
                    "out": prior.get("out", []),
                }
                for prior in steps[:index]
                if prior.get("role") in {"new_error", "propagated_error"}
            ]
            if not eligible_errors:
                continue
            prepared = {
                "schema": INPUT_SCHEMA,
                "id": record["id"],
                "context": record.get("context", {}),
                "reference_answer": record.get("reference_answer", []),
                "proposed_answer": record.get("proposed_answer"),
                "eligible_error_steps": eligible_errors,
                "prefix_steps": [evidence_step(item) for item in steps[: index + 1]],
                "candidate_step": evidence_step(step),
                "policy": {
                    "reference_answer_role": "advisory",
                    "later_steps_visible": False,
                    "upstream_annotations_visible": False,
                },
            }
            audit_context = {
                "candidate_reasons": candidate_reasons(
                    record, step, index, error_indices
                ),
                "existing_overall": record.get("overall"),
                "existing_first_error": first_error,
                "existing_candidate_annotation": existing_annotation(step),
            }
            order += 1
            key = f"{record['id']}::{step['id']}"
            candidates.append(
                Candidate(
                    order, key, record["id"], step["id"], source, prepared,
                    sha256_bytes(canonical_bytes(prepared)),
                    audit_context,
                )
            )
    return candidates


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    match = CODE_FENCE_RE.fullmatch(text)
    if match:
        text = match.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OutputValidationError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise OutputValidationError("response must be one JSON object")
    return value


def validate_decision(content: str, candidate: Candidate) -> tuple[dict[str, Any], str]:
    value = parse_json_object(content)
    expected_fields = {
        "schema", "id", "candidate_step", "is_cessation", "resolved_errors",
        "mode", "confidence", "note",
    }
    if set(value) != expected_fields:
        raise OutputValidationError(
            f"fields mismatch; missing={sorted(expected_fields-set(value))}, "
            f"extra={sorted(set(value)-expected_fields)}"
        )
    if value["schema"] != OUTPUT_SCHEMA:
        raise OutputValidationError(f"schema must be {OUTPUT_SCHEMA}")
    if value["id"] != candidate.record_id or value["candidate_step"] != candidate.step_id:
        raise OutputValidationError("id/candidate_step mismatch")
    if not isinstance(value["is_cessation"], bool):
        raise OutputValidationError("is_cessation must be boolean")
    if value["confidence"] not in CONFIDENCE_RANK:
        raise OutputValidationError("invalid confidence")
    if not isinstance(value["note"], str) or not value["note"].strip():
        raise OutputValidationError("note must be non-empty")
    resolved = value["resolved_errors"]
    if not isinstance(resolved, list) or len(resolved) != len(set(resolved)):
        raise OutputValidationError("resolved_errors must be a unique list")
    eligible = {item["id"] for item in candidate.prepared["eligible_error_steps"]}
    if any(not isinstance(item, str) or item not in eligible for item in resolved):
        raise OutputValidationError("resolved_errors contains an ineligible step")
    if value["is_cessation"]:
        if not resolved or value["mode"] not in {"correction", "bypass"}:
            raise OutputValidationError("true requires errors and correction/bypass mode")
    elif resolved or value["mode"] is not None:
        raise OutputValidationError("false requires resolved_errors=[] and mode=null")
    value["note"] = value["note"].strip()
    return value, sha256_bytes(content.encode("utf-8"))


def build_messages(
    prompt: str, candidate: Candidate, previous_content: str | None, previous_error: str | None
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(candidate.prepared, ensure_ascii=False)
            + "\n</CESSATION_CANDIDATE_INPUT_JSON>",
        },
    ]
    if previous_content is not None and previous_error is not None:
        messages.extend(
            [
                {"role": "assistant", "content": previous_content},
                {
                    "role": "user",
                    "content": "Correct the invalid response. Return only one JSON object. "
                    + "Validation error:\n" + previous_error[:6000],
                },
            ]
        )
    return messages


def make_payload(args: argparse.Namespace, messages: list[dict[str, str]]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if args.response_format == "json-object":
        value["response_format"] = {"type": "json_object"}
    return value


def post_completion(
    endpoint: str, payload: dict[str, Any], headers: dict[str, str], timeout: float
) -> tuple[str, dict[str, Any]]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, body = int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        status, body = int(exc.code), exc.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"endpoint connection failed: {exc.reason}") from exc
    try:
        value = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"non-JSON HTTP {status}: {body.decode('utf-8', errors='replace')[:1000]}"
        ) from exc
    if status >= 400:
        raise RuntimeError(f"endpoint HTTP {status}: {json.dumps(value, ensure_ascii=False)[:2000]}")
    try:
        choice = value["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("response lacks choices[0].message.content") from exc
    if not isinstance(content, str):
        raise RuntimeError("message.content is not a string")
    return content, {
        "response_id": value.get("id"),
        "response_model": value.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "usage": value.get("usage"),
    }


def process_one(
    candidate: Candidate,
    *,
    args: argparse.Namespace,
    headers: dict[str, str],
    prompt: str,
    run: dict[str, Any],
) -> WorkResult:
    started = time.perf_counter()
    previous_content: str | None = None
    previous_error: str | None = None
    attempt_errors: list[dict[str, Any]] = []
    usage_by_attempt: list[dict[str, Any]] = []
    total_attempts = 1 + args.max_retries
    last_meta: dict[str, Any] = {}
    for attempt in range(1, total_attempts + 1):
        attempt_started = time.perf_counter()
        content: str | None = None
        try:
            content, meta = post_completion(
                args.endpoint,
                make_payload(
                    args,
                    build_messages(prompt, candidate, previous_content, previous_error),
                ),
                headers,
                args.timeout,
            )
            last_meta = meta
            usage_by_attempt.append({"attempt": attempt, "usage": meta.get("usage")})
            decision, raw_hash = validate_decision(content, candidate)
            audit = {
                "schema": AUDIT_SCHEMA,
                "key": candidate.key,
                "id": candidate.record_id,
                "candidate_step": candidate.step_id,
                "candidate_reasons": candidate.audit_context["candidate_reasons"],
                "existing_overall": candidate.audit_context["existing_overall"],
                "existing_first_error": candidate.audit_context["existing_first_error"],
                "existing_candidate_annotation": candidate.audit_context[
                    "existing_candidate_annotation"
                ],
                "decision": decision,
                "lineage": {
                    "script_version": SCRIPT_VERSION,
                    "run_id": run["run_id"],
                    "input_line_number": candidate.source.line_number,
                    "input_record_sha256": candidate.source.sha256,
                    "prepared_input_sha256": candidate.prepared_sha256,
                    "prompt_sha256": run["prompt_sha256"],
                    "raw_response_sha256": raw_hash,
                },
                "generation": {
                    "endpoint": args.endpoint,
                    "requested_model": args.model,
                    "response_model": meta.get("response_model"),
                    "response_id": meta.get("response_id"),
                    "finish_reason": meta.get("finish_reason"),
                    "usage": meta.get("usage"),
                    "usage_by_attempt": usage_by_attempt,
                    "thinking_enabled": False,
                    "attempts": attempt,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                },
                "validation": {"valid": True, "retry_errors": attempt_errors},
            }
            return WorkResult(candidate, audit, None)
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            attempt_errors.append(
                {
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "message": error_text[:8000],
                    "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                }
            )
            if isinstance(content, str):
                previous_content, previous_error = content, error_text
            else:
                previous_content = previous_error = None
            if attempt < total_attempts:
                delay = args.retry_backoff * (2 ** (attempt - 1))
                delay += random.uniform(0, min(1.0, delay * 0.1))
                log(
                    f"[{candidate.key}] attempt {attempt}/{total_attempts} failed: "
                    f"{error_text[:500]}; retrying in {delay:.1f}s"
                )
                time.sleep(delay)
    error = {
        "schema": ERROR_SCHEMA,
        "key": candidate.key,
        "id": candidate.record_id,
        "candidate_step": candidate.step_id,
        "lineage": {
            "script_version": SCRIPT_VERSION,
            "run_id": run["run_id"],
            "input_line_number": candidate.source.line_number,
            "input_record_sha256": candidate.source.sha256,
            "prepared_input_sha256": candidate.prepared_sha256,
            "prompt_sha256": run["prompt_sha256"],
        },
        "generation": {
            "endpoint": args.endpoint,
            "requested_model": args.model,
            "response_model": last_meta.get("response_model"),
            "attempts": total_attempts,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "usage_by_attempt": usage_by_attempt,
        },
        "errors": attempt_errors,
        "last_response": previous_content,
    }
    return WorkResult(candidate, None, error)


def load_audits(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid audit JSON at {path}:{line_number}") from exc
            key = value.get("key") if isinstance(value, dict) else None
            if not isinstance(key, str):
                raise ValueError(f"audit line {line_number} lacks key")
            result[key] = value
    return result


def add_review_issue(record: dict[str, Any], issue: str, detail: dict[str, Any]) -> None:
    audit = record.setdefault("audit", {})
    audit["status"] = "review"
    flags = audit.setdefault("soft_flags", [])
    if issue not in flags:
        flags.append(issue)
    audit.setdefault("soft_issues", []).append({"code": issue, **detail})


def infer_via(step: dict[str, Any], source: str) -> list[str]:
    refs: list[str] = []
    for edge in step.get("audited_dependency_incoming", []):
        if isinstance(edge, dict) and edge.get("source_step") == source:
            ref = edge.get("source_ref")
            if isinstance(ref, str) and ref not in refs:
                refs.append(ref)
    return refs


def rebuild_causal_views(record: dict[str, Any]) -> None:
    steps = record["steps"]
    index = {step["id"]: i for i, step in enumerate(steps)}
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in record.get("causal_edges", []):
        if not isinstance(edge, dict):
            continue
        source, target, relation = edge.get("source"), edge.get("target"), edge.get("relation")
        if source not in index or target not in index or relation not in {"propagation", "cessation"}:
            continue
        gap = index[target] - index[source] - 1
        normalized = {
            "source": source,
            "target": target,
            "via": list(dict.fromkeys(edge.get("via", []))) if isinstance(edge.get("via"), list) else [],
            "relation": relation,
            "sequence_gap": gap,
            "is_skip": gap > 0,
        }
        unique[(source, target, relation)] = normalized
    edges = sorted(
        unique.values(), key=lambda e: (index[e["target"]], index[e["source"]], e["relation"])
    )
    record["causal_edges"] = edges
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        incoming[edge["target"]].append(edge)
        outgoing[edge["source"]].append(edge)
    for step in steps:
        step["causal_incoming"] = incoming[step["id"]]
        step["causal_outgoing"] = outgoing[step["id"]]
    metrics = record.setdefault("audit", {}).setdefault("metrics", {})
    metrics["causal_edge_count"] = len(edges)
    metrics["skip_causal_edge_count"] = sum(bool(edge["is_skip"]) for edge in edges)


def apply_audits(
    source: SourceRecord,
    candidates: list[Candidate],
    audits: dict[str, dict[str, Any]],
    min_confidence: str,
    run: dict[str, Any],
) -> tuple[dict[str, Any], bool, dict[str, int]]:
    record = json.loads(json.dumps(source.raw, ensure_ascii=False))
    upstream_status = record.get("audit", {}).get("status")
    upstream_review = upstream_status != "core"
    if upstream_review:
        add_review_issue(
            record,
            "upstream_materialization_not_core",
            {
                "upstream_status": upstream_status,
                "message": "Only upstream core records are eligible for automatic PRM output.",
            },
        )
    steps = {step["id"]: step for step in record["steps"]}
    step_index = {step["id"]: i for i, step in enumerate(record["steps"])}
    threshold = CONFIDENCE_RANK[min_confidence]
    promoted: list[str] = []
    validated_existing: list[str] = []
    review_steps: list[str] = []
    decision_counts: Counter[str] = Counter()
    for candidate in candidates:
        step = steps[candidate.step_id]
        audit_row = audits.get(candidate.key)
        if audit_row is None:
            review_steps.append(candidate.step_id)
            add_review_issue(
                record, "cessation_audit_missing",
                {"step": candidate.step_id, "message": "No valid binary audit is available."},
            )
            decision_counts["missing"] += 1
            continue
        decision = audit_row["decision"]
        confidence = decision["confidence"]
        accepted_confidence = CONFIDENCE_RANK[confidence] >= threshold
        decision_counts[f"{str(decision['is_cessation']).lower()}_{confidence}"] += 1
        if step.get("role") == "cessation":
            if decision["is_cessation"] and accepted_confidence:
                validated_existing.append(candidate.step_id)
            else:
                review_steps.append(candidate.step_id)
                add_review_issue(
                    record, "existing_cessation_not_confirmed",
                    {
                        "step": candidate.step_id,
                        "decision": decision,
                        "message": "Existing cessation is not auto-demoted; manual review is required.",
                    },
                )
            continue
        if decision["is_cessation"] and accepted_confidence:
            resolved = decision["resolved_errors"]
            step["role"] = "cessation"
            step["prm_label"] = "C"
            step["error_type"] = None
            step["caused_by"] = resolved
            step["note"] = decision["note"]
            step["mixed_error"] = False
            step["relation_to_first_error"] = "cessation"
            promoted.append(candidate.step_id)
            for source_id in resolved:
                gap = step_index[candidate.step_id] - step_index[source_id] - 1
                record.setdefault("causal_edges", []).append(
                    {
                        "source": source_id,
                        "target": candidate.step_id,
                        "via": infer_via(step, source_id),
                        "relation": "cessation",
                        "sequence_gap": gap,
                        "is_skip": gap > 0,
                    }
                )
        elif decision["is_cessation"] or confidence == "low":
            review_steps.append(candidate.step_id)
            add_review_issue(
                record, "cessation_decision_below_threshold",
                {"step": candidate.step_id, "decision": decision},
            )
    rebuild_causal_views(record)
    refinement = {
        "script_version": SCRIPT_VERSION,
        "run_id": run["run_id"],
        "prompt_sha256": run["prompt_sha256"],
        "min_confidence": min_confidence,
        "candidate_count": len(candidates),
        "promoted_steps": promoted,
        "validated_existing_steps": validated_existing,
        "review_steps": list(dict.fromkeys(review_steps)),
        "decision_counts": dict(sorted(decision_counts.items())),
    }
    record.setdefault("audit", {})["cessation_refinement"] = refinement
    record.setdefault("source_lineage", {})["cessation_refinement"] = {
        **refinement,
        "created_at": utc_now(),
        "input_line_number": source.line_number,
        "input_record_sha256": source.sha256,
    }
    counts = {
        "promoted": len(promoted),
        "validated_existing": len(validated_existing),
        "review_steps": len(set(review_steps)),
    }
    return record, upstream_review or bool(review_steps), counts


def prepare_paths(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    paths = [
        args.output, args.candidate_output, args.audit_output, args.review_output,
        args.error_output, args.stats_output,
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in paths:
            if path.exists():
                path.unlink()
    elif not args.resume:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError("outputs exist; pass --resume or --overwrite: " + ", ".join(existing))
    return load_audits(args.audit_output) if args.resume else {}


def reusable_audits(
    audits: dict[str, dict[str, Any]],
    candidates: list[Candidate],
    prompt_sha256: str,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Keep only audits made for the exact candidate payload and prompt."""
    by_key = {candidate.key: candidate for candidate in candidates}
    reusable: dict[str, dict[str, Any]] = {}
    stale = 0
    for key, audit in audits.items():
        candidate = by_key.get(key)
        lineage = audit.get("lineage", {}) if isinstance(audit, dict) else {}
        if (
            candidate is not None
            and lineage.get("prepared_input_sha256") == candidate.prepared_sha256
            and lineage.get("prompt_sha256") == prompt_sha256
            and isinstance(audit.get("decision"), dict)
        ):
            reusable[key] = audit
        else:
            stale += 1
    return reusable, stale


def main() -> int:
    args = parse_args()
    for path in (args.input, args.prompt):
        if not path.is_file():
            raise FileNotFoundError(f"file not found: {path}")
    prompt_bytes = args.prompt.read_bytes()
    prompt = prompt_bytes.decode("utf-8").strip()
    if OUTPUT_SCHEMA not in prompt or "</CESSATION_CANDIDATE_INPUT_JSON>" not in prompt:
        raise ValueError("prompt lacks the required output schema or input marker")
    output_paths = {
        args.output.resolve(), args.candidate_output.resolve(), args.audit_output.resolve(),
        args.review_output.resolve(), args.error_output.resolve(), args.stats_output.resolve(),
    }
    if len(output_paths) != 6:
        raise ValueError("output paths must be distinct")
    if args.input.resolve() in output_paths or args.prompt.resolve() in output_paths:
        raise ValueError("an output path must not overwrite --input or --prompt")
    loaded_audits = prepare_paths(args)
    all_records = read_materialized(args.input)
    records = all_records[args.start_line - 1 :]
    if args.limit is not None:
        records = records[: args.limit]
    candidates = build_candidates(records)
    candidate_by_record: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        candidate_by_record[candidate.record_id].append(candidate)
    with args.candidate_output.open("w", encoding="utf-8") as stream:
        for candidate in candidates:
            write_jsonl(stream, candidate.prepared)
    run = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8],
        "input_path": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "prompt_path": str(args.prompt.resolve()),
        "prompt_sha256": sha256_bytes(prompt_bytes),
    }
    existing_audits, stale_audits = reusable_audits(
        loaded_audits, candidates, run["prompt_sha256"]
    )
    pending = [candidate for candidate in candidates if candidate.key not in existing_audits]
    log(
        f"loaded {len(records)} record(s); recalled {len(candidates)} candidate(s) "
        f"from {len(candidate_by_record)} record(s); resume skipped {len(candidates)-len(pending)}; "
        f"stale audits ignored {stale_audits}; pending {len(pending)}"
    )
    log(
        f"runtime configuration: workers={args.workers}, max_in_flight={args.max_in_flight}, "
        f"timeout={args.timeout}s, max_attempts={1+args.max_retries}, "
        f"promotion_threshold={args.min_confidence}"
    )
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(args.api_key_env, "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    started_at, started_perf = utc_now(), time.perf_counter()
    valid_count = failed_count = first_attempt_valid = retried_valid = 0
    decision_counts: Counter[str] = Counter()
    retry_counts: Counter[str] = Counter()
    output_mode = "a" if args.resume else "w"
    with (
        args.audit_output.open(output_mode, encoding="utf-8") as audit_stream,
        args.error_output.open("w", encoding="utf-8") as error_stream,
        ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        iterator = iter(pending)
        futures: dict[Future[WorkResult], Candidate] = {}

        def submit_one() -> bool:
            try:
                candidate = next(iterator)
            except StopIteration:
                return False
            futures[executor.submit(
                process_one, candidate, args=args, headers=headers, prompt=prompt, run=run
            )] = candidate
            return True

        for _ in range(min(args.max_in_flight, len(pending))):
            submit_one()
        completed = written = last_logged = 0
        while futures:
            done, _ = wait(set(futures), timeout=args.heartbeat_seconds, return_when=FIRST_COMPLETED)
            if not done:
                preview = ", ".join(item.key for item in list(futures.values())[:10])
                log(
                    f"heartbeat: completed={completed}/{len(pending)}, written={written}, "
                    f"valid={valid_count}, failed={failed_count}, in_flight={len(futures)} "
                    f"[{preview}{' ...' if len(futures)>10 else ''}]"
                )
                continue
            for future in sorted(done, key=lambda item: futures[item].order):
                candidate = futures.pop(future)
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:
                    result = WorkResult(
                        candidate, None,
                        {
                            "schema": ERROR_SCHEMA, "key": candidate.key,
                            "id": candidate.record_id, "candidate_step": candidate.step_id,
                            "errors": [{"attempt": 0, "error_type": type(exc).__name__, "message": str(exc)}],
                        },
                    )
                if result.audit is not None:
                    write_jsonl(audit_stream, result.audit)
                    existing_audits[candidate.key] = result.audit
                    valid_count += 1
                    attempts = int(result.audit["generation"].get("attempts", 0))
                    first_attempt_valid += attempts == 1
                    retried_valid += attempts > 1
                    decision = result.audit["decision"]
                    decision_counts[f"{str(decision['is_cessation']).lower()}_{decision['confidence']}"] += 1
                    retries = result.audit["validation"]["retry_errors"]
                else:
                    assert result.error is not None
                    write_jsonl(error_stream, result.error)
                    failed_count += 1
                    retries = result.error.get("errors", [])
                written += 1
                for retry in retries:
                    if isinstance(retry.get("error_type"), str):
                        retry_counts[retry["error_type"]] += 1
                submit_one()
            if completed - last_logged >= args.log_every or completed == len(pending):
                log(
                    f"completed {completed}/{len(pending)}: written={written}, "
                    f"valid={valid_count}, failed={failed_count}, in_flight={len(futures)}"
                )
                last_logged = completed

    refined_count = review_count = promoted_count = validated_existing_count = 0
    review_step_count = 0
    with (
        args.output.open("w", encoding="utf-8") as output_stream,
        args.review_output.open("w", encoding="utf-8") as review_stream,
    ):
        for source in records:
            refined, needs_review, counts = apply_audits(
                source, candidate_by_record.get(source.raw["id"], []), existing_audits,
                args.min_confidence, run,
            )
            promoted_count += counts["promoted"]
            validated_existing_count += counts["validated_existing"]
            review_step_count += counts["review_steps"]
            if needs_review:
                write_jsonl(review_stream, refined)
                review_count += 1
            else:
                write_jsonl(output_stream, refined)
                refined_count += 1

    stats = {
        "schema_version": "cessation_candidate_audit_stats_v1",
        "status": "ok" if failed_count == 0 and review_count == 0 else "completed_with_review",
        "script_version": SCRIPT_VERSION,
        "run_id": run["run_id"],
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(time.perf_counter() - started_perf, 3),
        "input": run["input_path"],
        "prompt": run["prompt_path"],
        "output": str(args.output.resolve()),
        "candidate_output": str(args.candidate_output.resolve()),
        "audit_output": str(args.audit_output.resolve()),
        "review_output": str(args.review_output.resolve()),
        "error_output": str(args.error_output.resolve()),
        "selected_records": len(records),
        "records_with_candidates": len(candidate_by_record),
        "candidate_count": len(candidates),
        "resume_skipped_candidates": len(candidates) - len(pending),
        "stale_audits_ignored": stale_audits,
        "requested_candidates": len(pending),
        "valid_audits_this_run": valid_count,
        "failed_audits_this_run": failed_count,
        "first_attempt_valid": first_attempt_valid,
        "retried_valid": retried_valid,
        "decision_counts_this_run": dict(sorted(decision_counts.items())),
        "retry_error_type_counts": dict(sorted(retry_counts.items())),
        "refined_records": refined_count,
        "review_records": review_count,
        "review_steps": review_step_count,
        "promoted_cessation_steps": promoted_count,
        "validated_existing_cessation_steps": validated_existing_count,
        "min_confidence": args.min_confidence,
        "endpoint": args.endpoint,
        "model": args.model,
        "thinking_enabled": False,
        "workers": args.workers,
        "max_in_flight": args.max_in_flight,
        "max_retries": args.max_retries,
        "max_attempts": 1 + args.max_retries,
        "timeout_seconds": args.timeout,
        "output_order": "input",
        "audit_order": "completion",
    }
    args.stats_output.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    # Review records are an expected product of conservative annotation, not a
    # process failure.  A non-zero status is reserved for terminal API/validation
    # failures so production schedulers do not mistake normal review routing for
    # a crashed job.
    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted by user")
        raise SystemExit(130)
    except Exception as exc:
        log(f"error: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
