#!/usr/bin/env python3
"""Call Teacher B to judge fixed reasoning steps and audit error causality.

This client joins raw records, SaT candidate spans, and valid Teacher-A v3
outputs by id; builds compact ``judge_graph_input_v3`` records; calls an
OpenAI-compatible chat-completions endpoint with Qwen thinking disabled;
validates ``judge_graph_output_v3`` with Pydantic and graph constraints; and
writes enriched, traceable JSONL results.

``--max-retries 3`` means one initial request plus at most three retries.

Install:
    pip install "pydantic>=2.6"
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
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

try:
    import pydantic
    from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
except ImportError as exc:  # pragma: no cover
    raise SystemExit('missing dependency: pip install "pydantic>=2.6"') from exc


SCRIPT_VERSION = "teacher_b_judge_graph_client_v0.2"
INPUT_SCHEMA = "judge_graph_input_v3"
TEACHER_A_SCHEMA = "merge_io_output_v3"
SAT_SCHEMA = "merge_io_input_v1"
OUTPUT_SCHEMA = "judge_graph_output_v3"
ERROR_SCHEMA = "teacher_b_error_v1"
DEFAULT_ENDPOINT = (
    "http://127.0.0.1:4000/v1/chat/completions"
)
DEFAULT_MODEL = "Qwen3.8-27B"

STEP_RE = re.compile(r"^s(\d+)$")
SPAN_RE = re.compile(r"^c(\d+)$")
ENTITY_RE = re.compile(r"^q(\d+)$")
OUTPUT_RE = re.compile(r"^(s\d+)\.o(\d+)$")
ASSUME_RE = re.compile(r"^(s\d+)\.a(\d+)$")
ANSWER_BLOCK_RE = re.compile(r"<answer\b[^>]*>([\s\S]*?)</answer\s*>", re.I)
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>[\s\S]*?</think\s*>", re.I)
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.I)

DependencyType = Literal["data", "formula", "result", "condition"]
Role = Literal[
    "correct", "new_error", "propagated_error", "cessation", "unverifiable"
]
ErrorType = Literal[
    "problem_understanding",
    "method_or_formula",
    "derivation_or_logic",
    "calculation",
    "unit",
    "assumption_or_condition",
    "missing_evidence",
    "answer_mismatch",
    "other",
]
StepLabel = Literal["C", "I", "U"]
EvaluationStatus = Literal["complete", "limited", "not_evaluable"]
LOG_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    with LOG_LOCK:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}",
            file=sys.stderr,
            flush=True,
        )


def canonical_json_bytes(value: Any) -> bytes:
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


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


class CompiledStep(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    spans: list[str]
    op: str
    inputs: list[tuple[str, DependencyType]] = Field(alias="in")
    assume: list[tuple[str, str]]
    out: list[tuple[str, str]]

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not STEP_RE.fullmatch(value):
            raise ValueError("step id must match s01, s02, ...")
        return value


class TeacherARecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_: Literal[TEACHER_A_SCHEMA] = Field(alias="schema")
    id: str
    quality_flags: list[str]
    problem_entities: list[tuple[str, str]]
    steps: list[CompiledStep]

    @field_validator("steps")
    @classmethod
    def valid_steps(cls, value: list[CompiledStep]) -> list[CompiledStep]:
        if not value:
            raise ValueError("Teacher-A steps must not be empty")
        expected = [f"s{i:02d}" for i in range(1, len(value) + 1)]
        if [step.id for step in value] != expected:
            raise ValueError(f"Teacher-A step ids must be sequential: {expected}")
        return value


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    role: Role
    error_type: ErrorType | None
    caused_by: list[str]
    note: str | None

    @field_validator("step")
    @classmethod
    def valid_step(cls, value: str) -> str:
        if not STEP_RE.fullmatch(value):
            raise ValueError("judgment step must match s01, s02, ...")
        return value

    @field_validator("caused_by")
    @classmethod
    def unique_causes(cls, value: list[str]) -> list[str]:
        if any(not STEP_RE.fullmatch(item) for item in value):
            raise ValueError("caused_by values must be step ids")
        if len(value) != len(set(value)):
            raise ValueError("caused_by must not contain duplicates")
        return value


class TeacherBOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal[OUTPUT_SCHEMA] = Field(alias="schema")
    id: str
    evaluation_status: EvaluationStatus
    judgments: list[Judgment]
    first_error: str | None
    dependency_add: list[tuple[str, str, DependencyType]]
    dependency_remove: list[tuple[str, str, DependencyType]]
    answers: list[tuple[str, StepLabel, list[str]]]
    overall: StepLabel


@dataclass(frozen=True)
class IndexedRecord:
    line_number: int
    raw: dict[str, Any]
    record_sha256: str


@dataclass(frozen=True)
class WorkItem:
    order: int
    record_id: str
    teacher_a: IndexedRecord
    sat: IndexedRecord
    raw: IndexedRecord
    compiled: TeacherARecord
    prepared: dict[str, Any]
    prepared_sha256: str


@dataclass
class GenerationResult:
    order: int
    record_id: str
    valid_record: dict[str, Any] | None
    error_record: dict[str, Any] | None


class OutputValidationError(ValueError):
    """The response is JSON but violates the annotation contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call Teacher B on Teacher-A steps and audit error dependencies."
    )
    parser.add_argument("--teacher-a-input", type=Path, required=True)
    parser.add_argument("--sat-input", type=Path, required=True)
    parser.add_argument("--raw-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--prepared-input-output", type=Path, default=None)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-attempt HTTP timeout in seconds (default: 180).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument(
        "--response-format", choices=["json-object", "none"], default="json-object"
    )
    parser.add_argument("--error-output", type=Path, default=None)
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="Log pending request ids after this many seconds without completion.",
    )
    parser.add_argument(
        "--allow-lineage-mismatch",
        action="store_true",
        help="Allow Teacher-A source hashes to differ from the supplied SaT file.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_retries < 0 or args.retry_backoff < 0:
        parser.error("retry values cannot be negative")
    if args.timeout <= 0 or args.max_tokens < 1:
        parser.error("--timeout and --max-tokens must be positive")
    if not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1:
        parser.error("invalid sampling parameters")
    if args.start_line < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--start-line and --limit must be positive")
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    if args.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds must be positive")
    if args.error_output is None:
        args.error_output = args.output.with_name(args.output.stem + ".errors.jsonl")
    if args.stats_output is None:
        args.stats_output = args.output.with_name(args.output.stem + ".stats.json")
    if args.prepared_input_output is None:
        args.prepared_input_output = args.output.with_name(
            args.output.stem + ".inputs.jsonl"
        )
    return args


def read_jsonl_index(path: Path, label: str) -> tuple[list[str], dict[str, IndexedRecord]]:
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"{label} must be a .jsonl file: {path}")
    order: list[str] = []
    index: dict[str, IndexedRecord] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise TypeError(f"{label} line {line_number} is not an object")
            record_id = value.get("id")
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"{label} line {line_number} lacks a valid id")
            if record_id in index:
                raise ValueError(f"duplicate id in {label}: {record_id}")
            order.append(record_id)
            index[record_id] = IndexedRecord(
                line_number,
                value,
                sha256_bytes(canonical_json_bytes(value)),
            )
    return order, index


def answer_pairs(value: Any, prefix: str) -> list[list[str]]:
    if value is None or value == [] or value == "":
        return []
    values = value if isinstance(value, list) else [value]
    result: list[list[str]] = []
    for item in values:
        if isinstance(item, (dict, list)):
            text = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        else:
            text = normalize_text(item)
        if text:
            result.append([f"{prefix}{len(result) + 1:02d}", text])
    return result


def proposed_answer_pairs(raw: dict[str, Any], compiled: TeacherARecord) -> list[list[str]]:
    solution = raw.get("solution")
    if isinstance(solution, str):
        match = ANSWER_BLOCK_RE.search(solution)
        if match:
            body = match.group(1).strip()
            lines = [normalize_text(line) for line in body.splitlines() if line.strip()]
            if not lines and body:
                lines = [normalize_text(body)]
            if lines:
                return [[f"a{i:02d}", text] for i, text in enumerate(lines, 1)]
    final_outputs = compiled.steps[-1].out
    return [
        [f"a{i:02d}", normalize_text(content)]
        for i, (_, content) in enumerate(final_outputs, 1)
        if normalize_text(content)
    ]


def build_prepared_record(
    teacher_a_raw: dict[str, Any],
    sat_raw: dict[str, Any],
    raw: dict[str, Any],
    compiled: TeacherARecord,
) -> dict[str, Any]:
    context = sat_raw.get("context")
    spans = sat_raw.get("candidate_spans")
    if sat_raw.get("schema") != SAT_SCHEMA or not isinstance(context, dict):
        raise ValueError(f"{compiled.id}: invalid SaT {SAT_SCHEMA} record")
    if not isinstance(spans, list) or not spans:
        raise ValueError(f"{compiled.id}: SaT candidate_spans is empty")
    if teacher_a_raw.get("schema") != TEACHER_A_SCHEMA:
        raise ValueError(f"{compiled.id}: invalid Teacher-A schema")
    return {
        "schema": INPUT_SCHEMA,
        "id": compiled.id,
        "context": {
            "subject": context.get("subject", raw.get("subject")),
            "type": context.get("type", raw.get("type")),
            "problem": context.get("problem", raw.get("problem", "")),
            "options": context.get("options", raw.get("options", [])),
            "reference_answers": answer_pairs(raw.get("answer"), "r"),
            "proposed_answers": proposed_answer_pairs(raw, compiled),
        },
        "quality_flags": compiled.quality_flags,
        "candidate_spans": spans,
        "compiled": {
            "schema": TEACHER_A_SCHEMA,
            "problem_entities": compiled.problem_entities,
            "steps": [step.model_dump(by_alias=True) for step in compiled.steps],
        },
        "policy": {
            "rubric_id": "math_science_step_judgment_v2",
            "reference_answer_role": "advisory",
        },
    }


def lineage_source_hash(teacher_a_raw: dict[str, Any]) -> str | None:
    lineage = teacher_a_raw.get("lineage")
    value = lineage.get("source_record_sha256") if isinstance(lineage, dict) else None
    return value if isinstance(value, str) else None


def build_work_items(args: argparse.Namespace) -> tuple[list[WorkItem], dict[str, Any]]:
    teacher_order, teacher_index = read_jsonl_index(args.teacher_a_input, "Teacher-A")
    _, sat_index = read_jsonl_index(args.sat_input, "SaT")
    _, raw_index = read_jsonl_index(args.raw_input, "raw input")
    selected_ids = teacher_order[args.start_line - 1 :]
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]
    missing_sat = [item for item in selected_ids if item not in sat_index]
    missing_raw = [item for item in selected_ids if item not in raw_index]
    if missing_sat or missing_raw:
        raise ValueError(
            f"join failure: missing_sat={missing_sat[:10]}, missing_raw={missing_raw[:10]}"
        )
    items: list[WorkItem] = []
    lineage_mismatches: list[str] = []
    for order, record_id in enumerate(selected_ids, start=1):
        teacher = teacher_index[record_id]
        sat = sat_index[record_id]
        raw_record = raw_index[record_id]
        try:
            compiled = TeacherARecord.model_validate(teacher.raw)
        except ValidationError as exc:
            raise ValueError(f"invalid Teacher-A record {record_id}: {exc}") from exc
        source_hash = lineage_source_hash(teacher.raw)
        if source_hash is not None and source_hash != sat.record_sha256:
            lineage_mismatches.append(record_id)
        prepared = build_prepared_record(
            teacher.raw, sat.raw, raw_record.raw, compiled
        )
        items.append(
            WorkItem(
                order,
                record_id,
                teacher,
                sat,
                raw_record,
                compiled,
                prepared,
                sha256_bytes(canonical_json_bytes(prepared)),
            )
        )
    if lineage_mismatches and not args.allow_lineage_mismatch:
        raise ValueError(
            "Teacher-A lineage does not match supplied SaT records for ids "
            f"{lineage_mismatches[:10]}; use the exact upstream SaT file or pass "
            "--allow-lineage-mismatch for an intentional pilot"
        )
    diagnostics = {
        "teacher_a_records": len(teacher_index),
        "sat_records": len(sat_index),
        "raw_records": len(raw_index),
        "selected_records": len(items),
        "lineage_mismatch_records": len(lineage_mismatches),
    }
    return items, diagnostics


def strip_and_parse_json(content: str) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise OutputValidationError("assistant content is empty")
    cleaned = THINK_BLOCK_RE.sub("", content).strip()
    fenced = CODE_FENCE_RE.fullmatch(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise OutputValidationError(
            f"assistant content is not one JSON object: {exc.msg} at "
            f"line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(value, dict):
        raise OutputValidationError("assistant JSON root must be an object")
    return value


def reference_catalog(compiled: TeacherARecord) -> tuple[dict[str, str], dict[str, str]]:
    reference_owner: dict[str, str] = {}
    reference_kind: dict[str, str] = {}
    for entity_id, _ in compiled.problem_entities:
        if not ENTITY_RE.fullmatch(entity_id):
            raise OutputValidationError(f"invalid problem entity id: {entity_id}")
        reference_owner[entity_id] = "problem"
        reference_kind[entity_id] = "entity"
    for step in compiled.steps:
        for assumption_id, _ in step.assume:
            reference_owner[assumption_id] = step.id
            reference_kind[assumption_id] = "assumption"
        for output_id, _ in step.out:
            reference_owner[output_id] = step.id
            reference_kind[output_id] = "output"
    return reference_owner, reference_kind


def role_label(role: str) -> str:
    return {
        "correct": "C",
        "cessation": "C",
        "new_error": "I",
        "propagated_error": "I",
        "unverifiable": "U",
    }[role]


def validate_role_fields(judgment: Judgment) -> list[str]:
    errors: list[str] = []
    note_present = isinstance(judgment.note, str) and bool(judgment.note.strip())
    if judgment.role == "correct":
        if judgment.error_type is not None or judgment.caused_by or judgment.note is not None:
            errors.append(f"{judgment.step}: correct requires null error/note and [] causes")
    elif judgment.role == "new_error":
        if judgment.error_type is None or not note_present:
            errors.append(f"{judgment.step}: new_error requires error_type and note")
    elif judgment.role in {"propagated_error", "cessation"}:
        if judgment.error_type is not None or not judgment.caused_by or not note_present:
            errors.append(
                f"{judgment.step}: {judgment.role} requires null error_type, causes, note"
            )
    elif judgment.role == "unverifiable":
        if judgment.error_type != "missing_evidence" or judgment.caused_by or not note_present:
            errors.append(
                f"{judgment.step}: unverifiable requires missing_evidence, [] causes, note"
            )
    return errors


def apply_dependency_edits(
    output: TeacherBOutput, compiled: TeacherARecord
) -> tuple[dict[str, list[tuple[str, str]]], list[dict[str, Any]]]:
    step_index = {step.id: i for i, step in enumerate(compiled.steps, 1)}
    owner, kind = reference_catalog(compiled)
    graph = {step.id: list(step.inputs) for step in compiled.steps}
    errors: list[str] = []
    adds = list(output.dependency_add)
    removes = list(output.dependency_remove)
    if len(adds) != len(set(adds)):
        errors.append("dependency_add contains duplicates")
    if len(removes) != len(set(removes)):
        errors.append("dependency_remove contains duplicates")
    if set(adds) & set(removes):
        errors.append("the same dependency appears in add and remove")

    def validate_edge(edge: tuple[str, str, str], action: str) -> None:
        consumer, ref, dependency_type = edge
        if consumer not in step_index:
            errors.append(f"{action}: undefined consumer {consumer}")
            return
        if ref not in owner:
            errors.append(f"{action}: undefined reference {ref}")
            return
        ref_kind = kind[ref]
        if ref_kind == "entity" and dependency_type not in {"data", "formula", "condition"}:
            errors.append(f"{action}: qXX cannot use type {dependency_type}")
        if ref_kind == "output" and dependency_type not in {"result", "formula", "condition"}:
            errors.append(f"{action}: output cannot use type {dependency_type}")
        if ref_kind == "assumption" and dependency_type != "condition":
            errors.append(f"{action}: assumption requires condition type")
        source = owner[ref]
        if source != "problem" and step_index[source] >= step_index[consumer]:
            errors.append(f"{action}: {consumer} references self/future {ref}")

    for edge in removes:
        validate_edge(edge, "remove")
        consumer, ref, dependency_type = edge
        if consumer in graph and (ref, dependency_type) not in graph[consumer]:
            errors.append(f"remove edge does not exist: {edge}")
        elif consumer in graph:
            graph[consumer].remove((ref, dependency_type))
    for edge in adds:
        validate_edge(edge, "add")
        consumer, ref, dependency_type = edge
        if consumer in graph and (ref, dependency_type) in graph[consumer]:
            errors.append(f"add edge already exists: {edge}")
        elif consumer in graph:
            if any(existing_ref == ref for existing_ref, _ in graph[consumer]):
                errors.append(
                    f"add changes type for {consumer}/{ref} without removing old edge"
                )
            graph[consumer].append((ref, dependency_type))
    for consumer, edges in graph.items():
        refs = [ref for ref, _ in edges]
        if len(refs) != len(set(refs)):
            errors.append(f"{consumer}: final dependencies contain duplicate references")
    if errors:
        raise OutputValidationError("; ".join(errors))

    audited_edges: list[dict[str, Any]] = []
    for target, edges in graph.items():
        for ref, dependency_type in edges:
            source = owner[ref]
            gap = None if source == "problem" else step_index[target] - step_index[source] - 1
            audited_edges.append(
                {
                    "source_ref": ref,
                    "source_step": None if source == "problem" else source,
                    "target_step": target,
                    "dependency_type": dependency_type,
                    "sequence_gap": gap,
                    "is_skip": isinstance(gap, int) and gap > 0,
                }
            )
    return graph, audited_edges


def reaches_origin(
    step_id: str, judgments: dict[str, Judgment], memo: dict[str, bool]
) -> bool:
    if step_id in memo:
        return memo[step_id]
    judgment = judgments[step_id]
    if judgment.role == "new_error":
        memo[step_id] = True
        return True
    if judgment.role != "propagated_error":
        memo[step_id] = False
        return False
    memo[step_id] = any(reaches_origin(source, judgments, memo) for source in judgment.caused_by)
    return memo[step_id]


def validate_and_enrich(output: TeacherBOutput, item: WorkItem) -> tuple[dict[str, Any], dict[str, Any]]:
    compiled = item.compiled
    expected_steps = [step.id for step in compiled.steps]
    errors: list[str] = []
    if output.id != item.record_id:
        errors.append(f"output id {output.id!r} != input id {item.record_id!r}")
    observed_steps = [judgment.step for judgment in output.judgments]
    if observed_steps != expected_steps:
        errors.append(
            f"judgment steps/order mismatch: expected={expected_steps}, observed={observed_steps}"
        )
    judgments = {judgment.step: judgment for judgment in output.judgments}
    step_index = {step_id: i for i, step_id in enumerate(expected_steps, 1)}
    for judgment in output.judgments:
        errors.extend(validate_role_fields(judgment))
        for source in judgment.caused_by:
            if source not in step_index or step_index[source] >= step_index.get(judgment.step, 0):
                errors.append(f"{judgment.step}: invalid earlier cause {source}")
            elif source not in judgments:
                errors.append(f"{judgment.step}: cause {source} lacks a judgment")
            elif role_label(judgments[source].role) != "I":
                errors.append(f"{judgment.step}: cause {source} is not erroneous")

    earliest = next(
        (judgment.step for judgment in output.judgments if judgment.role == "new_error"),
        None,
    )
    if output.first_error != earliest:
        errors.append(f"first_error must be {earliest!r}, got {output.first_error!r}")

    if output.evaluation_status == "complete" and any(
        judgment.role == "unverifiable" for judgment in output.judgments
    ):
        errors.append("complete status cannot contain unverifiable steps")
    if "non_reasoning" in compiled.quality_flags and output.evaluation_status == "complete":
        errors.append("non_reasoning input cannot have complete evaluation status")

    try:
        graph, audited_edges = apply_dependency_edits(output, compiled)
    except OutputValidationError as exc:
        errors.append(str(exc))
        graph, audited_edges = {}, []
    owner, _ = reference_catalog(compiled)
    causal_edges: list[dict[str, Any]] = []
    if graph:
        for judgment in output.judgments:
            for source in judgment.caused_by:
                via = [ref for ref, _ in graph[judgment.step] if owner.get(ref) == source]
                if not via and judgment.role in {"new_error", "propagated_error"}:
                    errors.append(
                        f"{judgment.step}: caused_by {source} is not a direct audited parent"
                    )
                relation = {
                    "new_error": "mixed",
                    "propagated_error": "propagation",
                    "cessation": "cessation",
                }.get(judgment.role)
                causal_edges.append(
                    {
                        "source": source,
                        "target": judgment.step,
                        "via": via,
                        "relation": relation,
                        "sequence_gap": step_index[judgment.step] - step_index[source] - 1,
                        "is_skip": step_index[judgment.step] - step_index[source] - 1 > 0,
                    }
                )

    for judgment in output.judgments:
        causes_resolvable = all(source in judgments for source in judgment.caused_by)
        if (
            judgment.role == "propagated_error"
            and causes_resolvable
            and not reaches_origin(judgment.step, judgments, {})
        ):
            errors.append(f"{judgment.step}: propagated_error has no new_error ancestor")

    proposed_ids = [item_[0] for item_ in item.prepared["context"]["proposed_answers"]]
    answer_ids = [answer[0] for answer in output.answers]
    if answer_ids != proposed_ids:
        errors.append(
            f"answer ids/order mismatch: expected={proposed_ids}, observed={answer_ids}"
        )
    for answer_id, _, support in output.answers:
        if len(support) != len(set(support)) or any(s not in step_index for s in support):
            errors.append(f"{answer_id}: invalid/duplicate supporting steps")
    if output.answers:
        expected_overall = (
            "I"
            if any(label == "I" for _, label, _ in output.answers)
            else "U"
            if any(label == "U" for _, label, _ in output.answers)
            else "C"
        )
        if output.overall != expected_overall:
            errors.append(
                f"overall must be {expected_overall} from answer labels, got {output.overall}"
            )
    if errors:
        raise OutputValidationError("; ".join(errors))

    first = output.first_error
    causal_parent = {
        judgment.step: list(judgment.caused_by) for judgment in output.judgments
    }

    def reachable_from_first(target: str, visiting: set[str] | None = None) -> bool:
        if first is None:
            return False
        if target == first:
            return True
        visiting = set() if visiting is None else visiting
        if target in visiting:
            return False
        visiting.add(target)
        return any(reachable_from_first(parent, visiting) for parent in causal_parent[target])

    enriched_judgments: list[dict[str, Any]] = []
    for judgment in output.judgments:
        row = judgment.model_dump()
        row["prm_label"] = role_label(judgment.role)
        row["mixed_error"] = judgment.role == "new_error" and bool(judgment.caused_by)
        if first is None:
            relation = "unknown" if judgment.role == "unverifiable" else "none"
        elif judgment.step == first:
            relation = "first_error"
        elif judgment.role == "cessation":
            relation = "cessation"
        elif judgment.role == "unverifiable":
            relation = "unknown"
        elif judgment.role in {"new_error", "propagated_error"}:
            if first in judgment.caused_by:
                relation = "direct_propagation"
            elif reachable_from_first(judgment.step):
                relation = "indirect_propagation"
            else:
                relation = "independent_error"
        elif step_index[judgment.step] > step_index[first]:
            relation = "unaffected"
        else:
            relation = "none"
        row["relation_to_first_error"] = relation
        enriched_judgments.append(row)

    result = output.model_dump(by_alias=True)
    result["judgments"] = enriched_judgments
    result["labels"] = [
        [judgment.step, role_label(judgment.role)] for judgment in output.judgments
    ]
    result["audited_dependency_edges"] = audited_edges
    result["causal_edges"] = causal_edges
    checks = {
        "step_count": len(expected_steps),
        "answer_count": len(output.answers),
        "dependency_add_count": len(output.dependency_add),
        "dependency_remove_count": len(output.dependency_remove),
        "audited_dependency_edge_count": len(audited_edges),
        "causal_edge_count": len(causal_edges),
        "skip_causal_edge_count": sum(edge["is_skip"] for edge in causal_edges),
    }
    return result, checks


def validate_model_content(content: str, item: WorkItem) -> tuple[dict[str, Any], dict[str, Any], str]:
    parsed = strip_and_parse_json(content)
    try:
        output = TeacherBOutput.model_validate(parsed)
    except ValidationError as exc:
        raise OutputValidationError(f"Pydantic validation failed: {exc}") from exc
    result, checks = validate_and_enrich(output, item)
    return result, checks, sha256_bytes(content.encode("utf-8"))


def build_messages(
    prompt: str,
    prepared: dict[str, Any],
    previous_content: str | None,
    previous_error: str | None,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(prepared, ensure_ascii=False)
            + "\n</JUDGE_GRAPH_INPUT_JSON>",
        },
    ]
    if previous_content is not None and previous_error is not None:
        messages.extend(
            [
                {"role": "assistant", "content": previous_content},
                {
                    "role": "user",
                    "content": (
                        "The previous response is invalid. Correct it and return only "
                        f"one JSON object matching {OUTPUT_SCHEMA}. Validation errors:\n"
                        + previous_error[:8000]
                    ),
                },
            ]
        )
    return messages


def make_payload(args: argparse.Namespace, messages: list[dict[str, str]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if args.response_format == "json-object":
        payload["response_format"] = {"type": "json_object"}
    return payload


def post_chat_completion(
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
        response = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"endpoint returned non-JSON HTTP {status}: "
            + body.decode("utf-8", errors="replace")[:1000]
        ) from exc
    if status >= 400:
        raise RuntimeError(
            f"endpoint HTTP {status}: {json.dumps(response, ensure_ascii=False)[:2000]}"
        )
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("response lacks choices[0].message.content") from exc
    if not isinstance(content, str):
        raise RuntimeError("endpoint message.content is not a string")
    return content, {
        "response_id": response.get("id"),
        "response_model": response.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
    }


def aggregate_attempt_usage(entries: list[dict[str, Any]]) -> dict[str, int] | None:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    observed = False
    for entry in entries:
        usage = entry.get("usage")
        if isinstance(usage, dict):
            for key in totals:
                if isinstance(usage.get(key), int):
                    totals[key] += usage[key]
                    observed = True
    return totals if observed else None


def process_one(
    item: WorkItem,
    *,
    args: argparse.Namespace,
    headers: dict[str, str],
    prompt: str,
    run: dict[str, Any],
) -> GenerationResult:
    started = time.perf_counter()
    previous_content: str | None = None
    previous_error: str | None = None
    attempt_errors: list[dict[str, Any]] = []
    usage_by_attempt: list[dict[str, Any]] = []
    last_meta: dict[str, Any] = {}
    total_attempts = 1 + args.max_retries
    for attempt in range(1, total_attempts + 1):
        attempt_started = time.perf_counter()
        content: str | None = None
        try:
            log(
                f"[{item.record_id}] attempt {attempt}/{total_attempts}: "
                "sending request"
            )
            content, meta = post_chat_completion(
                args.endpoint,
                make_payload(
                    args,
                    build_messages(prompt, item.prepared, previous_content, previous_error),
                ),
                headers,
                args.timeout,
            )
            log(
                f"[{item.record_id}] attempt {attempt}/{total_attempts}: "
                "response received, validating"
            )
            last_meta = meta
            usage_by_attempt.append({"attempt": attempt, "usage": meta.get("usage")})
            result, checks, response_hash = validate_model_content(content, item)
            result["lineage"] = {
                "pipeline_stage": "teacher_b_judge_and_audit_error_graph",
                "script_version": SCRIPT_VERSION,
                "run_id": run["run_id"],
                "teacher_a_file": run["teacher_a_path"],
                "teacher_a_file_sha256": run["teacher_a_sha256"],
                "teacher_a_line_number": item.teacher_a.line_number,
                "teacher_a_record_sha256": item.teacher_a.record_sha256,
                "sat_file": run["sat_path"],
                "sat_file_sha256": run["sat_sha256"],
                "sat_line_number": item.sat.line_number,
                "sat_record_sha256": item.sat.record_sha256,
                "raw_file": run["raw_path"],
                "raw_file_sha256": run["raw_sha256"],
                "raw_line_number": item.raw.line_number,
                "raw_record_sha256": item.raw.record_sha256,
                "prepared_input_sha256": item.prepared_sha256,
                "prompt_file": run["prompt_path"],
                "prompt_sha256": run["prompt_sha256"],
            }
            result["generation"] = {
                "endpoint": args.endpoint,
                "requested_model": args.model,
                "response_model": meta.get("response_model"),
                "response_id": meta.get("response_id"),
                "finish_reason": meta.get("finish_reason"),
                "usage": meta.get("usage"),
                "usage_all_attempts": aggregate_attempt_usage(usage_by_attempt),
                "usage_by_attempt": usage_by_attempt,
                "thinking_enabled": False,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "response_format": args.response_format,
                "attempts": attempt,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "raw_response_sha256": response_hash,
            }
            result["validation"] = {
                "valid": True,
                "validator": SCRIPT_VERSION,
                "pydantic_version": pydantic.__version__,
                "checks": checks,
                "retry_errors": attempt_errors,
            }
            log(
                f"[{item.record_id}] valid after {attempt} attempt(s), "
                f"latency={round(time.perf_counter() - started, 2)}s"
            )
            return GenerationResult(item.order, item.record_id, result, None)
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            attempt_errors.append(
                {
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "message": error_text[:8000],
                    "latency_ms": round(
                        (time.perf_counter() - attempt_started) * 1000, 3
                    ),
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
                    f"[{item.record_id}] attempt {attempt}/{total_attempts} "
                    f"failed: {error_text[:500]}; retrying in {delay:.1f}s"
                )
                time.sleep(delay)
            else:
                log(
                    f"[{item.record_id}] terminal failure after {total_attempts} "
                    f"attempt(s): {error_text[:500]}"
                )
    error = {
        "schema": ERROR_SCHEMA,
        "id": item.record_id,
        "lineage": {
            "pipeline_stage": "teacher_b_judge_and_audit_error_graph",
            "script_version": SCRIPT_VERSION,
            "run_id": run["run_id"],
            "teacher_a_line_number": item.teacher_a.line_number,
            "teacher_a_record_sha256": item.teacher_a.record_sha256,
            "sat_line_number": item.sat.line_number,
            "sat_record_sha256": item.sat.record_sha256,
            "raw_line_number": item.raw.line_number,
            "raw_record_sha256": item.raw.record_sha256,
            "prepared_input_sha256": item.prepared_sha256,
            "prompt_sha256": run["prompt_sha256"],
        },
        "generation": {
            "endpoint": args.endpoint,
            "requested_model": args.model,
            "response_model": last_meta.get("response_model"),
            "thinking_enabled": False,
            "attempts": total_attempts,
            "max_retries": args.max_retries,
            "usage_all_attempts": aggregate_attempt_usage(usage_by_attempt),
            "usage_by_attempt": usage_by_attempt,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        },
        "errors": attempt_errors,
        "last_response": previous_content,
    }
    return GenerationResult(item.order, item.record_id, None, error)


def load_terminal_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"cannot resume from invalid JSON at {path}:{line_number}") from exc
            if isinstance(value, dict) and isinstance(value.get("id"), str):
                ids.add(value["id"])
    return ids


def prepare_files(args: argparse.Namespace) -> set[str]:
    paths = [args.output, args.error_output, args.stats_output, args.prepared_input_output]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in paths:
            if path.exists():
                path.unlink()
        return set()
    if args.resume:
        return load_terminal_ids(args.output) | load_terminal_ids(args.error_output)
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "output files exist; pass --resume or --overwrite: " + ", ".join(existing)
        )
    return set()


def write_jsonl(stream: Any, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    stream.flush()


def main() -> int:
    args = parse_args()
    for path in [args.teacher_a_input, args.sat_input, args.raw_input, args.prompt]:
        if not path.is_file():
            raise FileNotFoundError(f"file not found: {path}")
    prompt_bytes = args.prompt.read_bytes()
    prompt = prompt_bytes.decode("utf-8").strip()
    if OUTPUT_SCHEMA not in prompt or "<JUDGE_GRAPH_INPUT_JSON>" not in prompt:
        raise ValueError(f"prompt must specify {OUTPUT_SCHEMA} and input marker")
    completed_ids = prepare_files(args)
    items, join_diagnostics = build_work_items(args)
    selected = [item for item in items if item.record_id not in completed_ids]
    run = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8],
        "teacher_a_path": str(args.teacher_a_input.resolve()),
        "teacher_a_sha256": sha256_file(args.teacher_a_input),
        "sat_path": str(args.sat_input.resolve()),
        "sat_sha256": sha256_file(args.sat_input),
        "raw_path": str(args.raw_input.resolve()),
        "raw_sha256": sha256_file(args.raw_input),
        "prompt_path": str(args.prompt.resolve()),
        "prompt_sha256": sha256_bytes(prompt_bytes),
    }
    log(
        f"joined {len(items)} record(s); resume skipped {len(items)-len(selected)}; "
        f"pending {len(selected)}"
    )

    if args.resume and args.prepared_input_output.exists():
        log(f"reusing existing model input file: {args.prepared_input_output}")
    else:
        with args.prepared_input_output.open("w", encoding="utf-8") as stream:
            for item in items:
                write_jsonl(stream, item.prepared)
        log(f"wrote {len(items)} model input record(s) to {args.prepared_input_output}")

    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(args.api_key_env, "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    started_at, started_perf = utc_now(), time.perf_counter()
    valid_count = failed_count = total_attempts = first_attempt_valid = retried_valid = 0
    role_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    retry_counts: Counter[str] = Counter()
    total_causal_edges = total_skip_edges = 0
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage_records = 0
    output_mode = "a" if args.resume else "w"
    with (
        args.output.open(output_mode, encoding="utf-8") as valid_stream,
        args.error_output.open(output_mode, encoding="utf-8") as error_stream,
        ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        futures: dict[Future[GenerationResult], WorkItem] = {
            executor.submit(
                process_one,
                item,
                args=args,
                headers=headers,
                prompt=prompt,
                run=run,
            ): item
            for item in selected
        }
        pending: set[Future[GenerationResult]] = set(futures)
        completed = 0
        while pending:
            done, pending = wait(
                pending,
                timeout=args.heartbeat_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                pending_ids = sorted(futures[future].record_id for future in pending)
                preview = ", ".join(pending_ids[:10])
                suffix = " ..." if len(pending_ids) > 10 else ""
                log(
                    f"heartbeat: completed={completed}/{len(selected)}, "
                    f"valid={valid_count}, failed={failed_count}, "
                    f"pending={len(pending_ids)} [{preview}{suffix}]"
                )
                continue

            for future in sorted(done, key=lambda value: futures[value].order):
                item = futures[future]
                completed += 1
                try:
                    ready = future.result()
                except Exception as exc:  # defensive
                    ready = GenerationResult(
                        item.order,
                        item.record_id,
                        None,
                        {
                            "schema": ERROR_SCHEMA,
                            "id": item.record_id,
                            "generation": {"attempts": 0},
                            "errors": [
                                {
                                    "attempt": 0,
                                    "error_type": type(exc).__name__,
                                    "message": str(exc),
                                }
                            ],
                        },
                    )

                # Completion-order writes avoid head-of-line blocking when one
                # record is waiting on HTTP timeout or repeated validation.
                if ready.valid_record is not None:
                    write_jsonl(valid_stream, ready.valid_record)
                    valid_count += 1
                    generation = ready.valid_record["generation"]
                    attempts = int(generation.get("attempts", 0))
                    first_attempt_valid += attempts == 1
                    retried_valid += attempts > 1
                    for judgment in ready.valid_record["judgments"]:
                        role_counts[judgment["role"]] += 1
                        label_counts[judgment["prm_label"]] += 1
                    status_counts[ready.valid_record["evaluation_status"]] += 1
                    total_causal_edges += len(ready.valid_record["causal_edges"])
                    total_skip_edges += sum(
                        edge["is_skip"] for edge in ready.valid_record["causal_edges"]
                    )
                    retries = ready.valid_record["validation"]["retry_errors"]
                else:
                    assert ready.error_record is not None
                    write_jsonl(error_stream, ready.error_record)
                    failed_count += 1
                    generation = ready.error_record.get("generation", {})
                    attempts = int(generation.get("attempts", 0))
                    retries = ready.error_record.get("errors", [])

                total_attempts += attempts
                for retry in retries:
                    if isinstance(retry.get("error_type"), str):
                        retry_counts[retry["error_type"]] += 1
                usage = generation.get("usage_all_attempts")
                if isinstance(usage, dict):
                    usage_records += 1
                    for key in usage_totals:
                        if isinstance(usage.get(key), int):
                            usage_totals[key] += usage[key]

                if completed % args.log_every == 0 or completed == len(selected):
                    log(
                        f"completed {completed}/{len(selected)}: "
                        f"valid={valid_count}, failed={failed_count}, "
                        f"pending={len(pending)}"
                    )

    stats = {
        "schema_version": "teacher_b_run_stats_v1",
        "status": "ok" if failed_count == 0 else "completed_with_errors",
        "script_version": SCRIPT_VERSION,
        "run_id": run["run_id"],
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(time.perf_counter() - started_perf, 3),
        "teacher_a_input": run["teacher_a_path"],
        "sat_input": run["sat_path"],
        "raw_input": run["raw_path"],
        "prompt": run["prompt_path"],
        "output": str(args.output.resolve()),
        "prepared_input_output": str(args.prepared_input_output.resolve()),
        "error_output": str(args.error_output.resolve()),
        "join_diagnostics": join_diagnostics,
        "selected_records": len(items),
        "resume_skipped_records": len(items) - len(selected),
        "requested_records": len(selected),
        "valid_records": valid_count,
        "failed_records": failed_count,
        "valid_rate": valid_count / len(selected) if selected else None,
        "total_api_attempts": total_attempts,
        "first_attempt_valid_records": first_attempt_valid,
        "retried_valid_records": retried_valid,
        "role_counts": dict(sorted(role_counts.items())),
        "prm_label_counts": dict(sorted(label_counts.items())),
        "evaluation_status_counts": dict(sorted(status_counts.items())),
        "causal_edges": total_causal_edges,
        "skip_causal_edges": total_skip_edges,
        "retry_error_type_counts": dict(sorted(retry_counts.items())),
        "usage_available_records": usage_records,
        "usage_totals": usage_totals,
        "endpoint": args.endpoint,
        "model": args.model,
        "thinking_enabled": False,
        "workers": args.workers,
        "max_retries": args.max_retries,
        "max_attempts": 1 + args.max_retries,
        "http_timeout_seconds": args.timeout,
        "heartbeat_seconds": args.heartbeat_seconds,
        "output_order": "completion",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "response_format": args.response_format,
        "pydantic_version": pydantic.__version__,
    }
    args.stats_output.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False), flush=True)
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
