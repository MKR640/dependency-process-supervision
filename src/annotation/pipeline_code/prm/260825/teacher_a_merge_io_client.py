#!/usr/bin/env python3
"""Call an OpenAI-compatible Teacher-A model for span merging and step I/O.

The program reads ``merge_io_input_v1`` JSONL records produced by the SaT
splitter, calls a locally deployed chat-completions endpoint with thinking
disabled, validates ``merge_io_output_v3`` with Pydantic and deterministic
cross-record constraints, and writes only valid results to the main JSONL.

Validation failures, malformed JSON, HTTP errors, and transient endpoint
failures are retried. ``--max-retries 3`` means one initial request plus at
most three retries. Terminal failures are written to a separate error JSONL.

Install:
    pip install "pydantic>=2.6"

Example:
    python -u teacher_a_merge_io_client.py \
      --input data/prm/260825/example.sat.jsonl \
      --output data/prm/260825/example.teacher_a.jsonl \
      --prompt scripts/prm/260825/teacher_a_prompt_v2_1.txt \
      --endpoint http://host/v1/chat/completions \
      --model qwen3.8-27b \
      --workers 8 \
      --max-retries 3
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
import uuid
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

try:
    import pydantic
    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        ValidationError,
        field_validator,
        model_validator,
    )
except ImportError as exc:  # pragma: no cover - exercised in deployment
    raise SystemExit('missing dependency: pip install "pydantic>=2.6"') from exc


SCRIPT_VERSION = "teacher_a_merge_io_client_v0.3"
INPUT_SCHEMA = "merge_io_input_v1"
OUTPUT_SCHEMA = "merge_io_output_v3"
ERROR_SCHEMA = "teacher_a_error_v1"
DEFAULT_ENDPOINT = (
    "http://127.0.0.1:4000/v1/chat/completions"
)
DEFAULT_MODEL = "qwen3.8-27b"

ALLOWED_OPERATIONS = Literal[
    "given_preparation",
    "unit_conversion",
    "formula_setup",
    "substitution",
    "calculation",
    "case_analysis",
    "uncertainty_propagation",
    "conclusion",
    "other",
]
DEPENDENCY_TYPES = Literal["data", "formula", "result", "condition"]
QUALITY_FLAGS = Literal[
    "insufficient_segmentation",
    "missing_external_content",
    "non_reasoning",
]

ENTITY_ID_RE = re.compile(r"^q(\d+)$")
STEP_ID_RE = re.compile(r"^s(\d+)$")
SPAN_ID_RE = re.compile(r"^c(\d+)$")
OUTPUT_ID_RE = re.compile(r"^(s\d+)\.o(\d+)$")
ASSUMPTION_ID_RE = re.compile(r"^(s\d+)\.a(\d+)$")
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>[\s\S]*?</think\s*>", re.I)
CODE_FENCE_RE = re.compile(
    r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.IGNORECASE
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class TeacherStep(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    spans: list[str]
    op: ALLOWED_OPERATIONS
    inputs: list[tuple[str, DEPENDENCY_TYPES]] = Field(alias="in")
    assume: list[tuple[str, str]]
    out: list[tuple[str, str]]

    @field_validator("id")
    @classmethod
    def validate_step_id(cls, value: str) -> str:
        if not STEP_ID_RE.fullmatch(value):
            raise ValueError("step id must match s01, s02, ...")
        return value

    @field_validator("spans")
    @classmethod
    def validate_spans(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("every step must contain at least one span")
        if any(not SPAN_ID_RE.fullmatch(item) for item in value):
            raise ValueError("all span ids must match c0001, c0002, ...")
        if len(value) != len(set(value)):
            raise ValueError("a step contains duplicate span ids")
        return value

    @field_validator("inputs")
    @classmethod
    def validate_inputs(
        cls, value: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        references = [item[0] for item in value]
        if any(not item.strip() for item in references):
            raise ValueError("input references must be non-empty")
        if len(references) != len(set(references)):
            raise ValueError("a step contains duplicate input references")
        return value

    @field_validator("assume")
    @classmethod
    def validate_assumptions(
        cls, value: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        identifiers = [item[0] for item in value]
        if any(not ASSUMPTION_ID_RE.fullmatch(item) for item in identifiers):
            raise ValueError("assumption ids must match s01.a01, s01.a02, ...")
        if any(not item[1].strip() for item in value):
            raise ValueError("assumption content must be non-empty")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("a step contains duplicate assumption ids")
        return value

    @field_validator("out")
    @classmethod
    def validate_outputs(
        cls, value: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        identifiers = [item[0] for item in value]
        if any(not OUTPUT_ID_RE.fullmatch(item) for item in identifiers):
            raise ValueError("output ids must match s01.o01, s01.o02, ...")
        if any(not item[1].strip() for item in value):
            raise ValueError("output content must be non-empty")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("a step contains duplicate output ids")
        return value

    @model_validator(mode="after")
    def validate_product_namespaces(self) -> "TeacherStep":
        expected_assumptions = [
            f"{self.id}.a{index:02d}" for index in range(1, len(self.assume) + 1)
        ]
        observed_assumptions = [item[0] for item in self.assume]
        if observed_assumptions != expected_assumptions:
            raise ValueError(
                f"assumptions for {self.id} must be sequential: "
                f"{expected_assumptions}"
            )
        expected_ids = [
            f"{self.id}.o{index:02d}" for index in range(1, len(self.out) + 1)
        ]
        observed_ids = [item[0] for item in self.out]
        if observed_ids != expected_ids:
            raise ValueError(
                f"outputs for {self.id} must be sequential: {expected_ids}"
            )
        if not self.assume and not self.out:
            raise ValueError(
                f"{self.id} is inert; merge it with an adjacent operational step"
            )
        return self


class TeacherOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_: Literal[OUTPUT_SCHEMA] = Field(alias="schema")
    id: str
    quality_flags: list[QUALITY_FLAGS]
    problem_entities: list[tuple[str, str]]
    steps: list[TeacherStep]

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("id must be non-empty")
        return value

    @field_validator("quality_flags")
    @classmethod
    def validate_quality_flags(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("quality_flags must be unique")
        return value

    @field_validator("problem_entities")
    @classmethod
    def validate_problem_entities(
        cls, value: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        identifiers = [item[0] for item in value]
        if any(not ENTITY_ID_RE.fullmatch(item) for item in identifiers):
            raise ValueError("problem entity ids must match q01, q02, ...")
        if any(not item[1].strip() for item in value):
            raise ValueError("problem entity content must be non-empty")
        expected = [f"q{index:02d}" for index in range(1, len(value) + 1)]
        if identifiers != expected:
            raise ValueError(f"problem entity ids must be sequential: {expected}")
        return value

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, value: list[TeacherStep]) -> list[TeacherStep]:
        if not value:
            raise ValueError("steps must not be empty")
        expected = [f"s{index:02d}" for index in range(1, len(value) + 1)]
        observed = [item.id for item in value]
        if observed != expected:
            raise ValueError(f"step ids must be sequential: {expected}")
        return value


class InputContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    subject: Any = None
    type: Any = None
    problem: str
    options: list[Any] = Field(default_factory=list)


class TeacherInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_: Literal[INPUT_SCHEMA] = Field(alias="schema")
    id: str
    context: InputContext
    candidate_spans: list[tuple[str, str]]
    rules: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_input_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("input id must be non-empty")
        return value

    @field_validator("candidate_spans")
    @classmethod
    def validate_candidate_spans(
        cls, value: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        if not value:
            raise ValueError("candidate_spans must not be empty")
        ids = [item[0] for item in value]
        if any(not SPAN_ID_RE.fullmatch(item) for item in ids):
            raise ValueError("candidate span ids must match c0001, c0002, ...")
        if len(ids) != len(set(ids)):
            raise ValueError("candidate span ids must be unique")
        if any(not item[1] for item in value):
            raise ValueError("candidate span text must not be empty")
        numeric = [int(SPAN_ID_RE.fullmatch(item).group(1)) for item in ids]
        if numeric != list(range(numeric[0], numeric[0] + len(numeric))):
            raise ValueError("candidate span ids must be numerically consecutive")
        return value


@dataclass(frozen=True)
class InputItem:
    line_number: int
    raw: dict[str, Any]
    validated: TeacherInput
    record_sha256: str


@dataclass
class GenerationResult:
    line_number: int
    record_id: str
    valid_record: dict[str, Any] | None
    error_record: dict[str, Any] | None


class OutputValidationError(ValueError):
    """Raised when JSON is syntactically valid but violates compiler rules."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call Teacher A to merge SaT spans and compile step I/O."
    )
    parser.add_argument("--input", type=Path, required=True, help="SaT JSONL input.")
    parser.add_argument("--output", type=Path, required=True, help="Valid JSONL output.")
    parser.add_argument(
        "--prompt",
        type=Path,
        required=True,
        help="Teacher-A system prompt, e.g. teacher_a_prompt_v2_1.txt.",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing an optional bearer token.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries after the initial request (default: 3).",
    )
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-request HTTP timeout in seconds (default: 600).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument(
        "--response-format",
        choices=["json-object", "none"],
        default="json-object",
        help="Request JSON mode from the endpoint when supported.",
    )
    parser.add_argument(
        "--error-output",
        type=Path,
        default=None,
        help="Terminal failure JSONL; defaults beside --output.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Run statistics JSON; defaults beside --output.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Append and skip ids already present in valid/error outputs.",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output/error/stats files.",
    )
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument(
        "--limit", type=int, default=None, help="Process at most N selected rows."
    )
    parser.add_argument(
        "--log-every", type=int, default=10, help="Progress interval (default: 10)."
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    if args.retry_backoff < 0:
        parser.error("--retry-backoff cannot be negative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.start_line < 1:
        parser.error("--start-line must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.log_every < 1:
        parser.error("--log-every must be positive")

    if args.error_output is None:
        args.error_output = args.output.with_name(
            args.output.stem + ".errors.jsonl"
        )
    if args.stats_output is None:
        args.stats_output = args.output.with_name(args.output.stem + ".stats.json")
    return args


def read_inputs(path: Path, start_line: int, limit: int | None) -> list[InputItem]:
    if path.suffix.lower() != ".jsonl":
        raise ValueError("--input must be a .jsonl file")
    items: list[InputItem] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line_number < start_line or not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(raw, dict):
                raise TypeError(f"record at line {line_number} is not an object")
            try:
                validated = TeacherInput.model_validate(raw)
            except ValidationError as exc:
                raise ValueError(
                    f"invalid Teacher-A input at line {line_number}: {exc}"
                ) from exc
            if validated.id in seen_ids:
                raise ValueError(f"duplicate input id: {validated.id!r}")
            seen_ids.add(validated.id)
            items.append(
                InputItem(
                    line_number=line_number,
                    raw=raw,
                    validated=validated,
                    record_sha256=sha256_bytes(canonical_json_bytes(raw)),
                )
            )
            if limit is not None and len(items) >= limit:
                break
    return items


def strip_and_parse_json(content: str) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise OutputValidationError("assistant content is empty or not a string")
    cleaned = THINK_BLOCK_RE.sub("", content).strip()
    fenced = CODE_FENCE_RE.fullmatch(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise OutputValidationError(
            f"assistant content is not one valid JSON object: {exc.msg} "
            f"at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(value, dict):
        raise OutputValidationError("assistant JSON root must be an object")
    return value


def validate_output_against_input(
    output: TeacherOutput,
    source: TeacherInput,
) -> dict[str, int]:
    errors: list[str] = []
    if output.id != source.id:
        errors.append(f"output id {output.id!r} does not match input id {source.id!r}")

    source_span_ids = [item[0] for item in source.candidate_spans]
    output_span_ids = [span for step in output.steps for span in step.spans]
    if output_span_ids != source_span_ids:
        missing = [item for item in source_span_ids if item not in output_span_ids]
        extra = [item for item in output_span_ids if item not in source_span_ids]
        duplicates = sorted(
            {item for item in output_span_ids if output_span_ids.count(item) > 1}
        )
        errors.append(
            "span coverage/order mismatch; "
            f"missing={missing[:20]}, extra={extra[:20]}, "
            f"duplicates={duplicates[:20]}"
        )

    entity_ids = {item[0] for item in output.problem_entities}
    output_owner: dict[str, int] = {}
    assumption_owner: dict[str, int] = {}
    all_output_ids: list[str] = []
    all_assumption_ids: list[str] = []
    used_entity_ids: set[str] = set()
    dependency_count = 0
    for step_index, step in enumerate(output.steps, start=1):
        for output_id, _ in step.out:
            all_output_ids.append(output_id)
            output_owner[output_id] = step_index
        for assumption_id, _ in step.assume:
            all_assumption_ids.append(assumption_id)
            assumption_owner[assumption_id] = step_index
    if len(all_output_ids) != len(set(all_output_ids)):
        errors.append("output identifiers are not globally unique")
    if len(all_assumption_ids) != len(set(all_assumption_ids)):
        errors.append("assumption identifiers are not globally unique")
    product_ids = all_output_ids + all_assumption_ids
    if len(product_ids) != len(set(product_ids)):
        errors.append("output and assumption identifiers overlap")

    for step_index, step in enumerate(output.steps, start=1):
        for reference_id, dependency_type in step.inputs:
            dependency_count += 1
            if reference_id in entity_ids:
                used_entity_ids.add(reference_id)
                if dependency_type not in {"data", "formula", "condition"}:
                    errors.append(
                        f"{step.id} labels root entity {reference_id!r} as "
                        f"{dependency_type!r}; qXX permits data/formula/condition"
                    )
                continue

            output_step = output_owner.get(reference_id)
            if output_step is not None:
                if dependency_type not in {"result", "formula", "condition"}:
                    errors.append(
                        f"{step.id} labels prior output {reference_id!r} as "
                        f"{dependency_type!r}; sXX.oXX permits "
                        "result/formula/condition"
                    )
                if output_step >= step_index:
                    errors.append(
                        f"{step.id} references self/future output {reference_id!r}"
                    )
                continue

            assumption_step = assumption_owner.get(reference_id)
            if assumption_step is not None:
                if dependency_type != "condition":
                    errors.append(
                        f"{step.id} labels prior assumption {reference_id!r} as "
                        f"{dependency_type!r}; sXX.aXX permits only condition"
                    )
                if assumption_step >= step_index:
                    errors.append(
                        f"{step.id} references self/future assumption "
                        f"{reference_id!r}"
                    )
                continue

            errors.append(
                f"{step.id} has unresolved input reference {reference_id!r}"
            )

    unused_entity_ids = sorted(entity_ids - used_entity_ids)
    if unused_entity_ids:
        errors.append(
            "problem_entities contains unused root entities: "
            + ", ".join(unused_entity_ids)
        )

    if errors:
        raise OutputValidationError("; ".join(errors))
    return {
        "candidate_span_count": len(source_span_ids),
        "covered_span_count": len(output_span_ids),
        "problem_entity_count": len(output.problem_entities),
        "step_count": len(output.steps),
        "dependency_reference_count": dependency_count,
        "assumption_count": len(all_assumption_ids),
        "output_count": len(all_output_ids),
        "quality_flag_count": len(output.quality_flags),
    }


def validate_model_content(
    content: str,
    source: TeacherInput,
) -> tuple[TeacherOutput, dict[str, int], str]:
    parsed = strip_and_parse_json(content)
    try:
        output = TeacherOutput.model_validate(parsed)
    except ValidationError as exc:
        raise OutputValidationError(f"Pydantic validation failed: {exc}") from exc
    checks = validate_output_against_input(output, source)
    return output, checks, sha256_bytes(content.encode("utf-8"))


def load_terminal_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"cannot resume from invalid JSON at {path}:{line_number}"
                ) from exc
            identifier = value.get("id") if isinstance(value, dict) else None
            if isinstance(identifier, str):
                ids.add(identifier)
    return ids


def prepare_output_files(args: argparse.Namespace) -> set[str]:
    paths = [args.output, args.error_output, args.stats_output]
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
            "output files already exist; pass --resume or --overwrite: "
            + ", ".join(existing)
        )
    return set()


def build_messages(
    prompt: str,
    source_record: dict[str, Any],
    previous_content: str | None,
    previous_error: str | None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(source_record, ensure_ascii=False) + "\n</INSTANCE_JSON>",
        },
    ]
    if previous_content is not None and previous_error is not None:
        messages.extend(
            [
                {"role": "assistant", "content": previous_content},
                {
                    "role": "user",
                    "content": (
                        "The previous response is invalid. Correct it and return "
                        "only one JSON object matching merge_io_output_v3. Do not "
                        "include Markdown or explanations. Validation errors:\n"
                        + previous_error[:6000]
                    ),
                },
            ]
        )
    return messages


def extract_api_response(
    status_code: int,
    response_body: bytes,
) -> tuple[str, dict[str, Any]]:
    try:
        payload = json.loads(response_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"endpoint returned non-JSON HTTP {status_code}: "
            f"{response_body.decode('utf-8', errors='replace')[:1000]}"
        ) from exc
    if status_code >= 400:
        raise RuntimeError(
            f"endpoint HTTP {status_code}: "
            + json.dumps(payload, ensure_ascii=False)[:2000]
        )
    try:
        choice = payload["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("endpoint response lacks choices[0].message.content") from exc
    if not isinstance(content, str):
        raise RuntimeError("endpoint message.content is not a string")
    metadata = {
        "response_id": payload.get("id"),
        "response_model": payload.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage"),
    }
    return content, metadata


def post_chat_completion(
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = int(response.status)
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        status_code = int(exc.code)
        response_body = exc.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"endpoint connection failed: {exc.reason}") from exc
    return extract_api_response(status_code, response_body)


def make_payload(
    args: argparse.Namespace,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        # vLLM/Qwen chat-template switch. This is serialized at the top level,
        # equivalent to OpenAI client's extra_body={"chat_template_kwargs": ...}.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if args.response_format == "json-object":
        payload["response_format"] = {"type": "json_object"}
    return payload


def correction_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def aggregate_attempt_usage(
    usage_by_attempt: list[dict[str, Any]],
) -> dict[str, int] | None:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    observed = False
    for entry in usage_by_attempt:
        usage = entry.get("usage")
        if not isinstance(usage, dict):
            continue
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value
                observed = True
    return totals if observed else None


def process_one(
    item: InputItem,
    *,
    headers: dict[str, str],
    args: argparse.Namespace,
    prompt: str,
    run_context: dict[str, Any],
) -> GenerationResult:
    started = time.perf_counter()
    previous_content: str | None = None
    previous_error: str | None = None
    last_api_metadata: dict[str, Any] = {}
    attempt_errors: list[dict[str, Any]] = []
    usage_by_attempt: list[dict[str, Any]] = []
    total_attempts = 1 + args.max_retries

    for attempt in range(1, total_attempts + 1):
        attempt_started = time.perf_counter()
        content: str | None = None
        try:
            messages = build_messages(
                prompt,
                item.raw,
                previous_content,
                previous_error,
            )
            content, api_metadata = post_chat_completion(
                args.endpoint,
                make_payload(args, messages),
                headers,
                args.timeout,
            )
            last_api_metadata = api_metadata
            usage_by_attempt.append(
                {"attempt": attempt, "usage": api_metadata.get("usage")}
            )
            output, checks, response_sha256 = validate_model_content(
                content, item.validated
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            result = output.model_dump(by_alias=True)
            result["lineage"] = {
                "pipeline_stage": "teacher_a_merge_spans_and_compile_io",
                "script_version": SCRIPT_VERSION,
                "run_id": run_context["run_id"],
                "source_file": run_context["input_path"],
                "source_file_sha256": run_context["input_sha256"],
                "source_line_number": item.line_number,
                "source_schema": INPUT_SCHEMA,
                "source_record_sha256": item.record_sha256,
                "prompt_file": run_context["prompt_path"],
                "prompt_sha256": run_context["prompt_sha256"],
                "upstream_candidate_span_count": len(
                    item.validated.candidate_spans
                ),
            }
            result["generation"] = {
                "endpoint": args.endpoint,
                "requested_model": args.model,
                "response_model": api_metadata.get("response_model"),
                "response_id": api_metadata.get("response_id"),
                "finish_reason": api_metadata.get("finish_reason"),
                "usage": api_metadata.get("usage"),
                "usage_all_attempts": aggregate_attempt_usage(usage_by_attempt),
                "usage_by_attempt": usage_by_attempt,
                "thinking_enabled": False,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "response_format": args.response_format,
                "attempts": attempt,
                "latency_ms": elapsed_ms,
                "raw_response_sha256": response_sha256,
            }
            result["validation"] = {
                "valid": True,
                "validator": SCRIPT_VERSION,
                "pydantic_version": pydantic.__version__,
                "checks": checks,
                "retry_errors": attempt_errors,
            }
            return GenerationResult(item.line_number, item.validated.id, result, None)
        except Exception as exc:  # retry is intentionally shared across failure types
            error_text = correction_error(exc)
            attempt_errors.append(
                {
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "message": error_text[:6000],
                    "latency_ms": round(
                        (time.perf_counter() - attempt_started) * 1000, 3
                    ),
                }
            )
            if isinstance(content, str):
                previous_content = content
                previous_error = error_text
            else:
                previous_content = None
                previous_error = None
            if attempt < total_attempts:
                delay = args.retry_backoff * (2 ** (attempt - 1))
                delay += random.uniform(0, min(1.0, delay * 0.1))
                time.sleep(delay)

    error_record = {
        "schema": ERROR_SCHEMA,
        "id": item.validated.id,
        "lineage": {
            "pipeline_stage": "teacher_a_merge_spans_and_compile_io",
            "script_version": SCRIPT_VERSION,
            "run_id": run_context["run_id"],
            "source_file": run_context["input_path"],
            "source_file_sha256": run_context["input_sha256"],
            "source_line_number": item.line_number,
            "source_schema": INPUT_SCHEMA,
            "source_record_sha256": item.record_sha256,
            "prompt_file": run_context["prompt_path"],
            "prompt_sha256": run_context["prompt_sha256"],
        },
        "generation": {
            "endpoint": args.endpoint,
            "requested_model": args.model,
            "response_model": last_api_metadata.get("response_model"),
            "thinking_enabled": False,
            "max_retries": args.max_retries,
            "attempts": total_attempts,
            "usage_all_attempts": aggregate_attempt_usage(usage_by_attempt),
            "usage_by_attempt": usage_by_attempt,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        },
        "errors": attempt_errors,
        "last_response": previous_content,
    }
    return GenerationResult(item.line_number, item.validated.id, None, error_record)


def write_jsonl_line(stream: Any, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    stream.flush()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"input file not found: {args.input}")
    if not args.prompt.is_file():
        raise FileNotFoundError(f"prompt file not found: {args.prompt}")

    completed_ids = prepare_output_files(args)
    prompt_bytes = args.prompt.read_bytes()
    prompt = prompt_bytes.decode("utf-8").strip()
    if OUTPUT_SCHEMA not in prompt:
        raise ValueError(f"prompt does not specify {OUTPUT_SCHEMA}")
    if "<INSTANCE_JSON>" not in prompt:
        raise ValueError("prompt must end the instance preamble with <INSTANCE_JSON>")

    input_sha256 = sha256_file(args.input)
    prompt_sha256 = sha256_bytes(prompt_bytes)
    items = read_inputs(args.input, args.start_line, args.limit)
    selected = [item for item in items if item.validated.id not in completed_ids]
    run_context = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8],
        "input_path": str(args.input.resolve()),
        "input_sha256": input_sha256,
        "prompt_path": str(args.prompt.resolve()),
        "prompt_sha256": prompt_sha256,
    }

    log(
        f"loaded {len(items)} selected input record(s); "
        f"resume skipped {len(items) - len(selected)}; pending {len(selected)}"
    )
    if not selected:
        log("nothing to process")

    api_key = os.environ.get(args.api_key_env, "").strip()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    output_mode = "a" if args.resume else "w"
    error_mode = "a" if args.resume else "w"
    started_at = utc_now()
    started_perf = time.perf_counter()
    valid_count = 0
    error_count = 0
    total_attempts = 0
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage_available_records = 0
    usage_available_attempts = 0
    first_attempt_valid_count = 0
    retried_valid_count = 0
    retry_error_type_counts: Counter[str] = Counter()

    with (
        args.output.open(output_mode, encoding="utf-8") as valid_stream,
        args.error_output.open(error_mode, encoding="utf-8") as error_stream,
        ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        futures: dict[Future[GenerationResult], InputItem] = {
            executor.submit(
                process_one,
                item,
                headers=headers,
                args=args,
                prompt=prompt,
                run_context=run_context,
            ): item
            for item in selected
        }
        expected_line_numbers = [item.line_number for item in selected]
        buffered_results: dict[int, GenerationResult] = {}
        next_write_position = 0
        written_count = 0
        for completed, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # defensive: process_one normally captures all
                result = GenerationResult(
                    line_number=item.line_number,
                    record_id=item.validated.id,
                    valid_record=None,
                    error_record={
                        "schema": ERROR_SCHEMA,
                        "id": item.validated.id,
                        "lineage": {
                            **run_context,
                            "source_line_number": item.line_number,
                            "source_record_sha256": item.record_sha256,
                        },
                        "errors": [
                            {
                                "attempt": 0,
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            }
                        ],
                    },
                )
            buffered_results[result.line_number] = result

            while (
                next_write_position < len(expected_line_numbers)
                and expected_line_numbers[next_write_position] in buffered_results
            ):
                ready_line = expected_line_numbers[next_write_position]
                ready = buffered_results.pop(ready_line)
                next_write_position += 1
                written_count += 1

                if ready.valid_record is not None:
                    write_jsonl_line(valid_stream, ready.valid_record)
                    valid_count += 1
                    generation = ready.valid_record.get("generation", {})
                    attempts = int(generation.get("attempts") or 0)
                    total_attempts += attempts
                    if attempts == 1:
                        first_attempt_valid_count += 1
                    elif attempts > 1:
                        retried_valid_count += 1
                    retry_errors = ready.valid_record.get("validation", {}).get(
                        "retry_errors", []
                    )
                    usage = generation.get("usage_all_attempts")
                    usage_by_attempt = generation.get("usage_by_attempt", [])
                else:
                    assert ready.error_record is not None
                    write_jsonl_line(error_stream, ready.error_record)
                    error_count += 1
                    generation = ready.error_record.get("generation", {})
                    total_attempts += int(generation.get("attempts") or 0)
                    retry_errors = ready.error_record.get("errors", [])
                    usage = generation.get("usage_all_attempts")
                    usage_by_attempt = generation.get("usage_by_attempt", [])

                for retry_error in retry_errors:
                    error_type = retry_error.get("error_type")
                    if isinstance(error_type, str):
                        retry_error_type_counts[error_type] += 1

                if isinstance(usage, dict):
                    usage_available_records += 1
                    for key in usage_totals:
                        value = usage.get(key)
                        if isinstance(value, int):
                            usage_totals[key] += value
                if isinstance(usage_by_attempt, list):
                    usage_available_attempts += sum(
                        isinstance(entry, dict)
                        and isinstance(entry.get("usage"), dict)
                        for entry in usage_by_attempt
                    )

            if completed % args.log_every == 0 or completed == len(selected):
                log(
                    f"completed {completed}/{len(selected)}: "
                    f"written={written_count}, valid={valid_count}, "
                    f"failed={error_count}"
                )

        if buffered_results or next_write_position != len(expected_line_numbers):
            raise RuntimeError("internal output-order buffer did not fully flush")

    stats = {
        "schema_version": "teacher_a_run_stats_v2",
        "status": "ok" if error_count == 0 else "completed_with_errors",
        "script_version": SCRIPT_VERSION,
        "run_id": run_context["run_id"],
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(time.perf_counter() - started_perf, 3),
        "input": str(args.input.resolve()),
        "input_sha256": input_sha256,
        "prompt": str(args.prompt.resolve()),
        "prompt_sha256": prompt_sha256,
        "output": str(args.output.resolve()),
        "error_output": str(args.error_output.resolve()),
        "input_records_selected": len(items),
        "resume_skipped_records": len(items) - len(selected),
        "requested_records": len(selected),
        "valid_records": valid_count,
        "failed_records": error_count,
        "valid_rate": valid_count / len(selected) if selected else None,
        "total_api_attempts": total_attempts,
        "first_attempt_valid_records": first_attempt_valid_count,
        "retried_valid_records": retried_valid_count,
        "first_attempt_valid_rate": (
            first_attempt_valid_count / len(selected) if selected else None
        ),
        "retry_error_type_counts": dict(sorted(retry_error_type_counts.items())),
        "usage_available_records": usage_available_records,
        "usage_available_attempts": usage_available_attempts,
        "usage_totals": usage_totals,
        "usage_scope": "all attempts with endpoint-reported usage",
        "output_order": "source_line_number",
        "endpoint": args.endpoint,
        "model": args.model,
        "thinking_enabled": False,
        "workers": args.workers,
        "max_retries": args.max_retries,
        "max_attempts": 1 + args.max_retries,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "response_format": args.response_format,
        "pydantic_version": pydantic.__version__,
    }
    args.stats_output.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return 0 if error_count == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted by user")
        raise SystemExit(130)
    except Exception as exc:
        log(f"error: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
