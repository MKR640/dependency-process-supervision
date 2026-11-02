#!/usr/bin/env python3
"""Audit SaT/Teacher-A/Teacher-B lineage and materialize PRM step text.

This is a deterministic, offline postprocessor.  It never merges different
Teacher-A steps.  For every step ``sXX`` it concatenates exactly the ordered
SaT spans referenced by ``steps[].spans`` and attaches the corresponding
Teacher-B judgment, PRM label, audited dependency edges, and error-causal
edges.

Three JSONL outputs are produced:

* ``--output``: structurally valid, directly usable records;
* ``--review-output``: structurally valid records with soft quality flags;
* ``--reject-output``: ineligible Teacher-B records or hard audit failures.

Compact SaT records prove that the candidate spans form the text supplied to
Teacher A.  They cannot by themselves prove that SaT omitted nothing from the
pre-SaT solution.  Supply ``--raw-input`` to re-extract ``<think>`` with the
same policy as the splitter and require an exact character-for-character
comparison.  Use ``--require-raw-match`` when raw verification is mandatory.

The implementation indexes JSONL byte offsets instead of retaining all large
records in memory, so it is suitable for tens of thousands of long solutions.
Only the current joined record is materialized.

Example:
    python -u prm_step_materializer.py \
      --sat-input task.sat.jsonl \
      --teacher-a-input task.teacher_a_v2_1.jsonl \
      --teacher-b-input task.teacher_b.jsonl \
      --raw-input task.raw.jsonl \
      --output task.prm_materialized.jsonl \
      --require-raw-match \
      --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO


SCRIPT_VERSION = "prm_step_materializer_v1.1"
OUTPUT_SCHEMA = "prm_materialized_trace_v1"
REJECT_SCHEMA = "prm_materialization_reject_v1"
SAT_SCHEMA = "merge_io_input_v1"
TEACHER_A_SCHEMA = "merge_io_output_v3"
TEACHER_B_SCHEMA = "judge_graph_output_v3"

STEP_RE = re.compile(r"^s(\d+)$")
SPAN_RE = re.compile(r"^c(\d+)$")
ENTITY_RE = re.compile(r"^q(\d+)$")
STEP_REF_RE = re.compile(r"^(s\d+)\.(?:o|a)\d+$")
THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.I | re.S)
FINAL_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?final\s+answer(?:\*\*)?\s*:\s*", re.I
)
ANSWER_RE = re.compile(r"<answer\b[^>]*>(.*?)</answer\s*>", re.I | re.S)

ROLE_TO_LABEL = {
    "correct": "C",
    "cessation": "C",
    "new_error": "I",
    "propagated_error": "I",
    "unverifiable": "U",
}
VALID_ROLES = set(ROLE_TO_LABEL)


@dataclass(frozen=True)
class IndexEntry:
    offset: int
    line_number: int
    record_sha256: str


@dataclass(frozen=True)
class Span:
    span_id: str
    text: str
    start: int
    end: int
    declared_start: int | None = None
    declared_end: int | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def record_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def add_issue(
    issues: list[dict[str, Any]], code: str, message: str, **details: Any
) -> None:
    issue = {"code": code, "message": message}
    issue.update({key: value for key, value in details.items() if value is not None})
    issues.append(issue)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join and audit SaT, Teacher A, and Teacher B for PRM training."
    )
    parser.add_argument("--sat-input", type=Path, required=True)
    parser.add_argument("--teacher-a-input", type=Path, required=True)
    parser.add_argument("--teacher-b-input", type=Path, required=True)
    parser.add_argument(
        "--raw-input",
        type=Path,
        help="Optional pre-SaT raw JSONL for exact reasoning-text verification.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--reject-output", type=Path)
    parser.add_argument("--stats-output", type=Path)
    parser.add_argument(
        "--require-raw-match",
        action="store_true",
        help="Reject records without a raw record or exact raw/SaT text match.",
    )
    parser.add_argument(
        "--no-think-required",
        action="store_true",
        help="Use the complete solution if the raw record has no <think> block.",
    )
    parser.add_argument(
        "--keep-final-summary",
        action="store_true",
        help="Keep a trailing 'Final Answer:' section inside <think>.",
    )
    parser.add_argument(
        "--ignore-preprocessed-fields",
        action="store_true",
        help="Use raw.solution rather than sat_preprocessing.solution.",
    )
    parser.add_argument(
        "--min-reasoning-chars",
        type=int,
        default=200,
        help="Hard-reject reasoning shorter than this (default: 200; 0 disables).",
    )
    parser.add_argument(
        "--min-candidate-spans",
        type=int,
        default=10,
        help="Hard-reject SaT records with fewer spans (default: 10; 0 disables).",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=5,
        help="Hard-reject Teacher-A records with fewer merged steps (default: 5; 0 disables).",
    )
    parser.add_argument(
        "--max-step-chars",
        type=int,
        default=4000,
        help="Soft-review a materialized step longer than this; 0 disables.",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.92,
        help="Soft-review near-duplicate step text using character trigrams; 0 disables.",
    )
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def derived_path(output: Path, marker: str) -> Path:
    if output.suffix == ".jsonl":
        return output.with_name(output.name[:-6] + marker)
    return output.with_name(output.name + marker)


def resolve_output_paths(args: argparse.Namespace) -> None:
    args.review_output = args.review_output or derived_path(args.output, ".review.jsonl")
    args.reject_output = args.reject_output or derived_path(args.output, ".rejected.jsonl")
    args.stats_output = args.stats_output or derived_path(args.output, ".stats.json")


def prepare_outputs(args: argparse.Namespace) -> None:
    paths = [args.output, args.review_output, args.reject_output, args.stats_output]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("output, review, reject, and stats paths must be distinct")
    existing = [str(path) for path in paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "output files exist; pass --overwrite: " + ", ".join(existing)
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in paths:
            if path.exists():
                path.unlink()


def decode_record(raw_line: bytes, path: Path, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"record at {path}:{line_number} is not an object")
    return value


def build_offset_index(path: Path, name: str) -> tuple[dict[str, IndexEntry], str]:
    index: dict[str, IndexEntry] = {}
    file_digest = hashlib.sha256()
    with path.open("rb") as stream:
        line_number = 0
        while True:
            offset = stream.tell()
            raw_line = stream.readline()
            if not raw_line:
                break
            file_digest.update(raw_line)
            line_number += 1
            if not raw_line.strip():
                continue
            value = decode_record(raw_line, path, line_number)
            record_id = value.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{name} record at {path}:{line_number} lacks string id")
            if record_id in index:
                prior = index[record_id].line_number
                raise ValueError(
                    f"duplicate {name} id {record_id!r} at lines {prior} and {line_number}"
                )
            index[record_id] = IndexEntry(
                offset=offset,
                line_number=line_number,
                record_sha256=record_sha256(value),
            )
    log(f"indexed {len(index)} {name} record(s): {path}")
    return index, file_digest.hexdigest()


def read_at(
    stream: BinaryIO, entry: IndexEntry, path: Path
) -> dict[str, Any]:
    stream.seek(entry.offset)
    raw_line = stream.readline()
    return decode_record(raw_line, path, entry.line_number)


def get_nested(record: dict[str, Any], dotted: str) -> Any:
    current: Any = record
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def is_meaningful(value: Any) -> bool:
    return value not in (None, "", [], -1, "-1")


def preferred_solution(raw: dict[str, Any], ignore_preprocessed: bool) -> Any:
    if not ignore_preprocessed:
        value = get_nested(raw, "sat_preprocessing.solution")
        if is_meaningful(value):
            return value
    return raw.get("solution")


def extract_reasoning(
    solution: str, require_think: bool, keep_final_summary: bool
) -> tuple[str, dict[str, Any]]:
    matches = list(THINK_RE.finditer(solution))
    if len(matches) > 1:
        raise ValueError(f"solution has {len(matches)} <think> blocks; expected one")
    if matches:
        reasoning = matches[0].group(1)
        source = "solution.think"
    elif require_think:
        raise ValueError("solution has no <think>...</think> block")
    else:
        reasoning = solution
        source = "solution"
    removed = False
    if not keep_final_summary:
        found = list(FINAL_ANSWER_RE.finditer(reasoning))
        if found:
            reasoning = reasoning[: found[-1].start()]
            removed = True
    reasoning = reasoning.strip()
    if not reasoning:
        raise ValueError("extracted reasoning is empty")
    return reasoning, {"source": source, "removed_final_summary": removed}


def extract_proposed_answer(raw: dict[str, Any], ignore_preprocessed: bool) -> str | None:
    solution = preferred_solution(raw, ignore_preprocessed)
    if not isinstance(solution, str):
        return None
    matches = list(ANSWER_RE.finditer(solution))
    return matches[-1].group(1).strip() if matches else None


def first_mismatch(left: str, right: str) -> dict[str, Any] | None:
    limit = min(len(left), len(right))
    position = next((i for i in range(limit) if left[i] != right[i]), limit)
    if position == len(left) == len(right):
        return None
    radius = 60
    return {
        "index": position,
        "sat_context": left[max(0, position - radius) : position + radius],
        "raw_context": right[max(0, position - radius) : position + radius],
        "sat_length": len(left),
        "raw_length": len(right),
    }


def abnormal_text_characters(text: str, limit: int = 20) -> list[dict[str, Any]]:
    """Locate control/replacement characters unsafe for PRM training.

    Newline, carriage return, and tab are intentional formatting characters.
    Other Unicode ``Cc`` characters and U+FFFD replacement markers indicate
    damaged or unexpectedly encoded source text.
    """
    found: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character == "\ufffd" or (
            unicodedata.category(character) == "Cc"
            and character not in {"\n", "\r", "\t"}
        ):
            found.append(
                {
                    "index": index,
                    "codepoint": f"U+{ord(character):04X}",
                    "name": unicodedata.name(character, "UNNAMED"),
                }
            )
            if len(found) >= limit:
                break
    return found


def teacher_b_prefilter(record: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if record.get("schema") != TEACHER_B_SCHEMA:
        add_issue(issues, "teacher_b_schema", f"expected {TEACHER_B_SCHEMA!r}")
    if record.get("evaluation_status") != "complete":
        add_issue(
            issues,
            "teacher_b_not_complete",
            "Teacher B evaluation_status must be complete",
            observed=record.get("evaluation_status"),
        )
    validation = record.get("validation")
    if isinstance(validation, dict) and validation.get("valid") is not True:
        add_issue(issues, "teacher_b_invalid", "Teacher B validation.valid is not true")
    judgments = record.get("judgments")
    if not isinstance(judgments, list) or not judgments:
        add_issue(issues, "teacher_b_judgments", "Teacher B judgments are missing/empty")
    else:
        for row in judgments:
            if not isinstance(row, dict):
                add_issue(issues, "teacher_b_judgment_type", "judgment is not an object")
                continue
            if row.get("role") == "unverifiable" or row.get("prm_label") == "U":
                add_issue(
                    issues,
                    "teacher_b_unverifiable",
                    "unverifiable/U steps are excluded from PRM materialization",
                    step=row.get("step"),
                )
    return issues


def parse_spans(
    sat: dict[str, Any], hard: list[dict[str, Any]]
) -> tuple[list[Span], str]:
    raw_spans = sat.get("candidate_spans")
    if not isinstance(raw_spans, list) or not raw_spans:
        add_issue(hard, "candidate_spans_missing", "candidate_spans must be non-empty")
        return [], ""
    spans: list[Span] = []
    cursor = 0
    seen: set[str] = set()
    for position, item in enumerate(raw_spans, 1):
        span_id: Any = None
        text: Any = None
        declared_start: Any = None
        declared_end: Any = None
        if isinstance(item, (list, tuple)) and len(item) == 2:
            span_id, text = item
        elif isinstance(item, dict):
            span_id = item.get("span_id", item.get("id"))
            text = item.get("text")
            declared_start = item.get("start_char", item.get("start"))
            declared_end = item.get("end_char", item.get("end"))
        else:
            add_issue(
                hard,
                "candidate_span_shape",
                "span must be [id,text] or an object",
                position=position,
            )
            continue
        if not isinstance(span_id, str) or not SPAN_RE.fullmatch(span_id):
            add_issue(hard, "candidate_span_id", "invalid span id", position=position)
            continue
        expected = f"c{position:04d}"
        if span_id != expected:
            add_issue(
                hard,
                "candidate_span_order",
                "span ids must be sequential",
                expected=expected,
                observed=span_id,
            )
        if span_id in seen:
            add_issue(hard, "candidate_span_duplicate_id", "duplicate span id", span=span_id)
        seen.add(span_id)
        if not isinstance(text, str) or text == "":
            add_issue(hard, "candidate_span_text", "span text must be non-empty", span=span_id)
            text = text if isinstance(text, str) else ""
        start, end = cursor, cursor + len(text)
        if declared_start is not None or declared_end is not None:
            if not isinstance(declared_start, int) or not isinstance(declared_end, int):
                add_issue(hard, "candidate_span_offsets", "declared offsets must be integers", span=span_id)
            elif declared_start != start or declared_end != end:
                add_issue(
                    hard,
                    "candidate_span_offset_gap_or_overlap",
                    "declared offsets do not form a contiguous partition",
                    span=span_id,
                    expected=[start, end],
                    observed=[declared_start, declared_end],
                )
        spans.append(
            Span(span_id, text, start, end,
                 declared_start if isinstance(declared_start, int) else None,
                 declared_end if isinstance(declared_end, int) else None)
        )
        cursor = end
    joined = "".join(span.text for span in spans)
    declared_reasoning = sat.get("reasoning_text")
    if isinstance(declared_reasoning, str) and declared_reasoning != joined:
        add_issue(
            hard,
            "sat_reasoning_text_mismatch",
            "candidate spans do not reproduce SAT reasoning_text",
            mismatch=first_mismatch(joined, declared_reasoning),
        )
    return spans, joined


def sequential_ids(values: list[str], prefix: str) -> bool:
    return values == [f"{prefix}{i:02d}" for i in range(1, len(values) + 1)]


def normalized_for_similarity(text: str) -> str:
    return " ".join(text.lower().split())


def trigrams(text: str) -> set[str]:
    compact = normalized_for_similarity(text)
    if len(compact) < 3:
        return {compact} if compact else set()
    return {compact[i : i + 3] for i in range(len(compact) - 2)}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def delimiter_flags(step_id: str, text: str) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    # These are conservative warnings only; escaped currency dollars remain a
    # known ambiguous case and therefore never cause rejection.
    unescaped_dollars = len(re.findall(r"(?<!\\)\$", text))
    if unescaped_dollars % 2:
        add_issue(
            flags,
            "unbalanced_dollar_math",
            "odd number of unescaped $ delimiters",
            step=step_id,
            count=unescaped_dollars,
        )
    if text.count("\\[") != text.count("\\]"):
        add_issue(flags, "unbalanced_display_math", "\\[ and \\] counts differ", step=step_id)
    if text.count("\\(") != text.count("\\)"):
        add_issue(flags, "unbalanced_inline_math", "\\( and \\) counts differ", step=step_id)
    if text.count("```") % 2:
        add_issue(flags, "unbalanced_code_fence", "odd number of code fences", step=step_id)
    return flags


def materialize_and_audit(
    *,
    record_id: str,
    sat: dict[str, Any],
    sat_entry: IndexEntry,
    teacher_a: dict[str, Any],
    teacher_a_entry: IndexEntry,
    teacher_b: dict[str, Any],
    teacher_b_line: int,
    teacher_b_hash: str,
    raw: dict[str, Any] | None,
    raw_entry: IndexEntry | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    hard: list[dict[str, Any]] = []
    soft: list[dict[str, Any]] = []

    augmented_schema = get_nested(sat, "candidate_span_metadata.schema_version")
    if sat.get("schema") != SAT_SCHEMA and augmented_schema != "sat_candidate_spans_v0.4":
        add_issue(hard, "sat_schema", "unsupported SaT schema", observed=sat.get("schema"))
    if teacher_a.get("schema") != TEACHER_A_SCHEMA:
        add_issue(hard, "teacher_a_schema", "unsupported Teacher A schema", observed=teacher_a.get("schema"))
    if any(value.get("id") != record_id for value in (sat, teacher_a, teacher_b)):
        add_issue(hard, "join_id_mismatch", "joined record ids differ")

    a_validation = teacher_a.get("validation")
    if isinstance(a_validation, dict) and a_validation.get("valid") is not True:
        add_issue(hard, "teacher_a_invalid", "Teacher A validation.valid is not true")
    elif not isinstance(a_validation, dict):
        add_issue(soft, "teacher_a_validation_missing", "Teacher A validation metadata is missing")
    b_validation = teacher_b.get("validation")
    if not isinstance(b_validation, dict):
        add_issue(soft, "teacher_b_validation_missing", "Teacher B validation metadata is missing")

    quality_flags = teacher_a.get("quality_flags", [])
    if not isinstance(quality_flags, list):
        add_issue(hard, "teacher_a_quality_flags", "quality_flags must be a list")
        quality_flags = []
    if "non_reasoning" in quality_flags:
        add_issue(hard, "non_reasoning", "Teacher A marks the sample as non_reasoning")
    for flag in quality_flags:
        if flag != "non_reasoning":
            add_issue(soft, f"teacher_a_{flag}", "Teacher A quality flag", flag=flag)

    a_lineage = teacher_a.get("lineage")
    if isinstance(a_lineage, dict):
        expected = a_lineage.get("source_record_sha256")
        if isinstance(expected, str) and expected != sat_entry.record_sha256:
            add_issue(
                hard,
                "teacher_a_sat_lineage_mismatch",
                "Teacher A source_record_sha256 differs from supplied SaT record",
                expected=expected,
                observed=sat_entry.record_sha256,
            )
        elif not isinstance(expected, str):
            add_issue(soft, "teacher_a_sat_lineage_missing", "Teacher A SaT record hash is missing")
    else:
        add_issue(soft, "teacher_a_lineage_missing", "Teacher A lineage is missing")

    b_lineage = teacher_b.get("lineage")
    if isinstance(b_lineage, dict):
        expected_a = b_lineage.get("teacher_a_record_sha256")
        expected_sat = b_lineage.get("sat_record_sha256")
        if isinstance(expected_a, str) and expected_a != teacher_a_entry.record_sha256:
            add_issue(hard, "teacher_b_teacher_a_lineage_mismatch", "Teacher B points to another Teacher A record")
        elif not isinstance(expected_a, str):
            add_issue(soft, "teacher_b_teacher_a_lineage_missing", "Teacher B Teacher-A record hash is missing")
        if isinstance(expected_sat, str) and expected_sat != sat_entry.record_sha256:
            add_issue(hard, "teacher_b_sat_lineage_mismatch", "Teacher B points to another SaT record")
        elif not isinstance(expected_sat, str):
            add_issue(soft, "teacher_b_sat_lineage_missing", "Teacher B SaT record hash is missing")
    else:
        add_issue(soft, "teacher_b_lineage_missing", "Teacher B lineage is missing")

    spans, reasoning_text = parse_spans(sat, hard)
    span_by_id = {span.span_id: span for span in spans}
    span_ids = [span.span_id for span in spans]
    if not reasoning_text.strip():
        add_issue(
            hard,
            "source_reasoning_empty",
            "reconstructed reasoning contains no non-whitespace content",
        )
    abnormal_characters = abnormal_text_characters(reasoning_text)
    if abnormal_characters:
        add_issue(
            hard,
            "source_abnormal_control_characters",
            "reasoning contains invalid control or replacement characters",
            occurrences=abnormal_characters,
        )
    if (
        args.min_reasoning_chars > 0
        and len(reasoning_text) < args.min_reasoning_chars
    ):
        add_issue(
            hard,
            "sat_reasoning_too_short",
            "reasoning is below the SaT complexity threshold",
            observed=len(reasoning_text),
            required=args.min_reasoning_chars,
        )
    if args.min_candidate_spans > 0 and len(spans) < args.min_candidate_spans:
        add_issue(
            hard,
            "sat_too_few_candidate_spans",
            "candidate span count is below the SaT complexity threshold",
            observed=len(spans),
            required=args.min_candidate_spans,
        )

    steps = teacher_a.get("steps")
    if not isinstance(steps, list) or not steps:
        add_issue(hard, "teacher_a_steps", "Teacher A steps must be a non-empty list")
        steps = []
    if args.min_steps > 0 and len(steps) < args.min_steps:
        add_issue(
            hard,
            "teacher_a_too_few_steps",
            "merged reasoning step count is below the Teacher-A threshold",
            observed=len(steps),
            required=args.min_steps,
        )
    step_ids = [step.get("id") for step in steps if isinstance(step, dict)]
    if len(step_ids) != len(steps) or not all(isinstance(step_id, str) for step_id in step_ids):
        add_issue(hard, "teacher_a_step_id", "all Teacher A steps need string ids")
    elif not sequential_ids(step_ids, "s"):
        add_issue(hard, "teacher_a_step_order", "Teacher A step ids must be sequential")

    flattened: list[str] = []
    materialized: list[dict[str, Any]] = []
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            add_issue(hard, "teacher_a_step_shape", "Teacher A step is not an object", position=index)
            continue
        step_id = step.get("id")
        refs = step.get("spans")
        if not isinstance(refs, list) or not refs or not all(isinstance(ref, str) for ref in refs):
            add_issue(hard, "teacher_a_step_spans", "step spans must be non-empty string list", step=step_id)
            continue
        flattened.extend(refs)
        positions = [span_ids.index(ref) for ref in refs if ref in span_by_id]
        missing = [ref for ref in refs if ref not in span_by_id]
        if missing:
            add_issue(hard, "teacher_a_unknown_span", "step references unknown spans", step=step_id, spans=missing)
            continue
        if positions != list(range(positions[0], positions[0] + len(positions))):
            add_issue(hard, "teacher_a_noncontiguous_spans", "one step must use one contiguous c-span interval", step=step_id, spans=refs)
        text = "".join(span_by_id[ref].text for ref in refs)
        first_span, last_span = span_by_id[refs[0]], span_by_id[refs[-1]]
        materialized.append(
            {
                "id": step_id,
                "index": index,
                "span_ids": refs,
                "start_char": first_span.start,
                "end_char": last_span.end,
                "text": text,
                "text_sha256": text_sha256(text),
                "op": step.get("op"),
                "in": step.get("in", []),
                "assume": step.get("assume", []),
                "out": step.get("out", []),
            }
        )
        if args.max_step_chars > 0 and len(text) > args.max_step_chars:
            add_issue(soft, "step_too_long", "materialized step exceeds configured length", step=step_id, chars=len(text), threshold=args.max_step_chars)
        soft.extend(delimiter_flags(str(step_id), text))

    if flattened != span_ids:
        add_issue(
            hard,
            "teacher_a_span_partition",
            "flattened Teacher A step spans do not exactly equal all SaT span ids",
            expected_count=len(span_ids),
            observed_count=len(flattened),
        )
    rebuilt_from_steps = "".join(step["text"] for step in materialized)
    if rebuilt_from_steps != reasoning_text:
        add_issue(hard, "step_text_partition", "materialized step texts do not exactly reconstruct SaT text", mismatch=first_mismatch(rebuilt_from_steps, reasoning_text))

    # Equal text in distinct source intervals is not an overlap.  It is retained
    # but surfaced for review because repeated reasoning can bias PRM samples.
    normalized = [(step["id"], normalized_for_similarity(step["text"])) for step in materialized]
    gram_sets = [trigrams(text) for _, text in normalized]
    for left_i in range(len(normalized)):
        left_id, left_text = normalized[left_i]
        for right_i in range(left_i + 1, len(normalized)):
            right_id, right_text = normalized[right_i]
            if left_text and left_text == right_text:
                add_issue(soft, "duplicate_step_text", "two distinct steps have identical normalized text", steps=[left_id, right_id])
            elif args.near_duplicate_threshold > 0:
                score = jaccard(gram_sets[left_i], gram_sets[right_i])
                if score >= args.near_duplicate_threshold:
                    add_issue(soft, "near_duplicate_step_text", "two distinct steps have near-duplicate text", steps=[left_id, right_id], score=round(score, 4))

    raw_check: dict[str, Any] = {"status": "unavailable"}
    if raw is not None:
        solution = preferred_solution(raw, args.ignore_preprocessed_fields)
        if not isinstance(solution, str) or not solution:
            add_issue(hard if args.require_raw_match else soft, "raw_solution_missing", "raw solution is missing or not a string")
            raw_check = {"status": "not_extractable"}
        else:
            try:
                raw_reasoning, extraction = extract_reasoning(
                    solution,
                    require_think=not args.no_think_required,
                    keep_final_summary=args.keep_final_summary,
                )
                mismatch = first_mismatch(reasoning_text, raw_reasoning)
                raw_check = {
                    "status": "matched" if mismatch is None else "mismatched",
                    "sat_text_sha256": text_sha256(reasoning_text),
                    "raw_text_sha256": text_sha256(raw_reasoning),
                    "extraction": extraction,
                    "mismatch": mismatch,
                }
                if mismatch is not None:
                    add_issue(hard, "raw_sat_text_mismatch", "SaT spans do not exactly reproduce the pre-SaT reasoning", mismatch=mismatch)
            except Exception as exc:
                raw_check = {"status": "not_extractable", "error": f"{type(exc).__name__}: {exc}"}
                add_issue(hard if args.require_raw_match else soft, "raw_reasoning_extraction", "cannot extract raw reasoning", error=str(exc))
    elif args.require_raw_match:
        add_issue(hard, "raw_record_missing", "--require-raw-match was set but no matching raw record exists")
    else:
        add_issue(
            soft,
            "raw_reasoning_unverified",
            "no matching raw record; SaT omission/duplication relative to the pre-SaT solution is unverified",
        )

    judgments = teacher_b.get("judgments")
    if not isinstance(judgments, list):
        judgments = []
    judgment_ids = [row.get("step") for row in judgments if isinstance(row, dict)]
    if judgment_ids != step_ids:
        add_issue(hard, "teacher_b_step_order", "Teacher B judgments do not match Teacher A steps in order", expected=step_ids, observed=judgment_ids)
    judgment_by_step = {row.get("step"): row for row in judgments if isinstance(row, dict)}
    labels = teacher_b.get("labels")
    expected_labels: list[list[str]] = []
    for step_id in step_ids:
        judgment = judgment_by_step.get(step_id, {})
        role = judgment.get("role")
        if role not in VALID_ROLES:
            add_issue(hard, "teacher_b_role", "invalid role", step=step_id, observed=role)
            expected_label = None
        else:
            expected_label = ROLE_TO_LABEL[role]
        observed_label = judgment.get("prm_label")
        if observed_label != expected_label:
            add_issue(hard, "teacher_b_prm_label", "prm_label conflicts with role", step=step_id, expected=expected_label, observed=observed_label)
        if isinstance(expected_label, str):
            expected_labels.append([step_id, expected_label])
        error_type = judgment.get("error_type")
        caused_by = judgment.get("caused_by")
        note = judgment.get("note")
        note_present = isinstance(note, str) and bool(note.strip())
        if not isinstance(caused_by, list) or not all(isinstance(source, str) for source in caused_by):
            add_issue(hard, "teacher_b_caused_by", "caused_by must be a list of step ids", step=step_id)
            caused_by = []
        elif len(caused_by) != len(set(caused_by)):
            add_issue(hard, "teacher_b_duplicate_cause", "caused_by contains duplicates", step=step_id)
        if role == "correct" and (error_type is not None or caused_by or note is not None):
            add_issue(hard, "teacher_b_correct_fields", "correct requires null error_type/note and no causes", step=step_id)
        elif role == "new_error" and (error_type is None or not note_present):
            add_issue(hard, "teacher_b_new_error_fields", "new_error requires error_type and a non-empty note", step=step_id)
        elif role in {"propagated_error", "cessation"} and (error_type is not None or not caused_by or not note_present):
            add_issue(hard, "teacher_b_relational_role_fields", f"{role} requires null error_type, causes, and a note", step=step_id)
        expected_mixed = role == "new_error" and bool(caused_by)
        if judgment.get("mixed_error") != expected_mixed:
            add_issue(hard, "teacher_b_mixed_error", "mixed_error is inconsistent with role/caused_by", step=step_id, expected=expected_mixed, observed=judgment.get("mixed_error"))
    if labels != expected_labels:
        add_issue(hard, "teacher_b_labels", "top-level labels conflict with judgments", expected=expected_labels, observed=labels)

    first_expected = next(
        (step_id for step_id in step_ids if judgment_by_step.get(step_id, {}).get("role") == "new_error"),
        None,
    )
    if teacher_b.get("first_error") != first_expected:
        add_issue(hard, "teacher_b_first_error", "first_error is inconsistent with earliest new_error", expected=first_expected, observed=teacher_b.get("first_error"))

    step_position = {step_id: index for index, step_id in enumerate(step_ids, 1)}

    # Recompute the relation to the first error from the sparse causal parents.
    causal_parents = {
        step_id: judgment_by_step.get(step_id, {}).get("caused_by", [])
        if isinstance(judgment_by_step.get(step_id, {}).get("caused_by", []), list)
        else []
        for step_id in step_ids
    }

    def reachable_from_first(target: str, visiting: set[str] | None = None) -> bool:
        if first_expected is None:
            return False
        if target == first_expected:
            return True
        visiting = set() if visiting is None else set(visiting)
        if target in visiting:
            return False
        visiting.add(target)
        return any(
            parent in causal_parents and reachable_from_first(parent, visiting)
            for parent in causal_parents.get(target, [])
        )

    for step_id in step_ids:
        judgment = judgment_by_step.get(step_id, {})
        role = judgment.get("role")
        if first_expected is None:
            expected_relation = "unknown" if role == "unverifiable" else "none"
        elif step_id == first_expected:
            expected_relation = "first_error"
        elif role == "cessation":
            expected_relation = "cessation"
        elif role == "unverifiable":
            expected_relation = "unknown"
        elif role in {"new_error", "propagated_error"}:
            if first_expected in causal_parents.get(step_id, []):
                expected_relation = "direct_propagation"
            elif reachable_from_first(step_id):
                expected_relation = "indirect_propagation"
            else:
                expected_relation = "independent_error"
        elif step_position.get(step_id, 0) > step_position.get(first_expected, 10**9):
            expected_relation = "unaffected"
        else:
            expected_relation = "none"
        if judgment.get("relation_to_first_error") != expected_relation:
            add_issue(hard, "teacher_b_first_error_relation", "relation_to_first_error is inconsistent", step=step_id, expected=expected_relation, observed=judgment.get("relation_to_first_error"))
    problem_entities = teacher_a.get("problem_entities", [])
    ref_owner: dict[str, str | None] = {}
    if isinstance(problem_entities, list):
        for item in problem_entities:
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str):
                ref_owner[item[0]] = None
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("id"), str):
            continue
        for field in ("assume", "out"):
            values = step.get(field, [])
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str):
                        ref_owner[item[0]] = step["id"]

    audited_edges = teacher_b.get("audited_dependency_edges")
    if not isinstance(audited_edges, list):
        add_issue(hard, "audited_dependency_edges_missing", "Teacher B audited_dependency_edges must be a list")
        audited_edges = []
    for edge in audited_edges:
        if not isinstance(edge, dict):
            add_issue(hard, "audited_dependency_edge_shape", "dependency edge is not an object")
            continue
        source_ref = edge.get("source_ref")
        target = edge.get("target_step")
        source = edge.get("source_step")
        if source_ref not in ref_owner:
            add_issue(hard, "dependency_unknown_ref", "dependency edge has unknown source_ref", source_ref=source_ref)
            continue
        if target not in step_position:
            add_issue(hard, "dependency_unknown_target", "dependency edge has unknown target", target=target)
            continue
        expected_source = ref_owner[source_ref]
        if source != expected_source:
            add_issue(hard, "dependency_source_owner", "source_step does not own source_ref", source_ref=source_ref, expected=expected_source, observed=source)
        expected_gap = None if source is None else step_position[target] - step_position.get(source, 10**9) - 1
        if source is not None and (source not in step_position or expected_gap < 0):
            add_issue(hard, "dependency_not_prior", "dependency source must precede target", source=source, target=target)
        if edge.get("sequence_gap") != expected_gap or edge.get("is_skip") != (isinstance(expected_gap, int) and expected_gap > 0):
            add_issue(hard, "dependency_gap", "dependency gap/is_skip is inconsistent", source=source, target=target)

    # Independently apply Teacher B's sparse dependency edits to Teacher A and
    # compare the complete resulting edge list, including order and edge type.
    expected_graph: dict[str, list[tuple[str, str]]] = {}
    for step in steps:
        if not isinstance(step, dict) or step.get("id") not in step_position:
            continue
        parsed_inputs: list[tuple[str, str]] = []
        for item in step.get("in", []):
            if isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(value, str) for value in item):
                parsed_inputs.append((item[0], item[1]))
            else:
                add_issue(hard, "teacher_a_dependency_shape", "Teacher A input dependency must be [ref,type]", step=step.get("id"))
        expected_graph[step["id"]] = parsed_inputs

    def parse_edits(field: str) -> list[tuple[str, str, str]]:
        value = teacher_b.get(field, [])
        parsed: list[tuple[str, str, str]] = []
        if not isinstance(value, list):
            add_issue(hard, f"{field}_shape", f"{field} must be a list")
            return parsed
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 3 and all(isinstance(part, str) for part in item):
                parsed.append((item[0], item[1], item[2]))
            else:
                add_issue(hard, f"{field}_item", f"{field} item must be [target,ref,type]", observed=item)
        if len(parsed) != len(set(parsed)):
            add_issue(hard, f"{field}_duplicate", f"{field} contains duplicates")
        return parsed

    removes = parse_edits("dependency_remove")
    adds = parse_edits("dependency_add")
    if set(removes) & set(adds):
        add_issue(hard, "dependency_edit_conflict", "the same edge is added and removed")
    for target, ref, kind in removes:
        pair = (ref, kind)
        if target not in expected_graph or pair not in expected_graph[target]:
            add_issue(hard, "dependency_remove_missing", "dependency_remove edge does not exist", edge=[target, ref, kind])
        else:
            expected_graph[target].remove(pair)
    for target, ref, kind in adds:
        pair = (ref, kind)
        if target not in expected_graph or ref not in ref_owner:
            add_issue(hard, "dependency_add_unknown", "dependency_add uses an unknown endpoint/reference", edge=[target, ref, kind])
        elif pair in expected_graph[target] or any(old_ref == ref for old_ref, _ in expected_graph[target]):
            add_issue(hard, "dependency_add_duplicate", "dependency_add duplicates an existing reference", edge=[target, ref, kind])
        else:
            expected_graph[target].append(pair)
    expected_audited_edges: list[dict[str, Any]] = []
    for target in step_ids:
        for ref, kind in expected_graph.get(target, []):
            source = ref_owner.get(ref)
            gap = None if source is None else step_position[target] - step_position.get(source, 10**9) - 1
            expected_audited_edges.append(
                {
                    "source_ref": ref,
                    "source_step": source,
                    "target_step": target,
                    "dependency_type": kind,
                    "sequence_gap": gap,
                    "is_skip": isinstance(gap, int) and gap > 0,
                }
            )
    if audited_edges != expected_audited_edges:
        add_issue(hard, "audited_dependency_graph_mismatch", "audited_dependency_edges do not equal Teacher A dependencies after add/remove edits", expected_count=len(expected_audited_edges), observed_count=len(audited_edges))

    causal_edges = teacher_b.get("causal_edges")
    if not isinstance(causal_edges, list):
        add_issue(hard, "causal_edges_missing", "Teacher B causal_edges must be a list")
        causal_edges = []
    causes_from_edges: dict[str, list[str]] = {step_id: [] for step_id in step_ids}
    for edge in causal_edges:
        if not isinstance(edge, dict):
            add_issue(hard, "causal_edge_shape", "causal edge is not an object")
            continue
        source, target = edge.get("source"), edge.get("target")
        if source not in step_position or target not in step_position:
            add_issue(hard, "causal_edge_endpoint", "causal edge endpoint is unknown", source=source, target=target)
            continue
        if step_position[source] >= step_position[target]:
            add_issue(hard, "causal_edge_not_prior", "causal source must precede target", source=source, target=target)
        causes_from_edges[target].append(source)
        gap = step_position[target] - step_position[source] - 1
        if edge.get("sequence_gap") != gap or edge.get("is_skip") != (gap > 0):
            add_issue(hard, "causal_edge_gap", "causal gap/is_skip is inconsistent", source=source, target=target)
        source_role = judgment_by_step.get(source, {}).get("role")
        if ROLE_TO_LABEL.get(source_role) != "I":
            add_issue(hard, "causal_source_not_error", "causal source must be an erroneous step", source=source, target=target)
        target_role = judgment_by_step.get(target, {}).get("role")
        expected_edge_relation = {
            "new_error": "mixed",
            "propagated_error": "propagation",
            "cessation": "cessation",
        }.get(target_role)
        if edge.get("relation") != expected_edge_relation:
            add_issue(hard, "causal_edge_relation", "causal edge relation conflicts with target role", source=source, target=target, expected=expected_edge_relation, observed=edge.get("relation"))
        via = edge.get("via")
        if not isinstance(via, list) or not all(isinstance(ref, str) for ref in via):
            add_issue(hard, "causal_edge_via", "causal via must be a list of references", source=source, target=target)
            via = []
        matching_refs = {
            dep.get("source_ref")
            for dep in audited_edges
            if isinstance(dep, dict)
            and dep.get("source_step") == source
            and dep.get("target_step") == target
        }
        if target_role in {"new_error", "propagated_error"} and not via:
            add_issue(hard, "causal_edge_missing_via", "propagation/mixed edge must identify a consumed upstream reference", source=source, target=target)
        if any(ref not in matching_refs for ref in via):
            add_issue(hard, "causal_edge_via_dependency", "causal via is not an audited dependency between the same steps", source=source, target=target, via=via)
    for step_id in step_ids:
        caused_by = judgment_by_step.get(step_id, {}).get("caused_by", [])
        if caused_by != causes_from_edges.get(step_id, []):
            add_issue(hard, "caused_by_edges_mismatch", "judgment caused_by differs from incoming causal edges", step=step_id, caused_by=caused_by, edge_sources=causes_from_edges.get(step_id, []))

    incoming_dep = {step_id: [] for step_id in step_ids}
    incoming_causal = {step_id: [] for step_id in step_ids}
    outgoing_causal = {step_id: [] for step_id in step_ids}
    for edge in audited_edges:
        if isinstance(edge, dict) and edge.get("target_step") in incoming_dep:
            incoming_dep[edge["target_step"]].append(edge)
    for edge in causal_edges:
        if not isinstance(edge, dict):
            continue
        if edge.get("target") in incoming_causal:
            incoming_causal[edge["target"]].append(edge)
        if edge.get("source") in outgoing_causal:
            outgoing_causal[edge["source"]].append(edge)
    for step in materialized:
        judgment = judgment_by_step.get(step["id"], {})
        for key in (
            "prm_label", "role", "error_type", "caused_by", "note",
            "mixed_error", "relation_to_first_error",
        ):
            step[key] = judgment.get(key)
        step["audited_dependency_incoming"] = incoming_dep.get(step["id"], [])
        step["causal_incoming"] = incoming_causal.get(step["id"], [])
        step["causal_outgoing"] = outgoing_causal.get(step["id"], [])

    if hard:
        return None, hard, soft

    raw_reference_answer = None
    if isinstance(raw, dict):
        for candidate in (
            raw.get("answer"),
            get_nested(raw, "sat_preprocessing.answer"),
            get_nested(raw, "raw_str.answer"),
        ):
            if is_meaningful(candidate):
                raw_reference_answer = candidate
                break
    result = {
        "schema": OUTPUT_SCHEMA,
        "id": record_id,
        "context": sat.get("context", {}),
        "reference_answer": raw_reference_answer,
        "proposed_answer": extract_proposed_answer(raw, args.ignore_preprocessed_fields) if raw else None,
        "quality_flags": quality_flags,
        "audit": {
            "status": "review" if soft else "core",
            "raw_reasoning_check": raw_check,
            "soft_flags": sorted({issue["code"] for issue in soft}),
            "soft_issues": soft,
            "checks": {
                "sat_span_partition_exact": True,
                "teacher_a_span_coverage_exact": True,
                "step_text_reconstruction_exact": True,
                "teacher_b_step_alignment_exact": True,
                "teacher_b_graph_consistent": True,
                "raw_reasoning_exact": raw_check.get("status") == "matched",
                "source_text_clean": True,
                "sat_complexity_eligible": True,
                "teacher_a_step_count_eligible": True,
            },
            "metrics": {
                "reasoning_chars": len(reasoning_text),
                "candidate_span_count": len(spans),
                "step_count": len(materialized),
                "dependency_edge_count": len(audited_edges),
                "skip_dependency_edge_count": sum(bool(edge.get("is_skip")) for edge in audited_edges if isinstance(edge, dict)),
                "causal_edge_count": len(causal_edges),
                "skip_causal_edge_count": sum(bool(edge.get("is_skip")) for edge in causal_edges if isinstance(edge, dict)),
            },
        },
        "reasoning_text": reasoning_text,
        "problem_entities": problem_entities,
        "steps": materialized,
        "first_error": teacher_b.get("first_error"),
        "dependency_add": teacher_b.get("dependency_add", []),
        "dependency_remove": teacher_b.get("dependency_remove", []),
        "audited_dependency_edges": audited_edges,
        "causal_edges": causal_edges,
        "answers": teacher_b.get("answers", []),
        "overall": teacher_b.get("overall"),
        "source_lineage": {
            "script_version": SCRIPT_VERSION,
            "created_at": utc_now(),
            "sat_line_number": sat_entry.line_number,
            "sat_record_sha256": sat_entry.record_sha256,
            "teacher_a_line_number": teacher_a_entry.line_number,
            "teacher_a_record_sha256": teacher_a_entry.record_sha256,
            "teacher_b_line_number": teacher_b_line,
            "teacher_b_record_sha256": teacher_b_hash,
            "raw_line_number": raw_entry.line_number if raw_entry else None,
            "raw_record_sha256": raw_entry.record_sha256 if raw_entry else None,
        },
    }
    return result, hard, soft


def reject_record(
    record_id: Any,
    stage: str,
    issues: list[dict[str, Any]],
    teacher_b_line: int,
    soft_issues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema": REJECT_SCHEMA,
        "id": record_id,
        "stage": stage,
        "issues": issues,
        "soft_issues_observed": soft_issues or [],
        "teacher_b_line_number": teacher_b_line,
        "created_at": utc_now(),
    }


def rejection_stage(issues: list[dict[str, Any]]) -> str:
    """Return the earliest pipeline gate represented by the failed checks."""
    codes = {str(issue.get("code")) for issue in issues}
    if codes & {
        "source_reasoning_empty",
        "source_abnormal_control_characters",
        "raw_solution_missing",
        "raw_reasoning_extraction",
    }:
        return "source_text_filter"
    if codes & {
        "sat_reasoning_too_short",
        "sat_too_few_candidate_spans",
    }:
        return "sat_complexity_filter"
    if "teacher_a_too_few_steps" in codes:
        return "teacher_a_merge_filter"
    return "audit"


def main() -> int:
    args = parse_args()
    resolve_output_paths(args)
    for path in (args.sat_input, args.teacher_a_input, args.teacher_b_input):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.raw_input is not None and not args.raw_input.is_file():
        raise FileNotFoundError(args.raw_input)
    if args.require_raw_match and args.raw_input is None:
        raise ValueError("--require-raw-match requires --raw-input")
    if not 0 <= args.near_duplicate_threshold <= 1:
        raise ValueError("--near-duplicate-threshold must be between 0 and 1")
    if min(
        args.min_reasoning_chars,
        args.min_candidate_spans,
        args.min_steps,
        args.max_step_chars,
    ) < 0:
        raise ValueError("step/span/character thresholds cannot be negative")
    prepare_outputs(args)

    sat_index, sat_file_hash = build_offset_index(args.sat_input, "SaT")
    teacher_a_index, teacher_a_file_hash = build_offset_index(
        args.teacher_a_input, "Teacher A"
    )
    if args.raw_input:
        raw_index, raw_file_hash = build_offset_index(args.raw_input, "raw")
    else:
        raw_index, raw_file_hash = {}, None

    counters: Counter[str] = Counter()
    hard_codes: Counter[str] = Counter()
    soft_codes: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    seen_b: set[str] = set()
    teacher_b_file_digest = hashlib.sha256()
    started = datetime.now(timezone.utc)

    with (
        args.sat_input.open("rb") as sat_stream,
        args.teacher_a_input.open("rb") as a_stream,
        args.teacher_b_input.open("rb") as b_stream,
        args.output.open("w", encoding="utf-8") as core_out,
        args.review_output.open("w", encoding="utf-8") as review_out,
        args.reject_output.open("w", encoding="utf-8") as reject_out,
    ):
        raw_stream = args.raw_input.open("rb") if args.raw_input else None
        try:
            for b_line_number, raw_line in enumerate(b_stream, 1):
                teacher_b_file_digest.update(raw_line)
                if not raw_line.strip():
                    continue
                counters["teacher_b_records"] += 1
                try:
                    teacher_b = decode_record(raw_line, args.teacher_b_input, b_line_number)
                except Exception as exc:
                    issues: list[dict[str, Any]] = []
                    add_issue(issues, "teacher_b_json", "cannot parse Teacher B record", error=str(exc))
                    reject_out.write(json_line(reject_record(None, "teacher_b_filter", issues, b_line_number)))
                    counters["rejected"] += 1
                    hard_codes["teacher_b_json"] += 1
                    continue
                record_id = teacher_b.get("id")
                if not isinstance(record_id, str) or not record_id:
                    issues = []
                    add_issue(issues, "teacher_b_id", "Teacher B record lacks string id")
                    reject_out.write(json_line(reject_record(record_id, "teacher_b_filter", issues, b_line_number)))
                    counters["rejected"] += 1
                    hard_codes["teacher_b_id"] += 1
                    continue
                if record_id in seen_b:
                    issues = []
                    add_issue(issues, "teacher_b_duplicate_id", "duplicate Teacher B id")
                    reject_out.write(json_line(reject_record(record_id, "teacher_b_filter", issues, b_line_number)))
                    counters["rejected"] += 1
                    hard_codes["teacher_b_duplicate_id"] += 1
                    continue
                seen_b.add(record_id)

                prefilter = teacher_b_prefilter(teacher_b)
                if prefilter:
                    reject_out.write(json_line(reject_record(record_id, "teacher_b_filter", prefilter, b_line_number)))
                    counters["rejected"] += 1
                    for issue in prefilter:
                        hard_codes[issue["code"]] += 1
                    continue

                sat_entry = sat_index.get(record_id)
                a_entry = teacher_a_index.get(record_id)
                raw_entry = raw_index.get(record_id) if raw_index else None
                join_issues: list[dict[str, Any]] = []
                if sat_entry is None:
                    add_issue(join_issues, "sat_join_missing", "id is missing from SaT input")
                if a_entry is None:
                    add_issue(join_issues, "teacher_a_join_missing", "id is missing from Teacher A input")
                if args.require_raw_match and raw_entry is None:
                    add_issue(join_issues, "raw_join_missing", "id is missing from raw input")
                if join_issues:
                    reject_out.write(json_line(reject_record(record_id, "join", join_issues, b_line_number)))
                    counters["rejected"] += 1
                    for issue in join_issues:
                        hard_codes[issue["code"]] += 1
                    continue

                assert sat_entry is not None and a_entry is not None
                sat = read_at(sat_stream, sat_entry, args.sat_input)
                teacher_a = read_at(a_stream, a_entry, args.teacher_a_input)
                raw = read_at(raw_stream, raw_entry, args.raw_input) if raw_stream and raw_entry and args.raw_input else None
                result, hard, soft = materialize_and_audit(
                    record_id=record_id,
                    sat=sat,
                    sat_entry=sat_entry,
                    teacher_a=teacher_a,
                    teacher_a_entry=a_entry,
                    teacher_b=teacher_b,
                    teacher_b_line=b_line_number,
                    teacher_b_hash=record_sha256(teacher_b),
                    raw=raw,
                    raw_entry=raw_entry,
                    args=args,
                )
                if hard:
                    reject_out.write(
                        json_line(
                            reject_record(
                                record_id,
                                rejection_stage(hard),
                                hard,
                                b_line_number,
                                soft,
                            )
                        )
                    )
                    counters["rejected"] += 1
                    for issue in hard:
                        hard_codes[issue["code"]] += 1
                    for issue in soft:
                        soft_codes[issue["code"]] += 1
                else:
                    assert result is not None
                    for step in result["steps"]:
                        role_counts[str(step["role"])] += 1
                        label_counts[str(step["prm_label"])] += 1
                    if soft:
                        review_out.write(json_line(result))
                        counters["review"] += 1
                        for issue in soft:
                            soft_codes[issue["code"]] += 1
                    else:
                        core_out.write(json_line(result))
                        counters["core"] += 1

                processed = counters["teacher_b_records"]
                if args.log_every > 0 and processed % args.log_every == 0:
                    core_out.flush()
                    review_out.flush()
                    reject_out.flush()
                    log(
                        f"processed {processed}: core={counters['core']}, "
                        f"review={counters['review']}, rejected={counters['rejected']}"
                    )
        finally:
            if raw_stream is not None:
                raw_stream.close()

    finished = datetime.now(timezone.utc)
    stats = {
        "schema": "prm_materialization_stats_v1",
        "script_version": SCRIPT_VERSION,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": round((finished - started).total_seconds(), 3),
        "inputs": {
            "sat": str(args.sat_input),
            "teacher_a": str(args.teacher_a_input),
            "teacher_b": str(args.teacher_b_input),
            "raw": str(args.raw_input) if args.raw_input else None,
            "sat_sha256": sat_file_hash,
            "teacher_a_sha256": teacher_a_file_hash,
            "teacher_b_sha256": teacher_b_file_digest.hexdigest(),
            "raw_sha256": raw_file_hash,
        },
        "outputs": {
            "core": str(args.output),
            "review": str(args.review_output),
            "rejected": str(args.reject_output),
        },
        "settings": {
            "require_raw_match": args.require_raw_match,
            "require_think": not args.no_think_required,
            "keep_final_summary": args.keep_final_summary,
            "ignore_preprocessed_fields": args.ignore_preprocessed_fields,
            "max_step_chars": args.max_step_chars,
            "min_reasoning_chars": args.min_reasoning_chars,
            "min_candidate_spans": args.min_candidate_spans,
            "min_steps": args.min_steps,
            "near_duplicate_threshold": args.near_duplicate_threshold,
        },
        "counts": dict(sorted(counters.items())),
        "hard_issue_counts": dict(sorted(hard_codes.items())),
        "soft_issue_counts": dict(sorted(soft_codes.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "prm_label_counts": dict(sorted(label_counts.items())),
        "index_counts": {
            "sat": len(sat_index),
            "teacher_a": len(teacher_a_index),
            "raw": len(raw_index) if args.raw_input else None,
        },
    }
    args.stats_output.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(
        f"done: core={counters['core']}, review={counters['review']}, "
        f"rejected={counters['rejected']}; stats={args.stats_output}"
    )
    # Rejected records are a normal result of a filtering stage, not a process
    # failure.  Fatal I/O/schema errors still raise and produce a non-zero exit.
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted")
        raise SystemExit(130)
    except Exception as exc:
        log(f"fatal: {type(exc).__name__}: {exc}")
        raise
