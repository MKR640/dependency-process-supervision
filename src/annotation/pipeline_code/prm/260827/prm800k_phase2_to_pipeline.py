#!/usr/bin/env python3
"""Convert PRM800K Phase-2 JSONL into the standard SAT pipeline input.

The converter writes two aligned JSONL files:

1. ``--output`` contains only fields that may be consumed by the production
   pipeline: id, subject, type, problem, options, solution, answer and source
   lineage. Human ratings and original step boundaries are deliberately absent.
2. ``--gold-output`` is an evaluation-only sidecar containing the original
   PRM800K step spans, ratings, first-error location and evaluability masks.

For ``finish_reason=found_error``, Phase 2 normally sets
``chosen_completion=null`` at the terminal error. Therefore the source
completion is recovered by matching each ``pre_generated_steps[i]`` string
against the completions at label step ``i``; blindly following
``chosen_completion`` would omit the first error.

The default ``full`` trace mode keeps the complete pre-generated solution, so
the first error retains its position in the original trajectory. Steps after
the first error have gold_rating=null and evaluable=false. ``labeled-prefix``
is available for an oracle-prefix ablation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO


SCHEMA_VERSION = "prm800k_phase2_pipeline_adapter_v1.2"
GOLD_SCHEMA_VERSION = "prm800k_phase2_gold_v1"
STATS_SCHEMA_VERSION = "prm800k_phase2_adapter_stats_v1"
ALLOWED_RATINGS = {-1, 0, 1}
DEFAULT_FINISH_REASONS = {"solution", "found_error"}


class RecordError(ValueError):
    """A source row cannot be converted without an unsafe assumption."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PRM800K Phase-2 JSONL to SAT pipeline input plus gold sidecar."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gold-output", type=Path, default=None)
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument("--reject-output", type=Path, default=None)
    parser.add_argument(
        "--trace-mode",
        choices=("full", "labeled-prefix"),
        default="full",
        help=(
            "full keeps every pre_generated_step; labeled-prefix stops after the "
            "last human-labeled source step (default: full)"
        ),
    )
    parser.add_argument(
        "--split-name",
        default=None,
        help="Stable split name used in IDs (default: input stem).",
    )
    parser.add_argument("--subject", default="数学")
    parser.add_argument("--type", dest="problem_type", default="解答题")
    parser.add_argument(
        "--joiner",
        default="\n",
        help="Exact separator inserted between pre-generated steps (default: newline).",
    )
    parser.add_argument("--include-quality-control", action="store_true")
    parser.add_argument("--include-initial-screening", action="store_true")
    parser.add_argument(
        "--include-finish-reason",
        action="append",
        default=None,
        help=(
            "Allowed finish reason; repeat to provide multiple values. The default "
            "is solution + found_error."
        ),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop at the first invalid source record instead of writing it to rejects.",
    )
    parser.add_argument(
        "--fail-on-reject",
        action="store_true",
        help="Return exit status 2 when one or more rows were rejected.",
    )
    parser.add_argument(
        "--max-reject-warnings",
        type=int,
        default=20,
        help=(
            "Print at most this many per-record reject warnings; all rejects are "
            "still written and counted (default: 20, use 0 to suppress)."
        ),
    )
    args = parser.parse_args()

    if args.input.suffix.lower() != ".jsonl":
        parser.error("--input must end in .jsonl")
    for name in ("output", "gold_output", "reject_output"):
        value = getattr(args, name)
        if value is not None and value.suffix.lower() != ".jsonl":
            parser.error(f"--{name.replace('_', '-')} must end in .jsonl")
    if args.offset < 0:
        parser.error("--offset must be >= 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.max_reject_warnings < 0:
        parser.error("--max-reject-warnings must be >= 0")
    if args.joiner == "":
        parser.error("--joiner must not be empty")

    if args.gold_output is None:
        args.gold_output = args.output.with_name(args.output.stem + ".gold.jsonl")
    if args.stats_output is None:
        args.stats_output = args.output.with_name(args.output.stem + ".stats.json")
    if args.reject_output is None:
        args.reject_output = args.output.with_name(args.output.stem + ".rejects.jsonl")
    if args.split_name is None:
        args.split_name = re.sub(r"\.jsonl$", "", args.input.name, flags=re.IGNORECASE)
    args.split_name = sanitize_component(args.split_name)
    args.allowed_finish_reasons = set(
        args.include_finish_reason or sorted(DEFAULT_FINISH_REASONS)
    )
    return args


def sanitize_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    value = value.strip("_")
    return value or "split"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_object(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def normalize_line_endings(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def normalized_match_text(text: str) -> str:
    return " ".join(normalize_line_endings(text).split())


def require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecordError(f"{field} must be an object")
    return value


def require_string(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise RecordError(f"{field} must be a string")
    if not allow_empty and not value.strip():
        raise RecordError(f"{field} must not be empty")
    return normalize_line_endings(value)


def require_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RecordError(f"{field} must be a non-empty list")
    return [require_string(item, f"{field}[{index}]") for index, item in enumerate(value)]


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any], str]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise RecordError(
                    f"invalid JSON at line {line_number}: {error.msg} "
                    f"(column {error.colno})"
                ) from error
            if not isinstance(value, dict):
                raise RecordError(f"line {line_number} must contain a JSON object")
            yield line_number, value, raw_line


class AtomicTextWriter:
    def __init__(self, destination: Path):
        self.destination = destination
        self.temp_path: Path | None = None
        self.stream: TextIO | None = None

    def __enter__(self) -> TextIO:
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(
            prefix=self.destination.name + ".",
            suffix=".tmp",
            dir=self.destination.parent,
            text=True,
        )
        self.temp_path = Path(temp_name)
        self.stream = os.fdopen(handle, "w", encoding="utf-8")
        return self.stream

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        assert self.stream is not None and self.temp_path is not None
        try:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
            if exc_type is None:
                os.replace(self.temp_path, self.destination)
            else:
                self.temp_path.unlink(missing_ok=True)
        except Exception:
            self.temp_path.unlink(missing_ok=True)
            raise


def write_jsonl(stream: TextIO, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def completion_candidates(label_step: dict[str, Any]) -> list[dict[str, Any]]:
    completions = label_step.get("completions")
    if not isinstance(completions, list) or not completions:
        raise RecordError("label step completions must be a non-empty list")
    for index, completion in enumerate(completions):
        if not isinstance(completion, dict):
            raise RecordError(f"completion {index} must be an object")
        require_string(completion.get("text"), f"completion[{index}].text")
    return completions


def resolve_matching_completion(
    matches: list[int],
    completions: list[dict[str, Any]],
    chosen: Any,
    method: str,
) -> tuple[int, str]:
    """Resolve duplicate textual matches without inventing a gold label.

    A valid chosen_completion is authoritative. At terminal Phase-2 errors it
    is normally null; duplicated candidates are still safe when all matching
    copies agree on rating and flagged/unflagged evaluability. Only a genuine
    label conflict remains ambiguous.
    """
    if len(matches) == 1:
        return matches[0], method
    if isinstance(chosen, int) and chosen in matches:
        return chosen, f"{method}_duplicate_chosen"

    signatures = {
        (completions[index].get("rating"), completions[index].get("flagged") is True)
        for index in matches
    }
    if len(signatures) == 1:
        return min(matches), f"{method}_duplicate_consensus"

    details = [
        {
            "index": index,
            "rating": completions[index].get("rating"),
            "flagged": completions[index].get("flagged"),
        }
        for index in matches
    ]
    raise RecordError(
        "multiple textual matches have conflicting labels: "
        + json.dumps(details, ensure_ascii=False, separators=(",", ":"))
    )


def match_source_step(
    source_text: str, label_step: dict[str, Any]
) -> dict[str, Any]:
    """Match one pre-generated source step to its labeled completion."""
    completions = completion_candidates(label_step)
    source_nfc = normalize_line_endings(source_text)
    chosen = label_step.get("chosen_completion")
    if chosen is not None and (
        not isinstance(chosen, int) or chosen < 0 or chosen >= len(completions)
    ):
        raise RecordError(f"chosen_completion is out of range: {chosen!r}")

    exact_matches = [
        index
        for index, completion in enumerate(completions)
        if normalize_line_endings(completion["text"]) == source_nfc
    ]
    if exact_matches:
        index, method = resolve_matching_completion(
            exact_matches, completions, chosen, "exact"
        )
    else:
        normalized_source = normalized_match_text(source_nfc)
        normalized_matches = [
            index
            for index, completion in enumerate(completions)
            if normalized_match_text(completion["text"]) == normalized_source
        ]
        if normalized_matches:
            index, method = resolve_matching_completion(
                normalized_matches,
                completions,
                chosen,
                "normalized_whitespace",
            )
        else:
            human_completion = label_step.get("human_completion")
            if isinstance(human_completion, str) and normalized_match_text(
                human_completion
            ) == normalized_source:
                return {
                    "source": "human_completion",
                    "completion_index": None,
                    "chosen_completion": label_step.get("chosen_completion"),
                    "rating": 1,
                    "flagged": False,
                    "match_method": "human_completion_normalized",
                }
            raise RecordError("no completion matches the corresponding pre_generated_step")

    completion = completions[index]
    rating = completion.get("rating")
    if rating is not None and rating not in ALLOWED_RATINGS:
        raise RecordError(
            f"matched completion[{index}].rating must be -1, 0, 1 or null; "
            f"got {rating!r}"
        )
    return {
        "source": "completion",
        "completion_index": index,
        "chosen_completion": chosen,
        "rating": rating,
        "flagged": completion.get("flagged"),
        "match_method": method,
    }


def make_identifier(
    record: dict[str, Any], split_name: str, trace_mode: str, problem: str, steps: list[str]
) -> str:
    identity = {
        "labeler": record.get("labeler"),
        "timestamp": record.get("timestamp"),
        "generation": record.get("generation"),
        "problem": problem,
        "pre_generated_steps": steps,
    }
    digest = sha256_object(identity)[:20]
    mode = "full" if trace_mode == "full" else "prefix"
    return f"prm800k_{split_name}_{mode}_{digest}"


def build_spans(
    steps: list[str], joiner: str, matches: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    reasoning_text = joiner.join(steps)
    spans: list[dict[str, Any]] = []
    cursor = 0
    for index, text in enumerate(steps):
        start = cursor
        end = start + len(text)
        match = matches[index] if index < len(matches) else None
        rating = match["rating"] if match is not None else None
        flagged = match["flagged"] if match is not None else None
        evaluable = (
            match is not None and rating in ALLOWED_RATINGS and flagged is not True
        )
        if rating == -1:
            gold_class = "error"
        elif rating == 0:
            gold_class = "non_error_neutral"
        elif rating == 1:
            gold_class = "non_error_progress"
        elif rating is None and flagged is True:
            gold_class = "unverifiable_flagged"
        else:
            gold_class = "unlabeled"
        spans.append(
            {
                "id": f"g{index + 1:04d}",
                "index": index + 1,
                "text": text,
                "text_sha256": sha256_text(text),
                "start_char": start,
                "end_char": end,
                "gold_rating": rating,
                "gold_class": gold_class,
                "flagged": flagged,
                "evaluable": evaluable,
                "match": match,
            }
        )
        cursor = end + (len(joiner) if index + 1 < len(steps) else 0)
    if cursor != len(reasoning_text):
        raise AssertionError("internal span accounting mismatch")
    return reasoning_text, spans


def convert_record(
    record: dict[str, Any], line_number: int, raw_line: str, args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    question = require_dict(record.get("question"), "question")
    label = require_dict(record.get("label"), "label")
    problem = require_string(question.get("problem"), "question.problem")
    ground_truth_answer = require_string(
        question.get("ground_truth_answer"), "question.ground_truth_answer"
    )
    pre_generated_answer = require_string(
        question.get("pre_generated_answer"),
        "question.pre_generated_answer",
        allow_empty=True,
    )
    source_steps = require_string_list(
        question.get("pre_generated_steps"), "question.pre_generated_steps"
    )
    label_steps = label.get("steps")
    if not isinstance(label_steps, list) or not label_steps:
        raise RecordError("label.steps must be a non-empty list")
    if len(label_steps) > len(source_steps):
        raise RecordError(
            f"label step count {len(label_steps)} exceeds pre-generated step count "
            f"{len(source_steps)}"
        )

    matches: list[dict[str, Any]] = []
    for index, label_step_value in enumerate(label_steps):
        label_step = require_dict(label_step_value, f"label.steps[{index}]")
        try:
            match = match_source_step(source_steps[index], label_step)
        except RecordError as error:
            raise RecordError(f"label.steps[{index}]: {error}") from error
        matches.append(match)

    finish_reason = label.get("finish_reason")
    if not isinstance(finish_reason, str):
        raise RecordError("label.finish_reason must be a string")

    issues: list[str] = []
    trajectory_tags: list[str] = []
    for match in matches:
        if match["rating"] is None:
            issues.append(
                "matched_source_flagged_without_rating"
                if match["flagged"] is True
                else "matched_source_missing_rating"
            )

    negative_indices = [index for index, match in enumerate(matches) if match["rating"] == -1]
    first_error_zero = negative_indices[0] if negative_indices else None
    if finish_reason == "found_error":
        if first_error_zero is None:
            issues.append("found_error_without_matched_negative_rating")
        elif first_error_zero != len(matches) - 1:
            issues.append("first_error_is_not_last_labeled_step")
    if finish_reason == "solution" and first_error_zero is not None:
        # This may be a genuine error-then-recovery trajectory. It is not a
        # structural conversion failure: PRM800K says the trajectory ultimately
        # reached a correct answer, while the local -1 still supplies first-error
        # supervision.
        trajectory_tags.append("solution_with_labeled_error")

    if args.trace_mode == "full":
        emitted_steps = list(source_steps)
    else:
        emitted_steps = list(source_steps[: len(matches)])
    # sat_candidate_span_splitter.extract_reasoning() strips only the outer
    # whitespace of the complete <think> block. Mirror that deterministic view
    # here so sidecar offsets and hashes remain exact even for an unusual source
    # row whose first/last step has surrounding whitespace.
    emitted_steps[0] = emitted_steps[0].lstrip()
    emitted_steps[-1] = emitted_steps[-1].rstrip()
    if not emitted_steps[0] or not emitted_steps[-1]:
        raise RecordError("outer whitespace normalization produced an empty step")
    reasoning_text, gold_steps = build_spans(emitted_steps, args.joiner, matches)

    identifier = make_identifier(
        record, args.split_name, args.trace_mode, problem, source_steps
    )
    source_record_sha256 = hashlib.sha256(raw_line.rstrip("\n").encode("utf-8")).hexdigest()
    if first_error_zero is not None and first_error_zero < len(gold_steps):
        first_step = gold_steps[first_error_zero]
        first_error = {
            "gold_step_id": first_step["id"],
            "gold_step_index": first_step["index"],
            "start_char": first_step["start_char"],
            "end_char": first_step["end_char"],
            "normalized_start": (
                first_step["start_char"] / len(reasoning_text) if reasoning_text else None
            ),
            "rating": -1,
        }
    else:
        first_error = None

    # Overall outcome and strict process purity are intentionally separate.
    # A solution can contain a local error and later recover (cessation), so the
    # final overall target follows finish_reason. strict_process_gold answers the
    # different question: did any labeled source step receive -1?
    if finish_reason == "solution":
        overall_gold = "C"
    elif finish_reason == "found_error" and first_error is not None:
        overall_gold = "I"
    else:
        overall_gold = "U"
    if first_error is not None:
        strict_process_gold = "I"
    elif finish_reason == "solution":
        strict_process_gold = "C"
    else:
        strict_process_gold = "U"

    lineage = {
        "dataset": "PRM800K",
        "phase": 2,
        "split": args.split_name,
        "source_line_number": line_number,
        "source_record_sha256": source_record_sha256,
        "adapter_schema": SCHEMA_VERSION,
        "trace_mode": args.trace_mode,
    }
    pipeline_record = {
        "id": identifier,
        "subject": args.subject,
        "type": args.problem_type,
        "problem": problem,
        "options": [],
        "solution": f"<think>{reasoning_text}</think>\n<answer>{pre_generated_answer}</answer>",
        "answer": [ground_truth_answer],
        "source_lineage": lineage,
    }
    gold_record = {
        "schema": GOLD_SCHEMA_VERSION,
        "id": identifier,
        "source_lineage": lineage,
        "prm800k_metadata": {
            "labeler": record.get("labeler"),
            "timestamp": record.get("timestamp"),
            "generation": record.get("generation"),
            "is_quality_control_question": record.get("is_quality_control_question"),
            "is_initial_screening_question": record.get(
                "is_initial_screening_question"
            ),
            "finish_reason": finish_reason,
        },
        "trace_mode": args.trace_mode,
        "reasoning_text_sha256": sha256_text(reasoning_text),
        "reasoning_chars": len(reasoning_text),
        "pre_generated_step_count": len(source_steps),
        "emitted_step_count": len(emitted_steps),
        "labeled_source_step_count": len(matches),
        "evaluable_step_count": sum(step["evaluable"] for step in gold_steps),
        "unlabeled_after_stop_count": sum(
            step["gold_rating"] is None for step in gold_steps
        ),
        "gold_steps": gold_steps,
        "first_error": first_error,
        "overall_gold": overall_gold,
        "strict_process_gold": strict_process_gold,
        "trajectory_tags": trajectory_tags,
        "evaluation_issues": issues,
    }
    summary = {
        "finish_reason": finish_reason,
        "overall_gold": overall_gold,
        "strict_process_gold": strict_process_gold,
        "source_steps": len(source_steps),
        "emitted_steps": len(emitted_steps),
        "labeled_steps": len(matches),
        "evaluable_steps": sum(step["evaluable"] for step in gold_steps),
        "unlabeled_steps": sum(step["gold_rating"] is None for step in gold_steps),
        "ratings": Counter(match["rating"] for match in matches),
        "match_methods": Counter(match["match_method"] for match in matches),
        "issues": issues,
        "trajectory_tags": trajectory_tags,
        "first_error_index": first_error["gold_step_index"] if first_error else None,
    }
    return pipeline_record, gold_record, summary


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"error: input does not exist: {args.input}", file=sys.stderr)
        return 1

    counters: Counter[str] = Counter()
    finish_counts: Counter[str] = Counter()
    overall_counts: Counter[str] = Counter()
    strict_process_counts: Counter[str] = Counter()
    rating_counts: Counter[int] = Counter()
    match_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    trajectory_tag_counts: Counter[str] = Counter()
    identifiers: set[str] = set()
    first_error_indices: list[int] = []
    started_at = datetime.now(timezone.utc)

    try:
        with (
            AtomicTextWriter(args.output) as output_stream,
            AtomicTextWriter(args.gold_output) as gold_stream,
            AtomicTextWriter(args.reject_output) as reject_stream,
        ):
            selected_index = 0
            for line_number, record, raw_line in iter_jsonl(args.input):
                counters["input_records"] += 1
                if record.get("is_quality_control_question") is True and not args.include_quality_control:
                    counters["excluded_quality_control"] += 1
                    continue
                if record.get("is_initial_screening_question") is True and not args.include_initial_screening:
                    counters["excluded_initial_screening"] += 1
                    continue
                finish_reason = (
                    record.get("label", {}).get("finish_reason")
                    if isinstance(record.get("label"), dict)
                    else None
                )
                if finish_reason not in args.allowed_finish_reasons:
                    counters["excluded_finish_reason"] += 1
                    continue
                if selected_index < args.offset:
                    selected_index += 1
                    counters["offset_skipped"] += 1
                    continue
                if args.limit is not None and counters["selected_records"] >= args.limit:
                    break
                selected_index += 1
                counters["selected_records"] += 1

                try:
                    pipeline_record, gold_record, summary = convert_record(
                        record, line_number, raw_line, args
                    )
                    identifier = pipeline_record["id"]
                    if identifier in identifiers:
                        raise RecordError(f"duplicate generated id: {identifier}")
                    identifiers.add(identifier)
                    write_jsonl(output_stream, pipeline_record)
                    write_jsonl(gold_stream, gold_record)
                    counters["written_records"] += 1
                    counters["source_steps"] += summary["source_steps"]
                    counters["emitted_steps"] += summary["emitted_steps"]
                    counters["labeled_steps"] += summary["labeled_steps"]
                    counters["evaluable_steps"] += summary["evaluable_steps"]
                    counters["unlabeled_steps"] += summary["unlabeled_steps"]
                    finish_counts[summary["finish_reason"]] += 1
                    overall_counts[summary["overall_gold"]] += 1
                    strict_process_counts[summary["strict_process_gold"]] += 1
                    rating_counts.update(summary["ratings"])
                    match_counts.update(summary["match_methods"])
                    issue_counts.update(summary["issues"])
                    trajectory_tag_counts.update(summary["trajectory_tags"])
                    if summary["first_error_index"] is not None:
                        first_error_indices.append(summary["first_error_index"])
                except Exception as error:
                    if args.strict:
                        raise
                    counters["rejected_records"] += 1
                    reject = {
                        "schema": "prm800k_phase2_adapter_reject_v1",
                        "source_line_number": line_number,
                        "source_record_sha256": hashlib.sha256(
                            raw_line.rstrip("\n").encode("utf-8")
                        ).hexdigest(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    write_jsonl(reject_stream, reject)
                    if counters["rejected_records"] <= args.max_reject_warnings:
                        print(
                            f"warning: rejected source line {line_number}: {error}",
                            file=sys.stderr,
                        )
                    elif (
                        args.max_reject_warnings > 0
                        and counters["rejected_records"]
                        == args.max_reject_warnings + 1
                    ):
                        print(
                            "warning: additional per-record reject warnings suppressed; "
                            "see the reject JSONL and final statistics",
                            file=sys.stderr,
                        )
    except Exception as error:
        print(f"error: conversion failed: {error}", file=sys.stderr)
        return 1

    finished_at = datetime.now(timezone.utc)
    stats = {
        "schema": STATS_SCHEMA_VERSION,
        "script_version": SCHEMA_VERSION,
        "status": "completed_with_rejects" if counters["rejected_records"] else "ok",
        "input": str(args.input),
        "output": str(args.output),
        "gold_output": str(args.gold_output),
        "reject_output": str(args.reject_output),
        "trace_mode": args.trace_mode,
        "split_name": args.split_name,
        "allowed_finish_reasons": sorted(args.allowed_finish_reasons),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "counts": dict(sorted(counters.items())),
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "overall_gold_counts": dict(sorted(overall_counts.items())),
        "strict_process_gold_counts": dict(sorted(strict_process_counts.items())),
        "rating_counts": {
            ("null" if key is None else str(key)): value
            for key, value in sorted(rating_counts.items(), key=lambda item: str(item[0]))
        },
        "match_method_counts": dict(sorted(match_counts.items())),
        "evaluation_issue_counts": dict(sorted(issue_counts.items())),
        "trajectory_tag_counts": dict(sorted(trajectory_tag_counts.items())),
        "first_error_step_index": {
            "count": len(first_error_indices),
            "min": min(first_error_indices) if first_error_indices else None,
            "max": max(first_error_indices) if first_error_indices else None,
            "mean": (
                sum(first_error_indices) / len(first_error_indices)
                if first_error_indices
                else None
            ),
        },
    }
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    with AtomicTextWriter(args.stats_output) as stats_stream:
        json.dump(stats, stats_stream, ensure_ascii=False, indent=2)
        stats_stream.write("\n")

    print(json.dumps(stats, ensure_ascii=False, indent=2), file=sys.stderr)
    if counters["rejected_records"] and args.fail_on_reject:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
