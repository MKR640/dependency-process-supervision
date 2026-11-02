#!/usr/bin/env python3
"""Annotate raw solution JSONL before Segment any Text (SaT) inference.

The script preserves every original field and appends ``source_check``. When a
format defect can be repaired deterministically, it also writes only the
changed fields under ``sat_preprocessing``; the raw problem/solution/options
remain untouched. Data-quality failures set ``source_check.sat_eligible`` to
``false``; research-relevant reasoning defects do not block SaT.

Example:
    python sat_source_quality_filter.py \
      --input raw.jsonl \
      --output raw.sat_checked.jsonl \
      --stats-output raw.sat_checked.stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "sat_source_check_v2"
PREPROCESSING_SCHEMA_VERSION = "sat_preprocessing_v1"

THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>(.*?)</think\s*>", re.IGNORECASE | re.DOTALL
)
ANSWER_OPEN_RE = re.compile(r"<answer\b[^>]*>", re.IGNORECASE)
ANSWER_CLOSE_RE = re.compile(r"</answer\s*>", re.IGNORECASE)
ANSWER_BLOCK_RE = re.compile(
    r"<answer\b[^>]*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL
)

FINAL_ANSWER_IN_THINK_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?final\s+answer(?:\*\*)?\s*:", re.IGNORECASE
)
ANSWER_ASSERTION_RE = re.compile(
    r"\b(?:the\s+)?(?:correct\s+)?(?:answer|option)\s*(?:is|:|-|=)|"
    r"\b(?:correct\s+)?option\s*(?:\([A-I]\)|[A-I])?\s+is\s+correct\b|"
    r"\bcorrect\s+option\b",
    re.IGNORECASE,
)
DERIVATION_CUE_RE = re.compile(
    r"\b(?:because|since|therefore|thus|hence|calculate|derive|substitut|"
    r"solve|let|given|we\s+(?:have|obtain|get|find)|implies|from\s+which)\b",
    re.IGNORECASE,
)
SOCIAL_NOISE_RE = re.compile(
    r"\b(?:pls|plz)\s+(?:upvote|ask)|\bupvote\b|\bhave\s+a\s+nice\s+day\b|"
    r"\bteam\s+chegg\b|\bif\s+(?:u|you)\s+have\s+any\s+doubts?\b",
    re.IGNORECASE,
)

MISSING_PRIOR_DERIVATION_RE = re.compile(
    r"\bbased\s+on\s+(?:the\s+)?(?:provided|given|above)\s+solution\s+steps?\b|"
    r"\bprevious\s+(?:equations?|steps?|calculations?)\s+(?:are|were|have\s+not\s+been)\s+"
    r"(?:not\s+)?provided\b|"
    r"\busing\s+(?:the\s+)?values?\s+(?:from|in)\s+(?:the\s+)?(?:provided|previous)\s+"
    r"solution\b",
    re.IGNORECASE,
)
REFERENCE_RATIONALIZATION_RE = re.compile(
    r"\b(?:follow|use)\s+(?:the\s+)?(?:provided|original)\s+(?:answer|solution)\b|"
    r"\bto\s+match\s+(?:the\s+)?(?:provided|given|stated)\s+(?:answer|option)\b|"
    r"\bbased\s+on\s+(?:the\s+)?provided\s+(?:answer|options?)\b|"
    r"\bdespite\s+not\s+matching\b|"
    r"\b(?:provided|original)\s+(?:answer|explanation|solution)(?:'s)?\s+"
    r"(?:is|seems?|appears?|calculations?)\b|"
    r"\bchegg\s+solution\b|"
    r"\b(?:apparently|implicitly)\s+(?:used|assumed)\s+in\s+(?:the\s+)?"
    r"(?:provided|original)\s+solution\b",
    re.IGNORECASE,
)
UNSUPPORTED_INTERMEDIATE_RE = re.compile(
    r"\baccording\s+to\s+(?:the\s+)?(?:provided|original)\s+solution\b|"
    r"\bfrom\s+(?:the\s+)?provided\s+solution(?:\s+steps?)?\b|"
    r"\bas\s+(?:given|shown|stated)\s+in\s+(?:the\s+)?(?:provided|original)\s+"
    r"(?:answer|solution|explanation)\b",
    re.IGNORECASE,
)
EXTERNAL_DATA_RE = re.compile(
    r"\b(?:the\s+)?(?:retail|data|spreadsheet|csv|excel)\s+file\s+includes?\b|"
    r"\bclick\s+here\s+for\s+(?:the\s+)?(?:data|file)\b|"
    r"\b(?:attached|external|download(?:ed)?)\s+(?:data|dataset|file|spreadsheet)\b|"
    r"\bdata\s+file\b",
    re.IGNORECASE,
)
EXTERNAL_VISUAL_RE = re.compile(
    r"\[image\s*:|\baccompanying\s+images?\b|\battached\s+(?:image|figure|graph)\b|"
    r"\bprovided\s+(?:image|figure|graph|diagram)\b|\bsee\s+(?:the\s+)?"
    r"(?:image|figure|graph|diagram)\b|\bgraph\s+[A-Z]\s+visually\b",
    re.IGNORECASE,
)
PROBABLE_LITERAL_NEWLINE_RE = re.compile(r"\\n(?=\\n|[\sA-Z0-9$*#|>\-])")
ZERO_WIDTH_CHARACTERS = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}
DISPLAY_BLOCK_RE = re.compile(
    r"\$\$[\s\S]*?\$\$|"
    r"\\\[[\s\S]*?\\\]|"
    r"\\begin\{(?P<env>[A-Za-z*]+)\}[\s\S]*?\\end\{(?P=env)\}"
)
INLINE_MATH_RE = re.compile(
    r"(?<![\\$])\$(?!\$)(?:\\.|[^$\\])+?(?<!\\)\$(?!\$)"
)
SENTENCE_BOUNDARY_RE = re.compile(
    r"[.!?](?P<space>[ \t]+)(?=[\"'\u201c\u2018(\[]*[A-Z0-9$*#])"
)
COMMON_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "fig.",
    "eq.",
    "no.",
    "mr.",
    "mrs.",
    "dr.",
    "prof.",
)


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preserve raw JSONL records and annotate their SaT eligibility."
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw input JSONL.")
    parser.add_argument(
        "--output", type=Path, required=True, help="Annotated output JSONL."
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Statistics JSON path (default: <output>.stats.json).",
    )
    parser.add_argument("--id-key", default="id")
    parser.add_argument("--subject-key", default="subject")
    parser.add_argument("--type-key", default="type")
    parser.add_argument("--problem-key", default="problem")
    parser.add_argument("--options-key", default="options")
    parser.add_argument("--solution-key", default="solution")
    parser.add_argument(
        "--check-key",
        default="source_check",
        help="Top-level field used for annotations (default: source_check).",
    )
    parser.add_argument(
        "--replace-existing-check",
        action="store_true",
        help="Replace an existing --check-key field instead of failing.",
    )
    parser.add_argument(
        "--preprocessing-key",
        default="sat_preprocessing",
        help="Top-level field containing repaired SaT views (default: sat_preprocessing).",
    )
    parser.add_argument(
        "--max-source-line-chars",
        type=int,
        default=600,
        help=(
            "Insert safe structural boundaries into longer solution lines "
            "without changing the raw solution (default: 600)."
        ),
    )
    parser.add_argument(
        "--disable-repair",
        action="store_true",
        help="Audit only: do not create repaired fields or recover repairable records.",
    )
    args = parser.parse_args()
    if args.input.suffix.lower() != ".jsonl":
        parser.error("--input must use the .jsonl suffix")
    if args.output.suffix.lower() != ".jsonl":
        parser.error("--output must use the .jsonl suffix")
    if args.max_source_line_chars < 200:
        parser.error("--max-source-line-chars must be at least 200")
    if args.stats_output is None:
        args.stats_output = args.output.with_suffix(args.output.suffix + ".stats.json")
    return args


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {error.msg} "
                    f"(column {error.colno})"
                ) from error
            if not isinstance(value, dict):
                raise TypeError(
                    f"JSONL record at {path}:{line_number} is "
                    f"{type(value).__name__}, expected an object"
                )
            yield line_number, value


def count_ids(path: Path, id_key: str) -> tuple[Counter[Any], int, int]:
    counts: Counter[Any] = Counter()
    records = 0
    with path.open("r", encoding="utf-8") as stream:
        physical_lines = 0
        for physical_lines, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path}:{physical_lines}: {error.msg} "
                    f"(column {error.colno})"
                ) from error
            if not isinstance(value, dict):
                raise TypeError(
                    f"JSONL record at {path}:{physical_lines} is "
                    f"{type(value).__name__}, expected an object"
                )
            records += 1
            identifier = value.get(id_key)
            if isinstance(identifier, (str, int)) and str(identifier).strip():
                counts[identifier] += 1
    return counts, records, physical_lines


def add_unique(target: list[str], code: str) -> None:
    if code not in target:
        target.append(code)


def textual_fields(record: dict[str, Any], args: argparse.Namespace) -> Iterable[tuple[str, str]]:
    for key in (args.problem_key, args.solution_key):
        value = record.get(key)
        if isinstance(value, str):
            yield key, value
    options = record.get(args.options_key)
    if isinstance(options, list):
        for index, value in enumerate(options):
            if isinstance(value, str):
                yield f"{args.options_key}[{index}]", value


def prepared_textual_fields(
    problem: Any,
    solution: Any,
    options: Any,
    args: argparse.Namespace,
) -> Iterable[tuple[str, str]]:
    if isinstance(problem, str):
        yield args.problem_key, problem
    if isinstance(solution, str):
        yield args.solution_key, solution
    if isinstance(options, list):
        for index, value in enumerate(options):
            if isinstance(value, str):
                yield f"{args.options_key}[{index}]", value


def repair_control_characters(text: str) -> tuple[str, Counter[str]]:
    """Repair unambiguous JSON-escape damage and normalize benign controls."""
    repaired: list[str] = []
    counts: Counter[str] = Counter()
    index = 0
    escaped_repairs = {
        "\x0c": (("rac", r"\frac", "restore_latex_frac"),),
        "\r": (("ight", r"\right", "restore_latex_right"),),
        "\x08": (
            ("eta", r"\beta", "restore_latex_beta"),
            ("ar", r"\bar", "restore_latex_bar"),
        ),
        "\t": (
            ("ext", r"\text", "restore_latex_text"),
            ("imes", r"\times", "restore_latex_times"),
            ("op", r"\top", "restore_latex_top"),
            ("heta", r"\theta", "restore_latex_theta"),
        ),
    }
    while index < len(text):
        character = text[index]
        matched = False
        for suffix, replacement, code in escaped_repairs.get(character, ()):
            if text.startswith(suffix, index + 1):
                repaired.append(replacement)
                index += 1 + len(suffix)
                counts[code] += 1
                matched = True
                break
        if matched:
            continue
        if character == "\t":
            # Tabs are often real column delimiters in copied data tables. SaT
            # and its tokenizer only need whitespace here, so retain the sample
            # through a deterministic view-level normalization.
            repaired.append("    ")
            counts["normalize_tab_whitespace"] += 1
            index += 1
            continue
        if character == "\r":
            if index + 1 < len(text) and text[index + 1] == "\n":
                repaired.append("\n")
                index += 2
            else:
                repaired.append("\n")
                index += 1
            counts["normalize_carriage_return"] += 1
            continue
        if character == "\x0c":
            repaired.append("\n")
            counts["normalize_form_feed"] += 1
            index += 1
            continue
        if character in ZERO_WIDTH_CHARACTERS:
            counts["remove_zero_width_character"] += 1
            index += 1
            continue
        repaired.append(character)
        index += 1
    return "".join(repaired), counts


def protected_intervals(text: str) -> list[tuple[int, int]]:
    block_patterns = [
        re.compile(r"```[\s\S]*?```"),
        re.compile(r"\$\$[\s\S]*?\$\$"),
        re.compile(r"\\\[[\s\S]*?\\\]"),
        re.compile(r"\\\([\s\S]*?\\\)"),
        re.compile(r"\\begin\{(?P<env>[A-Za-z*]+)\}[\s\S]*?\\end\{(?P=env)\}"),
    ]
    intervals: list[tuple[int, int]] = []
    for pattern in block_patterns:
        intervals.extend((match.start(), match.end()) for match in pattern.finditer(text))
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    # Do not protect single-dollar regions here. In exercise text, currency
    # markers such as "$100 ... $20" are easily mistaken for one giant inline
    # formula and would hide genuine sentence boundaries. The boundary regex
    # itself is conservative enough not to split decimals or ordinary formulae.
    return merged


def position_is_protected(position: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start < position < end for start, end in intervals)


def add_boundaries_to_long_line(line: str, max_chars: int) -> tuple[str, int]:
    """Add newline hints at safe semantic boundaries in one long source line."""
    if len(line) <= max_chars:
        return line, 0
    intervals = protected_intervals(line)
    positions: set[int] = set()
    display_intervals = [
        (match.start(), match.end()) for match in DISPLAY_BLOCK_RE.finditer(line)
    ]

    for match in DISPLAY_BLOCK_RE.finditer(line):
        if match.start() > 0 and line[match.start() - 1] != "\n":
            positions.add(match.start())
        if match.end() < len(line) and line[match.end()] != "\n":
            positions.add(match.end())

    for match in INLINE_MATH_RE.finditer(line):
        if any(
            match.start() < block_end and block_start < match.end()
            for block_start, block_end in display_intervals
        ):
            continue
        prefix = line[: match.start()].rstrip()
        suffix = line[match.end() :]
        if prefix.endswith((':', ';')):
            positions.add(match.start())
        if re.match(r"[ \t]+(?=[A-Z*#])", suffix):
            positions.add(match.end())

    for match in SENTENCE_BOUNDARY_RE.finditer(line):
        boundary = match.start("space")
        if position_is_protected(boundary, intervals):
            continue
        prefix = line[max(0, match.start() - 8) : match.start() + 1].lower()
        if any(prefix.endswith(abbreviation) for abbreviation in COMMON_ABBREVIATIONS):
            continue
        positions.add(boundary)

    if not positions:
        return line, 0
    pieces: list[str] = []
    cursor = 0
    inserted = 0
    for position in sorted(positions):
        if position <= cursor or position >= len(line):
            continue
        pieces.append(line[cursor:position])
        if not pieces[-1].endswith("\n"):
            pieces.append("\n")
            inserted += 1
        cursor = position
    pieces.append(line[cursor:])
    return "".join(pieces), inserted


def normalize_solution_structure(
    solution: str,
    max_line_chars: int,
) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    literal_count = len(PROBABLE_LITERAL_NEWLINE_RE.findall(solution))
    if literal_count:
        solution = PROBABLE_LITERAL_NEWLINE_RE.sub("\n", solution)
        counts["decode_literal_newline"] += literal_count

    normalized_lines: list[str] = []
    for physical_line in solution.splitlines(keepends=True):
        if physical_line.endswith("\r\n"):
            core, ending = physical_line[:-2], "\r\n"
        elif physical_line.endswith(("\n", "\r")):
            core, ending = physical_line[:-1], physical_line[-1]
        else:
            core, ending = physical_line, ""
        normalized, inserted = add_boundaries_to_long_line(core, max_line_chars)
        normalized_lines.append(normalized + ending)
        counts["insert_safe_boundary"] += inserted
    return "".join(normalized_lines), counts


def preprocess_record(
    record: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Any, Any, Any, dict[str, Any] | None, Counter[str]]:
    """Create repaired views while leaving every raw source field unchanged."""
    raw_problem = record.get(args.problem_key)
    raw_solution = record.get(args.solution_key)
    raw_options = record.get(args.options_key)
    if args.disable_repair:
        return raw_problem, raw_solution, raw_options, None, Counter()

    repair_counts: Counter[str] = Counter()
    problem = raw_problem
    solution = raw_solution
    options = raw_options
    changed_fields: dict[str, Any] = {}

    if isinstance(raw_problem, str):
        problem, counts = repair_control_characters(raw_problem)
        repair_counts.update(counts)
        if problem != raw_problem:
            changed_fields[args.problem_key] = problem

    if isinstance(raw_solution, str):
        solution, counts = repair_control_characters(raw_solution)
        repair_counts.update(counts)
        solution, counts = normalize_solution_structure(
            solution, args.max_source_line_chars
        )
        repair_counts.update(counts)
        if solution != raw_solution:
            changed_fields[args.solution_key] = solution

    if isinstance(raw_options, list):
        normalized_options: list[Any] = []
        for value in raw_options:
            if isinstance(value, str):
                normalized, counts = repair_control_characters(value)
                repair_counts.update(counts)
                normalized_options.append(normalized)
            else:
                normalized_options.append(value)
        options = normalized_options
        if normalized_options != raw_options:
            changed_fields[args.options_key] = normalized_options

    if not changed_fields:
        return problem, solution, options, None, repair_counts
    preprocessing: dict[str, Any] = {
        "schema_version": PREPROCESSING_SCHEMA_VERSION,
        "repair_codes": sorted(code for code, count in repair_counts.items() if count),
        "repair_counts": dict(sorted(repair_counts.items())),
        **changed_fields,
    }
    return problem, solution, options, preprocessing, repair_counts


def invalid_control_characters(
    fields: Iterable[tuple[str, str]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for field, text in fields:
        for position, character in enumerate(text):
            codepoint = ord(character)
            if (codepoint < 32 and character != "\n") or 127 <= codepoint <= 159:
                findings.append(
                    {
                        "field": field,
                        "position": position,
                        "codepoint": f"U+{codepoint:04X}",
                    }
                )
    return findings


def latex_warnings(text: str, warnings: list[str]) -> None:
    # These are warnings only: dollar signs may denote currency, and malformed
    # mathematics can itself be a research-relevant model error.
    if text.count("{") != text.count("}"):
        add_unique(warnings, "latex_brace_mismatch")
    if text.count("$$") % 2:
        add_unique(warnings, "latex_display_delimiter_mismatch")
    if text.count(r"\[") != text.count(r"\]"):
        add_unique(warnings, "latex_display_delimiter_mismatch")
    if text.count(r"\(") != text.count(r"\)"):
        add_unique(warnings, "latex_inline_delimiter_mismatch")
    begins = Counter(re.findall(r"\\begin\{([A-Za-z*]+)\}", text))
    ends = Counter(re.findall(r"\\end\{([A-Za-z*]+)\}", text))
    if begins != ends:
        add_unique(warnings, "latex_environment_mismatch")


def analyze_record(
    record: dict[str, Any],
    line_number: int,
    id_counts: Counter[Any],
    prepared_problem: Any,
    prepared_solution: Any,
    prepared_options: Any,
    repair_counts: Counter[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    hard: list[str] = []
    research: list[str] = []
    warning: list[str] = []
    details: dict[str, Any] = {}

    identifier = record.get(args.id_key)
    if not isinstance(identifier, (str, int)) or not str(identifier).strip():
        add_unique(hard, "missing_or_invalid_id")
    elif id_counts[identifier] > 1:
        add_unique(hard, "duplicate_id")

    problem = prepared_problem
    if not isinstance(problem, str) or not problem.strip():
        add_unique(hard, "missing_or_invalid_problem")
        problem_text = ""
    else:
        problem_text = problem

    solution = prepared_solution
    reasoning = ""
    answer_text = ""
    think_open_count = think_close_count = answer_open_count = answer_close_count = 0
    think_block_count = answer_block_count = 0
    if not isinstance(solution, str) or not solution.strip():
        add_unique(hard, "missing_or_invalid_solution")
        solution_text = ""
    else:
        solution_text = solution
        think_open_count = len(THINK_OPEN_RE.findall(solution))
        think_close_count = len(THINK_CLOSE_RE.findall(solution))
        answer_open_count = len(ANSWER_OPEN_RE.findall(solution))
        answer_close_count = len(ANSWER_CLOSE_RE.findall(solution))
        think_matches = list(THINK_BLOCK_RE.finditer(solution))
        answer_matches = list(ANSWER_BLOCK_RE.finditer(solution))
        think_block_count = len(think_matches)
        answer_block_count = len(answer_matches)

        if think_open_count == 0 and think_close_count == 0:
            add_unique(hard, "missing_think_block")
        elif think_open_count != think_close_count or think_block_count != think_open_count:
            add_unique(hard, "malformed_think_tags")
        elif think_block_count > 1:
            add_unique(hard, "multiple_think_blocks")

        if answer_open_count == 0 and answer_close_count == 0:
            add_unique(hard, "missing_answer_block")
        elif answer_open_count != answer_close_count or answer_block_count != answer_open_count:
            add_unique(hard, "malformed_answer_tags")
        elif answer_block_count > 1:
            add_unique(hard, "multiple_answer_blocks")

        if think_block_count == 1:
            reasoning = think_matches[0].group(1).strip()
            if not reasoning:
                add_unique(hard, "empty_reasoning")
        if answer_block_count == 1:
            answer_text = answer_matches[0].group(1).strip()
            if not answer_text:
                add_unique(warning, "empty_answer_block")
        if think_block_count == 1 and answer_block_count == 1:
            if think_matches[0].end() > answer_matches[0].start():
                add_unique(hard, "invalid_think_answer_order")

    raw_controls = invalid_control_characters(textual_fields(record, args))
    residual_controls = invalid_control_characters(
        prepared_textual_fields(problem, solution, prepared_options, args)
    )
    if residual_controls:
        add_unique(hard, "unrepairable_control_character")
    if raw_controls:
        details["source_control_characters"] = raw_controls
    if residual_controls:
        details["residual_control_characters"] = residual_controls
    if raw_controls and not residual_controls:
        add_unique(warning, "repaired_control_character")

    social_noise = bool(SOCIAL_NOISE_RE.search(reasoning))
    if social_noise:
        add_unique(hard, "social_platform_noise")

    compact_reasoning = re.sub(r"\s+", " ", reasoning).strip()
    has_answer_assertion = bool(ANSWER_ASSERTION_RE.search(compact_reasoning))
    has_derivation_cue = bool(DERIVATION_CUE_RE.search(compact_reasoning))
    equation_markers = len(re.findall(r"(?:=|\\frac|\\sum|\\int|\\lim)", reasoning))
    answer_only = bool(
        compact_reasoning
        and len(compact_reasoning) <= 160
        and has_answer_assertion
        and (social_noise or not has_derivation_cue)
        and equation_markers <= 1
    )
    if answer_only:
        add_unique(hard, "answer_only")

    combined = f"{problem_text}\n{reasoning}"
    if MISSING_PRIOR_DERIVATION_RE.search(reasoning):
        add_unique(research, "missing_prior_derivation")
    if REFERENCE_RATIONALIZATION_RE.search(reasoning):
        add_unique(research, "reference_answer_rationalization")
    if UNSUPPORTED_INTERMEDIATE_RE.search(reasoning):
        add_unique(research, "unsupported_intermediate_claim")
    if EXTERNAL_DATA_RE.search(combined):
        add_unique(research, "missing_external_data")
    if EXTERNAL_VISUAL_RE.search(combined):
        add_unique(research, "external_visual_dependency")

    raw_solution = record.get(args.solution_key)
    literal_newlines = (
        len(PROBABLE_LITERAL_NEWLINE_RE.findall(raw_solution))
        if isinstance(raw_solution, str)
        else 0
    )
    if literal_newlines:
        if repair_counts.get("decode_literal_newline", 0):
            add_unique(warning, "repaired_literal_newline_escape")
        else:
            add_unique(warning, "probable_literal_newline_escape")
    zero_width_count = sum(
        text.count(character)
        for _, text in textual_fields(record, args)
        for character in ZERO_WIDTH_CHARACTERS
    )
    if zero_width_count:
        if repair_counts.get("remove_zero_width_character", 0):
            add_unique(warning, "removed_zero_width_character")
        else:
            add_unique(warning, "zero_width_character")
    if repair_counts.get("insert_safe_boundary", 0):
        add_unique(warning, "inserted_safe_source_boundaries")
    if FINAL_ANSWER_IN_THINK_RE.search(reasoning):
        add_unique(warning, "final_answer_inside_think")

    question_type = record.get(args.type_key)
    if question_type in (None, "", -1, "-1"):
        add_unique(warning, "unknown_question_type")
    options = prepared_options
    options_count = len(options) if isinstance(options, list) else 0
    if not isinstance(options, list):
        add_unique(warning, "invalid_options_field")
    if (
        isinstance(question_type, str)
        and any(marker in question_type.lower() for marker in ("选择", "客观", "choice"))
        and options_count == 0
    ):
        add_unique(warning, "empty_options_for_choice_question")

    latex_warnings(problem_text, warning)
    latex_warnings(solution_text, warning)

    metrics = {
        "input_line_number": line_number,
        "problem_chars": len(problem_text),
        "solution_chars": len(solution_text),
        "reasoning_chars": len(reasoning),
        "reasoning_lines": reasoning.count("\n") + (1 if reasoning else 0),
        "answer_chars": len(answer_text),
        "options_count": options_count,
        "source_control_character_count": len(raw_controls),
        "residual_control_character_count": len(residual_controls),
        "probable_literal_newline_count": literal_newlines,
        "zero_width_character_count": zero_width_count,
        "inserted_safe_boundary_count": repair_counts.get(
            "insert_safe_boundary", 0
        ),
    }
    details["tag_counts"] = {
        "think_open": think_open_count,
        "think_close": think_close_count,
        "think_blocks": think_block_count,
        "answer_open": answer_open_count,
        "answer_close": answer_close_count,
        "answer_blocks": answer_block_count,
    }

    sat_eligible = not hard
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if sat_eligible else "reject",
        "sat_eligible": sat_eligible,
        "hard_error_codes": hard,
        "research_error_signals": research,
        "warning_codes": warning,
        "repair_codes": sorted(code for code, count in repair_counts.items() if count),
        "metrics": metrics,
        "details": details,
    }
    return result


def atomic_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


def sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def main() -> int:
    args = parse_args()
    id_counts, input_records, physical_lines = count_ids(args.input, args.id_key)
    if input_records == 0:
        raise ValueError("input contains no JSON records")
    duplicate_ids = sum(1 for count in id_counts.values() if count > 1)
    log(
        f"validated {input_records} JSON record(s) across {physical_lines} physical line(s)"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = atomic_path(args.output)
    stats_tmp = atomic_path(args.stats_output)

    hard_counts: Counter[str] = Counter()
    research_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    repair_counts_total: Counter[str] = Counter()
    hard_combinations: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    eligible = 0
    rejected = 0
    preprocessed_records = 0
    written = 0

    try:
        with output_tmp.open("w", encoding="utf-8") as output_stream:
            for line_number, record in iter_jsonl(args.input):
                if args.check_key in record and not args.replace_existing_check:
                    raise ValueError(
                        f"record at line {line_number} already contains {args.check_key!r}; "
                        "pass --replace-existing-check to replace it"
                    )
                if (
                    args.preprocessing_key in record
                    and not args.replace_existing_check
                ):
                    raise ValueError(
                        f"record at line {line_number} already contains "
                        f"{args.preprocessing_key!r}; pass --replace-existing-check "
                        "to replace it"
                    )
                (
                    prepared_problem,
                    prepared_solution,
                    prepared_options,
                    preprocessing,
                    repair_counts,
                ) = preprocess_record(record, args)
                check = analyze_record(
                    record,
                    line_number,
                    id_counts,
                    prepared_problem,
                    prepared_solution,
                    prepared_options,
                    repair_counts,
                    args,
                )
                result = dict(record)
                result[args.check_key] = check
                if preprocessing is not None:
                    result[args.preprocessing_key] = preprocessing
                    preprocessed_records += 1
                elif args.preprocessing_key in result and args.replace_existing_check:
                    del result[args.preprocessing_key]
                output_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                written += 1

                if check["sat_eligible"]:
                    eligible += 1
                else:
                    rejected += 1
                    hard_combinations[" + ".join(check["hard_error_codes"])] += 1
                hard_counts.update(check["hard_error_codes"])
                research_counts.update(check["research_error_signals"])
                warning_counts.update(check["warning_codes"])
                repair_counts_total.update(repair_counts)
                subject_counts[str(record.get(args.subject_key))] += 1
                type_counts[str(record.get(args.type_key))] += 1

                if written % 10000 == 0:
                    log(f"annotated {written}/{input_records} record(s)")

        stats = {
            "schema_version": SCHEMA_VERSION,
            "input": str(args.input),
            "output": str(args.output),
            "input_records": input_records,
            "output_records": written,
            "physical_lines": physical_lines,
            "blank_lines": physical_lines - input_records,
            "sat_eligible_records": eligible,
            "sat_ineligible_records": rejected,
            "sat_eligibility_rate": eligible / input_records,
            "preprocessed_records": preprocessed_records,
            "unique_ids": len(id_counts),
            "duplicate_id_values": duplicate_ids,
            "hard_error_counts": sorted_counter(hard_counts),
            "hard_error_combinations": sorted_counter(hard_combinations),
            "research_error_signal_counts": sorted_counter(research_counts),
            "warning_counts": sorted_counter(warning_counts),
            "repair_counts": sorted_counter(repair_counts_total),
            "subject_counts": sorted_counter(subject_counts),
            "type_counts": sorted_counter(type_counts),
            "policy": {
                "hard_errors_block_sat": True,
                "research_error_signals_block_sat": False,
                "warnings_block_sat": False,
                "deterministic_repairs_block_sat": False,
                "raw_fields_preserved": True,
                "sat_uses_preprocessed_view": True,
                "max_source_line_chars": args.max_source_line_chars,
                "span_count_used_for_source_filtering": False,
            },
        }
        with stats_tmp.open("w", encoding="utf-8") as stats_stream:
            json.dump(stats, stats_stream, ensure_ascii=False, indent=2)
            stats_stream.write("\n")
        os.replace(output_tmp, args.output)
        os.replace(stats_tmp, args.stats_output)
    except Exception:
        for temporary in (output_tmp, stats_tmp):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise

    log(f"wrote {written} annotated record(s) to {args.output}")
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted by user")
        raise SystemExit(130)
    except Exception as error:
        log(f"error: {error}")
        raise SystemExit(1)
