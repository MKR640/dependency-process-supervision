#!/usr/bin/env python3
"""Build Teacher-A inputs by splitting solution reasoning with a local SaT model.

This utility reads raw JSONL records (one JSON object per non-empty line), extracts the reasoning inside
``<think>...</think>``, removes a duplicated ``Final Answer`` summary inside
``<think>`` by default, and losslessly segments the remaining reasoning with
Segment any Text (SaT).

The default output is the compact input expected by Teacher A:

    {
      "schema": "merge_io_input_v1",
      "id": "...",
      "context": {"subject": "...", "type": "...", "problem": "..."},
      "candidate_spans": [["c0001", "..."], ...],
      "rules": {...}
    }

SaT uses the XLM-R tokenizer. The standard BAAI/bge-m3 SentencePiece file is
byte-identical to XLM-R's file, so BGE-M3 may be used *only as the tokenizer
source*. Its embedding-model weights are never loaded. This script verifies the
SentencePiece SHA-256 before constructing SaT.

Install:
    pip install "wtpsplit>=2.2.1" torch transformers sentencepiece

Example:
    python sat_candidate_span_splitter.py \
      --input data/prm/260824/demo_input.sat_checked.jsonl \
      --output data/prm/260824/demo_teacher_a_input.jsonl \
      --model models/sat-3l-sm \
      --tokenizer models/bge-m3 \
      --device auto

By default, input records must contain a boolean
``source_check.sat_eligible`` field. Ineligible records are skipped before
reasoning extraction and before the SaT model is loaded. Repaired views under
``sat_preprocessing`` are preferred automatically while all raw fields remain
available in the audited input JSONL.

Use ``--emit augmented-record`` when full offsets and SaT metadata are needed
for auditing instead of the compact Teacher-A request.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence


SPAN_SCHEMA_VERSION = "sat_candidate_spans_v0.3"
TEACHER_A_SCHEMA_VERSION = "merge_io_input_v1"
DEFAULT_MODEL_PATH = "models/sat-3l-sm"
DEFAULT_TOKENIZER_PATH = "models/bge-m3"
DEFAULT_MAX_INLINE_MATH_CHARS = 512

# Standard XLM-R SentencePiece model; BAAI/bge-m3 publishes the same file.
XLMR_SENTENCEPIECE_SHA256 = (
    "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"
)
EXPECTED_XLMR_VOCAB_SIZE = 250002

THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.IGNORECASE | re.DOTALL)
FINAL_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?final\s+answer(?:\*\*)?\s*:\s*",
    re.IGNORECASE,
)

# sentencepiece currently emits Python 3.12 SWIG deprecation warnings during
# interpreter shutdown. They do not affect tokenization or SaT inference.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"builtin type (SwigPyPacked|SwigPyObject|swigvarlink).*",
)


def log(message: str) -> None:
    """Write progress to stderr without contaminating JSON output."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split <think> reasoning with local SaT and build Teacher-A JSON."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        help="Input .jsonl file; each non-empty line must be one JSON object.",
    )
    source.add_argument("--text", help="One reasoning string for a quick test.")

    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--emit",
        choices=["teacher-a-input", "augmented-record"],
        default="teacher-a-input",
        help="Compact Teacher-A input or audited source record (default: teacher-a-input).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Local SaT directory (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER_PATH,
        help=(
            "Local XLM-R-compatible tokenizer directory. Standard BGE-M3 is "
            f"supported (default: {DEFAULT_TOKENIZER_PATH})."
        ),
    )
    parser.add_argument(
        "--allow-unsafe-tokenizer",
        action="store_true",
        help="Allow a tokenizer whose SentencePiece hash differs from XLM-R.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face network access; local-only mode is the default.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="PyTorch device (default: auto).",
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "onnx-cpu", "onnx-gpu"],
        default="pytorch",
        help="Inference backend (default: pytorch).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Higher boundary threshold gives fewer/larger spans.",
    )
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--algorithm", choices=["viterbi", "greedy"], default="viterbi"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--newline-policy",
        choices=["boundary", "ignore"],
        default="boundary",
        help=(
            "Treat each non-empty source line as a candidate boundary, while "
            "using SaT within each line; 'ignore' reproduces whole-text SaT "
            "behavior (default: boundary)."
        ),
    )
    parser.add_argument(
        "--allow-math-splits",
        action="store_true",
        help="Keep boundaries inside LaTeX or code regions.",
    )
    parser.add_argument(
        "--max-inline-math-chars",
        type=int,
        default=DEFAULT_MAX_INLINE_MATH_CHARS,
        help=(
            "Maximum protected length of a single-line $...$ region. Inline "
            "math never crosses a physical newline (default: 512)."
        ),
    )
    parser.add_argument(
        "--markdown-table-policy",
        choices=["block", "rows"],
        default="block",
        help=(
            "Protect consecutive Markdown table rows as one structural span, "
            "or keep one candidate boundary per row (default: block)."
        ),
    )
    parser.add_argument(
        "--sentence-fallback",
        choices=["conservative", "off"],
        default="conservative",
        help=(
            "For a long one-line result containing at most two SaT spans, add "
            "unprotected sentence-final boundaries (default: conservative)."
        ),
    )
    parser.add_argument(
        "--keep-final-summary",
        action="store_true",
        help="Keep a 'Final Answer:' summary found at the end of <think>.",
    )
    parser.add_argument(
        "--no-think-required",
        action="store_true",
        help="Use the full solution when no <think> block exists.",
    )

    parser.add_argument("--id-key", default="id")
    parser.add_argument("--subject-key", default="subject")
    parser.add_argument("--type-key", default="type")
    parser.add_argument("--problem-key", default="problem")
    parser.add_argument("--options-key", default="options")
    parser.add_argument("--solution-key", default="solution")
    parser.add_argument(
        "--preprocessed-problem-key",
        default="sat_preprocessing.problem",
        help="Preferred repaired problem field when present.",
    )
    parser.add_argument(
        "--preprocessed-options-key",
        default="sat_preprocessing.options",
        help="Preferred repaired options field when present.",
    )
    parser.add_argument(
        "--preprocessed-solution-key",
        default="sat_preprocessing.solution",
        help="Preferred repaired solution field when present.",
    )
    parser.add_argument(
        "--ignore-preprocessed-fields",
        action="store_true",
        help="Use raw problem/options/solution even when repaired views exist.",
    )
    parser.add_argument(
        "--eligibility-key",
        default="source_check.sat_eligible",
        help=(
            "Dotted boolean field used to select records "
            "(default: source_check.sat_eligible)."
        ),
    )
    parser.add_argument(
        "--eligibility-policy",
        choices=["require", "if-present", "ignore"],
        default="require",
        help=(
            "'require' rejects unlabeled input, 'if-present' accepts unlabeled "
            "records, and 'ignore' processes every record (default: require)."
        ),
    )

    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.min_length is not None and args.min_length < 1:
        parser.error("--min-length must be positive")
    if args.max_length is not None and args.max_length < 1:
        parser.error("--max-length must be positive")
    if args.max_inline_math_chars < 1:
        parser.error("--max-inline-math-chars must be positive")
    if (
        args.min_length is not None
        and args.max_length is not None
        and args.min_length > args.max_length
    ):
        parser.error("--min-length cannot exceed --max-length")
    if args.threshold is not None and (
        args.min_length is not None or args.max_length is not None
    ):
        log("warning: SaT ignores --threshold when length constraints are used")
    return args


def get_nested(record: dict[str, Any], dotted_key: str) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_key)
        value = value[part]
    return value


def get_optional(record: dict[str, Any], dotted_key: str, default: Any) -> Any:
    try:
        return get_nested(record, dotted_key)
    except KeyError:
        return default


def get_preferred(
    record: dict[str, Any],
    preferred_key: str,
    raw_key: str,
    default: Any,
    *,
    ignore_preprocessed: bool,
) -> Any:
    if not ignore_preprocessed:
        preferred = get_optional(record, preferred_key, None)
        if preferred is not None:
            return preferred
    return get_optional(record, raw_key, default)


def read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix != ".jsonl":
        raise ValueError(
            "--input must be a .jsonl file with one JSON object per non-empty line; "
            "use --text for a single raw reasoning string"
        )

    records: list[dict[str, Any]] = []
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
            records.append(value)
    return records


def filter_sat_eligible_records(
    records: Sequence[dict[str, Any]],
    *,
    eligibility_key: str,
    eligibility_policy: str,
) -> tuple[list[dict[str, Any]], int]:
    """Return records selected for SaT and the number explicitly skipped."""
    if eligibility_policy == "ignore":
        return list(records), 0

    selected: list[dict[str, Any]] = []
    skipped = 0
    for index, record in enumerate(records, start=1):
        try:
            eligible = get_nested(record, eligibility_key)
        except KeyError:
            if eligibility_policy == "require":
                identifier = record.get("id", index)
                raise ValueError(
                    f"record {index} (id={identifier!r}) is missing required "
                    f"eligibility field {eligibility_key!r}; run the source-quality "
                    "filter first or pass --eligibility-policy if-present/ignore"
                )
            selected.append(record)
            continue
        if not isinstance(eligible, bool):
            identifier = record.get("id", index)
            raise TypeError(
                f"record {index} (id={identifier!r}) field {eligibility_key!r} "
                f"is {type(eligible).__name__}, expected a boolean"
            )
        if eligible:
            selected.append(record)
        else:
            skipped += 1
    return selected, skipped


def extract_reasoning(
    solution: str,
    *,
    require_think: bool,
    keep_final_summary: bool,
) -> tuple[str, dict[str, Any]]:
    matches = list(THINK_RE.finditer(solution))
    if len(matches) > 1:
        raise ValueError(f"solution contains {len(matches)} <think> blocks; expected one")
    if matches:
        reasoning = matches[0].group(1)
        source = "solution.think"
    elif require_think:
        raise ValueError("solution does not contain a <think>...</think> block")
    else:
        reasoning = solution
        source = "solution"

    removed_final_summary = False
    if not keep_final_summary:
        summary_matches = list(FINAL_ANSWER_RE.finditer(reasoning))
        if summary_matches:
            reasoning = reasoning[: summary_matches[-1].start()]
            removed_final_summary = True
    reasoning = reasoning.strip()
    if not reasoning:
        raise ValueError("extracted reasoning text is empty")
    return reasoning, {
        "source": source,
        "removed_final_summary": removed_final_summary,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_local_assets(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer)
    if not args.allow_network:
        if not model_path.is_dir():
            raise FileNotFoundError(f"local SaT model directory not found: {model_path}")
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"local tokenizer directory not found: {tokenizer_path}")

    if model_path.is_dir():
        if not (model_path / "config.json").exists():
            raise FileNotFoundError(f"required model file not found: {model_path / 'config.json'}")
        if not any(
            (model_path / name).exists()
            for name in ("model.safetensors", "pytorch_model.bin", "model_optimized.onnx")
        ):
            raise FileNotFoundError(
                f"no model weights found under {model_path}; expected model.safetensors, "
                "pytorch_model.bin, or model_optimized.onnx"
            )

    tokenizer_metadata: dict[str, Any] = {
        "path": str(tokenizer_path),
        "sentencepiece_sha256": None,
        "xlmr_sentencepiece_match": None,
    }
    if tokenizer_path.is_dir():
        sentencepiece_path = tokenizer_path / "sentencepiece.bpe.model"
        if not sentencepiece_path.exists():
            raise FileNotFoundError(
                f"tokenizer file not found: {sentencepiece_path}; "
                "SaT requires an XLM-R-compatible tokenizer"
            )
        sentencepiece_hash = sha256_file(sentencepiece_path)
        tokenizer_metadata["sentencepiece_sha256"] = sentencepiece_hash
        tokenizer_metadata["xlmr_sentencepiece_match"] = (
            sentencepiece_hash == XLMR_SENTENCEPIECE_SHA256
        )
        if sentencepiece_hash != XLMR_SENTENCEPIECE_SHA256:
            message = (
                "tokenizer SentencePiece model does not match standard XLM-R: "
                f"{sentencepiece_hash}"
            )
            if not args.allow_unsafe_tokenizer:
                raise ValueError(message + "; pass --allow-unsafe-tokenizer to override")
            log("warning: " + message)

        config_path = tokenizer_path / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            vocab_size = config.get("vocab_size")
            tokenizer_metadata["config_vocab_size"] = vocab_size
            if vocab_size not in (None, EXPECTED_XLMR_VOCAB_SIZE):
                message = (
                    f"tokenizer vocab_size={vocab_size}, expected "
                    f"{EXPECTED_XLMR_VOCAB_SIZE}"
                )
                if not args.allow_unsafe_tokenizer:
                    raise ValueError(message + "; pass --allow-unsafe-tokenizer to override")
                log("warning: " + message)
    return {
        "model_path": str(model_path),
        "tokenizer": tokenizer_metadata,
        "offline": not args.allow_network,
    }


def batched(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_sat(args: argparse.Namespace) -> tuple[Any, str]:
    if not args.allow_network:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from wtpsplit import SaT
    except ImportError as exc:
        raise RuntimeError(
            'install dependencies with: pip install "wtpsplit>=2.2.1" '
            "torch transformers sentencepiece"
        ) from exc
    # wtpsplit sets DeprecationWarning back to "default" during import, so the
    # sentencepiece/SWIG shutdown filters must be installed again afterwards.
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r"builtin type (SwigPyPacked|SwigPyObject|swigvarlink).*",
    )
    if "tokenizer_name_or_path" not in inspect.signature(SaT).parameters:
        raise RuntimeError(
            "installed wtpsplit lacks local-tokenizer support; upgrade with: "
            'pip install -U "wtpsplit>=2.2.1"'
        )

    sat_kwargs: dict[str, Any] = {
        "tokenizer_name_or_path": args.tokenizer,
        "from_pretrained_kwargs": {"local_files_only": not args.allow_network},
    }
    if args.backend == "onnx-cpu":
        sat_kwargs["ort_providers"] = ["CPUExecutionProvider"]
        return SaT(args.model, **sat_kwargs), "onnx-cpu"
    if args.backend == "onnx-gpu":
        sat_kwargs["ort_providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return SaT(args.model, **sat_kwargs), "onnx-gpu"

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch backend requires torch") from exc
    resolved_device = args.device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    sat = SaT(args.model, **sat_kwargs)
    if resolved_device == "cuda":
        sat.half().to("cuda")
    else:
        sat.to("cpu")
    return sat, resolved_device


def markdown_table_intervals(text: str) -> list[tuple[int, int]]:
    """Return offsets of consecutive Markdown table rows.

    A table is structural input rather than a sequence of reasoning steps. We
    only group runs of at least two pipe-delimited physical lines, so ordinary
    prose containing a pipe is unaffected.
    """

    intervals: list[tuple[int, int]] = []
    run_start: int | None = None
    run_end = 0
    run_rows = 0
    cursor = 0

    def finish_run() -> None:
        nonlocal run_start, run_end, run_rows
        if run_start is not None and run_rows >= 2:
            intervals.append((run_start, run_end))
        run_start = None
        run_end = 0
        run_rows = 0

    for line in text.splitlines(keepends=True):
        core, _ = split_line_ending(line)
        stripped = core.strip()
        is_table_row = (
            stripped.startswith("|")
            and stripped.endswith("|")
            and stripped.count("|") >= 3
        )
        if is_table_row:
            if run_start is None:
                run_start = cursor
            run_rows += 1
            run_end = cursor + len(line)
        else:
            finish_run()
        cursor += len(line)
    finish_run()
    return intervals


def protected_intervals(
    text: str,
    *,
    max_inline_math_chars: int = DEFAULT_MAX_INLINE_MATH_CHARS,
    protect_markdown_tables: bool = True,
) -> list[tuple[int, int]]:
    # Protect large/explicit constructs first. Inline-dollar matching is done
    # afterwards and any match overlapping these constructs is discarded. If
    # all patterns are collected independently, the second '$' in an opening
    # '$$' can be mistaken for an inline-math opener; overlapping matches then
    # bridge several display equations into one giant protected interval.
    block_patterns = [
        re.compile(r"```[\s\S]*?```"),
        # Same-line display math may appear after prose. A display block that
        # crosses lines must use line-oriented delimiters; otherwise a stray
        # trailing "$$" can pair with the next line's opener and chain many
        # unrelated equations back together.
        re.compile(r"\$\$[^\r\n]*?\$\$"),
        re.compile(
            r"(?m)^[ \t]*\$\$(?:(?!\$\$)[^\r\n])*(?:\r?\n)"
            r"[\s\S]*?\$\$[ \t]*(?=\r?$)"
        ),
        re.compile(r"\\\[[\s\S]*?\\\]"),
        re.compile(r"\\\([\s\S]*?\\\)"),
        re.compile(r"\\begin\{(?P<env>[A-Za-z*]+)\}[\s\S]*?\\end\{(?P=env)\}"),
    ]
    block_intervals: list[tuple[int, int]] = []
    for pattern in block_patterns:
        block_intervals.extend(
            (match.start(), match.end()) for match in pattern.finditer(text)
        )
    if protect_markdown_tables:
        block_intervals.extend(markdown_table_intervals(text))

    def merge_intervals(
        intervals: Sequence[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if not intervals:
            return []
        ordered = sorted(intervals)
        merged_intervals = [ordered[0]]
        for start, end in ordered[1:]:
            previous_start, previous_end = merged_intervals[-1]
            # Merge overlapping/nested constructs, but not merely adjacent
            # constructs. Adjacency is not evidence that two math blocks form
            # one reasoning unit.
            if start < previous_end:
                merged_intervals[-1] = (previous_start, max(previous_end, end))
            else:
                merged_intervals.append((start, end))
        return merged_intervals

    merged_blocks = merge_intervals(block_intervals)

    def overlaps_block(start: int, end: int) -> bool:
        return any(start < block_end and block_start < end for block_start, block_end in merged_blocks)

    # A malformed opener, a currency amount (e.g. "$13,000"), or an unmatched
    # dollar must never search later physical lines for a closing dollar. Such
    # cross-line matches previously re-merged dozens of valid SaT boundaries.
    inline_pattern = re.compile(
        rf"(?<![\\$])\$(?!\$)(?:\\[^\r\n]|[^\r\n$\\])"
        rf"{{1,{max_inline_math_chars}}}?(?<!\\)\$(?!\$)"
    )
    inline_intervals = [
        (match.start(), match.end())
        for match in inline_pattern.finditer(text)
        if not overlaps_block(match.start(), match.end())
    ]
    return merge_intervals([*merged_blocks, *inline_intervals])


def boundary_is_protected(position: int, intervals: Sequence[tuple[int, int]]) -> bool:
    return any(start < position < end for start, end in intervals)


def merge_protected_splits(
    text: str,
    segments: Sequence[str],
    *,
    max_inline_math_chars: int = DEFAULT_MAX_INLINE_MATH_CHARS,
    protect_markdown_tables: bool = True,
) -> list[str]:
    intervals = protected_intervals(
        text,
        max_inline_math_chars=max_inline_math_chars,
        protect_markdown_tables=protect_markdown_tables,
    )
    if not intervals or len(segments) < 2:
        return list(segments)
    result: list[str] = []
    cursor = 0
    for segment in segments:
        if not result:
            result.append(segment)
            cursor += len(segment)
            continue
        if boundary_is_protected(cursor, intervals):
            result[-1] += segment
        else:
            result.append(segment)
        cursor += len(segment)
    return result


SENTENCE_FALLBACK_RE = re.compile(
    r"(?<=[.!?])[ \t]+(?=(?:[A-Z]|\*\*|Step\b|Part\b))"
)


def conservative_sentence_fallback(
    text: str,
    segments: Sequence[str],
    *,
    max_initial_spans: int = 2,
    min_text_chars: int = 200,
    max_inline_math_chars: int = DEFAULT_MAX_INLINE_MATH_CHARS,
    protect_markdown_tables: bool = True,
) -> list[str]:
    """Split only conspicuously under-segmented, mostly one-line outputs.

    Teacher A can merge an extra sentence boundary, whereas a single candidate
    containing several complete claims cannot be recovered without rewriting.
    The fallback is deliberately limited to records with at most two current
    spans and never inserts a boundary inside protected math/code/table blocks.
    """

    if len(segments) > max_initial_spans or len(text) < min_text_chars:
        return list(segments)
    intervals = protected_intervals(
        text,
        max_inline_math_chars=max_inline_math_chars,
        protect_markdown_tables=protect_markdown_tables,
    )
    boundaries = {
        match.end()
        for match in SENTENCE_FALLBACK_RE.finditer(text)
        if not boundary_is_protected(match.end(), intervals)
    }
    if not boundaries:
        return list(segments)

    result: list[str] = []
    segment_start = 0
    for segment in segments:
        segment_end = segment_start + len(segment)
        cuts = sorted(
            position
            for position in boundaries
            if segment_start < position < segment_end
        )
        cursor = segment_start
        for position in cuts:
            result.append(text[cursor:position])
            cursor = position
        result.append(text[cursor:segment_end])
        segment_start = segment_end
    if "".join(result) != text:
        raise ValueError("sentence fallback did not preserve text exactly")
    return result


def postprocess_segments(
    text: str,
    segments: Sequence[str],
    args: argparse.Namespace,
) -> list[str]:
    result = list(segments)
    if not args.allow_math_splits:
        result = merge_protected_splits(
            text,
            result,
            max_inline_math_chars=args.max_inline_math_chars,
            protect_markdown_tables=args.markdown_table_policy == "block",
        )
    if args.sentence_fallback == "conservative":
        result = conservative_sentence_fallback(
            text,
            result,
            max_inline_math_chars=args.max_inline_math_chars,
            protect_markdown_tables=args.markdown_table_policy == "block",
        )
    if "".join(result) != text:
        raise ValueError("span postprocessing did not preserve text exactly")
    return result


def split_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "strip_whitespace": False,
    }
    if args.newline_policy == "ignore":
        # This option exists in wtpsplit 2.2.1. In boundary mode, individual
        # newline-free lines are sent to SaT and this option is unnecessary.
        kwargs["split_on_input_newlines"] = False
    if args.threshold is not None:
        kwargs["threshold"] = args.threshold
    if args.min_length is not None:
        kwargs["min_length"] = args.min_length
    if args.max_length is not None:
        kwargs["max_length"] = args.max_length
    if args.min_length is not None or args.max_length is not None:
        kwargs["algorithm"] = args.algorithm
    return kwargs


def split_line_ending(line: str) -> tuple[str, str]:
    """Return a physical line's content and exact line ending."""
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n") or line.endswith("\r"):
        return line[:-1], line[-1]
    return line, ""


def split_with_newline_boundaries(
    sat: Any,
    text: str,
    args: argparse.Namespace,
) -> list[str]:
    """Use structural newlines as boundaries and SaT within non-empty lines."""
    physical_lines = text.splitlines(keepends=True)
    if not physical_lines:
        return [text]

    line_parts = [split_line_ending(line) for line in physical_lines]
    nonempty_cores = [core for core, _ in line_parts if core.strip()]
    kwargs = split_kwargs(args)
    # All inputs are newline-free in this mode.
    kwargs.pop("split_on_input_newlines", None)
    if nonempty_cores:
        line_outputs = iter(list(sat.split(nonempty_cores, **kwargs)))
    else:
        line_outputs = iter(())

    segments: list[str] = []
    leading_whitespace = ""
    for core, ending in line_parts:
        if not core.strip():
            whitespace = core + ending
            if segments:
                segments[-1] += whitespace
            else:
                leading_whitespace += whitespace
            continue

        current = list(next(line_outputs))
        if not current:
            current = [core]
        if "".join(current) != core:
            raise ValueError("SaT did not preserve a physical line exactly")
        if leading_whitespace:
            current[0] = leading_whitespace + current[0]
            leading_whitespace = ""
        current[-1] += ending
        segments.extend(current)

    if leading_whitespace:
        if segments:
            segments[-1] += leading_whitespace
        else:
            segments = [leading_whitespace]
    if "".join(segments) != text:
        raise ValueError("newline-aware segmentation did not preserve text exactly")
    return segments


def make_candidate_spans(text: str, segments: Sequence[str]) -> list[dict[str, Any]]:
    reconstructed = "".join(segments)
    if reconstructed != text:
        mismatch = next(
            (
                index
                for index, (expected, actual) in enumerate(zip(text, reconstructed))
                if expected != actual
            ),
            min(len(text), len(reconstructed)),
        )
        raise ValueError(
            "SaT output did not preserve reasoning text exactly; "
            f"first mismatch at character {mismatch}"
        )
    spans: list[dict[str, Any]] = []
    cursor = 0
    for index, segment in enumerate(segments, start=1):
        end = cursor + len(segment)
        spans.append(
            {
                "span_id": f"c{index:04d}",
                "start_char": cursor,
                "end_char": end,
                "length_chars": len(segment),
                "text": segment,
            }
        )
        cursor = end
    return spans


def run_segmentation(
    sat: Any,
    texts: Sequence[str],
    args: argparse.Namespace,
) -> Iterator[list[str]]:
    if args.newline_policy == "boundary":
        for index, text in enumerate(texts, start=1):
            log(f"segmenting record {index}/{len(texts)} with newline-aware SaT")
            segments = split_with_newline_boundaries(sat, text, args)
            yield postprocess_segments(text, segments, args)
        return

    kwargs = split_kwargs(args)
    completed = 0
    for batch_index, text_batch in enumerate(batched(texts, args.batch_size), start=1):
        log(
            f"segmenting batch {batch_index}: records {completed + 1}-"
            f"{completed + len(text_batch)} of {len(texts)}"
        )
        outputs = list(sat.split(list(text_batch), **kwargs))
        if len(outputs) != len(text_batch):
            raise RuntimeError(
                f"SaT returned {len(outputs)} outputs for {len(text_batch)} inputs"
            )
        for text, segments in zip(text_batch, outputs):
            completed += 1
            yield postprocess_segments(text, segments, args)


def record_id(record: dict[str, Any], index: int, args: argparse.Namespace) -> Any:
    return get_optional(record, args.id_key, index)


def teacher_a_input(
    record: dict[str, Any],
    index: int,
    spans: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    options = get_preferred(
        record,
        args.preprocessed_options_key,
        args.options_key,
        [],
        ignore_preprocessed=args.ignore_preprocessed_fields,
    )
    if options is None:
        options = []
    return {
        "schema": TEACHER_A_SCHEMA_VERSION,
        "id": record_id(record, index, args),
        "context": {
            "subject": get_optional(record, args.subject_key, None),
            "type": get_optional(record, args.type_key, None),
            "problem": get_preferred(
                record,
                args.preprocessed_problem_key,
                args.problem_key,
                None,
                ignore_preprocessed=args.ignore_preprocessed_fields,
            ),
            "options": options,
        },
        "candidate_spans": [[span["span_id"], span["text"]] for span in spans],
        "rules": {
            "adjacent_merge_only": True,
            "cover_every_span_once": True,
            "preserve_span_order": True,
            "preserve_original_content": True,
            "extract_inputs_outputs_as_written": True,
        },
    }


def augmented_record(
    record: dict[str, Any],
    index: int,
    reasoning_text: str,
    extraction: dict[str, Any],
    spans: Sequence[dict[str, Any]],
    assets: dict[str, Any],
    runtime: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result = dict(record)
    result["reasoning_text"] = reasoning_text
    result["candidate_spans"] = list(spans)
    result["candidate_span_metadata"] = {
        "schema_version": SPAN_SCHEMA_VERSION,
        "record_id": record_id(record, index, args),
        "reasoning_extraction": extraction,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "tokenizer_validation": assets["tokenizer"],
        "runtime": runtime,
        "threshold": args.threshold,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "algorithm": args.algorithm,
        "newline_policy": args.newline_policy,
        "math_split_protection": not args.allow_math_splits,
        "inline_math_crosses_newlines": False,
        "max_inline_math_chars": args.max_inline_math_chars,
        "markdown_table_policy": args.markdown_table_policy,
        "sentence_fallback": args.sentence_fallback,
        "num_spans": len(spans),
        "text_preserved": True,
    }
    return result


def main() -> int:
    args = parse_args()
    if args.text is not None:
        records: list[dict[str, Any]] = [
            {
                "id": "text-0",
                "subject": None,
                "type": None,
                "problem": None,
                "options": [],
                "solution": args.text,
            }
        ]
        input_record_count = 1
        skipped_ineligible = 0
    else:
        records = read_records(args.input)
        input_record_count = len(records)
        log(f"loaded {input_record_count} input record(s)")
        records, skipped_ineligible = filter_sat_eligible_records(
            records,
            eligibility_key=args.eligibility_key,
            eligibility_policy=args.eligibility_policy,
        )
        log(
            f"eligibility selection kept {len(records)} and skipped "
            f"{skipped_ineligible} record(s)"
        )
    if input_record_count == 0:
        raise ValueError("input contains no records")

    if not records:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")
        summary = {
            "status": "ok",
            "input_records": input_record_count,
            "eligible_records": 0,
            "skipped_ineligible_records": skipped_ineligible,
            "output_records": 0,
            "output": str(args.output),
            "emit": args.emit,
            "model": args.model,
            "tokenizer": args.tokenizer,
            "runtime": None,
            "eligibility_key": args.eligibility_key,
            "eligibility_policy": args.eligibility_policy,
            "preprocessed_solution_records": 0,
            "ignored_preprocessed_fields": args.ignore_preprocessed_fields,
            "max_inline_math_chars": args.max_inline_math_chars,
            "markdown_table_policy": args.markdown_table_policy,
            "sentence_fallback": args.sentence_fallback,
        }
        log("no eligible records; wrote an empty output without loading SaT")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0

    reasoning_texts: list[str] = []
    extractions: list[dict[str, Any]] = []
    preprocessed_solution_records = 0
    for index, record in enumerate(records):
        preprocessed_solution = get_optional(
            record, args.preprocessed_solution_key, None
        )
        used_preprocessed_solution = bool(
            not args.ignore_preprocessed_fields
            and preprocessed_solution is not None
        )
        solution = get_preferred(
            record,
            args.preprocessed_solution_key,
            args.solution_key,
            None,
            ignore_preprocessed=args.ignore_preprocessed_fields,
        )
        if not isinstance(solution, str):
            raise TypeError(
                f"record {index} selected solution field is not a string"
            )
        reasoning, extraction = extract_reasoning(
            solution,
            require_think=not args.no_think_required and args.text is None,
            keep_final_summary=args.keep_final_summary,
        )
        extraction["used_preprocessed_solution"] = used_preprocessed_solution
        if used_preprocessed_solution:
            preprocessed_solution_records += 1
        reasoning_texts.append(reasoning)
        extractions.append(extraction)
    log("extracted reasoning text from all records")

    log("validating local model and tokenizer assets")
    assets = validate_local_assets(args)
    if assets["tokenizer"].get("xlmr_sentencepiece_match") is True:
        log("tokenizer validation passed: standard XLM-R SentencePiece hash matched")

    log(f"loading SaT model from {args.model}")
    log(f"loading tokenizer from {args.tokenizer}")
    load_started = time.perf_counter()
    sat, resolved_runtime = load_sat(args)
    log(
        f"SaT loaded in {time.perf_counter() - load_started:.2f}s "
        f"with runtime={resolved_runtime}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for index, (record, reasoning, extraction, segments) in enumerate(
            zip(
                records,
                reasoning_texts,
                extractions,
                run_segmentation(sat, reasoning_texts, args),
            )
        ):
            spans = make_candidate_spans(reasoning, segments)
            if args.emit == "teacher-a-input":
                result = teacher_a_input(record, index, spans, args)
            else:
                result = augmented_record(
                    record,
                    index,
                    reasoning,
                    extraction,
                    spans,
                    assets,
                    resolved_runtime,
                    args,
                )
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
            log(f"record {written}/{len(records)} -> {len(spans)} candidate spans")

    summary = {
        "status": "ok",
        "input_records": input_record_count,
        "eligible_records": len(records),
        "skipped_ineligible_records": skipped_ineligible,
        "output_records": written,
        "output": str(args.output),
        "emit": args.emit,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "runtime": resolved_runtime,
        "eligibility_key": args.eligibility_key,
        "eligibility_policy": args.eligibility_policy,
        "preprocessed_solution_records": preprocessed_solution_records,
        "ignored_preprocessed_fields": args.ignore_preprocessed_fields,
        "max_inline_math_chars": args.max_inline_math_chars,
        "markdown_table_policy": args.markdown_table_policy,
        "sentence_fallback": args.sentence_fallback,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted by user")
        raise SystemExit(130)
    except Exception as exc:
        log(f"error: {exc}")
        raise SystemExit(1)
