#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert materialized PRM800K phase-2 traces to hierarchical PRM data.

The source train/test split is preserved. A deterministic trajectory-level
validation set is optionally carved from the source training file, so steps
from one trajectory can never appear in both train and validation.

Outputs both:
  * trajectory JSONL: one record contains every step;
  * flat JSONL: one record contains one current step plus its preceding prefix.

The flat files are directly compatible with hierarchical_prm_train.py used in
this project. In particular, ``labels`` is the source of truth for correctness:
False -> ERROR (0), True -> NON_ERROR (1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


DEFAULT_TRAIN_INPUT = Path(
    "data/prm800k/"
    "prm800k_phase2_train.prm_materialized.cessation_merged.jsonl"
)
DEFAULT_TEST_INPUT = Path(
    "data/prm800k/"
    "prm800k_phase2_test.prm_materialized.cessation_merged.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/prm800k/hierarchical_prm"
)

IGNORE_INDEX = -100

# These IDs intentionally match hierarchical_prm_train.py.
# Do not use the older preprocessing convention 0=correct, 1=incorrect.
CORRECTNESS_TO_ID = {
    "I": 0,  # ERROR
    "C": 1,  # NON_ERROR
}
CORRECTNESS_TO_BOOL = {
    "I": False,
    "C": True,
}

ROLE_TO_ID = {
    "correct": 0,
    "new_error": 1,
    "propagated_error": 2,
}

# The current model has only three role classes. A cessation step has prm_label
# C and is therefore folded into correct, while original_role remains available.
ORIGINAL_ROLE_TO_TRAIN_ROLE = {
    "correct": "correct",
    "cessation": "correct",
    "new_error": "new_error",
    "propagated_error": "propagated_error",
}

ERROR_TYPE_TO_ID = {
    "calculation": 0,
    "derivation_or_logic": 1,
    "method_or_formula": 2,
    "problem_understanding": 3,
    "assumption_or_condition": 4,
    "answer_mismatch": 5,
    "missing_evidence": 6,
    "unit": 7,
}

# ``other`` is a legitimate source-side fallback, but it is deliberately not a
# ninth classifier class: its semantics are heterogeneous. Such new-error steps
# still supervise correctness and role, while the error-type loss is masked.
UNSUPERVISED_ERROR_TYPES = {None, "", "other"}


class RecordValidationError(ValueError):
    """Raised when one trajectory cannot be converted safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare PRM800K phase-2 hierarchical multi-task data."
    )
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.05,
        help="Fraction of source training trajectories assigned to validation.",
    )
    parser.add_argument(
        "--split-seed",
        type=str,
        default="prm800k_phase2_hierarchical_v1",
        help="Salt used by deterministic train/validation assignment.",
    )
    parser.add_argument(
        "--exclude-audit-status",
        action="append",
        default=[],
        help=(
            "Exclude records whose audit.status equals this value. May be used "
            "multiple times, for example --exclude-audit-status review."
        ),
    )
    parser.add_argument(
        "--invalid-record-policy",
        choices=("error", "drop"),
        default="error",
        help="Stop on an invalid trajectory or drop it and record statistics.",
    )
    parser.add_argument(
        "--error-type-keep-rates",
        type=str,
        default="{}",
        help=(
            "Optional JSON object for deterministic downsampling of high-frequency "
            "new_error rows in flat TRAIN data, e.g. "
            "'{\"calculation\":0.5,\"derivation_or_logic\":0.7}'. "
            "Trajectory files and validation/test files are never downsampled."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print progress every N source trajectories; 0 disables it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecordValidationError(
                    f"Invalid JSON at {path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise RecordValidationError(
                    f"Expected a JSON object at {path}:{line_no}."
                )
            yield line_no, value


def dump_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def stable_fraction(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def assign_training_split(sample_id: str, validation_ratio: float, seed: str) -> str:
    if stable_fraction(f"{seed}\0{sample_id}") < validation_ratio:
        return "validation"
    return "train"


def should_keep_by_rate(key: str, rate: float, seed: str) -> bool:
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return stable_fraction(f"{seed}\0downsample\0{key}") < rate


def parse_keep_rates(raw: str) -> Dict[str, float]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--error-type-keep-rates is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--error-type-keep-rates must be a JSON object.")

    result: Dict[str, float] = {}
    for name, rate in value.items():
        if name not in ERROR_TYPE_TO_ID:
            raise ValueError(
                f"Unknown error type in keep rates: {name!r}; "
                f"expected one of {sorted(ERROR_TYPE_TO_ID)}"
            )
        rate = float(rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Keep rate for {name!r} must be in [0,1], got {rate}.")
        result[name] = rate
    return result


def build_prompt(record: Mapping[str, Any]) -> str:
    context = record.get("context") or {}
    if not isinstance(context, dict):
        raise RecordValidationError("context must be an object.")

    problem = str(context.get("problem") or "").strip()
    if not problem:
        raise RecordValidationError("context.problem is empty.")

    options = context.get("options") or []
    if not isinstance(options, list):
        raise RecordValidationError("context.options must be a list when present.")
    option_text = [str(item).strip() for item in options if str(item).strip()]
    return problem if not option_text else problem + "\n\n" + "\n".join(option_text)


def normalize_step(
    step: Mapping[str, Any],
    sample_id: str,
    fallback_index: int,
) -> Dict[str, Any]:
    text = str(step.get("text") or "").strip()
    if not text:
        raise RecordValidationError(
            f"id={sample_id}: step {fallback_index + 1} has empty text."
        )

    prm_label = step.get("prm_label")
    original_role = step.get("role")
    raw_error_type = step.get("error_type")

    if prm_label not in CORRECTNESS_TO_BOOL:
        raise RecordValidationError(
            f"id={sample_id}: unknown prm_label={prm_label!r}."
        )
    if original_role not in ORIGINAL_ROLE_TO_TRAIN_ROLE:
        raise RecordValidationError(
            f"id={sample_id}: unknown role={original_role!r}."
        )

    # Detect contradictory source annotations before training.
    if prm_label == "C" and original_role not in {"correct", "cessation"}:
        raise RecordValidationError(
            f"id={sample_id}: prm_label=C conflicts with role={original_role!r}."
        )
    if prm_label == "I" and original_role not in {"new_error", "propagated_error"}:
        raise RecordValidationError(
            f"id={sample_id}: prm_label=I conflicts with role={original_role!r}."
        )

    train_role = ORIGINAL_ROLE_TO_TRAIN_ROLE[original_role]
    role_id = ROLE_TO_ID[train_role]

    if original_role == "new_error":
        if raw_error_type in ERROR_TYPE_TO_ID:
            error_type = str(raw_error_type)
            error_type_id = ERROR_TYPE_TO_ID[error_type]
            error_type_mask = True
        elif raw_error_type in UNSUPERVISED_ERROR_TYPES:
            # Keep this row for correctness/new_error supervision. It has no
            # well-defined target for the fixed eight-class error-type head.
            error_type = None
            error_type_id = IGNORE_INDEX
            error_type_mask = False
        else:
            raise RecordValidationError(
                f"id={sample_id}: new_error has unsupported "
                f"error_type={raw_error_type!r}; expected one of "
                f"{sorted(ERROR_TYPE_TO_ID)} or 'other'/null."
            )
    else:
        error_type = None
        error_type_id = IGNORE_INDEX
        error_type_mask = False

    raw_index = step.get("index", fallback_index + 1)
    try:
        source_step_index = int(raw_index)
    except (TypeError, ValueError):
        source_step_index = fallback_index + 1

    return {
        "text": text,
        "step_id": str(step.get("id") or f"s{fallback_index + 1:02d}"),
        "source_step_index": source_step_index,
        "prm_label": prm_label,
        "label_bool": CORRECTNESS_TO_BOOL[prm_label],
        "raw_rating": 1 if prm_label == "C" else -1,
        "correctness_id": CORRECTNESS_TO_ID[prm_label],
        "original_role": original_role,
        "role": train_role,
        "role_id": role_id,
        "raw_error_type": raw_error_type,
        "error_type": error_type,
        "error_type_id": error_type_id,
        "error_type_mask": error_type_mask,
        "caused_by": step.get("caused_by") or [],
        "relation_to_first_error": step.get("relation_to_first_error"),
        "mixed_error": bool(step.get("mixed_error", False)),
        "op": step.get("op"),
        "note": step.get("note"),
    }


def convert_record(
    record: Mapping[str, Any],
    source_split: str,
    output_split: str,
    source_file: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prompt = build_prompt(record)
    sample_id = str(record.get("id") or "").strip()
    if not sample_id:
        sample_id = hashlib.sha256(
            (source_split + "\0" + prompt).encode("utf-8")
        ).hexdigest()

    raw_steps = record.get("steps") or []
    if not isinstance(raw_steps, list) or not raw_steps:
        raise RecordValidationError(f"id={sample_id}: steps is empty or not a list.")

    normalized_steps = []
    for index, step in enumerate(raw_steps):
        if not isinstance(step, dict):
            raise RecordValidationError(f"id={sample_id}: step {index + 1} is not an object.")
        normalized_steps.append(normalize_step(step, sample_id, index))

    first_new_error = next(
        (step["step_id"] for step in normalized_steps if step["original_role"] == "new_error"),
        None,
    )
    declared_first_error = record.get("first_error")
    first_error_consistent = (
        declared_first_error in (None, "", first_new_error)
        if first_new_error is not None
        else declared_first_error in (None, "")
    )

    context = record.get("context") or {}
    audit = record.get("audit") or {}
    audit_status = audit.get("status") if isinstance(audit, dict) else None

    completions: List[str] = []
    flat_records: List[Dict[str, Any]] = []
    prefix: List[str] = []

    for zero_index, step in enumerate(normalized_steps):
        step_number = zero_index + 1
        completion = f"Step{step_number}: {step['text']}"
        completions.append(completion)

        flat_prompt = prompt + "\n\n"
        if prefix:
            flat_prompt += "\n\n".join(prefix) + "\n\n"

        flat_records.append(
            {
                "id": sample_id,
                "problem": prompt,
                "prefix": list(prefix),
                "prompt": flat_prompt,
                "completions": [completion],
                "labels": [step["label_bool"]],
                "raw_rating": step["raw_rating"],
                "prm_label": step["prm_label"],
                "correctness_id": step["correctness_id"],
                "correctness_mask": True,
                "role": step["role"],
                "role_id": step["role_id"],
                "role_mask": True,
                "original_role": step["original_role"],
                "error_type": step["error_type"],
                "error_type_id": step["error_type_id"],
                "error_type_mask": step["error_type_mask"],
                "raw_error_type": step["raw_error_type"],
                "phase": "phase2",
                "step_index": zero_index,
                "step_number": step_number,
                "source_step_index": step["source_step_index"],
                "step_id": step["step_id"],
                "source_split": source_split,
                "split": output_split,
                "source": "prm800k_phase2",
                "source_file": source_file.name,
                "first_error": declared_first_error,
                "derived_first_error": first_new_error,
                "first_error_consistent": first_error_consistent,
                "overall": record.get("overall"),
                "caused_by": step["caused_by"],
                "relation_to_first_error": step["relation_to_first_error"],
                "mixed_error": step["mixed_error"],
                "op": step["op"],
                "audit_status": audit_status,
                "quality_flags": record.get("quality_flags") or [],
            }
        )
        prefix.append(completion)

    trajectory = {
        "schema": "hierarchical_multitask_prm_v1",
        "id": sample_id,
        "prompt": prompt,
        "completions": completions,
        "labels": [step["label_bool"] for step in normalized_steps],
        "raw_ratings": [step["raw_rating"] for step in normalized_steps],
        "correctness_ids": [step["correctness_id"] for step in normalized_steps],
        "correctness_mask": [True] * len(normalized_steps),
        "roles": [step["role"] for step in normalized_steps],
        "role_ids": [step["role_id"] for step in normalized_steps],
        "role_mask": [True] * len(normalized_steps),
        "original_roles": [step["original_role"] for step in normalized_steps],
        "error_types": [step["error_type"] for step in normalized_steps],
        "raw_error_types": [step["raw_error_type"] for step in normalized_steps],
        "error_type_ids": [step["error_type_id"] for step in normalized_steps],
        "error_type_mask": [step["error_type_mask"] for step in normalized_steps],
        "step_ids": [step["step_id"] for step in normalized_steps],
        "step_numbers": list(range(1, len(normalized_steps) + 1)),
        "source_step_indices": [step["source_step_index"] for step in normalized_steps],
        "caused_by": [step["caused_by"] for step in normalized_steps],
        "relation_to_first_error": [
            step["relation_to_first_error"] for step in normalized_steps
        ],
        "mixed_error": [step["mixed_error"] for step in normalized_steps],
        "phase": "phase2",
        "source_split": source_split,
        "split": output_split,
        "source": "prm800k_phase2",
        "source_file": source_file.name,
        "subject": context.get("subject"),
        "type": context.get("type"),
        "reference_answer": record.get("reference_answer"),
        "proposed_answer": record.get("proposed_answer"),
        "overall": record.get("overall"),
        "first_error": declared_first_error,
        "derived_first_error": first_new_error,
        "first_error_consistent": first_error_consistent,
        "audit_status": audit_status,
        "quality_flags": record.get("quality_flags") or [],
    }

    expected = len(completions)
    aligned = (
        "labels",
        "raw_ratings",
        "correctness_ids",
        "roles",
        "role_ids",
        "original_roles",
        "error_types",
        "raw_error_types",
        "error_type_ids",
        "error_type_mask",
        "step_ids",
        "step_numbers",
        "source_step_indices",
        "caused_by",
        "relation_to_first_error",
        "mixed_error",
    )
    for field in aligned:
        if len(trajectory[field]) != expected:
            raise RuntimeError(
                f"id={sample_id}: alignment failure for {field}: "
                f"{len(trajectory[field])} != {expected}"
            )

    return trajectory, flat_records


def new_split_stats() -> Dict[str, Any]:
    return {
        "trajectories": 0,
        "flat_rows_before_downsampling": 0,
        "flat_rows_written": 0,
        "correctness": Counter(),
        "original_role": Counter(),
        "training_role": Counter(),
        "error_type_new_error_only": Counter(),
        "unsupervised_new_error_type": Counter(),
        "overall": Counter(),
        "audit_status": Counter(),
        "first_error_inconsistent": 0,
        "downsampled_error_type_rows": Counter(),
    }


def update_stats(stats: Dict[str, Any], trajectory: Mapping[str, Any]) -> None:
    stats["trajectories"] += 1
    stats["overall"][str(trajectory.get("overall"))] += 1
    stats["audit_status"][str(trajectory.get("audit_status"))] += 1
    if not trajectory.get("first_error_consistent", True):
        stats["first_error_inconsistent"] += 1

    for label, original_role, role, error_type, raw_error_type in zip(
        trajectory["labels"],
        trajectory["original_roles"],
        trajectory["roles"],
        trajectory["error_types"],
        trajectory["raw_error_types"],
    ):
        stats["correctness"]["C" if label is True else "I"] += 1
        stats["original_role"][str(original_role)] += 1
        stats["training_role"][str(role)] += 1
        if error_type is not None:
            stats["error_type_new_error_only"][str(error_type)] += 1
        elif original_role == "new_error":
            stats["unsupervised_new_error_type"][str(raw_error_type)] += 1


def to_plain(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, defaultdict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain(v) for v in value]
    return value


def output_paths(output_dir: Path) -> Dict[str, Path]:
    prefix = "prm800k_phase2_hierarchical_prm"
    result: Dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        result[f"trajectory_{split}"] = output_dir / f"{prefix}_trajectory_{split}.jsonl"
        result[f"flat_{split}"] = output_dir / f"{prefix}_flat_{split}.jsonl"
    result["statistics"] = output_dir / f"{prefix}_statistics.json"
    result["label_map"] = output_dir / f"{prefix}_label_map.json"
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be in [0,1).")

    keep_rates = parse_keep_rates(args.error_type_keep_rates)
    excluded_statuses = set(args.exclude_audit_status)

    for path in (args.train_input, args.test_input):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(args.output_dir)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        formatted = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Output files already exist; use --overwrite to replace them:\n  " + formatted
        )

    temporary = {
        name: path.with_name(f".{path.name}.tmp.{os.getpid()}")
        for name, path in paths.items()
    }

    stats: Dict[str, Any] = {
        "source_files": {
            "train": str(args.train_input),
            "test": str(args.test_input),
        },
        "validation_ratio": args.validation_ratio,
        "split_seed": args.split_seed,
        "exclude_audit_status": sorted(excluded_statuses),
        "error_type_keep_rates_train_only": keep_rates,
        "source_records_seen": Counter(),
        "excluded_by_audit_status": Counter(),
        "invalid_records": Counter(),
        "duplicate_ids_within_source": Counter(),
        "cross_source_id_overlap": 0,
        "splits": {name: new_split_stats() for name in ("train", "validation", "test")},
        "invalid_examples": [],
    }

    seen_train: set[str] = set()
    seen_test: set[str] = set()

    try:
        with ExitStack() as stack:
            writers = {
                name: stack.enter_context(path.open("w", encoding="utf-8"))
                for name, path in temporary.items()
                if name.startswith("trajectory_") or name.startswith("flat_")
            }

            for source_split, source_path in (
                ("train", args.train_input),
                ("test", args.test_input),
            ):
                source_seen = seen_train if source_split == "train" else seen_test
                other_seen = seen_test if source_split == "train" else seen_train

                for line_no, record in read_jsonl(source_path):
                    stats["source_records_seen"][source_split] += 1
                    audit = record.get("audit") or {}
                    audit_status = audit.get("status") if isinstance(audit, dict) else None
                    if audit_status in excluded_statuses:
                        stats["excluded_by_audit_status"][str(audit_status)] += 1
                        continue

                    sample_id = str(record.get("id") or "").strip()
                    if sample_id:
                        if sample_id in source_seen:
                            stats["duplicate_ids_within_source"][source_split] += 1
                        if sample_id in other_seen:
                            stats["cross_source_id_overlap"] += 1
                        source_seen.add(sample_id)

                    output_split = (
                        assign_training_split(sample_id, args.validation_ratio, args.split_seed)
                        if source_split == "train"
                        else "test"
                    )

                    try:
                        trajectory, flat_records = convert_record(
                            record=record,
                            source_split=source_split,
                            output_split=output_split,
                            source_file=source_path,
                        )
                    except RecordValidationError as exc:
                        stats["invalid_records"][source_split] += 1
                        if len(stats["invalid_examples"]) < 20:
                            stats["invalid_examples"].append(
                                {
                                    "source_split": source_split,
                                    "line": line_no,
                                    "id": sample_id,
                                    "error": str(exc),
                                }
                            )
                        if args.invalid_record_policy == "error":
                            raise RecordValidationError(
                                f"{source_path}:{line_no}: {exc}"
                            ) from exc
                        continue

                    writers[f"trajectory_{output_split}"].write(dump_line(trajectory))
                    split_stats = stats["splits"][output_split]
                    update_stats(split_stats, trajectory)
                    split_stats["flat_rows_before_downsampling"] += len(flat_records)

                    for flat in flat_records:
                        keep = True
                        error_type = flat.get("error_type")
                        if output_split == "train" and error_type in keep_rates:
                            rate = keep_rates[str(error_type)]
                            key = f"{flat['id']}\0{flat['step_id']}\0{error_type}"
                            keep = should_keep_by_rate(key, rate, args.split_seed)
                            if not keep:
                                split_stats["downsampled_error_type_rows"][str(error_type)] += 1
                        if keep:
                            writers[f"flat_{output_split}"].write(dump_line(flat))
                            split_stats["flat_rows_written"] += 1

                    if (
                        args.progress_every > 0
                        and stats["source_records_seen"][source_split] % args.progress_every == 0
                    ):
                        print(
                            f"[{source_split}] processed "
                            f"{stats['source_records_seen'][source_split]:,} trajectories",
                            flush=True,
                        )

        if stats["cross_source_id_overlap"]:
            raise RecordValidationError(
                f"Detected {stats['cross_source_id_overlap']} train/test ID overlaps. "
                "The temporary outputs were not committed."
            )

        label_map = {
            "ignore_index": IGNORE_INDEX,
            "correctness": {
                "id2label": {"0": "ERROR", "1": "NON_ERROR"},
                "source_prm_label": {
                    "I": {"bool": False, "id": 0, "raw_rating": -1},
                    "C": {"bool": True, "id": 1, "raw_rating": 1},
                },
                "source_of_truth_for_trainer": "labels[0]",
            },
            "role": {
                "label_to_id": ROLE_TO_ID,
                "source_to_training": ORIGINAL_ROLE_TO_TRAIN_ROLE,
                "note": "cessation is folded into correct; original_role is retained",
            },
            "error_type": {
                "label_to_id": ERROR_TYPE_TO_ID,
                "supervision": "only original_role == new_error",
                "other_steps": IGNORE_INDEX,
                "untyped_new_error": (
                    "source error_type='other'/null keeps correctness and role "
                    "supervision but uses error_type_id=-100 and mask=false"
                ),
            },
        }

        temporary["statistics"].write_text(
            json.dumps(to_plain(stats), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary["label_map"].write_text(
            json.dumps(label_map, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        for name, final_path in paths.items():
            os.replace(temporary[name], final_path)

    except Exception:
        for path in temporary.values():
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    print("=" * 100)
    print("PRM800K phase-2 hierarchical data preparation finished")
    print("=" * 100)
    for split in ("train", "validation", "test"):
        split_stats = stats["splits"][split]
        print(
            f"{split:10s}: trajectories={split_stats['trajectories']:,}, "
            f"flat_rows={split_stats['flat_rows_written']:,}, "
            f"correctness={dict(split_stats['correctness'])}, "
            f"roles={dict(split_stats['training_role'])}"
        )
    print(f"statistics: {paths['statistics']}")
    print("Training files:")
    print(f"  --train_file {paths['flat_train']}")
    print(f"  --validation_file {paths['flat_validation']}")
    print("Held-out test file:")
    print(f"  {paths['flat_test']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise