#!/usr/bin/env python3
"""Merge cessation-refined and review records back into a full materialized JSONL.

The base file remains the authoritative row set and order.  Records emitted by
``cessation_candidate_audit_client.py`` replace base records by ``id`` only
after their recorded input hash is verified against the exact base record.

Both outputs should normally be supplied:

* ``--refined`` contains accepted/core records.
* ``--review`` contains records routed to review and must also be merged so
  downstream consumers do not silently treat them as core.

The output is always a separate file; in-place replacement of ``--base`` is
rejected.  Untouched base lines are copied byte-for-byte (apart from ensuring a
final newline), while overlaid records are serialized as one JSON object/line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MATERIALIZED_SCHEMA = "prm_materialized_trace_v1"
SCRIPT_VERSION = "merge_cessation_refinement_v1.0"


@dataclass(frozen=True)
class OverlayRecord:
    record_id: str
    record: dict[str, Any]
    source_kind: str
    source_path: Path
    line_number: int
    expected_base_sha256: str


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay cessation refined/review records onto the full materialized "
            "JSONL while preserving base order and verifying lineage."
        )
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--refined", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output/report file; never permits overwriting inputs.",
    )
    args = parser.parse_args()
    if args.report is None:
        stem = args.output.name[:-6] if args.output.name.endswith(".jsonl") else args.output.stem
        args.report = args.output.with_name(stem + ".merge_report.json")
    return args


def validate_paths(args: argparse.Namespace) -> None:
    inputs = [args.base.resolve(), args.refined.resolve(), args.review.resolve()]
    outputs = [args.output.resolve(), args.report.resolve()]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"input file not found: {path}")
    if len(set(inputs)) != len(inputs):
        raise ValueError("--base, --refined, and --review must be distinct files")
    if len(set(outputs)) != len(outputs):
        raise ValueError("--output and --report must be distinct files")
    if set(inputs) & set(outputs):
        raise ValueError("outputs must not overwrite --base, --refined, or --review")
    if not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "outputs already exist; use a new path or pass --overwrite: "
                + ", ".join(existing)
            )


def read_json_line(path: Path, line_number: int, line: str) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}:{line_number}")
    if value.get("schema") != MATERIALIZED_SCHEMA:
        raise ValueError(
            f"expected schema={MATERIALIZED_SCHEMA} at {path}:{line_number}"
        )
    record_id = value.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"missing/invalid id at {path}:{line_number}")
    return value


def get_lineage_hash(record: dict[str, Any], path: Path, line_number: int) -> str:
    lineage = record.get("source_lineage", {}).get("cessation_refinement", {})
    expected = lineage.get("input_record_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(
            f"missing cessation_refinement.input_record_sha256 at "
            f"{path}:{line_number}"
        )
    return expected


def load_overlay(
    path: Path,
    source_kind: str,
    expected_status: str,
    combined: dict[str, OverlayRecord],
) -> int:
    count = 0
    seen_local: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = read_json_line(path, line_number, line)
            record_id = record["id"]
            if record_id in seen_local:
                raise ValueError(f"duplicate id {record_id!r} in {path}")
            if record_id in combined:
                other = combined[record_id]
                raise ValueError(
                    f"id {record_id!r} appears in both {other.source_path} "
                    f"and {path}"
                )
            status = record.get("audit", {}).get("status")
            if status != expected_status:
                raise ValueError(
                    f"{source_kind} record {record_id!r} has audit.status={status!r}; "
                    f"expected {expected_status!r}"
                )
            seen_local.add(record_id)
            combined[record_id] = OverlayRecord(
                record_id=record_id,
                record=record,
                source_kind=source_kind,
                source_path=path,
                line_number=line_number,
                expected_base_sha256=get_lineage_hash(record, path, line_number),
            )
            count += 1
    return count


def serialized_line(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write_output(
    base: Path,
    output: Path,
    overlays: dict[str, OverlayRecord],
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_base: set[str] = set()
    applied: set[str] = set()
    status_counts: Counter[str] = Counter()
    promoted_steps = 0
    base_count = 0
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=output.name + ".tmp-", dir=output.parent, delete=False
        ) as target:
            temp_name = target.name
            with base.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    base_record = read_json_line(base, line_number, line)
                    record_id = base_record["id"]
                    if record_id in seen_base:
                        raise ValueError(f"duplicate id {record_id!r} in base file")
                    seen_base.add(record_id)
                    base_count += 1
                    overlay = overlays.get(record_id)
                    if overlay is None:
                        output_record = base_record
                        raw = line if line.endswith("\n") else line + "\n"
                        target.write(raw.encode("utf-8"))
                    else:
                        actual_hash = object_sha256(base_record)
                        if actual_hash != overlay.expected_base_sha256:
                            raise ValueError(
                                f"lineage mismatch for id {record_id!r}: overlay from "
                                f"{overlay.source_path}:{overlay.line_number} expects "
                                f"{overlay.expected_base_sha256}, base has {actual_hash}"
                            )
                        output_record = overlay.record
                        target.write(serialized_line(output_record))
                        applied.add(record_id)
                        refinement = output_record.get("audit", {}).get(
                            "cessation_refinement", {}
                        )
                        promoted = refinement.get("promoted_steps", [])
                        if isinstance(promoted, list):
                            promoted_steps += len(promoted)
                    status = output_record.get("audit", {}).get("status")
                    status_counts[str(status)] += 1
            missing = sorted(set(overlays) - applied)
            if missing:
                preview = ", ".join(missing[:20])
                raise ValueError(
                    f"{len(missing)} overlay id(s) do not exist in the base file: "
                    f"{preview}{' ...' if len(missing) > 20 else ''}"
                )
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp_name, output)
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
    return {
        "base_records": base_count,
        "overlay_records": len(overlays),
        "applied_overlay_records": len(applied),
        "unchanged_base_records": base_count - len(applied),
        "output_records": base_count,
        "promoted_cessation_steps": promoted_steps,
        "output_audit_status_counts": dict(sorted(status_counts.items())),
    }


def write_report(path: Path, report: dict[str, Any], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if overwrite else os.O_EXCL)
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    validate_paths(args)
    overlays: dict[str, OverlayRecord] = {}
    refined_count = load_overlay(args.refined, "refined", "core", overlays)
    review_count = load_overlay(args.review, "review", "review", overlays)
    counts = atomic_write_output(args.base, args.output, overlays)
    if counts["output_records"] != counts["base_records"]:
        raise RuntimeError("output row count differs from base row count")
    if counts["applied_overlay_records"] != refined_count + review_count:
        raise RuntimeError("not every overlay record was applied")
    report = {
        "schema_version": "cessation_refinement_merge_report_v1",
        "script_version": SCRIPT_VERSION,
        "status": "ok",
        "base": str(args.base.resolve()),
        "base_sha256": file_sha256(args.base),
        "refined": str(args.refined.resolve()),
        "refined_sha256": file_sha256(args.refined),
        "review": str(args.review.resolve()),
        "review_sha256": file_sha256(args.review),
        "output": str(args.output.resolve()),
        "output_sha256": file_sha256(args.output),
        "refined_overlay_records": refined_count,
        "review_overlay_records": review_count,
        **counts,
        "base_order_preserved": True,
        "lineage_verified": True,
    }
    write_report(args.report, report, args.overwrite)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        raise SystemExit(1)
