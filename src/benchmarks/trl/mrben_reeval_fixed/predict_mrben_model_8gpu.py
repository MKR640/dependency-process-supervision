#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8-GPU distributed inference for one PRM checkpoint on the frozen MRBEN test set.

Launch:
    torchrun --standalone --nproc_per_node=8 predict_mrben_model_8gpu.py ...

Each rank evaluates a disjoint subset of reference rows:
    global_row_index % WORLD_SIZE == RANK

Rank-local predictions are written to temporary shards. After all ranks finish,
rank 0 validates and merges them back into the ORIGINAL reference order, writes
one final prediction JSONL + .meta.json, and removes shards unless
--keep-shards is specified.

Model semantics
---------------
three_class:
    p_error = softmax(ERROR) from ERROR / NEUTRAL / CORRECT.

binary:
    p_error = softmax(ERROR) from ERROR / NON_ERROR.

full_hierarchy:
    p_error = P(ERROR) from correctness head.

    [FIX] role_probs = softmax(role_head), in the original training order:
        correct / new_error / propagated_error.
    No multiplication by correctness probabilities. No conditional 2-class head.

All three models use the same tokenizer and exactly the same packing policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import ModelOutput
from checkpoint_contract import inspect_checkpoint, check_loading_info


ROLES = ["correct", "new_error", "propagated_error"]
ERROR_TYPES = [
    "calculation",
    "derivation_or_logic",
    "method_or_formula",
    "problem_understanding",
    "assumption_or_condition",
    "answer_mismatch",
    "missing_evidence",
    "unit",
]
TRUNC = (
    "problem_tokens_dropped",
    "current_tokens_dropped",
    "prefix_steps_dropped",
)


# =============================================================================
# I/O
# =============================================================================

def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield row


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def tokenizer_hash(tokenizer) -> str:
    payload = {
        "class": tokenizer.__class__.__name__,
        "vocab": tokenizer.get_vocab(),
        "added_vocab": tokenizer.get_added_vocab(),
        "special_tokens_map": tokenizer.special_tokens_map,
        "padding_side": tokenizer.padding_side,
    }
    return digest(payload)


# =============================================================================
# Full-hierarchy model definition
# =============================================================================

@dataclass
class HierarchicalPRMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    correctness_logits: Optional[torch.Tensor] = None
    role_logits: Optional[torch.Tensor] = None
    error_type_logits: Optional[torch.Tensor] = None


class HierarchicalPRMModel(PreTrainedModel):
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def __init__(
        self,
        config: PretrainedConfig,
        backbone: Optional[nn.Module] = None,
        classifier_dropout: Optional[float] = None,
    ):
        super().__init__(config)

        self.config.correctness_num_labels = 2
        self.config.role_num_labels = 3
        self.config.error_type_num_labels = 8

        hidden_size = int(getattr(config, "hidden_size"))

        self.backbone = (
            AutoModel.from_config(config, trust_remote_code=True)
            if backbone is None
            else backbone
        )

        if classifier_dropout is None:
            classifier_dropout = getattr(
                config,
                "hierarchical_prm_classifier_dropout",
                getattr(config, "classifier_dropout", 0.1),
            )
        if classifier_dropout is None:
            classifier_dropout = 0.1

        self.dropout = nn.Dropout(float(classifier_dropout))
        self.correctness_head = nn.Linear(hidden_size, 2, bias=True)
        self.role_head = nn.Linear(hidden_size, 3, bias=True)
        self.error_type_head = nn.Linear(hidden_size, 8, bias=True)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        **kwargs,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            **kwargs,
        )
        hidden = self.dropout(outputs.last_hidden_state)
        return HierarchicalPRMOutput(
            correctness_logits=self.correctness_head(hidden),
            role_logits=self.role_head(hidden),
            error_type_logits=self.error_type_head(hidden),
        )


# =============================================================================
# Distributed utilities
# =============================================================================

def init_distributed() -> Tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-compatible GPU runtime is required")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

    return rank, local_rank, world_size, device


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def cleanup(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Tokenization / packing
# =============================================================================

def load_common_tokenizer(path: str):
    common = dict(
        use_fast=True,
        trust_remote_code=True,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            fix_mistral_regex=True,
            **common,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            **common,
        )

    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither PAD nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_reference_row(
    row: Dict[str, Any],
    tokenizer,
    max_length: int,
    separator: str,
) -> Dict[str, Any]:
    problem = str(row["problem"])
    prefix = row["prefix"]
    current = str(row["current_step"])

    if not isinstance(prefix, list) or not all(isinstance(x, str) for x in prefix):
        raise ValueError(
            f"{row.get('trace_id')}:{row.get('step_index')}: invalid prefix"
        )
    if not current.strip():
        raise ValueError(
            f"{row.get('trace_id')}:{row.get('step_index')}: empty current step"
        )

    sep_ids = tokenizer.encode(
        separator,
        add_special_tokens=False,
    )
    if not sep_ids:
        raise ValueError("step separator tokenized to an empty sequence")

    bos_ids = (
        [tokenizer.bos_token_id]
        if tokenizer.bos_token_id is not None
        else []
    )

    raw_problem_ids = tokenizer(
        problem,
        add_special_tokens=False,
    )["input_ids"]

    raw_current_ids = tokenizer(
        current,
        add_special_tokens=False,
    )["input_ids"]

    problem_ids = raw_problem_ids + sep_ids
    current_ids = raw_current_ids + sep_ids

    problem_dropped = 0
    current_dropped = 0

    minimum = (
        len(bos_ids)
        + len(problem_ids)
        + len(current_ids)
    )

    if minimum > max_length:
        available_problem = (
            max_length
            - len(bos_ids)
            - len(current_ids)
        )

        if available_problem <= len(sep_ids):
            available_current = (
                max_length
                - len(bos_ids)
                - len(sep_ids)
            )
            keep_current = max(1, available_current)

            current_dropped = max(
                0,
                len(raw_current_ids) - keep_current,
            )
            raw_current_ids = raw_current_ids[-keep_current:]
            current_ids = raw_current_ids + sep_ids

            problem_dropped = len(raw_problem_ids)
            problem_ids = []
        else:
            keep_problem = max(
                0,
                available_problem - len(sep_ids),
            )
            problem_dropped = max(
                0,
                len(raw_problem_ids) - keep_problem,
            )
            problem_ids = (
                raw_problem_ids[:keep_problem]
                + sep_ids
            )

    remaining = (
        max_length
        - len(bos_ids)
        - len(problem_ids)
        - len(current_ids)
    )

    selected_prefix_pieces: List[List[int]] = []
    prefix_dropped = 0

    for step in reversed(prefix):
        piece = (
            tokenizer(
                str(step),
                add_special_tokens=False,
            )["input_ids"]
            + sep_ids
        )
        if len(piece) <= remaining:
            selected_prefix_pieces.append(piece)
            remaining -= len(piece)
        else:
            prefix_dropped += 1

    selected_prefix_pieces.reverse()

    prefix_ids = [
        token
        for piece in selected_prefix_pieces
        for token in piece
    ]

    input_ids = (
        bos_ids
        + problem_ids
        + prefix_ids
        + current_ids
    )

    if len(input_ids) > max_length:
        raise RuntimeError(
            f"Encoded length {len(input_ids)} > max_length={max_length}"
        )

    return {
        "input_ids": input_ids,
        "target_pos": len(input_ids) - 1,
        "truncation": {
            "problem_tokens_dropped": int(problem_dropped),
            "current_tokens_dropped": int(current_dropped),
            "prefix_steps_dropped": int(prefix_dropped),
        },
    }


def normalized_id2label(config) -> Dict[int, str]:
    raw = getattr(config, "id2label", {}) or {}
    result: Dict[int, str] = {}
    for key, value in raw.items():
        try:
            idx = int(key)
        except Exception:
            continue
        result[idx] = (
            str(value)
            .strip()
            .upper()
            .replace("-", "_")
            .replace(" ", "_")
        )
    return result


def resolve_error_class(
    config,
    explicit: Optional[int],
) -> int:
    n = int(getattr(config, "num_labels"))

    if explicit is not None:
        if not 0 <= explicit < n:
            raise ValueError(
                f"--error-class-id={explicit} out of range for num_labels={n}"
            )
        named = normalized_id2label(config)
        known = [i for i, label in named.items() if label in {"ERROR", "INCORRECT", "WRONG", "NEGATIVE", "LABEL_ERROR"}]
        if known and known != [explicit]:
            raise ValueError(f"Explicit ERROR class {explicit} conflicts with config {named}")
        return explicit

    labels = normalized_id2label(config)
    accepted = {
        "ERROR",
        "INCORRECT",
        "WRONG",
        "NEGATIVE",
        "LABEL_ERROR",
    }
    hits = [
        idx
        for idx, name in labels.items()
        if name in accepted
    ]

    if len(hits) == 1:
        return hits[0]

    raise ValueError(
        f"Cannot resolve ERROR class: num_labels={n}, id2label={labels}. "
        "Pass --error-class-id explicitly."
    )


def parse_dtype(name: str):
    name = name.lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def batches(items: List[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


# =============================================================================
# Merge
# =============================================================================

def merge_rank_shards(
    output: Path,
    shard_dir: Path,
    world_size: int,
    total_rows: int,
) -> List[Dict[str, Any]]:
    merged: Dict[int, Dict[str, Any]] = {}

    for rank in range(world_size):
        shard = shard_dir / f"rank_{rank:02d}.jsonl"
        if not shard.is_file():
            raise FileNotFoundError(f"Missing prediction shard: {shard}")

        for row in read_jsonl(shard):
            idx = row.pop("_global_index")
            if type(idx) is not int:
                raise ValueError(f"Invalid _global_index in {shard}: {idx!r}")
            if idx in merged:
                raise ValueError(f"Duplicate global prediction index: {idx}")
            merged[idx] = row

    expected = set(range(total_rows))
    actual = set(merged)

    if actual != expected:
        missing = sorted(expected - actual)[:20]
        extra = sorted(actual - expected)[:20]
        raise ValueError(
            f"Distributed merge mismatch: missing={missing}, extra={extra}"
        )

    ordered = [merged[i] for i in range(total_rows)]
    write_jsonl(output, ordered)
    return ordered


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--name", required=True)
    p.add_argument(
        "--kind",
        required=True,
        choices=[
            "three_class",
            "binary",
            "full_hierarchy",
        ],
    )
    p.add_argument("--model-path", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--output", required=True)

    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--step-separator", default="\n\n")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )
    p.add_argument(
        "--attn-implementation",
        default="eager",
    )
    p.add_argument("--error-class-id", type=int)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-shards", action="store_true")

    args = p.parse_args()
    if args.max_length < 8 or args.batch_size < 1:
        raise ValueError("Invalid max_length/batch_size")
    torch.manual_seed(42)


    rank, local_rank, world_size, device = init_distributed()

    try:
        reference = Path(args.reference)
        reference_meta = Path(str(reference) + ".meta.json")

        if not reference.is_file():
            raise FileNotFoundError(reference)
        if not reference_meta.is_file():
            raise FileNotFoundError(reference_meta)

        output = Path(args.output)
        output_meta = Path(str(output) + ".meta.json")
        shard_dir = Path(str(output) + ".shards")

        if rank == 0:
            if (
                not args.overwrite
                and (
                    output.exists()
                    or output_meta.exists()
                    or shard_dir.exists()
                )
            ):
                raise FileExistsError(
                    f"Prediction output exists for {output}; "
                    "use --overwrite or a new output path."
                )

            output.parent.mkdir(parents=True, exist_ok=True)

            if shard_dir.exists():
                shutil.rmtree(shard_dir)

            shard_dir.mkdir(
                parents=True,
                exist_ok=False,
            )

        barrier(world_size)

        ref_meta = json.loads(
            reference_meta.read_text(encoding="utf-8")
        )

        reference_split = ref_meta.get("split")
        if reference_split not in {"test", "validation", "all"}:
            raise ValueError(
                f"Unsupported reference split={reference_split!r}; "
                "expected test, validation, or all"
            )

        if ref_meta.get("reference_sha256") != file_hash(reference):
            raise ValueError(
                "Reference hash does not match reference .meta.json"
            )

        rows = list(read_jsonl(reference))
        if not rows:
            raise ValueError("Empty reference")

        # Round-robin distribution gives balanced shard row counts even if the
        # total number of rows is not divisible by 8.
        shard_indices = list(
            range(rank, len(rows), world_size)
        )
        shard_rows = [
            rows[i]
            for i in shard_indices
        ]

        print(
            f"[{args.name}] rank={rank}/{world_size} "
            f"local_rank={local_rank} device={device} "
            f"rows={len(shard_rows)}",
            flush=True,
        )

        tokenizer = load_common_tokenizer(
            args.tokenizer_path
        )
        tok_sha = tokenizer_hash(tokenizer)

        encoded = [
            encode_reference_row(
                row,
                tokenizer,
                args.max_length,
                args.step_separator,
            )
            for row in shard_rows
        ]

        dtype = parse_dtype(args.dtype)

        common_model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if args.attn_implementation:
            common_model_kwargs[
                "attn_implementation"
            ] = args.attn_implementation

        error_class_id = None
        model_labels = None

        if args.kind == "full_hierarchy":
            config = AutoConfig.from_pretrained(
                args.model_path,
                trust_remote_code=True,
            )
            # [FIX] Validate the actual checkpoint, then require complete loading.
            checkpoint_audit = inspect_checkpoint(args.model_path)
            model, loading_info = HierarchicalPRMModel.from_pretrained(
                args.model_path, config=config, torch_dtype=dtype,
                attn_implementation=args.attn_implementation,
                output_loading_info=True,
            )
            loading_audit = check_loading_info(loading_info)
            active_heads = [
                "correctness",
                "role",
                "error_type",
            ]
        else:
            model, loading_info = AutoModelForTokenClassification.from_pretrained(
                args.model_path, output_loading_info=True, **common_model_kwargs,
            )
            loading_audit = check_loading_info(loading_info)
            expected_labels = 3 if args.kind == "three_class" else 2
            if model.config.num_labels != expected_labels:
                raise ValueError(f"Expected {expected_labels} labels for {args.kind}")
            checkpoint_audit = None
            error_class_id = resolve_error_class(
                model.config,
                args.error_class_id,
            )
            model_labels = normalized_id2label(
                model.config
            )
            active_heads = ["correctness"]

        if rank == 0:
            write_json(str(output) + ".loading.json", {
                "name": args.name, "model_path": args.model_path,
                "loading": loading_audit, "checkpoint": checkpoint_audit,
            })
            print(f"[{args.name}] STRICT LOAD PASS: missing=0 unexpected=0 mismatched=0", flush=True)
        model.to(device)
        model.eval()

        local_predictions: List[Dict[str, Any]] = []

        with torch.inference_mode():
            for batch_start, batch_items in batches(
                encoded,
                args.batch_size,
            ):
                batch_rows = shard_rows[
                    batch_start:
                    batch_start + len(batch_items)
                ]
                batch_global_indices = shard_indices[
                    batch_start:
                    batch_start + len(batch_items)
                ]

                padded = tokenizer.pad(
                    [
                        {"input_ids": item["input_ids"]}
                        for item in batch_items
                    ],
                    padding=True,
                    return_tensors="pt",
                )
                padded = {
                    k: v.to(
                        device,
                        non_blocking=True,
                    )
                    for k, v in padded.items()
                }

                target_pos = torch.tensor(
                    [
                        item["target_pos"]
                        for item in batch_items
                    ],
                    dtype=torch.long,
                    device=device,
                )
                batch_index = torch.arange(
                    len(batch_items),
                    device=device,
                )

                outputs = model(**padded)

                if args.kind == "full_hierarchy":
                    corr_logits = (
                        outputs.correctness_logits[
                            batch_index,
                            target_pos,
                        ]
                    )
                    role_logits = (
                        outputs.role_logits[
                            batch_index,
                            target_pos,
                        ]
                    )
                    type_logits = (
                        outputs.error_type_logits[
                            batch_index,
                            target_pos,
                        ]
                    )

                    corr_prob = torch.softmax(
                        corr_logits.float(),
                        dim=-1,
                    )
                    # [FIX] Independent 3-way head, exactly as trained.
                    role_prob = torch.softmax(role_logits.float(), dim=-1)
                    type_prob = torch.softmax(type_logits.float(), dim=-1)
                    p_error = corr_prob[:, 0]
                    p_non_error = corr_prob[:, 1]
                    if role_prob.shape[-1] != 3:
                        raise RuntimeError("Expected 3-way role_head output")

                else:
                    logits = outputs.logits[
                        batch_index,
                        target_pos,
                    ]
                    prob = torch.softmax(
                        logits.float(),
                        dim=-1,
                    )
                    p_error = prob[:, error_class_id]
                    role_prob = None
                    type_prob = None

                for j, row in enumerate(batch_rows):
                    record = {
                        "_global_index": int(
                            batch_global_indices[j]
                        ),
                        "trace_id": row["trace_id"],
                        "step_index": int(
                            row["step_index"]
                        ),
                        "p_error": float(
                            p_error[j].item()
                        ),
                        "role_probs": (
                            [
                                float(x)
                                for x in role_prob[j].tolist()
                            ]
                            if role_prob is not None
                            else None
                        ),
                        "error_type_probs": (
                            [
                                float(x)
                                for x in type_prob[j].tolist()
                            ]
                            if type_prob is not None
                            else None
                        ),
                        "input_sha256": digest(
                            [
                                row["problem"],
                                row["prefix"],
                                row["current_step"],
                            ]
                        ),
                        "truncation": batch_items[j][
                            "truncation"
                        ],
                    }
                    local_predictions.append(
                        record
                    )

        shard_path = (
            shard_dir
            / f"rank_{rank:02d}.jsonl"
        )
        write_jsonl(
            shard_path,
            local_predictions,
        )

        print(
            f"[{args.name}] rank={rank}: "
            f"wrote {len(local_predictions)} rows -> {shard_path}",
            flush=True,
        )

        # Release GPU memory before waiting for merge / before next model launch.
        del model
        torch.cuda.empty_cache()

        barrier(world_size)

        if rank == 0:
            predictions = merge_rank_shards(
                output=output,
                shard_dir=shard_dir,
                world_size=world_size,
                total_rows=len(rows),
            )

            meta = {
                "schema": "mrben_three_model_prediction_v3_strict_role3",
                "runtime": {"torch": torch.__version__, "transformers": __import__("transformers").__version__, "dtype": args.dtype},
                "loading_audit": loading_audit,
                "checkpoint_audit": checkpoint_audit,
                "name": args.name,
                "kind": args.kind,
                "split": reference_split,
                "model_path": str(
                    Path(args.model_path)
                ),
                "reference": str(
                    reference.resolve()
                ),
                "reference_sha256": (
                    ref_meta["reference_sha256"]
                ),
                "predictions_sha256": (
                    file_hash(output)
                ),
                "active_heads": active_heads,
                "distributed_inference": {
                    "world_size": world_size,
                    "sharding": (
                        "global_row_index % world_size == rank"
                    ),
                    "merged_in_original_reference_order": True,
                },
                "packing": {
                    "format": (
                        "problem + SEP + recent_complete_prefix "
                        "+ current_step + SEP"
                    ),
                    "target": (
                        "last token of trailing separator"
                    ),
                    "max_length": args.max_length,
                    "step_separator": args.step_separator,
                    "truncation": (
                        "drop oldest complete prefix first; "
                        "preserve current step"
                    ),
                },
                "tokenizer_path": str(
                    Path(args.tokenizer_path)
                ),
                "tokenizer_sha256": tok_sha,
                "role_order": ROLES,
                "error_type_order": ERROR_TYPES,
                "baseline_classifier": (
                    {
                        "num_labels": int(
                            model_labels
                            and len(model_labels)
                            or (
                                3
                                if args.kind == "three_class"
                                else 2
                            )
                        ),
                        "id2label": model_labels,
                        "error_class_id": (
                            int(error_class_id)
                        ),
                        "p_error_rule": (
                            "softmax(ERROR)"
                        ),
                    }
                    if args.kind != "full_hierarchy"
                    else None
                ),
                "full_hierarchy_probability_rule": (
                    {
                        "p_error": (
                            "P(ERROR) from correctness head"
                        ),
                        "role_probs": "softmax(role_head): correct,new_error,propagated_error",
                    }
                    if args.kind == "full_hierarchy"
                    else None
                ),
                "truncation_step_counts": {
                    key: sum(
                        int(
                            record["truncation"][key]
                            > 0
                        )
                        for record in predictions
                    )
                    for key in TRUNC
                },
            }

            write_json(
                output_meta,
                meta,
            )

            if not args.keep_shards:
                shutil.rmtree(
                    shard_dir,
                    ignore_errors=True,
                )

            print("=" * 100)
            print(
                f"Distributed prediction finished: {args.name}"
            )
            print(
                f"world_size : {world_size}"
            )
            print(
                f"rows       : {len(predictions)}"
            )
            print(
                f"output     : {output}"
            )
            print(
                f"metadata   : {output_meta}"
            )
            print("=" * 100)

        barrier(world_size)

    finally:
        cleanup(world_size)


if __name__ == "__main__":
    main()
