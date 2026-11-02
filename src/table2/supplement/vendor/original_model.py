#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hierarchical multi-task PRM training for PRM800K phase-2 data.

Data:
    prm800k_phase2_hierarchical_prm_flat_train.jsonl
    prm800k_phase2_hierarchical_prm_flat_validation.jsonl
    prm800k_phase2_hierarchical_prm_flat_test.jsonl (optional held-out test)

Three token-level heads share one Qwen3 backbone:
    1) correctness:
         0 = ERROR
         1 = NON_ERROR
       A two-class or three-class PRM checkpoint can initialize this head.
       For a three-class ERROR / NEUTRAL / CORRECT source, the ERROR and
       CORRECT rows are copied to ERROR / NON_ERROR; NEUTRAL is intentionally
       not copied.
       The prepared dataset's bool field `labels[0]` is used as source of truth:
         False -> 0 (ERROR)
         True  -> 1 (NON_ERROR)

    2) role:
         0 = correct
         1 = new_error
         2 = propagated_error
       Source `cessation` was already folded into correct during preprocessing.

    3) error_type (ONLY supervised for new_error with a known concrete type):
         0 = calculation
         1 = derivation_or_logic
         2 = method_or_formula
         3 = problem_understanding
         4 = assumption_or_condition
         5 = answer_mismatch
         6 = missing_evidence
         7 = unit
       All non-new-error rows and new_error rows with source type other/null
       use IGNORE_INDEX=-100. Those untyped new_error rows still supervise the
       correctness and role heads.

Design choices:
- one supervised token per current Step: the LAST token of the trailing step separator;
- preserve the full current candidate whenever possible;
- when max_length is exceeded, drop the OLDEST prefix Steps first;
- hierarchical multi-task loss:
      L = 1.0 * L_correctness
        + 0.5 * L_role
        + 0.5 * L_error_type
- error_type loss is safely skipped for batches containing no new_error rows;
- class imbalance uses capped sqrt-inverse-frequency weights by default;
- can initialize the backbone + correctness head from an existing two-class or
  three-class PRM checkpoint.

This script uses transformers.Trainer and is compatible with Accelerate + DeepSpeed ZeRO-2.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import (
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import ModelOutput


IGNORE_INDEX = -100

CORRECTNESS_ID2LABEL = {
    0: "ERROR",
    1: "NON_ERROR",
}
CORRECTNESS_LABEL2ID = {
    v: k for k, v in CORRECTNESS_ID2LABEL.items()
}

ROLE_ID2LABEL = {
    0: "correct",
    1: "new_error",
    2: "propagated_error",
}
ROLE_LABEL2ID = {
    v: k for k, v in ROLE_ID2LABEL.items()
}

ERROR_TYPE_ID2LABEL = {
    0: "calculation",
    1: "derivation_or_logic",
    2: "method_or_formula",
    3: "problem_understanding",
    4: "assumption_or_condition",
    5: "answer_mismatch",
    6: "missing_evidence",
    7: "unit",
}
ERROR_TYPE_LABEL2ID = {
    v: k for k, v in ERROR_TYPE_ID2LABEL.items()
}


# =============================================================================
# Arguments
# =============================================================================

@dataclass
class DataArguments:
    train_file: str = field(
        metadata={
            "help": "Flat hierarchical PRM training JSONL."
        }
    )
    validation_file: str = field(
        metadata={
            "help": "Flat hierarchical PRM validation JSONL."
        }
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional held-out flat hierarchical PRM test JSONL."
        },
    )
    max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum token length."
        },
    )
    step_separator: str = field(
        default="\n\n",
        metadata={
            "help": "Separator placed after problem/prefix/current Step."
        },
    )
    dataset_num_proc: int = field(
        default=8,
        metadata={
            "help": "Number of processes for dataset tokenization."
        },
    )
    fix_mistral_regex: bool = field(
        default=True,
        metadata={
            "help": "Pass fix_mistral_regex=True when loading tokenizer."
        },
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": (
                "Existing two-class or three-class PRM checkpoint. "
                "Recommended for this run: "
                "Qwen3-4B-base-PRM800K-final-three-class-v2."
            )
        }
    )
    init_from_prm: bool = field(
        default=True,
        metadata={
            "help": (
                "Load an existing AutoModelForTokenClassification checkpoint, "
                "reuse its backbone, and initialize the two-class correctness "
                "head from a two-class head or from the ERROR/CORRECT rows of "
                "a three-class head."
            )
        },
    )
    trust_remote_code: bool = field(
        default=True,
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
    )


@dataclass
class MultiTaskArguments:
    correctness_loss_weight: float = field(
        default=1.0
    )
    role_loss_weight: float = field(
        default=0.5
    )
    error_type_loss_weight: float = field(
        default=0.5
    )

    use_class_weights: bool = field(
        default=True,
        metadata={
            "help": "Enable class-weighted CE for all three heads."
        },
    )
    class_weight_mode: str = field(
        default="sqrt_inverse",
        metadata={
            "help": "none | inverse | sqrt_inverse"
        },
    )
    max_class_weight: float = field(
        default=4.0,
        metadata={
            "help": (
                "Cap normalized class weights. "
                "Useful because `unit` is extremely rare."
            )
        },
    )
    classifier_dropout: float = field(
        default=0.1,
    )


# =============================================================================
# Model output
# =============================================================================

@dataclass
class HierarchicalPRMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    correctness_logits: Optional[torch.Tensor] = None
    role_logits: Optional[torch.Tensor] = None
    error_type_logits: Optional[torch.Tensor] = None


# =============================================================================
# Hierarchical model
# =============================================================================

class HierarchicalPRMModel(PreTrainedModel):
    """
    Shared decoder backbone + 3 independent token-classification heads.

    Saved checkpoints can be reloaded with:
        HierarchicalPRMModel.from_pretrained(path)
    as long as this class definition is available.
    """

    supports_gradient_checkpointing = True
    # This wrapper contains no attention layer of its own; attention is delegated
    # to the Qwen backbone. Declaring SDPA support prevents Transformers from
    # rejecting a config that was loaded with attn_implementation="sdpa".
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

        self.config.correctness_id2label = CORRECTNESS_ID2LABEL
        self.config.correctness_label2id = CORRECTNESS_LABEL2ID
        self.config.role_id2label = ROLE_ID2LABEL
        self.config.role_label2id = ROLE_LABEL2ID
        self.config.error_type_id2label = ERROR_TYPE_ID2LABEL
        self.config.error_type_label2id = ERROR_TYPE_LABEL2ID

        hidden_size = int(
            getattr(config, "hidden_size")
        )

        if backbone is None:
            self.backbone = AutoModel.from_config(
                config,
                trust_remote_code=True,
            )
        else:
            self.backbone = backbone

        if classifier_dropout is None:
            classifier_dropout = getattr(
                config,
                "classifier_dropout",
                0.1,
            )
            if classifier_dropout is None:
                classifier_dropout = 0.1

        self.config.hierarchical_prm_classifier_dropout = float(
            classifier_dropout
        )

        self.dropout = nn.Dropout(
            float(classifier_dropout)
        )

        self.correctness_head = nn.Linear(
            hidden_size,
            2,
            bias=True,
        )
        self.role_head = nn.Linear(
            hidden_size,
            3,
            bias=True,
        )
        self.error_type_head = nn.Linear(
            hidden_size,
            8,
            bias=True,
        )

        self._init_head(
            self.correctness_head
        )
        self._init_head(
            self.role_head
        )
        self._init_head(
            self.error_type_head
        )

    def _init_head(self, head: nn.Linear):
        std = float(
            getattr(
                self.config,
                "initializer_range",
                0.02,
            )
        )
        nn.init.normal_(
            head.weight,
            mean=0.0,
            std=std,
        )
        if head.bias is not None:
            nn.init.zeros_(
                head.bias
            )

    @classmethod
    def from_source_checkpoint(
        cls,
        model_name_or_path: str,
        init_from_prm: bool,
        classifier_dropout: float,
        trust_remote_code: bool = True,
        attn_implementation: Optional[str] = "sdpa",
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
    ):
        common_kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
        }

        if torch_dtype is not None:
            common_kwargs["dtype"] = torch_dtype

        if attn_implementation:
            common_kwargs["attn_implementation"] = (
                attn_implementation
            )

        if init_from_prm:
            print(
                "Initializing from PRM token-classification checkpoint:"
            )
            print(
                f"  {model_name_or_path}"
            )

            source = (
                AutoModelForTokenClassification
                .from_pretrained(
                    model_name_or_path,
                    **common_kwargs,
                )
            )

            config = source.config

            # PreTrainedModel.base_model resolves to the underlying Qwen3 model.
            backbone = source.base_model

            model = cls(
                config=config,
                backbone=backbone,
                classifier_dropout=classifier_dropout,
            )

            source_head = None
            source_head_name = None

            # Qwen-style token classifiers commonly use `score`; other model
            # families may expose the same layer as `classifier`.
            for attr_name in (
                "score",
                "classifier",
            ):
                old_head = getattr(
                    source,
                    attr_name,
                    None,
                )

                if isinstance(old_head, nn.Linear):
                    source_head = old_head
                    source_head_name = attr_name
                    break

            if source_head is None:
                raise RuntimeError(
                    "Cannot find a linear PRM classifier head named `score` "
                    "or `classifier`; refusing to continue with a randomly "
                    "initialized correctness head."
                )

            if source_head.in_features != model.correctness_head.in_features:
                raise RuntimeError(
                    "Source classifier hidden size does not match the new "
                    "correctness head: "
                    f"source={tuple(source_head.weight.shape)}, "
                    f"target={tuple(model.correctness_head.weight.shape)}"
                )

            source_num_labels = int(source_head.out_features)
            raw_id2label = getattr(source.config, "id2label", {}) or {}
            source_id2label = {
                int(class_id): str(label).strip().upper()
                for class_id, label in raw_id2label.items()
            }

            def resolve_source_id(
                accepted_names: Tuple[str, ...],
                fallback_id: int,
            ) -> int:
                matches = [
                    class_id
                    for class_id, label in source_id2label.items()
                    if label in accepted_names
                ]
                if len(matches) == 1:
                    return matches[0]
                if not matches and 0 <= fallback_id < source_num_labels:
                    print(
                        "WARNING: could not resolve labels "
                        f"{accepted_names} from id2label={source_id2label}; "
                        f"using fallback class ID {fallback_id}."
                    )
                    return fallback_id
                raise RuntimeError(
                    f"Cannot uniquely resolve {accepted_names} from "
                    f"id2label={source_id2label}."
                )

            if source_num_labels == 2:
                # Expected two-class semantics: ERROR / NON_ERROR.  Also accept
                # CORRECT as the positive name for compatibility.
                error_source_id = resolve_source_id(
                    ("ERROR",),
                    fallback_id=0,
                )
                correct_source_id = resolve_source_id(
                    ("NON_ERROR", "CORRECT"),
                    fallback_id=1,
                )
                source_mode = "two_class"
            elif source_num_labels == 3:
                # Three-class checkpoint semantics:
                #   0=ERROR, 1=NEUTRAL, 2=CORRECT (fallback ordering).
                # The hierarchical correctness target is binary, so NEUTRAL is
                # deliberately excluded rather than averaged into NON_ERROR.
                error_source_id = resolve_source_id(
                    ("ERROR",),
                    fallback_id=0,
                )
                correct_source_id = resolve_source_id(
                    ("CORRECT", "NON_ERROR"),
                    fallback_id=2,
                )
                source_mode = "three_class_extremes"
            else:
                raise RuntimeError(
                    "PRM initialization supports only two-class or three-class "
                    f"classification heads; got {source_num_labels} labels."
                )

            if error_source_id == correct_source_id:
                raise RuntimeError(
                    "ERROR and CORRECT/NON_ERROR resolved to the same source "
                    f"class ID {error_source_id}."
                )

            with torch.no_grad():
                model.correctness_head.weight[0].copy_(
                    source_head.weight[error_source_id]
                )
                model.correctness_head.weight[1].copy_(
                    source_head.weight[correct_source_id]
                )

                if model.correctness_head.bias is not None:
                    if source_head.bias is not None:
                        model.correctness_head.bias[0].copy_(
                            source_head.bias[error_source_id]
                        )
                        model.correctness_head.bias[1].copy_(
                            source_head.bias[correct_source_id]
                        )
                    else:
                        model.correctness_head.bias.zero_()

            model.config.hierarchical_prm_source_head = source_head_name
            model.config.hierarchical_prm_source_num_labels = source_num_labels
            model.config.hierarchical_prm_source_id2label = source_id2label
            model.config.hierarchical_prm_correctness_init = {
                "target_ERROR": error_source_id,
                "target_NON_ERROR": correct_source_id,
                "source_mode": source_mode,
                "neutral_policy": (
                    "excluded"
                    if source_num_labels == 3
                    else "not_applicable"
                ),
            }

            print(
                "Copied PRM classifier rows -> correctness_head "
                f"from `{source_head_name}`: "
                f"source[{error_source_id}]={source_id2label.get(error_source_id)} "
                "-> target[0]=ERROR; "
                f"source[{correct_source_id}]={source_id2label.get(correct_source_id)} "
                "-> target[1]=NON_ERROR."
            )
            if source_num_labels == 3:
                print(
                    "Three-class initialization: the NEUTRAL classifier row "
                    "is intentionally excluded; the shared backbone is fully reused."
                )

            # source still references backbone. Removing the wrapper is safe because
            # model.backbone owns a reference to the same module.
            del source

            return model

        print(
            "Initializing from causal/base backbone:"
        )
        print(
            f"  {model_name_or_path}"
        )

        backbone = AutoModel.from_pretrained(
            model_name_or_path,
            **common_kwargs,
        )

        return cls(
            config=backbone.config,
            backbone=backbone,
            classifier_dropout=classifier_dropout,
        )

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.backbone.set_input_embeddings(
            value
        )

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs=None,
    ):
        if hasattr(
            self.backbone,
            "gradient_checkpointing_enable",
        ):
            return (
                self.backbone
                .gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=(
                        gradient_checkpointing_kwargs
                    )
                )
            )

    def gradient_checkpointing_disable(self):
        if hasattr(
            self.backbone,
            "gradient_checkpointing_disable",
        ):
            return (
                self.backbone
                .gradient_checkpointing_disable()
            )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        correctness_labels=None,
        role_labels=None,
        error_type_labels=None,
        **kwargs,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            **kwargs,
        )

        hidden = self.dropout(
            outputs.last_hidden_state
        )

        correctness_logits = (
            self.correctness_head(
                hidden
            )
        )
        role_logits = self.role_head(
            hidden
        )
        error_type_logits = (
            self.error_type_head(
                hidden
            )
        )

        return HierarchicalPRMOutput(
            correctness_logits=correctness_logits,
            role_logits=role_logits,
            error_type_logits=error_type_logits,
        )


# =============================================================================
# Dataset encoding
# =============================================================================

def _ensure_one(
    row: Dict[str, Any],
    key: str,
):
    value = row.get(key)

    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(
            f"Expected `{key}` to contain exactly one item; "
            f"got {value!r}"
        )

    return value[0]


def encode_row(
    row: Dict[str, Any],
    tokenizer,
    max_length: int,
    separator: str,
):
    """
    Build:
        problem + SEP + recent_prefix + current_step + SEP

    Supervise ONLY the final token of the current Step separator.

    Truncation policy:
      1) current candidate is highest priority;
      2) keep beginning of problem as much as possible;
      3) keep most RECENT complete prefix Steps;
      4) drop OLDEST prefix Steps first.

    Correctness mapping intentionally follows the old binary PRM:
        False -> ERROR     -> 0
        True  -> NON_ERROR -> 1
    """

    problem = str(
        row.get("problem") or ""
    )

    prefix = row.get("prefix") or []
    if not isinstance(prefix, list):
        raise ValueError(
            "`prefix` must be a list"
        )

    candidate = str(
        _ensure_one(
            row,
            "completions",
        )
    )

    bool_label = _ensure_one(
        row,
        "labels",
    )

    if bool_label is True:
        correctness_target = 1
    elif bool_label is False:
        correctness_target = 0
    else:
        correctness_target = IGNORE_INDEX

    role_target = int(
        row.get(
            "role_id",
            IGNORE_INDEX,
        )
    )

    error_type_mask = bool(
        row.get(
            "error_type_mask",
            False,
        )
    )

    error_type_target = int(
        row.get(
            "error_type_id",
            IGNORE_INDEX,
        )
    )

    if not error_type_mask:
        error_type_target = IGNORE_INDEX

    # Hierarchy consistency check. A known concrete error type may only be
    # attached to new_error. Conversely, new_error is allowed to have
    # IGNORE_INDEX when the source type is `other` or null.
    train_role = row.get(
        "role"
    )

    if error_type_target != IGNORE_INDEX:
        if train_role != "new_error":
            raise ValueError(
                "error_type is supervised but role != new_error: "
                f"id={row.get('id')}, role={train_role}, "
                f"error_type_id={error_type_target}"
            )

    sep_ids = tokenizer.encode(
        separator,
        add_special_tokens=False,
    )

    if not sep_ids:
        raise ValueError(
            "step_separator tokenized to an empty sequence"
        )

    bos_ids = (
        [tokenizer.bos_token_id]
        if tokenizer.bos_token_id is not None
        else []
    )

    raw_problem_ids = tokenizer(
        problem,
        add_special_tokens=False,
    )["input_ids"]

    raw_candidate_ids = tokenizer(
        candidate,
        add_special_tokens=False,
    )["input_ids"]

    problem_ids = (
        raw_problem_ids
        + sep_ids
    )

    candidate_ids = (
        raw_candidate_ids
        + sep_ids
    )

    problem_truncated = 0
    candidate_truncated = 0

    minimum = (
        len(bos_ids)
        + len(problem_ids)
        + len(candidate_ids)
    )

    if minimum > max_length:
        available_problem = (
            max_length
            - len(bos_ids)
            - len(candidate_ids)
        )

        if available_problem <= len(sep_ids):
            # Current candidate itself is too long.
            # Keep candidate suffix because its terminal supervision must survive.
            available_candidate = (
                max_length
                - len(bos_ids)
                - len(sep_ids)
            )

            keep_candidate = max(
                1,
                available_candidate,
            )

            candidate_truncated = max(
                0,
                len(raw_candidate_ids)
                - keep_candidate,
            )

            raw_candidate_ids = (
                raw_candidate_ids[
                    -keep_candidate:
                ]
            )

            candidate_ids = (
                raw_candidate_ids
                + sep_ids
            )

            problem_truncated = len(
                raw_problem_ids
            )
            problem_ids = []

        else:
            keep_problem = max(
                0,
                available_problem
                - len(sep_ids),
            )

            problem_truncated = max(
                0,
                len(raw_problem_ids)
                - keep_problem,
            )

            problem_ids = (
                raw_problem_ids[
                    :keep_problem
                ]
                + sep_ids
            )

    remaining = (
        max_length
        - len(bos_ids)
        - len(problem_ids)
        - len(candidate_ids)
    )

    selected_prefix_pieces = []
    dropped_prefix_steps = 0

    for step in reversed(prefix):
        piece = (
            tokenizer(
                str(step),
                add_special_tokens=False,
            )["input_ids"]
            + sep_ids
        )

        if len(piece) <= remaining:
            selected_prefix_pieces.append(
                piece
            )
            remaining -= len(piece)
        else:
            dropped_prefix_steps += 1

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
        + candidate_ids
    )

    if len(input_ids) > max_length:
        raise RuntimeError(
            f"Encoded length {len(input_ids)} exceeds max_length={max_length}"
        )

    correctness_labels = [
        IGNORE_INDEX
    ] * len(input_ids)

    role_labels = [
        IGNORE_INDEX
    ] * len(input_ids)

    error_type_labels = [
        IGNORE_INDEX
    ] * len(input_ids)

    # Only one token is supervised: final token of current Step separator.
    target_pos = len(input_ids) - 1

    correctness_labels[
        target_pos
    ] = correctness_target

    role_labels[
        target_pos
    ] = role_target

    error_type_labels[
        target_pos
    ] = error_type_target

    return {
        "input_ids": input_ids,
        "correctness_labels": correctness_labels,
        "role_labels": role_labels,
        "error_type_labels": error_type_labels,
        "dropped_prefix_steps": dropped_prefix_steps,
        "problem_truncated_tokens": problem_truncated,
        "candidate_truncated_tokens": candidate_truncated,
    }


def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    data_args: DataArguments,
    desc: str,
) -> Dataset:
    original_columns = dataset.column_names

    encoded = dataset.map(
        encode_row,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": data_args.max_length,
            "separator": data_args.step_separator,
        },
        num_proc=data_args.dataset_num_proc,
        remove_columns=original_columns,
        desc=desc,
    )

    dropped_prefix = encoded[
        "dropped_prefix_steps"
    ]
    problem_trunc = encoded[
        "problem_truncated_tokens"
    ]
    candidate_trunc = encoded[
        "candidate_truncated_tokens"
    ]

    rows_trimmed = sum(
        x > 0
        for x in dropped_prefix
    )
    total_dropped = sum(
        dropped_prefix
    )

    rows_problem_trunc = sum(
        x > 0
        for x in problem_trunc
    )
    rows_candidate_trunc = sum(
        x > 0
        for x in candidate_trunc
    )

    print(
        f"{desc}: {len(encoded):,} rows"
    )
    print(
        f"  prefix trimmed rows : "
        f"{rows_trimmed:,} "
        f"({rows_trimmed/max(1,len(encoded)):.2%})"
    )
    print(
        f"  dropped prefix Steps: "
        f"{total_dropped:,}"
    )
    print(
        f"  problem truncated   : "
        f"{rows_problem_trunc:,} "
        f"({rows_problem_trunc/max(1,len(encoded)):.2%})"
    )
    print(
        f"  candidate truncated : "
        f"{rows_candidate_trunc:,} "
        f"({rows_candidate_trunc/max(1,len(encoded)):.2%})"
    )

    return encoded.remove_columns(
        [
            "dropped_prefix_steps",
            "problem_truncated_tokens",
            "candidate_truncated_tokens",
        ]
    )


# =============================================================================
# Collator
# =============================================================================

@dataclass
class HierarchicalDataCollator:
    tokenizer: Any
    label_pad_token_id: int = IGNORE_INDEX

    def __call__(
        self,
        features: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:

        label_names = [
            "correctness_labels",
            "role_labels",
            "error_type_labels",
        ]

        input_features = [
            {
                "input_ids": f["input_ids"],
            }
            for f in features
        ]

        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        max_len = batch[
            "input_ids"
        ].shape[1]

        padding_side = (
            self.tokenizer.padding_side
        )

        for label_name in label_names:
            padded_labels = []

            for f in features:
                values = list(
                    f[label_name]
                )

                pad_len = (
                    max_len
                    - len(values)
                )

                if padding_side == "right":
                    values = (
                        values
                        + [
                            self.label_pad_token_id
                        ] * pad_len
                    )
                else:
                    values = (
                        [
                            self.label_pad_token_id
                        ] * pad_len
                        + values
                    )

                padded_labels.append(
                    values
                )

            batch[label_name] = torch.tensor(
                padded_labels,
                dtype=torch.long,
            )

        return batch


# =============================================================================
# Class weights
# =============================================================================

def _normalized_class_weights(
    counts: List[int],
    mode: str,
    cap: float,
) -> Optional[List[float]]:

    mode = str(mode).lower()

    if mode == "none":
        return None

    if mode not in {
        "inverse",
        "sqrt_inverse",
    }:
        raise ValueError(
            f"Unknown class_weight_mode={mode}"
        )

    if any(
        c <= 0
        for c in counts
    ):
        print(
            "WARNING: at least one class has zero training samples; "
            f"counts={counts}. Class weights disabled for this head."
        )
        return None

    total = float(
        sum(counts)
    )
    k = float(
        len(counts)
    )

    raw = []

    for count in counts:
        base = (
            total
            / (k * float(count))
        )

        if mode == "sqrt_inverse":
            base = math.sqrt(
                base
            )

        raw.append(
            base
        )

    # Normalize arithmetic mean to 1.
    mean_weight = (
        sum(raw)
        / len(raw)
    )

    weights = [
        x / mean_weight
        for x in raw
    ]

    if cap is not None and cap > 0:
        weights = [
            min(
                x,
                float(cap),
            )
            for x in weights
        ]

    return weights


def get_training_distributions(
    raw_train: Dataset,
    multitask_args: MultiTaskArguments,
):
    correctness_counts = Counter()
    role_counts = Counter()
    error_type_counts = Counter()
    unsupervised_new_error_types = Counter()

    for row in raw_train:
        label = _ensure_one(
            row,
            "labels",
        )

        if label is True:
            correctness_counts[1] += 1
        elif label is False:
            correctness_counts[0] += 1

        role_id = int(
            row.get(
                "role_id",
                IGNORE_INDEX,
            )
        )
        if role_id != IGNORE_INDEX:
            role_counts[
                role_id
            ] += 1

        has_error_type_target = bool(
            row.get(
                "error_type_mask",
                False,
            )
        )

        if has_error_type_target:
            error_type_id = int(
                row.get(
                    "error_type_id",
                    IGNORE_INDEX,
                )
            )
            if error_type_id != IGNORE_INDEX:
                error_type_counts[
                    error_type_id
                ] += 1
        elif role_id == ROLE_LABEL2ID["new_error"]:
            raw_error_type = row.get(
                "raw_error_type"
            )
            unsupervised_new_error_types[
                str(raw_error_type)
            ] += 1

    corr_counts = [
        correctness_counts.get(i, 0)
        for i in range(2)
    ]
    role_count_list = [
        role_counts.get(i, 0)
        for i in range(3)
    ]
    type_count_list = [
        error_type_counts.get(i, 0)
        for i in range(8)
    ]

    print("=" * 100)
    print("TRAINING LABEL DISTRIBUTION")
    print("=" * 100)

    print(
        "Correctness "
        "(0=ERROR, 1=NON_ERROR):"
    )
    for i, count in enumerate(
        corr_counts
    ):
        print(
            f"  {i} {CORRECTNESS_ID2LABEL[i]:20s}: "
            f"{count:,}"
        )

    print(
        "Role:"
    )
    for i, count in enumerate(
        role_count_list
    ):
        print(
            f"  {i} {ROLE_ID2LABEL[i]:20s}: "
            f"{count:,}"
        )

    print(
        "Error type "
        "(new_error only):"
    )
    for i, count in enumerate(
        type_count_list
    ):
        print(
            f"  {i} {ERROR_TYPE_ID2LABEL[i]:28s}: "
            f"{count:,}"
        )

    print(
        "Untyped new_error rows "
        "(correctness/role only; no error_type loss):"
    )
    if unsupervised_new_error_types:
        for name, count in sorted(
            unsupervised_new_error_types.items()
        ):
            print(
                f"  {name:28s}: {count:,}"
            )
    else:
        print("  none")

    weights = {
        "correctness": None,
        "role": None,
        "error_type": None,
    }

    if multitask_args.use_class_weights:
        weights["correctness"] = (
            _normalized_class_weights(
                corr_counts,
                multitask_args.class_weight_mode,
                multitask_args.max_class_weight,
            )
        )
        weights["role"] = (
            _normalized_class_weights(
                role_count_list,
                multitask_args.class_weight_mode,
                multitask_args.max_class_weight,
            )
        )
        weights["error_type"] = (
            _normalized_class_weights(
                type_count_list,
                multitask_args.class_weight_mode,
                multitask_args.max_class_weight,
            )
        )

    print(
        "Class-weight mode:",
        multitask_args.class_weight_mode
        if multitask_args.use_class_weights
        else "disabled",
    )

    for name, values in weights.items():
        if values is None:
            print(
                f"  {name}: None"
            )
        else:
            print(
                f"  {name}: "
                + ", ".join(
                    f"{x:.4f}"
                    for x in values
                )
            )

    print("=" * 100)

    distribution = {
        "counts": {
            "correctness": corr_counts,
            "role": role_count_list,
            "error_type": type_count_list,
            "unsupervised_new_error_type": dict(
                unsupervised_new_error_types
            ),
        },
        "weights": weights,
    }

    return distribution


# =============================================================================
# Metrics
# =============================================================================

def _multiclass_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: Dict[int, str],
    prefix: str,
):
    if len(y_true) == 0:
        return {
            f"{prefix}_accuracy": 0.0,
            f"{prefix}_balanced_accuracy": 0.0,
            f"{prefix}_macro_f1": 0.0,
        }

    metrics: Dict[str, float] = {}

    accuracy = float(
        (y_true == y_pred).mean()
    )

    recalls = []
    f1s = []

    for class_id in range(
        num_classes
    ):
        tp = int(
            (
                (y_true == class_id)
                & (y_pred == class_id)
            ).sum()
        )
        fp = int(
            (
                (y_true != class_id)
                & (y_pred == class_id)
            ).sum()
        )
        fn = int(
            (
                (y_true == class_id)
                & (y_pred != class_id)
            ).sum()
        )
        support = int(
            (y_true == class_id).sum()
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp)
            else 0.0
        )
        recall = (
            tp / (tp + fn)
            if (tp + fn)
            else 0.0
        )
        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        safe_name = (
            class_names[class_id]
            .replace("/", "_")
            .replace(" ", "_")
        )

        metrics[
            f"{prefix}_{safe_name}_precision"
        ] = precision
        metrics[
            f"{prefix}_{safe_name}_recall"
        ] = recall
        metrics[
            f"{prefix}_{safe_name}_f1"
        ] = f1
        metrics[
            f"{prefix}_{safe_name}_support"
        ] = float(
            support
        )

        if support > 0:
            recalls.append(
                recall
            )
            f1s.append(
                f1
            )

    metrics[
        f"{prefix}_accuracy"
    ] = accuracy

    metrics[
        f"{prefix}_balanced_accuracy"
    ] = (
        float(
            np.mean(recalls)
        )
        if recalls
        else 0.0
    )

    metrics[
        f"{prefix}_macro_f1"
    ] = (
        float(
            np.mean(f1s)
        )
        if f1s
        else 0.0
    )

    return metrics


def compute_hierarchical_metrics(
    eval_pred,
):
    predictions = (
        eval_pred.predictions
    )
    label_ids = (
        eval_pred.label_ids
    )

    if not isinstance(
        predictions,
        (tuple, list),
    ) or len(predictions) != 3:
        raise ValueError(
            "Expected three prediction tensors: "
            "correctness, role, error_type."
        )

    if not isinstance(
        label_ids,
        (tuple, list),
    ) or len(label_ids) != 3:
        raise ValueError(
            "Expected three label tensors: "
            "correctness, role, error_type."
        )

    (
        corr_logits,
        role_logits,
        type_logits,
    ) = predictions

    (
        corr_labels,
        role_labels,
        type_labels,
    ) = label_ids

    metrics: Dict[str, float] = {}

    # Correctness
    corr_mask = (
        corr_labels
        != IGNORE_INDEX
    )
    corr_true = corr_labels[
        corr_mask
    ].astype(
        np.int64
    )
    corr_pred = np.argmax(
        corr_logits,
        axis=-1,
    )[
        corr_mask
    ].astype(
        np.int64
    )

    metrics.update(
        _multiclass_stats(
            corr_true,
            corr_pred,
            num_classes=2,
            class_names=CORRECTNESS_ID2LABEL,
            prefix="correctness",
        )
    )

    # Role
    role_mask = (
        role_labels
        != IGNORE_INDEX
    )
    role_true = role_labels[
        role_mask
    ].astype(
        np.int64
    )
    role_pred = np.argmax(
        role_logits,
        axis=-1,
    )[
        role_mask
    ].astype(
        np.int64
    )

    metrics.update(
        _multiclass_stats(
            role_true,
            role_pred,
            num_classes=3,
            class_names=ROLE_ID2LABEL,
            prefix="role",
        )
    )

    # Error type: valid ONLY for new_error.
    type_mask = (
        type_labels
        != IGNORE_INDEX
    )
    type_true = type_labels[
        type_mask
    ].astype(
        np.int64
    )
    type_pred = np.argmax(
        type_logits,
        axis=-1,
    )[
        type_mask
    ].astype(
        np.int64
    )

    metrics.update(
        _multiclass_stats(
            type_true,
            type_pred,
            num_classes=8,
            class_names=ERROR_TYPE_ID2LABEL,
            prefix="error_type",
        )
    )

    return metrics


# =============================================================================
# Trainer
# =============================================================================

class HierarchicalPRMTrainer(Trainer):
    def __init__(
        self,
        *args,
        correctness_weights=None,
        role_weights=None,
        error_type_weights=None,
        correctness_loss_weight=1.0,
        role_loss_weight=0.5,
        error_type_loss_weight=0.5,
        **kwargs,
    ):
        self._correctness_weights = (
            torch.tensor(
                correctness_weights,
                dtype=torch.float32,
            )
            if correctness_weights is not None
            else None
        )
        self._role_weights = (
            torch.tensor(
                role_weights,
                dtype=torch.float32,
            )
            if role_weights is not None
            else None
        )
        self._error_type_weights = (
            torch.tensor(
                error_type_weights,
                dtype=torch.float32,
            )
            if error_type_weights is not None
            else None
        )

        self.correctness_loss_weight = float(
            correctness_loss_weight
        )
        self.role_loss_weight = float(
            role_loss_weight
        )
        self.error_type_loss_weight = float(
            error_type_loss_weight
        )

        super().__init__(
            *args,
            **kwargs,
        )

        # Critical for Trainer prediction_step / compute_metrics.
        self.label_names = [
            "correctness_labels",
            "role_labels",
            "error_type_labels",
        ]

    @staticmethod
    def _masked_ce(
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
    ):
        valid = (
            labels
            != IGNORE_INDEX
        )

        # This is essential for the error_type head:
        # many micro-batches contain zero new_error rows.
        if not bool(
            valid.any()
        ):
            return (
                logits.sum()
                * 0.0
            )

        valid_logits = logits[
            valid
        ].float()

        valid_labels = labels[
            valid
        ]

        weight = None

        if weights is not None:
            weight = weights.to(
                device=logits.device,
                dtype=torch.float32,
            )

        return F.cross_entropy(
            valid_logits,
            valid_labels,
            weight=weight,
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        correctness_labels = inputs[
            "correctness_labels"
        ]
        role_labels = inputs[
            "role_labels"
        ]
        error_type_labels = inputs[
            "error_type_labels"
        ]

        model_inputs = {
            k: v
            for k, v in inputs.items()
            if k not in {
                "correctness_labels",
                "role_labels",
                "error_type_labels",
            }
        }

        outputs = model(
            **model_inputs
        )

        loss_correctness = (
            self._masked_ce(
                outputs.correctness_logits,
                correctness_labels,
                self._correctness_weights,
            )
        )

        loss_role = self._masked_ce(
            outputs.role_logits,
            role_labels,
            self._role_weights,
        )

        loss_error_type = (
            self._masked_ce(
                outputs.error_type_logits,
                error_type_labels,
                self._error_type_weights,
            )
        )

        loss = (
            self.correctness_loss_weight
            * loss_correctness
            + self.role_loss_weight
            * loss_role
            + self.error_type_loss_weight
            * loss_error_type
        )

        outputs.loss = loss

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            MultiTaskArguments,
            TrainingArguments,
        )
    )

    (
        model_args,
        data_args,
        multitask_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    train_file = Path(
        data_args.train_file
    )
    validation_file = Path(
        data_args.validation_file
    )
    test_file = (
        Path(data_args.test_file)
        if data_args.test_file
        else None
    )

    if not train_file.exists():
        raise FileNotFoundError(
            f"Train file not found: {train_file}"
        )

    if not validation_file.exists():
        raise FileNotFoundError(
            f"Validation file not found: {validation_file}"
        )
    if test_file is not None and not test_file.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_file}"
        )

    print("=" * 100)
    print("HIERARCHICAL MULTI-TASK PRM TRAINING")
    print("=" * 100)
    print(
        f"Train file : {train_file}"
    )
    print(
        f"Val file   : {validation_file}"
    )
    print(
        f"Test file  : {test_file if test_file is not None else 'disabled'}"
    )
    print(
        f"Init model : {model_args.model_name_or_path}"
    )
    print(
        f"Init from PRM classifier: "
        f"{model_args.init_from_prm}"
    )
    print(
        f"Loss weights: correctness="
        f"{multitask_args.correctness_loss_weight}, "
        f"role={multitask_args.role_loss_weight}, "
        f"error_type={multitask_args.error_type_loss_weight}"
    )
    print("=" * 100)

    tokenizer_kwargs = {
        "use_fast": True,
        "trust_remote_code": model_args.trust_remote_code,
    }

    if data_args.fix_mistral_regex:
        tokenizer_kwargs[
            "fix_mistral_regex"
        ] = True

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_args.model_name_or_path,
            **tokenizer_kwargs,
        )
    )

    tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print(
        f"Tokenizer EOS: "
        f"{tokenizer.eos_token!r} "
        f"({tokenizer.eos_token_id})"
    )
    print(
        f"Tokenizer PAD: "
        f"{tokenizer.pad_token!r} "
        f"({tokenizer.pad_token_id})"
    )

    data_files = {
        "train": str(train_file),
        "validation": str(validation_file),
    }
    if test_file is not None:
        data_files["test"] = str(test_file)

    dataset = load_dataset(
        "json",
        data_files=data_files,
    )

    raw_train = dataset[
        "train"
    ]
    raw_eval = dataset[
        "validation"
    ]
    raw_test = (
        dataset["test"]
        if test_file is not None
        else None
    )

    # Hard validation of the flat format.
    raw_train = raw_train.filter(
        lambda x: (
            isinstance(
                x["completions"],
                list,
            )
            and len(
                x["completions"]
            ) == 1
            and isinstance(
                x["labels"],
                list,
            )
            and len(
                x["labels"]
            ) == 1
        ),
        desc="Validating train flat rows",
    )

    raw_eval = raw_eval.filter(
        lambda x: (
            isinstance(
                x["completions"],
                list,
            )
            and len(
                x["completions"]
            ) == 1
            and isinstance(
                x["labels"],
                list,
            )
            and len(
                x["labels"]
            ) == 1
        ),
        desc="Validating val flat rows",
    )

    if raw_test is not None:
        raw_test = raw_test.filter(
            lambda x: (
                isinstance(
                    x["completions"],
                    list,
                )
                and len(
                    x["completions"]
                ) == 1
                and isinstance(
                    x["labels"],
                    list,
                )
                and len(
                    x["labels"]
                ) == 1
            ),
            desc="Validating test flat rows",
        )

    print(
        f"Raw train rows: "
        f"{len(raw_train):,}"
    )
    print(
        f"Raw val rows  : "
        f"{len(raw_eval):,}"
    )
    if raw_test is not None:
        print(
            f"Raw test rows : "
            f"{len(raw_test):,}"
        )

    distribution = (
        get_training_distributions(
            raw_train,
            multitask_args,
        )
    )

    train_dataset = tokenize_dataset(
        raw_train,
        tokenizer,
        data_args,
        desc="Tokenizing train",
    )

    eval_dataset = tokenize_dataset(
        raw_eval,
        tokenizer,
        data_args,
        desc="Tokenizing validation",
    )

    test_dataset = (
        tokenize_dataset(
            raw_test,
            tokenizer,
            data_args,
            desc="Tokenizing held-out test",
        )
        if raw_test is not None
        else None
    )

    # Model is loaded AFTER tokenization statistics to avoid holding a 4B model
    # while CPU tokenization workers are busy.
    dtype = (
        torch.bfloat16
        if training_args.bf16
        else (
            torch.float16
            if training_args.fp16
            else None
        )
    )

    model = (
        HierarchicalPRMModel
        .from_source_checkpoint(
            model_name_or_path=(
                model_args.model_name_or_path
            ),
            init_from_prm=(
                model_args.init_from_prm
            ),
            classifier_dropout=(
                multitask_args.classifier_dropout
            ),
            trust_remote_code=(
                model_args.trust_remote_code
            ),
            attn_implementation=(
                model_args.attn_implementation
            ),
            torch_dtype=dtype,
        )
    )

    model.config.use_cache = not (
        training_args.gradient_checkpointing
    )

    data_collator = (
        HierarchicalDataCollator(
            tokenizer=tokenizer
        )
    )

    trainer = HierarchicalPRMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_hierarchical_metrics,
        correctness_weights=(
            distribution["weights"][
                "correctness"
            ]
        ),
        role_weights=(
            distribution["weights"][
                "role"
            ]
        ),
        error_type_weights=(
            distribution["weights"][
                "error_type"
            ]
        ),
        correctness_loss_weight=(
            multitask_args
            .correctness_loss_weight
        ),
        role_loss_weight=(
            multitask_args
            .role_loss_weight
        ),
        error_type_loss_weight=(
            multitask_args
            .error_type_loss_weight
        ),
    )

    train_result = trainer.train()

    trainer.log_metrics(
        "train",
        train_result.metrics,
    )
    trainer.save_metrics(
        "train",
        train_result.metrics,
    )

    eval_metrics = trainer.evaluate()

    trainer.log_metrics(
        "eval",
        eval_metrics,
    )
    trainer.save_metrics(
        "eval",
        eval_metrics,
    )

    test_metrics = {}
    if test_dataset is not None:
        test_metrics = trainer.evaluate(
            eval_dataset=test_dataset,
            metric_key_prefix="test",
        )
        trainer.log_metrics(
            "test",
            test_metrics,
        )
        trainer.save_metrics(
            "test",
            test_metrics,
        )

    trainer.save_model(
        training_args.output_dir
    )

    if trainer.is_world_process_zero():
        output_dir = Path(
            training_args.output_dir
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        meta = {
            "model_type": "hierarchical_multitask_prm",
            "source_checkpoint": model_args.model_name_or_path,
            "init_from_prm": model_args.init_from_prm,
            "source_prm_initialization": {
                "source_head": getattr(
                    trainer.model.config,
                    "hierarchical_prm_source_head",
                    None,
                ),
                "source_num_labels": getattr(
                    trainer.model.config,
                    "hierarchical_prm_source_num_labels",
                    None,
                ),
                "source_id2label": getattr(
                    trainer.model.config,
                    "hierarchical_prm_source_id2label",
                    None,
                ),
                "correctness_mapping": getattr(
                    trainer.model.config,
                    "hierarchical_prm_correctness_init",
                    None,
                ),
            },
            "data": {
                "train_file": str(train_file),
                "validation_file": str(validation_file),
                "test_file": (
                    str(test_file)
                    if test_file is not None
                    else None
                ),
                "max_length": data_args.max_length,
                "step_separator": data_args.step_separator,
            },
            "label_maps": {
                "correctness": {
                    "id2label": CORRECTNESS_ID2LABEL,
                    "note": (
                        "0=ERROR, 1=NON_ERROR; "
                        "this matches PRM800K phase-2 hierarchical preprocessing."
                    ),
                },
                "role": {
                    "id2label": ROLE_ID2LABEL,
                },
                "error_type": {
                    "id2label": ERROR_TYPE_ID2LABEL,
                    "supervision": (
                        "new_error with known concrete type only; "
                        "other/null uses IGNORE_INDEX"
                    ),
                },
            },
            "loss": {
                "correctness_weight": multitask_args.correctness_loss_weight,
                "role_weight": multitask_args.role_loss_weight,
                "error_type_weight": multitask_args.error_type_loss_weight,
                "class_weight_mode": (
                    multitask_args.class_weight_mode
                    if multitask_args.use_class_weights
                    else "none"
                ),
                "max_class_weight": multitask_args.max_class_weight,
            },
            "training_distribution": distribution,
            "eval_metrics": {
                k: float(v)
                if isinstance(
                    v,
                    (int, float, np.floating),
                )
                else v
                for k, v in eval_metrics.items()
            },
            "test_metrics": {
                k: float(v)
                if isinstance(
                    v,
                    (int, float, np.floating),
                )
                else v
                for k, v in test_metrics.items()
            },
        }

        with (
            output_dir
            / "hierarchical_prm_meta.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                meta,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            "Saved metadata:"
        )
        print(
            output_dir
            / "hierarchical_prm_meta.json"
        )

    if trainer.is_world_process_zero():
        print("=" * 100)
        print("TRAINING FINISHED")
        print("=" * 100)


if __name__ == "__main__":
    main()
