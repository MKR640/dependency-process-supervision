import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.utils import ModelOutput

from .base_prm import BasePRM


@dataclass
class HierarchicalPRMOutput(ModelOutput):
    """Outputs of the hierarchical multi-task PRM."""
    correctness_logits: torch.FloatTensor = None
    role_logits: torch.FloatTensor = None
    error_type_logits: torch.FloatTensor = None
    hidden_states: tuple = None
    attentions: tuple = None


class HierarchicalPRMModel(PreTrainedModel):
    """
    Checkpoint-compatible hierarchical PRM.

    The supplied checkpoint stores parameters as:
        backbone.*
        correctness_head.*
        role_head.*
        error_type_head.*

    Head semantics from hierarchical_prm_meta.json:
        correctness: 0=ERROR, 1=NON_ERROR
        role:        0=correct, 1=new_error, 2=propagated_error
        error_type:  8-way auxiliary classification
    """

    base_model_prefix = "backbone"
    _no_split_modules = ["Qwen3DecoderLayer"]

    def __init__(
        self,
        config,
        classifier_dropout=None,
    ):
        super().__init__(config)

        self.backbone = AutoModel.from_config(
            config,
            trust_remote_code=True,
        )

        hidden_size = int(
            getattr(config, "hidden_size")
        )

        if classifier_dropout is None:
            classifier_dropout = getattr(
                config,
                "hierarchical_prm_classifier_dropout",
                None,
            )

        if classifier_dropout is None:
            classifier_dropout = getattr(
                config,
                "classifier_dropout",
                0.1,
            )

        if classifier_dropout is None:
            classifier_dropout = 0.1

        self.dropout = torch.nn.Dropout(
            float(classifier_dropout)
        )

        self.correctness_head = torch.nn.Linear(
            hidden_size,
            2,
            bias=True,
        )

        self.role_head = torch.nn.Linear(
            hidden_size,
            3,
            bias=True,
        )

        self.error_type_head = torch.nn.Linear(
            hidden_size,
            8,
            bias=True,
        )

        # from_pretrained loads the real checkpoint after initialization.
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.backbone.set_input_embeddings(value)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=False,
        )

        hidden = self.dropout(
            outputs.last_hidden_state
        )

        return HierarchicalPRMOutput(
            correctness_logits=self.correctness_head(hidden),
            role_logits=self.role_head(hidden),
            error_type_logits=self.error_type_head(hidden),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )


class Qwen3MathPRM(BasePRM):
    """
    ProcessBench evaluator for the Qwen3 binary token-classification PRM.

    This implementation matches the training format of:

        Qwen3-4B-base-PRM800K-final-fixed

    Training format:

        BOS
        + problem
        + STEP_SEPARATOR
        + prefix_step_1
        + STEP_SEPARATOR
        + ...
        + candidate_step
        + STEP_SEPARATOR

    The classification target is placed at the FINAL TOKEN of the
    candidate's STEP_SEPARATOR.

    Label convention:

        0 = ERROR
        1 = NON_ERROR

    Therefore:

        P(class=1) = probability that the current step is valid.

    ProcessBench prediction is the first step whose NON_ERROR
    probability is below threshold.
    """

    STEP_SEPARATOR = "\n\n"

    ERROR_LABEL = 0
    NON_ERROR_LABEL = 1

    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        threshold: float = 0.5,
        max_length: int = 4096,
        step_separator: str = "\n\n",
        error_label_ids=None,
        non_error_label_ids=None,
        hierarchical_score_mode: str = "joint_first_error",
        hierarchical_alpha: float = 0.5,
    ):
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            threshold=threshold,
            max_length=max_length,
        )

        self.step_separator = step_separator
        self.sep_ids = None
        self.input_device = None

        # Multi-class checkpoints are supported by mapping their token-classification
        # probabilities to a single ProcessBench P(NON_ERROR) score.
        #
        # If neither mapping is given:
        #   * 2 labels -> preserve the original convention: class 1 = NON_ERROR.
        #   * >2 labels -> try to infer the mapping from config.id2label.
        #                  If the labels are generic (LABEL_0, ...), fail loudly
        #                  instead of silently evaluating with the wrong semantics.
        self.error_label_ids = (
            list(error_label_ids)
            if error_label_ids is not None
            else None
        )
        self.non_error_label_ids = (
            list(non_error_label_ids)
            if non_error_label_ids is not None
            else None
        )
        self.score_mode = None
        self.score_label_ids = None

        self.hierarchical_score_mode = hierarchical_score_mode
        self.hierarchical_alpha = float(hierarchical_alpha)
        if not 0.0 <= self.hierarchical_alpha <= 1.0:
            raise ValueError("hierarchical_alpha must be in [0, 1].")
        self._last_forward_aux = None
        self.is_hierarchical = False
        self.hierarchical_meta = {}
        self.correctness_error_id = 0
        self.correctness_non_error_id = 1
        self.role_correct_id = 0
        self.role_new_error_id = 1
        self.role_propagated_error_id = 2
        self.loading_info = {}

    # ============================================================
    # Model / tokenizer loading
    # ============================================================

    def _load_tokenizer(self):

        kwargs = {
            "trust_remote_code": True,
            "use_fast": True,
        }

        # Keep tokenizer behavior aligned with your training script.
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                fix_mistral_regex=True,
                **kwargs,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **kwargs,
            )

        if tokenizer.pad_token_id is None:

            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                raise ValueError(
                    "Tokenizer has neither pad_token nor eos_token."
                )

        return tokenizer

    def _load_hierarchical_meta(self):
        meta_path = (
            Path(self.model_name)
            / "hierarchical_prm_meta.json"
        )

        if not meta_path.is_file():
            raise FileNotFoundError(
                "Hierarchical checkpoint is missing "
                f"{meta_path}. This file is required to validate "
                "head semantics."
            )

        with meta_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            meta = json.load(f)

        label_maps = meta.get(
            "label_maps",
            {},
        )

        correctness = (
            label_maps
            .get("correctness", {})
            .get("id2label", {})
        )

        role = (
            label_maps
            .get("role", {})
            .get("id2label", {})
        )

        def invert(mapping):
            return {
                str(value).strip().lower():
                    int(key)
                for key, value
                in mapping.items()
            }

        correctness_inv = invert(
            correctness
        )

        role_inv = invert(
            role
        )

        required_correctness = {
            "error",
            "non_error",
        }

        required_role = {
            "correct",
            "new_error",
            "propagated_error",
        }

        if not required_correctness.issubset(
            correctness_inv
        ):
            raise ValueError(
                "Unexpected correctness label map in "
                f"{meta_path}: {correctness}"
            )

        if not required_role.issubset(
            role_inv
        ):
            raise ValueError(
                "Unexpected role label map in "
                f"{meta_path}: {role}"
            )

        self.correctness_error_id = (
            correctness_inv["error"]
        )

        self.correctness_non_error_id = (
            correctness_inv["non_error"]
        )

        self.role_correct_id = (
            role_inv["correct"]
        )

        self.role_new_error_id = (
            role_inv["new_error"]
        )

        self.role_propagated_error_id = (
            role_inv["propagated_error"]
        )

        return meta

    def _validate_hierarchical_loading(
        self,
        loading_info,
    ):
        missing = list(
            loading_info.get(
                "missing_keys",
                [],
            )
            or []
        )

        unexpected = list(
            loading_info.get(
                "unexpected_keys",
                [],
            )
            or []
        )

        mismatched = list(
            loading_info.get(
                "mismatched_keys",
                [],
            )
            or []
        )

        print(
            "Hierarchical checkpoint loading diagnostics:"
        )
        print(
            "  missing keys:",
            len(missing),
        )
        print(
            "  unexpected keys:",
            len(unexpected),
        )
        print(
            "  mismatched keys:",
            len(mismatched),
        )

        if missing:
            print(
                "  first missing keys:",
                missing[:12],
            )

        if unexpected:
            print(
                "  first unexpected keys:",
                unexpected[:12],
            )

        if mismatched:
            print(
                "  first mismatched keys:",
                mismatched[:12],
            )

        # For this checkpoint the state-dict namespace is known exactly:
        # backbone.*, correctness_head.*, role_head.*, error_type_head.*.
        # Missing trainable tensors means we are no longer evaluating the
        # trained checkpoint faithfully, so abort instead of reporting scores.
        trainable_prefixes = (
            "backbone.",
            "correctness_head.",
            "role_head.",
            "error_type_head.",
        )

        critical_missing = [
            key
            for key in missing
            if key.startswith(
                trainable_prefixes
            )
        ]

        if (
            critical_missing
            or mismatched
        ):
            raise RuntimeError(
                "Hierarchical checkpoint was not loaded faithfully. "
                "Refusing to evaluate a partially/randomly initialized model. "
                f"missing={critical_missing[:20]}, "
                f"mismatched={mismatched[:20]}"
            )

    def _load_model(self):

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available()
            else torch.float32
        )

        config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        architectures = list(
            getattr(
                config,
                "architectures",
                None,
            )
            or []
        )

        self.is_hierarchical = any(
            str(name)
            == "HierarchicalPRMModel"
            for name in architectures
        )

        print(
            "Checkpoint-declared architectures:",
            architectures,
        )

        if self.is_hierarchical:

            self.hierarchical_meta = (
                self._load_hierarchical_meta()
            )

            kwargs = {
                "config": config,
                "device_map": "auto",
                "output_loading_info": True,
            }

            # Newer transformers uses dtype=; older versions use torch_dtype=.
            try:
                model, loading_info = (
                    HierarchicalPRMModel.from_pretrained(
                        self.model_name,
                        dtype=dtype,
                        **kwargs,
                    )
                )
            except TypeError:
                model, loading_info = (
                    HierarchicalPRMModel.from_pretrained(
                        self.model_name,
                        torch_dtype=dtype,
                        **kwargs,
                    )
                )

            self.loading_info = (
                loading_info
                or {}
            )

            self._validate_hierarchical_loading(
                self.loading_info
            )

            return model

        kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }

        try:
            return (
                AutoModelForTokenClassification
                .from_pretrained(
                    self.model_name,
                    dtype=dtype,
                    **kwargs,
                )
            )
        except TypeError:
            return (
                AutoModelForTokenClassification
                .from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    **kwargs,
                )
            )

    def _post_load(self):

        self.sep_ids = self.tokenizer.encode(
            self.step_separator,
            add_special_tokens=False,
        )

        if not self.sep_ids:
            raise ValueError(
                "step_separator tokenized to an empty sequence."
            )

        try:
            self.input_device = (
                self.model
                .get_input_embeddings()
                .weight
                .device
            )
        except Exception:
            self.input_device = next(
                self.model.parameters()
            ).device

        print("=" * 80)
        print("Qwen3 PRM configuration")
        print("=" * 80)
        print(
            "architecture:",
            self.model.config.architectures,
        )
        print(
            "loaded class:",
            self.model.__class__.__name__,
        )
        print(
            "step_separator:",
            repr(self.step_separator),
        )
        print(
            "separator token ids:",
            self.sep_ids,
        )
        print(
            "threshold:",
            self.threshold,
        )
        print(
            "max_length:",
            self.max_length,
        )

        if self.is_hierarchical:

            allowed_modes = {
                "correctness",
                "new_error",
                "joint_first_error",
                "joint_first_error_raw",
                "logit_fusion",
                "correctness_role_product",
                "correctness_role_mean",
            }

            if (
                self.hierarchical_score_mode
                not in allowed_modes
            ):
                raise ValueError(
                    "Unsupported hierarchical_score_mode="
                    f"{self.hierarchical_score_mode!r}. "
                    f"Choose from {sorted(allowed_modes)}."
                )

            print(
                "hierarchical score mode:",
                self.hierarchical_score_mode,
            )
            print(
                "correctness labels:",
                {
                    self.correctness_error_id:
                        "ERROR",
                    self.correctness_non_error_id:
                        "NON_ERROR",
                },
            )
            print(
                "role labels:",
                {
                    self.role_correct_id:
                        "correct",
                    self.role_new_error_id:
                        "new_error",
                    self.role_propagated_error_id:
                        "propagated_error",
                },
            )

            if (
                self.hierarchical_score_mode
                == "new_error"
            ):
                description = (
                    "P(NON_ERROR) = 1 - P(role=new_error)."
                )
            elif (
                self.hierarchical_score_mode
                == "correctness"
            ):
                description = (
                    "P(NON_ERROR) = P(correctness=NON_ERROR)."
                )
            elif (
                self.hierarchical_score_mode
                == "joint_first_error"
            ):
                description = (
                    "P(first_error) = P(correctness=ERROR) * "
                    "P(role=new_error | role in {new_error, propagated_error}); "
                    "P(NON_ERROR)=1-P(first_error)."
                )
            elif (
                self.hierarchical_score_mode
                == "joint_first_error_raw"
            ):
                description = (
                    "P(first_error) = P(correctness=ERROR) * "
                    "P(role=new_error); P(NON_ERROR)=1-P(first_error)."
                )
            elif (
                self.hierarchical_score_mode
                == "logit_fusion"
            ):
                description = (
                    "P(first_error) = sigmoid(alpha*logit(P(ERROR)) + "
                    "(1-alpha)*logit(P(new_error|error-role))); "
                    f"alpha={self.hierarchical_alpha:.3f}."
                )
            elif (
                self.hierarchical_score_mode
                == "correctness_role_product"
            ):
                description = (
                    "P(NON_ERROR) = P(correctness=NON_ERROR) * "
                    "P(role=correct)."
                )
            else:
                description = (
                    "P(NON_ERROR) = 0.5 * "
                    "(P(correctness=NON_ERROR) + P(role=correct))."
                )

            print(
                "ProcessBench score mapping:",
                description,
            )

        else:

            num_labels = getattr(
                self.model.config,
                "num_labels",
                None,
            )

            if (
                not isinstance(
                    num_labels,
                    int,
                )
                or num_labels < 2
            ):
                raise ValueError(
                    "Qwen3 PRM requires a token-classification "
                    "head with at least 2 labels, but got "
                    f"num_labels={num_labels}."
                )

            self._resolve_score_mapping(
                num_labels
            )

            print(
                "num_labels:",
                num_labels,
            )
            print(
                "ProcessBench score mapping:",
                self._score_mapping_description(),
            )

        print("=" * 80)

    def _resolve_score_mapping(self, num_labels: int):
        if (
            self.error_label_ids is not None
            and self.non_error_label_ids is not None
        ):
            raise ValueError(
                "Specify only one of error_label_ids or non_error_label_ids."
            )

        def validate(ids, name):
            ids = sorted(set(int(x) for x in ids))
            bad = [x for x in ids if x < 0 or x >= num_labels]
            if bad:
                raise ValueError(
                    f"{name} contains out-of-range ids {bad}; "
                    f"num_labels={num_labels}."
                )
            if not ids:
                raise ValueError(f"{name} cannot be empty.")
            return ids

        if self.non_error_label_ids is not None:
            self.score_mode = "sum_non_error"
            self.score_label_ids = validate(
                self.non_error_label_ids,
                "non_error_label_ids",
            )
            return

        if self.error_label_ids is not None:
            self.score_mode = "one_minus_error"
            self.score_label_ids = validate(
                self.error_label_ids,
                "error_label_ids",
            )
            return

        # Preserve the original binary evaluator exactly.
        if num_labels == 2:
            self.score_mode = "sum_non_error"
            self.score_label_ids = [self.NON_ERROR_LABEL]
            return

        raw_id2label = getattr(self.model.config, "id2label", None) or {}
        id2label = {}
        for key, value in raw_id2label.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            id2label[idx] = str(value).strip().lower().replace("-", "_")

        error_words = (
            "error", "incorrect", "wrong", "invalid", "negative", "bad"
        )
        non_error_words = (
            "non_error", "nonerror", "correct", "valid", "positive", "good"
        )

        error_ids = [
            idx for idx, label in id2label.items()
            if any(word in label for word in error_words)
            and not ("non_error" in label or "nonerror" in label)
        ]
        non_error_ids = [
            idx for idx, label in id2label.items()
            if any(word in label for word in non_error_words)
        ]

        # Prefer an explicit ERROR class/set because P(NON_ERROR)=1-P(ERROR)
        # naturally supports 3-class models with multiple acceptable subclasses.
        if error_ids:
            self.score_mode = "one_minus_error"
            self.score_label_ids = validate(error_ids, "inferred error_label_ids")
            return

        if non_error_ids:
            self.score_mode = "sum_non_error"
            self.score_label_ids = validate(
                non_error_ids,
                "inferred non_error_label_ids",
            )
            return

        raise ValueError(
            "Cannot infer ProcessBench NON_ERROR probability for a multi-class "
            f"checkpoint (num_labels={num_labels}, id2label={raw_id2label}). "
            "Pass --error-label-ids (P(NON_ERROR)=1-sum(P(error))) or "
            "--non-error-label-ids (P(NON_ERROR)=sum(P(non-error)))."
        )

    def _score_mapping_description(self):
        if self.score_mode == "one_minus_error":
            return (
                "P(NON_ERROR) = 1 - sum(P(label)) for ERROR label ids "
                f"{self.score_label_ids}"
            )
        return (
            "P(NON_ERROR) = sum(P(label)) for NON_ERROR label ids "
            f"{self.score_label_ids}"
        )

    def _non_error_probability(self, position_probs):
        selected = position_probs[self.score_label_ids].sum()
        if self.score_mode == "one_minus_error":
            selected = 1.0 - selected
        return selected.clamp(0.0, 1.0)

    # ============================================================
    # Tokenization
    # ============================================================

    def _bos_ids(self):

        if self.tokenizer.bos_token_id is None:
            return []

        return [
            self.tokenizer.bos_token_id
        ]

    def _encode_text(
        self,
        text: str,
    ):

        return self.tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

    # ============================================================
    # Fast path:
    #
    # score an entire ProcessBench trajectory in one forward pass.
    #
    # This is valid because Qwen3 is causal:
    # the logits at step i cannot see future steps.
    # ============================================================

    def _encode_full_trajectory(
        self,
        problem: str,
        steps: List[str],
    ):

        input_ids = []

        input_ids.extend(
            self._bos_ids()
        )

        # problem + separator
        input_ids.extend(
            self._encode_text(problem)
        )
        input_ids.extend(
            self.sep_ids
        )

        score_positions = []

        for step in steps:

            input_ids.extend(
                self._encode_text(step)
            )

            input_ids.extend(
                self.sep_ids
            )

            # Exactly the same position used during training:
            # final token of the separator.
            score_positions.append(
                len(input_ids) - 1
            )

        if len(input_ids) > self.max_length:
            return None

        return {
            "input_ids": input_ids,
            "score_positions": score_positions,
        }

    # ============================================================
    # Long-sequence fallback
    #
    # Reproduces your prm_train_fixed.py behavior:
    # preserve problem + current candidate, then keep as many recent
    # complete prefix steps as possible.
    # ============================================================

    def _encode_candidate(
        self,
        problem: str,
        prefix: List[str],
        candidate: str,
    ):

        bos_ids = self._bos_ids()

        problem_ids = (
            self._encode_text(problem)
            + self.sep_ids
        )

        candidate_ids = (
            self._encode_text(candidate)
            + self.sep_ids
        )

        # Current candidate must survive truncation because
        # its final separator token is the scoring position.
        minimum = (
            len(bos_ids)
            + len(problem_ids)
            + len(candidate_ids)
        )

        if minimum > self.max_length:

            available_problem = (
                self.max_length
                - len(bos_ids)
                - len(candidate_ids)
            )

            if available_problem <= len(self.sep_ids):

                # Candidate itself is extremely long.
                # Keep candidate suffix + separator.
                available_candidate = (
                    self.max_length
                    - len(bos_ids)
                    - len(self.sep_ids)
                )

                available_candidate = max(
                    1,
                    available_candidate,
                )

                raw_candidate_ids = self._encode_text(
                    candidate
                )

                raw_candidate_ids = raw_candidate_ids[
                    -available_candidate:
                ]

                candidate_ids = (
                    raw_candidate_ids
                    + self.sep_ids
                )

                problem_ids = []

            else:

                raw_problem_ids = self._encode_text(
                    problem
                )

                keep_problem = max(
                    0,
                    available_problem
                    - len(self.sep_ids),
                )

                problem_ids = (
                    raw_problem_ids[:keep_problem]
                    + self.sep_ids
                )

        remaining = (
            self.max_length
            - len(bos_ids)
            - len(problem_ids)
            - len(candidate_ids)
        )

        # Drop oldest prefix steps first.
        selected_prefix = []

        for step in reversed(prefix):

            piece = (
                self._encode_text(step)
                + self.sep_ids
            )

            if len(piece) <= remaining:

                selected_prefix.append(
                    piece
                )

                remaining -= len(piece)

        selected_prefix.reverse()

        prefix_ids = []

        for piece in selected_prefix:
            prefix_ids.extend(piece)

        input_ids = (
            bos_ids
            + problem_ids
            + prefix_ids
            + candidate_ids
        )

        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]

        # Training target was labels[-1].
        score_position = len(input_ids) - 1

        return {
            "input_ids": input_ids,
            "score_positions": [
                score_position
            ],
        }

    # ============================================================
    # Batch inference
    # ============================================================

    @staticmethod
    def _safe_logit(prob):
        eps = 1e-6
        prob = prob.clamp(eps, 1.0 - eps)
        return torch.log(prob) - torch.log1p(-prob)

    def _hierarchical_step_values(
        self,
        correctness_probs,
        role_probs,
        i,
        position,
    ):
        p_error = correctness_probs[
            i,
            position,
            self.correctness_error_id,
        ]

        p_non_error = correctness_probs[
            i,
            position,
            self.correctness_non_error_id,
        ]

        p_role_correct = role_probs[
            i,
            position,
            self.role_correct_id,
        ]

        p_new_error = role_probs[
            i,
            position,
            self.role_new_error_id,
        ]

        p_propagated = role_probs[
            i,
            position,
            self.role_propagated_error_id,
        ]

        error_role_mass = (
            p_new_error
            + p_propagated
        ).clamp_min(1e-8)

        p_new_given_error_role = (
            p_new_error
            / error_role_mass
        )

        # Hierarchical factorization:
        #   level 1: ERROR vs NON_ERROR
        #   level 2: if error-like, NEW_ERROR vs PROPAGATED_ERROR
        p_joint_first_error = (
            p_error
            * p_new_given_error_role
        )

        p_joint_first_error_raw = (
            p_error
            * p_new_error
        )

        alpha = self.hierarchical_alpha

        fused_logit = (
            alpha
            * self._safe_logit(
                p_error
            )
            + (1.0 - alpha)
            * self._safe_logit(
                p_new_given_error_role
            )
        )

        p_logit_fusion = torch.sigmoid(
            fused_logit
        )

        aux = {
            "p_error":
                float(p_error.detach().cpu()),
            "p_non_error":
                float(p_non_error.detach().cpu()),
            "p_role_correct":
                float(p_role_correct.detach().cpu()),
            "p_new_error":
                float(p_new_error.detach().cpu()),
            "p_propagated_error":
                float(p_propagated.detach().cpu()),
            "p_new_given_error_role":
                float(p_new_given_error_role.detach().cpu()),
            "p_joint_first_error":
                float(p_joint_first_error.detach().cpu()),
            "p_joint_first_error_raw":
                float(p_joint_first_error_raw.detach().cpu()),
            "p_logit_fusion":
                float(p_logit_fusion.detach().cpu()),
        }

        if (
            self.hierarchical_score_mode
            == "correctness"
        ):
            score = p_non_error

        elif (
            self.hierarchical_score_mode
            == "new_error"
        ):
            score = 1.0 - p_new_error

        elif (
            self.hierarchical_score_mode
            == "joint_first_error"
        ):
            score = (
                1.0
                - p_joint_first_error
            )

        elif (
            self.hierarchical_score_mode
            == "joint_first_error_raw"
        ):
            score = (
                1.0
                - p_joint_first_error_raw
            )

        elif (
            self.hierarchical_score_mode
            == "logit_fusion"
        ):
            score = (
                1.0
                - p_logit_fusion
            )

        elif (
            self.hierarchical_score_mode
            == "correctness_role_product"
        ):
            score = (
                p_non_error
                * p_role_correct
            )

        elif (
            self.hierarchical_score_mode
            == "correctness_role_mean"
        ):
            score = 0.5 * (
                p_non_error
                + p_role_correct
            )

        else:
            raise RuntimeError(
                "Unexpected hierarchical score mode: "
                f"{self.hierarchical_score_mode}"
            )

        return (
            score.clamp(
                0.0,
                1.0,
            ),
            aux,
        )

    def _forward_sequences(
        self,
        encoded_sequences: List[Dict],
    ) -> List[List[float]]:

        if not encoded_sequences:
            self._last_forward_aux = []
            return []

        max_len = max(
            len(x["input_ids"])
            for x in encoded_sequences
        )

        batch_size = len(
            encoded_sequences
        )

        pad_token_id = (
            self.tokenizer.pad_token_id
        )

        input_ids = torch.full(
            (
                batch_size,
                max_len,
            ),
            pad_token_id,
            dtype=torch.long,
        )

        attention_mask = torch.zeros(
            (
                batch_size,
                max_len,
            ),
            dtype=torch.long,
        )

        for i, encoded in enumerate(
            encoded_sequences
        ):

            ids = torch.tensor(
                encoded["input_ids"],
                dtype=torch.long,
            )

            length = len(
                encoded["input_ids"]
            )

            input_ids[
                i,
                :length,
            ] = ids

            attention_mask[
                i,
                :length,
            ] = 1

        input_ids = input_ids.to(
            self.input_device
        )

        attention_mask = attention_mask.to(
            self.input_device
        )

        with torch.inference_mode():

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            if self.is_hierarchical:

                correctness_probs = F.softmax(
                    outputs.correctness_logits.float(),
                    dim=-1,
                )

                role_probs = F.softmax(
                    outputs.role_logits.float(),
                    dim=-1,
                )

            else:

                probabilities = F.softmax(
                    outputs.logits.float(),
                    dim=-1,
                )

        results = []
        aux_results = []

        for i, encoded in enumerate(
            encoded_sequences
        ):

            positions = encoded[
                "score_positions"
            ]

            sample_scores = []
            sample_aux = []

            for position in positions:

                if self.is_hierarchical:

                    score, aux = (
                        self._hierarchical_step_values(
                            correctness_probs,
                            role_probs,
                            i,
                            position,
                        )
                    )

                    sample_aux.append(
                        aux
                    )

                else:

                    score = (
                        self._non_error_probability(
                            probabilities[
                                i,
                                position,
                            ]
                        )
                    )

                sample_scores.append(
                    float(
                        score.detach().cpu()
                    )
                )

            results.append(
                sample_scores
            )

            aux_results.append(
                sample_aux
                if self.is_hierarchical
                else None
            )

        self._last_forward_aux = (
            aux_results
        )

        return results

    # ============================================================
    # Long example
    # ============================================================

    def _predict_long_example(
        self,
        problem: str,
        steps: List[str],
    ):

        encoded_candidates = []

        prefix = []

        for candidate in steps:

            encoded_candidates.append(
                self._encode_candidate(
                    problem=problem,
                    prefix=prefix,
                    candidate=candidate,
                )
            )

            prefix.append(
                candidate
            )

        step_scores = []
        step_aux = []

        for start in range(
            0,
            len(encoded_candidates),
            self.batch_size,
        ):

            batch = encoded_candidates[
                start:
                start + self.batch_size
            ]

            batch_scores = (
                self._forward_sequences(
                    batch
                )
            )

            batch_aux = (
                self._last_forward_aux
                or [None] * len(batch_scores)
            )

            for scores, aux in zip(
                batch_scores,
                batch_aux,
            ):

                # candidate-level encoding has exactly
                # one supervised/scored position.
                step_scores.append(
                    scores[0]
                )

                if (
                    self.is_hierarchical
                    and aux
                ):
                    step_aux.append(
                        aux[0]
                    )

        return (
            step_scores,
            step_aux
            if self.is_hierarchical
            else None,
        )

    # ============================================================
    # Public prediction API
    # ============================================================

    def predict_batch(
        self,
        examples: Sequence[Dict],
    ) -> List[Dict]:

        results = [
            None
            for _ in examples
        ]

        # -----------------------------
        # Short trajectories
        # -----------------------------

        short_indices = []
        short_encoded = []

        long_indices = []

        for index, example in enumerate(
            examples
        ):

            problem = example["problem"]

            steps = list(
                example["steps"]
            )

            encoded = (
                self._encode_full_trajectory(
                    problem=problem,
                    steps=steps,
                )
            )

            if encoded is None:

                long_indices.append(
                    index
                )

            else:

                short_indices.append(
                    index
                )

                short_encoded.append(
                    encoded
                )

        # One forward pass for normal ProcessBench samples.
        if short_encoded:

            short_scores = (
                self._forward_sequences(
                    short_encoded
                )
            )

            short_aux = (
                self._last_forward_aux
                or [None] * len(short_scores)
            )

            for (
                index,
                step_scores,
                step_aux,
            ) in zip(
                short_indices,
                short_scores,
                short_aux,
            ):

                step_predictions, prediction = (
                    self.scores_to_prediction(
                        step_scores
                    )
                )

                item = {
                    "step_scores":
                        step_scores,

                    "step_predictions":
                        step_predictions,

                    "prediction":
                        prediction,
                }

                if (
                    self.is_hierarchical
                    and step_aux is not None
                ):
                    item[
                        "step_aux"
                    ] = step_aux

                results[index] = item

        # -----------------------------
        # Long trajectories
        # -----------------------------

        for index in long_indices:

            example = examples[
                index
            ]

            (
                step_scores,
                step_aux,
            ) = (
                self._predict_long_example(
                    problem=example["problem"],
                    steps=list(
                        example["steps"]
                    ),
                )
            )

            step_predictions, prediction = (
                self.scores_to_prediction(
                    step_scores
                )
            )

            item = {
                "step_scores":
                    step_scores,

                "step_predictions":
                    step_predictions,

                "prediction":
                    prediction,
            }

            if (
                self.is_hierarchical
                and step_aux is not None
            ):
                item[
                    "step_aux"
                ] = step_aux

            results[index] = item

        return results