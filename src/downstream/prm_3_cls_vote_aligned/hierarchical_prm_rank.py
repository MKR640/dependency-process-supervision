#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score candidates with hierarchical or three-class PRMs using shared input packing.

Matches training-time truncation and scores the final token of each step.
Writes per-step probabilities and truncation statistics for each manifest shard.
"""
from __future__ import annotations

import argparse

import json

import math

import os

from dataclasses import dataclass

from pathlib import Path

from typing import Optional

import torch

import torch.nn as nn

from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)

from transformers.modeling_outputs import ModelOutput

CORRECTNESS_ID2LABEL = {0: "ERROR", 1: "NON_ERROR"}

ROLE_ID2LABEL = {0: "correct", 1: "new_error", 2: "propagated_error"}

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

@dataclass
class HierarchicalPRMOutput(ModelOutput):
    correctness_logits: Optional[torch.Tensor] = None
    role_logits: Optional[torch.Tensor] = None
    error_type_logits: Optional[torch.Tensor] = None
    three_class_logits: Optional[torch.Tensor] = None

class HierarchicalPRMModel(PreTrainedModel):
    """Inference-only definition matching the training checkpoint keys."""

    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def __init__(
        self,
        config: PretrainedConfig,
        backbone: Optional[nn.Module] = None,
        classifier_dropout: Optional[float] = None,
    ):
        super().__init__(config)
        hidden_size = int(getattr(config, "hidden_size"))
        if backbone is None:
            self.backbone = AutoModel.from_config(config, trust_remote_code=True)
        else:
            self.backbone = backbone

        if classifier_dropout is None:
            classifier_dropout = getattr(
                config, "hierarchical_prm_classifier_dropout", 0.1
            )
        self.dropout = nn.Dropout(float(classifier_dropout))
        self.correctness_head = nn.Linear(hidden_size, 2, bias=True)
        self.role_head = nn.Linear(hidden_size, 3, bias=True)
        self.error_type_head = nn.Linear(hidden_size, 8, bias=True)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
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

# 三分类模型只投影 ERROR/CORRECT logits；没有 Role/Type 头。
def resolve_threeclass_ids(config, explicit):
    if int(config.num_labels) != 3:
        raise ValueError(f'Expected 3 labels, got {config.num_labels}')
    mapping = {int(k): str(v).strip().upper() for k,v in config.id2label.items()}
    if any(v is not None for v in explicit.values()):
        if any(v is None for v in explicit.values()) or set(explicit.values()) != {0,1,2}:
            raise ValueError('Provide all three distinct IDs: ERROR, CORRECT, NEUTRAL = permutation of 0,1,2')
        for name, i in explicit.items():
            if mapping.get(i) in ('ERROR','CORRECT','NEUTRAL') and mapping[i] != name:
                raise ValueError(f'Explicit label IDs conflict with config: {mapping}')
        return explicit
    result = {}
    for name in ('ERROR','CORRECT','NEUTRAL'):
        ids = [i for i,label in mapping.items() if label == name]
        if len(ids) != 1:
            raise ValueError(f'Ambiguous label map {mapping}. Supply --error-id, --correct-id, --neutral-id from training records; do not guess.')
        result[name] = ids[0]
    if set(result.values()) != {0,1,2}:
        raise ValueError(f'Invalid three-class IDs: {result}')
    return result

class ThreeClassProjection(nn.Module):
    def __init__(self, model, ids):
        super().__init__()
        self.model, self.ids = model, ids
    def forward(self, **kwargs):
        logits = self.model(**kwargs).logits
        if logits.ndim != 3 or logits.shape[-1] != 3:
            raise ValueError('Expected per-token logits [batch, sequence, 3]')
        return HierarchicalPRMOutput(
            correctness_logits=logits[..., [self.ids['ERROR'], self.ids['CORRECT']]],
            three_class_logits=logits)

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows

def split_steps(solution: str) -> list[str]:
    """Use the same blank-line boundary as data preparation and training."""
    return [part.strip() for part in str(solution).split("\n\n") if part.strip()]

def normalize_label_map(raw_map, fallback: dict[int, str]) -> dict[int, str]:
    if not isinstance(raw_map, dict):
        return dict(fallback)
    normalized = {}
    for key, value in raw_map.items():
        try:
            normalized[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return normalized or dict(fallback)

def resolve_label_maps(config) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    correctness = normalize_label_map(
        getattr(config, "correctness_id2label", None), CORRECTNESS_ID2LABEL
    )
    role = normalize_label_map(getattr(config, "role_id2label", None), ROLE_ID2LABEL)
    error_type = normalize_label_map(
        getattr(config, "error_type_id2label", None), ERROR_TYPE_ID2LABEL
    )
    return correctness, role, error_type

def find_label_id(label_map: dict[int, str], wanted: str) -> int:
    matches = [
        class_id
        for class_id, label in label_map.items()
        if str(label).strip().lower() == wanted.lower()
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely resolve {wanted!r} from {label_map}")
    return matches[0]

def build_step_input_ids(
    tokenizer,
    problem: str,
    previous_steps: list[str],
    current_step: str,
    separator: str,
    max_length: int,
) -> list[int]:
    """Reproduce the training-time suffix-preserving truncation policy."""
    sep_ids = tokenizer.encode(separator, add_special_tokens=False)
    if not sep_ids:
        raise ValueError("The step separator tokenized to an empty sequence")

    bos_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    if max_length < len(bos_ids) + len(sep_ids) + 1:
        raise ValueError(f"max_length={max_length} is too small")

    raw_problem_ids = tokenizer(problem, add_special_tokens=False)["input_ids"]
    raw_step_ids = tokenizer(current_step, add_special_tokens=False)["input_ids"]
    problem_ids = raw_problem_ids + sep_ids
    step_ids = raw_step_ids + sep_ids

    if len(bos_ids) + len(problem_ids) + len(step_ids) > max_length:
        available_problem = max_length - len(bos_ids) - len(step_ids)
        if available_problem <= len(sep_ids):
            available_step = max_length - len(bos_ids) - len(sep_ids)
            step_ids = raw_step_ids[-max(1, available_step) :] + sep_ids
            problem_ids = []
        else:
            keep_problem = max(0, available_problem - len(sep_ids))
            problem_ids = raw_problem_ids[:keep_problem] + sep_ids

    remaining = max_length - len(bos_ids) - len(problem_ids) - len(step_ids)
    selected = []
    for previous_step in reversed(previous_steps):
        piece = tokenizer(previous_step, add_special_tokens=False)["input_ids"] + sep_ids
        if len(piece) > remaining:
            continue
        selected.append(piece)
        remaining -= len(piece)
    selected.reverse()

    prefix_ids = [token for piece in selected for token in piece]
    input_ids = bos_ids + problem_ids + prefix_ids + step_ids
    if not input_ids or len(input_ids) > max_length:
        raise RuntimeError(
            f"Invalid tokenized length={len(input_ids)}, max_length={max_length}"
        )
    return input_ids

def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(max(value, 1e-12)) for value in values) / len(values))

def aggregate_step_scores(
    step_scores: list[float], aggregation: str, aggregation_lambda: float
) -> float:
    if not step_scores:
        return 0.0
    minimum = min(step_scores)
    if aggregation == "min":
        return minimum
    if aggregation == "geometric_mean":
        return geometric_mean(step_scores)
    if aggregation == "mean":
        return sum(step_scores) / len(step_scores)
    if aggregation == "min_gmean_mix":
        return (
            aggregation_lambda * minimum
            + (1.0 - aggregation_lambda) * geometric_mean(step_scores)
        )
    raise ValueError(f"Unsupported aggregation={aggregation!r}")

def compute_step_score(
    correctness_probability: float,
    role_correct_probability: float,
    score_mode: str,
) -> float:
    if score_mode == "correctness":
        return correctness_probability
    if score_mode == "correctness_role_product":
        return correctness_probability * role_correct_probability
    if score_mode == "correctness_role_mean":
        return 0.5 * (correctness_probability + role_correct_probability)
    raise ValueError(f"Unsupported score_mode={score_mode!r}")

@torch.inference_mode()
def score_trajectory(
    model,
    tokenizer,
    device: torch.device,
    problem: str,
    solution: str,
    separator: str,
    max_length: int,
    batch_size: int,
    aggregation: str,
    aggregation_lambda: float,
    classification_threshold: float,
    score_mode: str,
    correctness_map: dict[int, str],
    role_map: dict[int, str],
    error_type_map: dict[int, str],
) -> dict:
    steps = split_steps(solution)
    if not steps:
        return {
            "score": 0.0,
            "predicted_label": "ERROR",
            "step_scores": [],
            "step_predictions": [],
            "first_predicted_error_step": None,
            "first_predicted_new_error_step": None,
        }

    encoded_steps = [
        build_step_input_ids(
            tokenizer=tokenizer,
            problem=problem,
            previous_steps=steps[:step_index],
            current_step=step,
            separator=separator,
            max_length=max_length,
        )
        for step_index, step in enumerate(steps)
    ]

    # 截断诊断；以下统计不改变模型实际输入。
    # Diagnostic only: keep the original training-compatible packing unchanged.
    sep_length = len(tokenizer.encode(separator, add_special_tokens=False))
    cumulative = (int(tokenizer.bos_token_id is not None)
                  + len(tokenizer(problem, add_special_tokens=False)['input_ids']) + sep_length)
    original_lengths = []
    for step in steps:
        cumulative += len(tokenizer(step, add_special_tokens=False)['input_ids']) + sep_length
        original_lengths.append(cumulative)

    non_error_id = find_label_id(correctness_map, "NON_ERROR")
    step_scores: list[float] = []
    step_predictions: list[dict] = []

    for start in range(0, len(encoded_steps), batch_size):
        batch_ids = encoded_steps[start : start + batch_size]
        batch = tokenizer.pad(
            {"input_ids": batch_ids}, padding=True, return_tensors="pt"
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        last_positions = batch["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(len(batch_ids), device=device)

        correctness_probs = torch.softmax(
            outputs.correctness_logits[batch_indices, last_positions].float(), dim=-1
        ).cpu()
        # 双头概率 softmax 等价于 p(C)/(p(C)+p(E))，更数值稳定。
        is_threeclass = outputs.three_class_logits is not None
        if is_threeclass:
            if score_mode != 'correctness':
                raise ValueError('Three-class PRM has no Role head; use correctness scoring')
            raw_probs = torch.softmax(outputs.three_class_logits[batch_indices, last_positions].float(), dim=-1).cpu()
        else:
            role_probs = torch.softmax(outputs.role_logits[batch_indices, last_positions].float(), dim=-1).cpu()
            error_type_probs = torch.softmax(outputs.error_type_logits[batch_indices, last_positions].float(), dim=-1).cpu()

        for offset in range(len(batch_ids)):
            step_index = start + offset
            correctness_row = correctness_probs[offset]
            if is_threeclass:
                correctness_id = int(torch.argmax(correctness_row))
                p_non_error = float(correctness_row[non_error_id])
                step_scores.append(p_non_error)
                step_predictions.append(dict(
                    step_index=step_index, step_text=steps[step_index],
                    input_token_count=len(encoded_steps[step_index]),
                    input_ids_sha256=digest(encoded_steps[step_index]),
                    untruncated_token_count=original_lengths[step_index],
                    context_truncated=original_lengths[step_index] > len(encoded_steps[step_index]),
                    score=p_non_error, predicted_label=correctness_map[correctness_id],
                    correct_probability=p_non_error, error_probability=1.0-p_non_error,
                    correctness_probabilities={'ERROR':1.0-p_non_error,'NON_ERROR':p_non_error},
                    three_class_probabilities={name:float(raw_probs[offset][i]) for name,i in model.ids.items()},
                    predicted_role=None, role_probabilities=None,
                    predicted_error_type=None, top_error_type=None, error_type_probabilities=None))
                continue
            role_row = role_probs[offset]
            error_type_row = error_type_probs[offset]

            correctness_id = int(torch.argmax(correctness_row))
            role_id = int(torch.argmax(role_row))
            error_type_id = int(torch.argmax(error_type_row))
            p_non_error = float(correctness_row[non_error_id])
            role_correct_id = find_label_id(role_map, "correct")
            p_role_correct = float(role_row[role_correct_id])
            step_score = compute_step_score(p_non_error, p_role_correct, score_mode)
            step_scores.append(step_score)

            correctness_probabilities = {
                correctness_map[class_id]: float(correctness_row[class_id])
                for class_id in sorted(correctness_map)
            }
            role_probabilities = {
                role_map[class_id]: float(role_row[class_id])
                for class_id in sorted(role_map)
            }
            error_type_probabilities = {
                error_type_map[class_id]: float(error_type_row[class_id])
                for class_id in sorted(error_type_map)
            }
            predicted_role = role_map[role_id]

            step_predictions.append(
                {
                    "step_index": step_index,
                    "step_text": steps[step_index],
                    "input_token_count": len(encoded_steps[step_index]),
                    "input_ids_sha256": digest(encoded_steps[step_index]),
                    "untruncated_token_count": original_lengths[step_index],
                    "context_truncated": original_lengths[step_index] > len(encoded_steps[step_index]),
                    "score": step_score,
                    "predicted_label": correctness_map[correctness_id],
                    "correct_probability": p_non_error,
                    "error_probability": 1.0 - p_non_error,
                    "correctness_probabilities": correctness_probabilities,
                    "predicted_role": predicted_role,
                    "role_probabilities": role_probabilities,
                    "predicted_error_type": (
                        error_type_map[error_type_id]
                        if predicted_role == "new_error"
                        else None
                    ),
                    "top_error_type": error_type_map[error_type_id],
                    "error_type_probabilities": error_type_probabilities,
                }
            )

    trajectory_score = aggregate_step_scores(
        step_scores, aggregation, aggregation_lambda
    )
    first_error = next(
        (
            row["step_index"]
            for row in step_predictions
            if row["predicted_label"].upper() == "ERROR"
        ),
        None,
    )
    first_new_error = next(
        (
            row["step_index"]
            for row in step_predictions
            if row["predicted_role"] == "new_error"
        ),
        None,
    )
    return {
        "score": trajectory_score,
        "predicted_label": (
            "CORRECT" if trajectory_score >= classification_threshold else "ERROR"
        ),
        "step_scores": step_scores,
        "step_predictions": step_predictions,
        "first_predicted_error_step": first_error,
        "first_predicted_new_error_step": first_new_error,
    }

# Manifest-based shard execution with resume support.
import argparse
import os
from pathlib import Path

from evaluate import (atomic_write, complete, file_hash, read_json, read_jsonl,
                        validate_candidates, validate_ranked, verified, code_hash, digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world != manifest['world_size']:
        raise ValueError('WORLD_SIZE does not match manifest')
    if file_hash(manifest['candidates']) != manifest['candidates_sha256']:
        raise ValueError('Candidate file changed after manifest creation')
    if code_hash() != manifest['code_sha256']:
        raise ValueError('Code changed after manifest creation')
    rows = read_jsonl(manifest['candidates'])
    validate_candidates(rows)
    shard = rows[rank::world]
    output = Path(manifest['shard_dir']) / f'rank{rank}.jsonl'
    identity = dict(run_id=manifest['run_id'], rank=rank, world_size=world)
    # 仅复用内容哈希与身份都匹配的完整分片。
    if verified(output, identity):
        validate_ranked(read_jsonl(output), shard, require_steps=True, allow_empty=True)
        print(f'[rank {rank}] reuse verified completed shard', flush=True)
        return
    if not shard:
        atomic_write(output, [], jsonl=True)
        complete(output, identity, 0)
        return

    # Import only here: offline commands and empty shards do not require Torch.
    import torch
    from transformers import AutoConfig, AutoTokenizer

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        if world != 1:
            raise RuntimeError('Multi-process inference requires CUDA')
        device = torch.device('cpu')
    dtype = getattr(torch, manifest['dtype'])
    path = manifest['model']['path']
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    # 显式读取 config，并核验三头模型参数完整性。
    kind = manifest.get('model_kind', 'hierarchical')
    model_class = AutoModelForTokenClassification if kind == 'threeclass' else HierarchicalPRMModel
    model, loading = model_class.from_pretrained(
        path, config=config, dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True, output_loading_info=True)
    if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise RuntimeError(f'Checkpoint architecture mismatch; refusing random/missing heads: {loading}')
    if kind == 'threeclass':
        ids = resolve_threeclass_ids(config, manifest['threeclass_ids'])
        print(f'[rank {rank}] resolved labels: {ids}; score=softmax(ERROR,CORRECT)[CORRECT]', flush=True)
        model = ThreeClassProjection(model, ids)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True,
                                              fix_mistral_regex=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer has neither PAD nor EOS')
        tokenizer.pad_token = tokenizer.eos_token
    correctness, role, error_type = (resolve_label_maps(config) if kind == 'hierarchical'
                                     else (CORRECTNESS_ID2LABEL, ROLE_ID2LABEL, ERROR_TYPE_ID2LABEL))
    # 词表语义和实际 token 序列都留存，供两模型输入一致性检查。
    tokenizer_fingerprint = digest(dict(vocab=tokenizer.get_vocab(), bos=tokenizer.bos_token_id,
                                        eos=tokenizer.eos_token_id, pad=tokenizer.pad_token_id))
    resolved_ids = ids if kind == 'threeclass' else correctness
    if set(correctness.values()) != {'ERROR', 'NON_ERROR'} or set(role.values()) != {'correct', 'new_error', 'propagated_error'}:
        raise ValueError(f'Unexpected label semantics: {correctness}, {role}')

    def outputs():
        for position, row in enumerate(shard, 1):
            ranking = []
            for candidate_id, candidate in enumerate(row['candidates']):
                result = score_trajectory(
                    model, tokenizer, device, row['problem'], candidate,
                    manifest['separator'], manifest['max_length'], manifest['batch_size'],
                    'min_gmean_mix', 0.5, 0.5, 'correctness', correctness, role, error_type)
                ranking.append(dict(candidate_id=candidate_id, candidate=candidate,
                                    prm_score=result.pop('score'),
                                    prm_predicted_label=result.pop('predicted_label'), **result))
            ranking.sort(key=lambda c: (-c['prm_score'], c['candidate_id']))
            yield dict(id=row['id'], problem=row['problem'], answer=row['answer'],
                       prm_model_type=('three_class_prm' if kind == 'threeclass' else 'hierarchical_multitask_prm'),
                       tokenizer_fingerprint=tokenizer_fingerprint, resolved_label_ids=resolved_ids,
                       inference_run_id=manifest['run_id'],
                       ranking=ranking)
            if position % 20 == 0 or position == len(shard):
                print(f'[rank {rank}] {position}/{len(shard)} questions', flush=True)

    # 原子写入 + 完成标记，半写结果不会进入后续合并。
    atomic_write(output, outputs(), jsonl=True)
    validate_ranked(read_jsonl(output), shard, require_steps=True, allow_empty=True)
    complete(output, identity, len(shard))
    print(f'[rank {rank}] completed: {output}', flush=True)


if __name__ == '__main__':
    main()
