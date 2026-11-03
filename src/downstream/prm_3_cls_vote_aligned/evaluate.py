#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate answer selection, weighted voting and validation-selected policies.

Examples:
  python evaluate.py --ranked result.jsonl --output metrics.json
  python evaluate.py rerank --scores scores.jsonl --candidates candidates.jsonl --allow-legacy --output reranked.jsonl
  python evaluate.py tune --validation-ranked val.jsonl --split validation --allow-legacy --output policy.json

Scoring, offline reranking and policy selection use the Python standard library.
GPU generation and inference require PyTorch.
"""
from __future__ import annotations



# ======================================================================
# 公共校验、内容哈希、原子写入和评分公式
# ======================================================================
import hashlib

import json

import math

import os

import tempfile

from pathlib import Path

VERSION = "prm-two-model-two-strategy-v3.2"
ALIGNMENT_PROTOCOL = "attachment-v3.1_skip-prefix_arithmetic-min-gmean"

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as e:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from e
    return rows

def atomic_write(path, value, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            if jsonl:
                for row in value:
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            else:
                json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
                f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def key(row):
    value = row.get("id")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Every problem needs a string/integer id")
    return (type(value).__name__, value)

def index_rows(rows):
    index = {}
    for row in rows:
        k = key(row)
        if k in index:
            raise ValueError(f"Duplicate problem id: {k}")
        index[k] = row
    return index

def validate_candidates(rows, expected_n=None, expected_questions=None):
    if not rows:
        raise ValueError("Empty candidate file")
    index_rows(rows)
    if expected_questions is not None and len(rows) != expected_questions:
        raise ValueError(f"Expected {expected_questions} questions, got {len(rows)}")
    for row in rows:
        if not isinstance(row.get("problem"), str) or not isinstance(row.get("answer"), str):
            raise ValueError(f"Invalid problem/answer for {row.get('id')}")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or not candidates or not all(isinstance(c, str) for c in candidates):
            raise ValueError(f"Invalid candidates for {row['id']}")
        if expected_n is not None and len(candidates) != expected_n:
            raise ValueError(f"{row['id']}: expected {expected_n} candidates, got {len(candidates)}")

def validate_ranked(rows, candidates=None, require_steps=False, allow_empty=False):
    if not rows and not allow_empty:
        raise ValueError("Empty ranked file")
    index_rows(rows)
    expected = index_rows(candidates) if candidates is not None else None
    if expected is not None and set(index_rows(rows)) != set(expected):
        raise ValueError("Ranked/candidate problem IDs do not match")
    for row in rows:
        ranking = row.get("ranking")
        if not isinstance(ranking, list) or not ranking:
            raise ValueError(f"Missing ranking: {row.get('id')}")
        ids = [c.get("candidate_id") for c in ranking]
        if any(type(i) is not int for i in ids) or sorted(ids) != list(range(len(ranking))):
            raise ValueError(f"Candidate IDs must be unique contiguous 0..N-1: {row['id']}")
        original = sorted(ranking, key=lambda c: c['candidate_id'])
        if not isinstance(row.get('problem'), str) or not isinstance(row.get('answer'), str):
            raise ValueError("Missing problem/answer")
        if expected is not None:
            source = expected[key(row)]
            if (row['problem'], row['answer'], [c.get('candidate') for c in original]) != (
                    source['problem'], source['answer'], source['candidates']):
                raise ValueError(f"Candidate content/order mismatch: {row['id']}")
        for candidate in ranking:
            if not isinstance(candidate.get('candidate'), str):
                raise ValueError("Missing candidate text")
            probability(candidate.get('prm_score'))
            if require_steps:
                steps = candidate.get('step_predictions')
                texts = [x.strip() for x in candidate['candidate'].split('\n\n') if x.strip()]
                if not isinstance(steps, list) or len(steps) != len(texts):
                    raise ValueError(f"Missing/incomplete step_predictions: {row['id']}")
                for i, step in enumerate(steps):
                    if step.get('step_index') != i or step.get('step_text') != texts[i]:
                        raise ValueError("Step order/text mismatch")
                    probability(step.get('correct_probability'))
                    # 三分类无 Role/Type；明确缺失，不填伪造概率。
                    if row.get('prm_model_type') == 'three_class_prm':
                        if step.get('role_probabilities') is not None or step.get('predicted_role') is not None:
                            raise ValueError('Three-class PRM must not fabricate Role outputs')
                        probs = step.get('three_class_probabilities', {})
                        for name in ('ERROR','CORRECT','NEUTRAL'):
                            probability(probs.get(name))
                        if abs(sum(probs.values())-1)>1e-4:
                            raise ValueError('Invalid three-class probability sum')
                    else:
                        role = step.get('role_probabilities') or {}
                        for name in ('correct', 'new_error', 'propagated_error'):
                            probability(role.get(name))
                        if abs(sum(role[n] for n in ('correct', 'new_error', 'propagated_error')) - 1) > 1e-4:
                            raise ValueError("Role probabilities do not sum to one")

def probability(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Expected finite probability in [0,1], got {value!r}")
    return float(value)

def scoring_config(mode='correctness', alpha=0.0, aggregation='min_gmean_mix', mix=0.5, threshold=0.5, tail_fraction=0.25):
    modes = ('correctness', 'correctness_role_mean', 'correctness_role_product', 'correctness_role_blend')
    if mode not in modes or aggregation not in ('min', 'mean', 'geometric_mean', 'min_gmean_mix', 'lower_tail_mean'):
        raise ValueError("Unknown score mode or aggregation")
    for v in (alpha, mix, threshold, tail_fraction):
        probability(v)
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be strictly between zero and one")
    if tail_fraction == 0:
        raise ValueError('tail_fraction must be positive')
    effective_alpha = {'correctness': 0., 'correctness_role_mean': .5}.get(mode, alpha)
    return dict(score_mode=mode, role_alpha=effective_alpha, aggregation=aggregation,
                aggregation_lambda=mix, classification_threshold=threshold, tail_fraction=tail_fraction)

def step_score(step, config):
    c = probability(step['correct_probability'])
    # correctness 模式不读取不存在的 Role 概率。
    if config['score_mode'] == 'correctness':
        return c
    if not step.get('role_probabilities'):
        raise ValueError('Role fusion requires a real Role head')
    r = probability(step['role_probabilities']['correct'])
    if config['score_mode'] == 'correctness_role_product':
        return c * r
    a = config['role_alpha']
    return (1 - a) * c + a * r

def aggregate(values, config):
    if not values:
        return 0.0
    minimum = min(values)
    gmean = math.exp(sum(math.log(max(v, 1e-12)) for v in values) / len(values))
    kind = config['aggregation']
    if kind == 'lower_tail_mean':
        k = max(1, math.ceil(len(values) * config['tail_fraction']))
        return sum(sorted(values)[:k]) / k
    if kind == 'min':
        return minimum
    if kind == 'mean':
        return sum(values) / len(values)
    if kind == 'geometric_mean':
        return gmean
    weight = config['aggregation_lambda']
    return weight * minimum + (1 - weight) * gmean

def complete(path, identity, rows):
    info = dict(identity=identity, rows=rows, sha256=file_hash(path))
    metadata = Path(str(path) + '.meta.json')
    if metadata.exists():
        info['metadata_sha256'] = file_hash(metadata)
    atomic_write(str(path) + '.done.json', info)

def verified(path, identity=None):
    try:
        info = read_json(str(path) + '.done.json')
        return ((identity is None or info['identity'] == identity)
                and info['sha256'] == file_hash(path)
                and ('metadata_sha256' not in info or info['metadata_sha256'] == file_hash(str(path) + '.meta.json')))
    except (OSError, ValueError, KeyError):
        return False

def code_hash():
    root = Path(__file__).parent
    return digest({p.name: file_hash(p) for p in sorted(root.glob('*.py'))})

def model_hash(path):
    root = Path(path).resolve()
    if not (root / 'config.json').is_file():
        raise ValueError('model_path must be a local saved checkpoint directory')
    # Hash all top-level checkpoint/tokenizer/code files, not nested checkpoint-* runs.
    paths = [p for p in sorted(root.iterdir()) if p.is_file() and not p.name.startswith('.')]
    if not any(p.suffix in ('.safetensors', '.bin') for p in paths):
        raise ValueError('No model weights found (adapter-only checkpoints are not supported)')
    print('Hashing checkpoint bytes once; large models may take a few minutes.', flush=True)
    return dict(path=str(root), files={p.name: file_hash(p) for p in paths})


# ======================================================================
# Answer extraction and baseline metrics.
# ======================================================================
import argparse

import json

import os

import random

import re

from collections import Counter

from decimal import Decimal, InvalidOperation

def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_no}: {e}"
                ) from e

    return rows

NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
)

def normalize_number(value):
    """
    Convert numeric strings into a canonical representation.

    Examples:
        "1,234" -> "1234"
        "18.0"  -> "18"
        "-2.50" -> "-2.5"
    """

    if value is None:
        return None

    value = str(value).strip()
    value = value.replace(",", "")
    value = value.replace("$", "")
    value = value.replace("%", "")

    try:
        number = Decimal(value)

        if number == number.to_integral():
            return str(number.quantize(Decimal("1")))

        normalized = format(number.normalize(), "f")

        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")

        return normalized

    except InvalidOperation:
        return value.strip().lower()

def extract_gt_answer(answer_text):
    """
    GSM8K official format:

        reasoning ...
        #### 18
    """

    if answer_text is None:
        return None

    text = str(answer_text)

    if "####" in text:
        final_part = text.split("####")[-1].strip()

        matches = NUMBER_PATTERN.findall(final_part)

        if matches:
            return normalize_number(matches[-1])

        return normalize_number(final_part)

    matches = NUMBER_PATTERN.findall(text)

    if not matches:
        return None

    return normalize_number(matches[-1])

def extract_pred_answer(solution):
    """
    Extract the final numeric answer from a model-generated GSM8K solution.

    Priority:
    1. #### ...
    2. \\boxed{...}
    3. "final answer is ..."
    4. "answer is ..."
    5. last number in the trajectory
    """

    if solution is None:
        return None

    text = str(solution).strip()

    if not text:
        return None

    # --------------------------------------------------------
    # 1. GSM8K-style explicit answer
    # --------------------------------------------------------

    if "####" in text:
        final_part = text.split("####")[-1]

        matches = NUMBER_PATTERN.findall(final_part)

        if matches:
            return normalize_number(matches[-1])

    # --------------------------------------------------------
    # 2. LaTeX boxed answer
    # --------------------------------------------------------

    boxed_matches = re.findall(
        r"\\boxed\s*\{([^{}]+)\}",
        text
    )

    if boxed_matches:
        values = NUMBER_PATTERN.findall(boxed_matches[-1])

        if values:
            return normalize_number(values[-1])

    # --------------------------------------------------------
    # 3. Explicit final-answer phrases
    # --------------------------------------------------------

    patterns = [
        r"final\s+answer\s*(?:is|:|=)\s*([^\n]+)",
        r"the\s+answer\s*(?:is|:|=)\s*([^\n]+)",
        r"answer\s*(?:is|:|=)\s*([^\n]+)",
        r"therefore[^\n]*?([-+]?\d[\d,]*(?:\.\d+)?)",
        r"thus[^\n]*?([-+]?\d[\d,]*(?:\.\d+)?)",
    ]

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if matches:
            nums = NUMBER_PATTERN.findall(
                str(matches[-1])
            )

            if nums:
                return normalize_number(nums[-1])

    # --------------------------------------------------------
    # 4. Fallback: last number in entire solution
    # --------------------------------------------------------

    matches = NUMBER_PATTERN.findall(text)

    if not matches:
        return None

    return normalize_number(matches[-1])

def is_correct(pred_answer, gt_answer):

    if pred_answer is None or gt_answer is None:
        return False

    return normalize_number(pred_answer) == normalize_number(gt_answer)

def restore_original_order(item):
    """
    prm_rank.py stores candidates sorted by PRM score.

    For Best-of-N curve:
        N=1,2,4,8

    we must reconstruct the ORIGINAL sampling order using candidate_id,
    then only allow the first N sampled trajectories.

    This is important:
    N=4 must NOT mean the top-4 PRM candidates out of N=8.
    """

    ranking = item["ranking"]

    return sorted(
        ranking,
        key=lambda x: x["candidate_id"]
    )

def majority_vote(candidates):
    """
    Majority voting over extracted final answers.

    Tie-breaking:
    choose the answer appearing earliest among the tied candidates.
    """

    answers = [
        extract_pred_answer(x["candidate"])
        for x in candidates
    ]

    valid_answers = [
        x for x in answers
        if x is not None
    ]

    if not valid_answers:
        return None

    counts = Counter(valid_answers)

    max_count = max(counts.values())

    tied = {
        answer
        for answer, count in counts.items()
        if count == max_count
    }

    # deterministic tie break: earliest generated answer
    for answer in answers:
        if answer in tied:
            return answer

    return None

def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0

def compute_correct_error_metrics(classification_pairs):
    """
    Args:
        classification_pairs: iterable of
            (true_is_correct, predicted_is_correct)

    A trajectory is ground-truth CORRECT when its extracted final answer equals
    the GSM8K reference answer; otherwise it is ground-truth ERROR. The PRM
    prediction is also binary because prm_rank.py excludes NEUTRAL.
    """
    correct_to_correct = 0
    correct_to_error = 0
    error_to_correct = 0
    error_to_error = 0

    for true_is_correct, predicted_is_correct in classification_pairs:
        if true_is_correct and predicted_is_correct:
            correct_to_correct += 1
        elif true_is_correct and not predicted_is_correct:
            correct_to_error += 1
        elif not true_is_correct and predicted_is_correct:
            error_to_correct += 1
        else:
            error_to_error += 1

    total = (
        correct_to_correct
        + correct_to_error
        + error_to_correct
        + error_to_error
    )

    correct_precision = safe_div(
        correct_to_correct,
        correct_to_correct + error_to_correct,
    )
    correct_recall = safe_div(
        correct_to_correct,
        correct_to_correct + correct_to_error,
    )
    correct_f1 = safe_div(
        2 * correct_precision * correct_recall,
        correct_precision + correct_recall,
    )

    error_precision = safe_div(
        error_to_error,
        error_to_error + correct_to_error,
    )
    error_recall = safe_div(
        error_to_error,
        error_to_error + error_to_correct,
    )
    error_f1 = safe_div(
        2 * error_precision * error_recall,
        error_precision + error_recall,
    )

    return {
        "num_classified_candidates": total,
        "binary_accuracy": safe_div(
            correct_to_correct + error_to_error,
            total,
        ),
        "binary_balanced_accuracy": 0.5 * (correct_recall + error_recall),
        "binary_macro_f1": 0.5 * (correct_f1 + error_f1),
        "correct_precision": correct_precision,
        "correct_recall": correct_recall,
        "correct_f1": correct_f1,
        "error_precision": error_precision,
        "error_recall": error_recall,
        "error_f1": error_f1,
        "correct_to_error_rate": safe_div(
            correct_to_error,
            correct_to_correct + correct_to_error,
        ),
        "error_to_correct_rate": safe_div(
            error_to_correct,
            error_to_error + error_to_correct,
        ),
        "confusion_matrix": {
            "true_correct_pred_correct": correct_to_correct,
            "true_correct_pred_error": correct_to_error,
            "true_error_pred_correct": error_to_correct,
            "true_error_pred_error": error_to_error,
        },
    }

def evaluate_n(data, n, seed=42, classification_threshold=0.5):

    rng = random.Random(seed)

    total = 0

    prm_correct = 0
    majority_correct = 0
    oracle_correct = 0
    random_correct = 0

    candidate0_correct = 0

    # Expected random selection accuracy:
    # for each problem = #correct candidates / N
    random_expected_sum = 0.0

    per_problem = []
    classification_pairs = []

    for item in data:

        gt = extract_gt_answer(
            item["answer"]
        )

        original_candidates = restore_original_order(
            item
        )

        if len(original_candidates) < n:
            continue

        subset = original_candidates[:n]

        # ----------------------------------------------------
        # Evaluate every candidate
        # ----------------------------------------------------

        candidate_infos = []

        for candidate in subset:

            pred = extract_pred_answer(
                candidate["candidate"]
            )

            correct = is_correct(
                pred,
                gt
            )

            prm_score = float(candidate["prm_score"])
            predicted_is_correct = prm_score >= classification_threshold
            classification_pairs.append((correct, predicted_is_correct))

            candidate_infos.append({
                "candidate_id": candidate["candidate_id"],
                "prm_score": prm_score,
                "prm_error_score": 1.0 - prm_score,
                "true_label": "CORRECT" if correct else "ERROR",
                "prm_predicted_label": (
                    "CORRECT" if predicted_is_correct else "ERROR"
                ),
                "pred_answer": pred,
                "correct": correct,
            })

        # ----------------------------------------------------
        # Sample@1 / first sampled candidate
        # ----------------------------------------------------

        candidate0_ok = candidate_infos[0]["correct"]

        candidate0_correct += int(
            candidate0_ok
        )

        # ----------------------------------------------------
        # PRM Best-of-N
        # ----------------------------------------------------

        prm_best = max(
            subset,
            key=lambda x: x["prm_score"]
        )

        prm_answer = extract_pred_answer(
            prm_best["candidate"]
        )

        prm_ok = is_correct(
            prm_answer,
            gt
        )

        prm_correct += int(prm_ok)

        # ----------------------------------------------------
        # Majority vote
        # ----------------------------------------------------

        majority_answer = majority_vote(
            subset
        )

        majority_ok = is_correct(
            majority_answer,
            gt
        )

        majority_correct += int(
            majority_ok
        )

        # ----------------------------------------------------
        # Oracle
        # ----------------------------------------------------

        candidate_correctness = [
            x["correct"]
            for x in candidate_infos
        ]

        oracle_ok = any(
            candidate_correctness
        )

        oracle_correct += int(
            oracle_ok
        )

        # ----------------------------------------------------
        # Random actual sample
        # ----------------------------------------------------

        random_idx = rng.randrange(n)

        random_ok = candidate_correctness[
            random_idx
        ]

        random_correct += int(
            random_ok
        )

        # ----------------------------------------------------
        # Expected random accuracy
        # ----------------------------------------------------

        random_expected = (
            sum(candidate_correctness)
            / n
        )

        random_expected_sum += (
            random_expected
        )

        total += 1

        per_problem.append({
            "id": item.get("id"),
            "gt_answer": gt,

            "n": n,

            "candidate0_correct":
                bool(candidate0_ok),

            "prm_selected_candidate_id":
                prm_best["candidate_id"],

            "prm_selected_score":
                prm_best["prm_score"],

            "prm_pred_answer":
                prm_answer,

            "prm_correct":
                bool(prm_ok),

            "majority_answer":
                majority_answer,

            "majority_correct":
                bool(majority_ok),

            "oracle_correct":
                bool(oracle_ok),

            "num_correct_candidates":
                int(sum(candidate_correctness)),

            "candidate_results":
                candidate_infos,
        })

    if total == 0:
        raise RuntimeError(
            f"No valid samples for N={n}"
        )

    binary_metrics = compute_correct_error_metrics(classification_pairs)

    metrics = {
        "N": n,

        "num_problems": total,

        # Equivalent to ordinary sampling / generator pass rate
        "sample_1_accuracy":
            candidate0_correct / total,

        "random_accuracy":
            random_correct / total,

        "random_expected_accuracy":
            random_expected_sum / total,

        "majority_vote_accuracy":
            majority_correct / total,

        "prm_best_of_n_accuracy":
            prm_correct / total,

        "oracle_accuracy":
            oracle_correct / total,

        "prm_gain_vs_sample1":
            (
                prm_correct
                - candidate0_correct
            ) / total,

        "prm_gain_vs_majority":
            (
                prm_correct
                - majority_correct
            ) / total,

        "prm_oracle_gap":
            (
                oracle_correct
                - prm_correct
            ) / total,

        "classification_threshold": classification_threshold,

        # Candidate-level classification using CORRECT and ERROR only.
        **binary_metrics,
    }

    return metrics, per_problem


# ======================================================================
# PRM 加权答案投票与冻结配置；选择函数不读取参考答案
# ======================================================================
import math

import unicodedata

from pathlib import Path

def selection_config(method='weighted_vote', beta=0.5, temperature=0.25):
    if method not in ('prm', 'majority', 'weighted_vote'):
        raise ValueError('Unknown selection method')
    probability(beta)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('vote temperature must be finite and positive')
    return dict(method=method, beta=beta, temperature=temperature)

def select_candidate(candidates, config):
    """Caller supplies only its first N candidates. Higher score wins ties within an answer."""
    if not candidates:
        raise ValueError('No candidates to select')
    ordered = sorted(candidates, key=lambda c: c['candidate_id'])
    best = max(ordered, key=lambda c: c['prm_score'])
    if config['method'] == 'prm':
        return best, dict(method='prm', answer=extract_pred_answer(best['candidate']), groups=[])
    valid = [(c, c.get('_extracted_answer', extract_pred_answer(c['candidate']))) for c in ordered]
    valid = [(c, answer) for c, answer in valid if answer is not None]
    if not valid:
        return best, dict(method=config['method'], fallback='no_parseable_answer', groups=[])
    beta = 0.0 if config['method'] == 'majority' else config['beta']
    top = max(c['prm_score'] for c, _ in valid)
    exps = [math.exp((c['prm_score'] - top) / config['temperature']) for c, _ in valid]
    denominator = sum(exps)
    groups = {}
    for (c, answer), e in zip(valid, exps):
        weight = (1 - beta) / len(valid) + beta * e / denominator
        group = groups.setdefault(answer, dict(answer=answer, mass=0., count=0,
                                               first_id=c['candidate_id'], candidates=[]))
        group['mass'] += weight
        group['count'] += 1
        group['candidates'].append(c)
    # Earliest generated answer wins an exact mass tie, as in legacy majority.
    winning = min(groups.values(), key=lambda g: (-g['mass'], g['first_id']))
    representative = max(winning['candidates'], key=lambda c:c['prm_score'])
    diagnostic = [dict(answer=g['answer'], mass=g['mass'], count=g['count'], first_id=g['first_id'])
                  for g in sorted(groups.values(), key=lambda g:(-g['mass'], g['first_id']))]
    return representative, dict(method=config['method'], answer=winning['answer'], groups=diagnostic)

def problem_hash(problem):
    return digest(' '.join(unicodedata.normalize('NFKC', problem).split()))

def inference_contract(path):
    meta_path = Path(str(path) + '.meta.json')
    if not meta_path.exists():
        return None
    meta = read_json(meta_path)
    while isinstance(meta, dict):
        if 'model' in meta and 'max_length' in meta:
            return dict(model_files=meta['model'].get('files'), max_length=meta['max_length'],
                        separator=meta.get('separator'), dtype=meta.get('dtype'))
        meta = meta.get('source_metadata')
    return None

def load_policy(path, target_rows=None, target_scores=None, enforce_disjoint=True):
    doc = read_json(path)
    if doc.get('schema') != 'hierarchical-prm-policy-v3' or doc.get('fit_split') != 'validation':
        raise ValueError('Policy must be produced by evaluate.py tune on validation data')
    if not doc.get('validation_problem_hashes'):
        raise ValueError('Policy has no validation provenance')
    s = doc['scoring']
    scoring = scoring_config(s['score_mode'], s['role_alpha'], s['aggregation'], s['aggregation_lambda'],
                             s['classification_threshold'], s['tail_fraction'])
    selection = selection_config(**doc['selection'])
    if target_rows is not None and enforce_disjoint:
        overlap = set(doc['validation_problem_hashes']) & {problem_hash(r['problem']) for r in target_rows}
        if overlap:
            raise ValueError(f'Validation/test question overlap: {len(overlap)}; refusing test evaluation')
    if target_scores and doc.get('inference_contract'):
        actual = inference_contract(target_scores)
        if actual is not None and actual != doc['inference_contract']:
            raise ValueError('Policy and test scores use different checkpoint/packing settings')
    return scoring, selection, doc


# ======================================================================
# CPU 离线重排，复用步骤概率
# ======================================================================
import argparse

import copy

from pathlib import Path

def rerank_rows(rows, config):
    for source in rows:
        row = copy.deepcopy(source)
        for candidate in row['ranking']:
            values = []
            for step in candidate['step_predictions']:
                step['score'] = step_score(step, config)
                # Keep argmax correctness/role diagnostics distinct from fused score.
                step['score_threshold_label'] = 'NON_ERROR' if step['score'] >= config['classification_threshold'] else 'ERROR'
                values.append(step['score'])
            score = aggregate(values, config)
            candidate.update(prm_score=score, step_scores=values,
                             prm_predicted_label='CORRECT' if score >= config['classification_threshold'] else 'ERROR')
            candidate['first_score_threshold_error_step'] = next(
                (s['step_index'] for s in candidate['step_predictions'] if s['score_threshold_label'] == 'ERROR'), None)
        row['ranking'].sort(key=lambda c: (-c['prm_score'], c['candidate_id']))
        best = row['ranking'][0]
        row.update(prm_score_mode='hierarchical_' + config['score_mode'],
                   role_alpha=config['role_alpha'], aggregation=config['aggregation'],
                   aggregation_lambda=config['aggregation_lambda'],
                   tail_fraction=config['tail_fraction'],
                   classification_threshold=config['classification_threshold'],
                   best_candidate=best['candidate'], best_score=best['prm_score'],
                   best_predicted_label=best['prm_predicted_label'])
        yield row

def rerank(source, candidates, output, config, allow_legacy=False):
    if Path(source).resolve() == Path(output).resolve() or Path(candidates).resolve() == Path(output).resolve():
        raise ValueError('Output must not overwrite input scores/candidates')
    expected = read_jsonl(candidates)
    validate_candidates(expected)
    rows = read_jsonl(source)
    validate_ranked(rows, expected, require_steps=True)
    has_marker = Path(str(source) + '.done.json').exists()
    if has_marker and not verified(source):
        raise ValueError('Source completion marker/hash invalid; legacy flag cannot bypass corruption')
    if not has_marker and not allow_legacy:
        raise ValueError('Legacy scores have no completion/provenance marker; pass --allow-legacy explicitly after checking checkpoint and packing settings')
    provenance = (read_json(str(source) + '.meta.json')
                  if has_marker and Path(str(source) + '.meta.json').exists() else None)
    meta = dict(version=VERSION, source=str(Path(source).resolve()), source_sha256=file_hash(source),
                candidates_sha256=file_hash(candidates), scoring=config,
                provenance_status='verified_artifact' if has_marker else 'unverified_legacy_checkpoint',
                source_metadata=provenance)
    identity = dict(rerank_id=digest(meta))
    atomic_write(output, rerank_rows(rows, config), jsonl=True)
    atomic_write(str(output) + '.meta.json', meta)
    complete(output, identity, len(rows))
    return output

def add_scoring_arguments(p):
    p.add_argument('--score-mode', '--score_mode', default='correctness', choices=[
        'correctness', 'correctness_role_mean', 'correctness_role_product', 'correctness_role_blend'])
    p.add_argument('--role-alpha', type=float, default=0.0)
    p.add_argument('--aggregation', default='min_gmean_mix', choices=['min', 'mean', 'geometric_mean', 'min_gmean_mix', 'lower_tail_mean'])
    p.add_argument('--tail-fraction', type=float, default=0.25)
    p.add_argument('--aggregation-lambda', '--aggregation_lambda', type=float, default=0.5)
    p.add_argument('--classification-threshold', '--classification_threshold', type=float, default=0.5)

def config_from_args(a):
    return scoring_config(a.score_mode, a.role_alpha, a.aggregation, a.aggregation_lambda, a.classification_threshold, a.tail_fraction)


# ======================================================================
# 仅用独立验证集选参，保留原配置参与比较
# ======================================================================
import argparse

import itertools

from pathlib import Path

def fit(path, output, target_n=8, allow_legacy=False, alphas=(0., .15, .3),
        aggregations=('min_gmean_mix', 'geometric_mean', 'lower_tail_mean'),
        betas=(.5, 1.), temperatures=(.1, .25, .5)):
    if Path(path).resolve() == Path(output).resolve():
        raise ValueError('Policy output must not overwrite input scores')
    if target_n < 1:
        raise ValueError('target_n must be positive')
    has_marker = Path(str(path)+'.done.json').exists()
    if has_marker and not verified(path):
        raise ValueError('Corrupted validation scores')
    if not has_marker and not allow_legacy:
        raise ValueError('Pass --allow-legacy for old scores without provenance markers')
    rows = read_jsonl(path)
    validate_ranked(rows, require_steps=True)
    if any(r.get('prm_model_type') == 'three_class_prm' for r in rows):
        raise ValueError('Role-fusion tune is hierarchical-only. Use fixed shared settings for the 2x2 comparison.')
    problems, truths, compact = [], [], []
    for row in rows:
        if len(row['ranking']) < target_n:
            raise ValueError('Validation candidates insufficient for target N')
        truth = extract_gt_answer(row['answer'])
        if truth is None:
            raise ValueError('Unparseable validation reference answer')
        problems.append(problem_hash(row['problem']))
        truths.append(truth)
        compact.append([dict(candidate_id=c['candidate_id'], candidate=c['candidate'],
                             _extracted_answer=extract_pred_answer(c['candidate']),
                             step_predictions=[dict(correct_probability=s['correct_probability'],
                                                    role_probabilities={'correct': s['role_probabilities']['correct']})
                                               for s in c['step_predictions']])
                        for c in sorted(row['ranking'], key=lambda c:c['candidate_id'])[:target_n]])
    if len(set(problems)) != len(problems):
        raise ValueError('Duplicate validation question text; deduplicate before tuning')
    # Exact original scoring and pure PRM selection always enter first. Ties keep
    # this baseline, then earlier/simpler policies; validation gains are not test gains.
    configs = [scoring_config()]
    for alpha, agg in itertools.product(alphas, aggregations):
        c = scoring_config('correctness' if alpha == 0 else 'correctness_role_blend', alpha, agg)
        if c not in configs:
            configs.append(c)
    selections = [selection_config('prm'), selection_config('majority')]
    selections += [selection_config('weighted_vote', b, t) for b, t in itertools.product(betas, temperatures)]
    results = []
    for scoring in configs:
        scored = []
        for candidates in compact:
            scored.append([dict(candidate_id=c['candidate_id'], candidate=c['candidate'],
                                _extracted_answer=c['_extracted_answer'],
                                prm_score=aggregate([step_score(s, scoring) for s in c['step_predictions']], scoring))
                           for c in candidates])
        for selector in selections:
            # Majority is score independent: include only once.
            if selector['method'] == 'majority' and scoring != configs[0]:
                continue
            correct = 0
            for candidates, truth in zip(scored, truths):
                selected, _ = select_candidate(candidates, selector)
                correct += is_correct(extract_pred_answer(selected['candidate']), truth)
            results.append(dict(scoring=scoring, selection=selector, correct=correct,
                                accuracy=correct/len(rows)))
    winner = max(results, key=lambda r:r['correct'])
    doc = dict(schema='hierarchical-prm-policy-v3', fit_split='validation', target_n=target_n,
               validation_source=str(Path(path).resolve()), validation_sha256=file_hash(path),
               validation_provenance='verified_artifact' if has_marker else 'legacy_checkpoint_unverified',
               validation_problem_hashes=problems, inference_contract=inference_contract(path),
               num_validation_questions=len(rows), num_policies=len(results),
               objective='final_answer_accuracy_at_target_n', scoring=winner['scoring'],
               selection=winner['selection'], validation_accuracy=winner['accuracy'],
               original_validation_accuracy=results[0]['accuracy'],
               warning='Validation selection score is optimistic; report independent test performance.',
               trials=sorted(results, key=lambda r:-r['correct']))
    atomic_write(output, doc)
    print(f'Validation: original={results[0]["accuracy"]:.2%}, selected={winner["accuracy"]:.2%}; {len(results)} policies')
    print('Frozen policy:', output, winner['scoring'], winner['selection'])
    return doc


# ======================================================================
# Evaluate and report each selection strategy separately.
# ======================================================================
import argparse

import random

from pathlib import Path

def percentile(values, q):
    values = sorted(values)
    at = (len(values) - 1) * q
    lo = int(at)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - at) + values[hi] * (at - lo) if hi != lo else values[lo]

def paired_comparison(current, baseline, repeats=2000, seed=42):
    base = index_rows(baseline)
    if set(index_rows(current)) != set(base):
        raise ValueError('Paired comparison requires identical problem IDs')
    changes = [int(r['prm_correct']) - int(base[key(r)]['prm_correct']) for r in current]
    wins, losses = changes.count(1), changes.count(-1)
    result = dict(wins=wins, losses=losses, ties=changes.count(0),
                  delta_pp=100 * sum(changes) / len(changes), bootstrap_samples=repeats,
                  resampling_unit='problem', ci_method='paired percentile bootstrap')
    if repeats:
        rng = random.Random(seed)
        samples = [100 * sum(rng.choices(changes, k=len(changes))) / len(changes) for _ in range(repeats)]
        result['delta_pp_ci95'] = [percentile(samples, .025), percentile(samples, .975)]
    return result

def diagnostics(rows, n, details, threshold):
    pairwise, pair_count, micro_sum = [], 0, 0.0
    extraction_failures, total, truncated, truncation_known = 0, 0, 0, 0
    length_groups = {}
    for row in rows:
        gt = extract_gt_answer(row['answer'])
        candidates = sorted(row['ranking'], key=lambda c: c['candidate_id'])[:n]
        positives, negatives = [], []
        for c in candidates:
            pred = extract_pred_answer(c['candidate'])
            ok = is_correct(pred, gt)
            (positives if ok else negatives).append(c['prm_score'])
            extraction_failures += pred is None
            total += 1
            steps = c.get('step_predictions')
            count = len(steps) if steps is not None else len([x for x in c['candidate'].split('\n\n') if x.strip()])
            bucket = '0' if count == 0 else '1-4' if count <= 4 else '5-8' if count <= 8 else '9+'
            group = length_groups.setdefault(bucket, dict(candidates=0, answer_correct=0, correct_answer_flagged_error=0))
            group['candidates'] += 1
            group['answer_correct'] += ok
            group['correct_answer_flagged_error'] += ok and c['prm_score'] < threshold
            if steps and all('context_truncated' in s for s in steps):
                truncation_known += 1
                truncated += any(s['context_truncated'] for s in steps)
        if positives and negatives:
            comparisons = [float(p > m) + .5 * (p == m) for p in positives for m in negatives]
            pairwise.append(sum(comparisons) / len(comparisons))
            pair_count += len(comparisons)
            micro_sum += sum(comparisons)
    oracle_count = sum(d['oracle_correct'] for d in details)
    selected_correct = sum(d['prm_correct'] for d in details)
    misses = [d['id'] for d in details if d['oracle_correct'] and not d['prm_correct']]
    for group in length_groups.values():
        group['correct_answer_false_error_rate'] = (group['correct_answer_flagged_error'] / group['answer_correct']
                                                    if group['answer_correct'] else None)
    return dict(oracle_solvable_problems=oracle_count,
                oracle_conditioned_selection_accuracy=selected_correct / oracle_count if oracle_count else None,
                oracle_miss_count=len(misses), oracle_miss_ids=misses,
                pairwise_mixed_problems=len(pairwise), pairwise_pairs=pair_count,
                pairwise_accuracy_macro=sum(pairwise) / len(pairwise) if pairwise else None,
                pairwise_accuracy_micro=micro_sum / pair_count if pair_count else None,
                pairwise_tie_credit=0.5, answer_extraction_failure_rate=extraction_failures / total,
                truncation_known_candidates=truncation_known,
                truncated_candidate_fraction=truncated / truncation_known if truncation_known else None,
                by_step_count=length_groups)

def evaluate(ranked, output, n_values, threshold=.5, seed=42, details_output=None,
             baseline=None, bootstrap_samples=2000, split='test', selection=None, policy=None):
    selection = selection_config(**selection) if selection else selection_config()
    if not 0 < probability(threshold) < 1:
        raise ValueError('Invalid threshold')
    if bootstrap_samples < 0:
        raise ValueError('bootstrap_samples must be non-negative')
    if not n_values or any(n < 1 for n in n_values) or len(set(n_values)) != len(n_values):
        raise ValueError('N values must be unique positive integers')
    for source in [ranked] + ([baseline] if baseline else []):
        if Path(str(source) + '.done.json').exists() and not verified(source):
            raise ValueError(f'Corrupted/incomplete ranked file: {source}')
    outputs = [p for p in (output, details_output) if p]
    sources = {Path(p).resolve() for p in (ranked, baseline) if p}
    if any(Path(p).resolve() in sources for p in outputs) or len({Path(p).resolve() for p in outputs}) != len(outputs):
        raise ValueError('Output paths must be distinct from each other and inputs')
    rows = read_jsonl(ranked)
    validate_ranked(rows)
    policy_doc = None
    if policy:
        expected_scoring, selection, policy_doc = load_policy(
            policy, rows, ranked, enforce_disjoint=(split == 'test'))
        threshold = expected_scoring['classification_threshold']
        for row in rows:
            if row.get('prm_score_mode') != 'hierarchical_' + expected_scoring['score_mode'] or any(
                    row.get(k) != v for k, v in expected_scoring.items() if k != 'score_mode'):
                raise ValueError('Ranked scores do not match frozen scoring policy; use evaluate.py run --policy to rerank first')
    for row in rows:
        if extract_gt_answer(row['answer']) is None:
            raise ValueError(f'Unparseable ground truth for {row["id"]}')
        if len(row['ranking']) < max(n_values):
            raise ValueError(f'{row["id"]}: insufficient candidates; refusing silent dataset shrinkage')
    baseline_rows = None
    if baseline:
        baseline_rows = read_jsonl(baseline)
        expected = [dict(id=r['id'], problem=r['problem'], answer=r['answer'],
                         candidates=[c['candidate'] for c in sorted(r['ranking'], key=lambda c:c['candidate_id'])]) for r in rows]
        validate_ranked(baseline_rows, expected)
        baseline_index = index_rows(baseline_rows)
        baseline_rows = [baseline_index[key(row)] for row in rows]
    metrics, details = [], {}
    for n in n_values:
        metric, per_problem = evaluate_n(rows, n, seed=seed, classification_threshold=threshold)
        metric['diagnostics'] = diagnostics(rows, n, per_problem, threshold)
        selected_pairs = []
        # The selector receives no ground-truth field, and only the first N rows.
        for row, detail in zip(rows, per_problem):
            subset = sorted(row['ranking'], key=lambda c:c['candidate_id'])[:n]
            chosen, explanation = select_candidate(subset, selection)
            answer = extract_pred_answer(chosen['candidate'])
            correct = is_correct(answer, extract_gt_answer(row['answer']))
            detail.update(selected_candidate_id=chosen['candidate_id'], selected_answer=answer,
                          selected_correct=correct, selection_explanation=explanation)
            selected_pairs.append(dict(id=row['id'], prm_correct=correct))
        selected_count = sum(d['selected_correct'] for d in per_problem)
        accuracy = selected_count / len(per_problem)
        metric.update(selection_method=selection['method'], selected_strategy_accuracy=accuracy,
                      selected_gain_vs_majority=accuracy-metric['majority_vote_accuracy'],
                      selected_gain_vs_current_prm=accuracy-metric['prm_best_of_n_accuracy'],
                      selected_oracle_gap=metric['oracle_accuracy']-accuracy,
                      selected_oracle_miss_ids=[d['id'] for d in per_problem if d['oracle_correct'] and not d['selected_correct']],
                      selected_vs_current_prm=paired_comparison(selected_pairs, per_problem, repeats=0, seed=seed))
        if baseline_rows is not None:
            base_metric, base_detail = evaluate_n(baseline_rows, n, seed=seed, classification_threshold=threshold)
            metric['baseline_prm_best_of_n_accuracy'] = base_metric['prm_best_of_n_accuracy']
            metric['paired_vs_baseline'] = paired_comparison(per_problem, base_detail, bootstrap_samples, seed)
            metric['selected_vs_baseline'] = paired_comparison(selected_pairs, base_detail, bootstrap_samples, seed)
        metrics.append(metric)
        details[str(n)] = per_problem
    summary = dict(ranked_file=str(Path(ranked).resolve()), ranked_sha256=file_hash(ranked),
                   baseline_file=str(Path(baseline).resolve()) if baseline else None,
                   baseline_sha256=file_hash(baseline) if baseline else None,
                   n_values=n_values, split=split, seed=seed,
                   classification_mode='CORRECT_vs_ERROR_only', classification_threshold=threshold,
                   label_scope='final_answer_correctness_proxy_not_step_correctness',
                   selection_policy='first N by candidate_id; score ties prefer smallest candidate_id',
                   optimized_selection=selection,
                   frozen_policy_sha256=file_hash(policy) if policy else None,
                   policy_status='validation_fitted' if policy_doc else 'manual_unvalidated',
                   policy_target_n=policy_doc['target_n'] if policy_doc else None,
                   metrics=metrics)
    atomic_write(output, summary)
    if details_output:
        atomic_write(details_output, details)
    print('| N | Sample@1 | Majority | Reference PRM | Current PRM | Selected strategy | Oracle | Selected vs Majority | Selected gap |')
    print('|---|---|---|---|---|---|---|---|---|')
    for m in metrics:
        print(f"| {m['N']} | {m['sample_1_accuracy']:.2%} | {m['majority_vote_accuracy']:.2%} | "
              f"{format(m['baseline_prm_best_of_n_accuracy'], '.2%') if baseline else '—'} | "
              f"{m['prm_best_of_n_accuracy']:.2%} | {m['selected_strategy_accuracy']:.2%} | {m['oracle_accuracy']:.2%} | "
              f"{100*m['selected_gain_vs_majority']:+.2f}pp | {100*m['selected_oracle_gap']:.2f}pp |")
    print(f'Saved {output}; classification labels refer to final answers, not individual steps.')
    return summary


# ======================================================================
# End-to-end evaluation entry.
# ======================================================================
import argparse

import importlib.metadata

import os

from pathlib import Path

import subprocess

import sys

def execute(a):
    # 延迟导入合并器，避免模块循环依赖。
    from merge_candidates import merge
    # Read inference settings from latest_run.json.
    matched = match_reference_run(a) if a.match_run else None
    config = config_from_args(a)
    selection = selection_config(a.selection_method, a.vote_beta, a.vote_temperature)
    n_values = [int(n) for n in a.n_values.split(',')]
    if not n_values or any(n < 1 for n in n_values) or len(set(n_values)) != len(n_values):
        raise ValueError('N values must be unique positive integers')
    if a.num_gpus < 1 or a.batch_size < 1 or a.max_length < 2:
        raise ValueError('num_gpus, batch_size, max_length must be positive')
    candidates = Path(a.candidates).resolve()
    data = read_jsonl(candidates)
    validate_candidates(data, a.num_samples, a.expected_questions)
    if a.policy:
        config, selection, _ = load_policy(a.policy, data, enforce_disjoint=(a.split == 'test'))
    if a.model_kind == 'threeclass' and config['score_mode'] != 'correctness':
        raise ValueError('Three-class comparison requires correctness-only scoring for both models')
    if min(len(r['candidates']) for r in data) < max(n_values):
        raise ValueError('Requested N exceeds candidate count')
    output = Path(a.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if a.scores:
        raw = Path(a.scores).resolve()
    else:
        if not a.model_path:
            raise ValueError('--model-path required unless --scores is provided')
        versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers')}
        # Model bytes are hashed every invocation: replacing a checkpoint at the
        # same path must invalidate inference cache. No path-only cache identity.
        model = model_hash(a.model_path)
        signature = dict(version=VERSION, candidates_sha256=file_hash(candidates), model=model,
                         model_kind=a.model_kind, alignment_protocol=ALIGNMENT_PROTOCOL,
                         threeclass_ids=dict(ERROR=a.error_id, CORRECT=a.correct_id, NEUTRAL=a.neutral_id),
                         max_length=a.max_length, separator=a.separator, batch_size=a.batch_size,
                         dtype=a.dtype, world_size=a.num_gpus, code_sha256=code_hash(), dependencies=versions)
        run_id = digest(signature)
        cache = output / 'inference_cache' / run_id
        cache.mkdir(parents=True, exist_ok=True)
        manifest_path = cache / 'manifest.json'
        manifest = dict(**signature, run_id=run_id, candidates=str(candidates), shard_dir=str(cache / 'shards'))
        atomic_write(manifest_path, manifest)
        raw = cache / 'raw_scores.jsonl'
        if verified(raw, dict(run_id=run_id)):
            validate_ranked(read_jsonl(raw), data, require_steps=True)
            print(f'Reusing verified probability cache: {raw}', flush=True)
        else:
            worker = str(Path(__file__).with_name('hierarchical_prm_rank.py'))
            command = ([sys.executable, worker] if a.num_gpus == 1 else
                       [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                        f'--nproc_per_node={a.num_gpus}', worker])
            env = os.environ.copy()
            # Single-process use should not inherit a stale outer launcher rank.
            if a.num_gpus == 1:
                env.update(RANK='0', LOCAL_RANK='0', WORLD_SIZE='1')
            subprocess.run(command + ['--manifest', str(manifest_path)], check=True, env=env)
            merge(manifest_path, raw)
    actual_rows = read_jsonl(raw)
    expected_kind = 'three_class_prm' if a.model_kind == 'threeclass' else 'hierarchical_multitask_prm'
    if any(r.get('prm_model_type') != expected_kind for r in actual_rows):
        raise ValueError('Input cache model type differs from --model-kind')
    # Reranking and evaluation are cheap relative to inference; always rebuild.
    # Content+config-derived directory names prevent different variants colliding.
    if a.policy:
        load_policy(a.policy, data, raw, enforce_disjoint=(a.split == 'test'))
    # Reference PRM comparisons require a separate reference prediction file.
    reference = a.baseline_ranked
    identity = dict(scores_sha256=file_hash(raw), scoring=config, selection=selection,
                    code_sha256=code_hash(), n_values=n_values, split=a.split, seed=a.seed,
                    baseline_sha256=file_hash(reference) if reference else None, bootstrap_samples=a.bootstrap_samples,
                    policy_sha256=file_hash(a.policy) if a.policy else None)
    variant = output / 'variants' / digest(identity)[:20]
    variant.mkdir(parents=True, exist_ok=True)
    ranked = variant / 'ranked.jsonl'
    rerank(raw, candidates, ranked, config, a.allow_legacy)
    summary = variant / 'metrics.json'
    details = variant / 'details.json'
    evaluate(ranked, summary, n_values, config['classification_threshold'], a.seed,
             details, reference, a.bootstrap_samples, a.split, selection, a.policy)
    result = dict(raw_scores=str(raw), ranked=str(ranked), metrics=str(summary), details=str(details),
                  scoring=config, selection=selection, policy=a.policy, split=a.split,
                  model_kind=a.model_kind, alignment_protocol=ALIGNMENT_PROTOCOL,
                  matched_reference=matched)
    atomic_write(output / 'latest_run.json', result)
    print('Run paths:', result, flush=True)



def evaluation_cli(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ranked', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--details-output', '--details_output')
    p.add_argument('--n-values', '--n_values', default='1,2,3,4,5,6,7,8')
    p.add_argument('--classification-threshold', '--classification_threshold', type=float, default=.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--baseline-ranked')
    p.add_argument('--bootstrap-samples', type=int, default=2000)
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--selection-method', choices=['prm', 'majority', 'weighted_vote'], default='weighted_vote')
    p.add_argument('--vote-beta', type=float, default=.5)
    p.add_argument('--vote-temperature', type=float, default=.25)
    p.add_argument('--policy')
    a = p.parse_args(argv)
    evaluate(a.ranked, a.output, [int(n) for n in a.n_values.split(',')], a.classification_threshold,
             a.seed, a.details_output, a.baseline_ranked, a.bootstrap_samples, a.split,
             selection_config(a.selection_method, a.vote_beta, a.vote_temperature), a.policy)



def pipeline_cli(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--model-path')
    p.add_argument('--model-kind', choices=['threeclass','hierarchical'], default='threeclass')
    p.add_argument('--error-id', type=int)
    p.add_argument('--correct-id', type=int)
    p.add_argument('--neutral-id', type=int)
    p.add_argument('--match-run', help='Hierarchical latest_run.json; reuse recorded scoring/inference settings')
    p.add_argument('--scores', help='Offline input, no model/GPU required')
    p.add_argument('--allow-legacy', action='store_true')
    p.add_argument('--num-gpus', type=int, default=1)
    p.add_argument('--num-samples', type=int, default=8)
    p.add_argument('--expected-questions', type=int)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    p.add_argument('--separator', default='\n\n')
    p.add_argument('--n-values', default='1,2,3,4,5,6,7,8')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--baseline-ranked')
    p.add_argument('--bootstrap-samples', type=int, default=2000)
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--selection-method', choices=['prm', 'majority', 'weighted_vote'], default='weighted_vote')
    p.add_argument('--vote-beta', type=float, default=.5)
    p.add_argument('--vote-temperature', type=float, default=.25)
    p.add_argument('--policy', help='Frozen validation policy; overrides scoring/selection CLI settings')
    add_scoring_arguments(p)
    execute(p.parse_args(argv))



def rerank_cli(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores', required=True)
    p.add_argument('--candidates', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--allow-legacy', action='store_true')
    add_scoring_arguments(p)
    a = p.parse_args(argv)
    rerank(a.scores, a.candidates, a.output, config_from_args(a), a.allow_legacy)



def tune_cli(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation-ranked', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['validation'], required=True)
    p.add_argument('--target-n', type=int, default=8)
    p.add_argument('--allow-legacy', action='store_true')
    a = p.parse_args(argv)
    fit(a.validation_ranked, a.output, a.target_n, a.allow_legacy)



ORIGINAL_ATTACHMENT_CODE_HASH = '32ba5c8472e243d5fdf1b3639519c13f535939842ab9ec528e29de04a2dab9f0'

# Compare two models and two selection strategies using their saved predictions.
def source_manifest(path):
    meta = read_json(str(path) + '.meta.json')
    while isinstance(meta, dict):
        if 'model' in meta and 'max_length' in meta:
            return meta
        meta = meta.get('source_metadata')
    raise ValueError(f'Missing inference metadata: {path}')


def check_protocol(meta):
    declared = meta.get('alignment_protocol')
    if declared == ALIGNMENT_PROTOCOL:
        return
    if declared is None and meta.get('code_sha256') == ORIGINAL_ATTACHMENT_CODE_HASH:
        return
    raise ValueError('Unrecognized packing/aggregation implementation. Use this bundle to rerun hierarchy; do not mix contiguous/geometric-mix caches with this skip/arithmetic-mix protocol.')


def match_reference_run(a):
    if a.policy:
        raise ValueError('--match-run and --policy cannot be combined; preserve the reference effective settings')
    run = read_json(a.match_run)
    raw = run['raw_scores']
    if not verified(raw):
        raise ValueError('Reference raw scores must have a valid completion marker')
    meta = source_manifest(raw)
    check_protocol(meta)
    rows = read_jsonl(raw)
    validate_ranked(rows, read_jsonl(a.candidates), require_steps=True)
    if any(r.get('prm_model_type') != 'hierarchical_multitask_prm' for r in rows):
        raise ValueError('--match-run must identify a hierarchical run')
    if meta['candidates_sha256'] != file_hash(a.candidates):
        raise ValueError('Reference and current candidate file hashes differ')
    if run['scoring']['score_mode'] != 'correctness' or run['scoring']['role_alpha'] != 0:
        raise ValueError('Reference uses Role fusion. Rerun both models in correctness mode for this comparison')
    if run['selection']['method'] != 'weighted_vote':
        raise ValueError('Reference must use weighted_vote to reproduce its voting settings')
    if run.get('split') != a.split:
        raise ValueError('Reference/current split differs')
    for name,value in run['scoring'].items():
        setattr(a, name, value)
    for name in ('max_length','separator','dtype','batch_size'):
        setattr(a, name, meta[name])
    a.selection_method = 'weighted_vote'
    a.vote_beta = run['selection']['beta']
    a.vote_temperature = run['selection']['temperature']
    old_metrics = read_json(run['metrics'])
    if old_metrics.get('ranked_sha256') != file_hash(run['ranked']):
        raise ValueError('Reference metrics/ranked file hash mismatch')
    a.seed = old_metrics['seed']
    a.n_values = ','.join(map(str,old_metrics['n_values']))
    # GPU count can change, but tokenizer IDs and packing will be checked in compare.
    print('Matched recorded hierarchical scoring, selection, packing and N budgets:',
          run['scoring'], run['selection'], flush=True)
    return dict(path=str(Path(a.match_run).resolve()), sha256=file_hash(a.match_run),
                raw_sha256=file_hash(raw), protocol=ALIGNMENT_PROTOCOL)


def compare_cli(argv):
    p = argparse.ArgumentParser(description='Strict same-candidate model x selection comparison, CPU only')
    p.add_argument('--threeclass-run', required=True, help='Three-class latest_run.json')
    p.add_argument('--hierarchical-run', required=True, help='Hierarchical latest_run.json')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--n-values', default='1,2,3,4,5,6,7,8')
    p.add_argument('--bootstrap-samples', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--allow-unverified-inputs', action='store_true',
                   help='Old attachment cache lacks per-step token hashes; explicitly mark comparison as not fully input-verified')
    a = p.parse_args(argv)
    if a.bootstrap_samples < 0:
        raise ValueError('bootstrap-samples must be nonnegative')
    docs = [read_json(a.threeclass_run), read_json(a.hierarchical_run)]
    if docs[0]['scoring'] != docs[1]['scoring']:
        raise ValueError('Scoring settings differ; rerun with --match-run or identical settings')
    scoring = docs[0]['scoring']
    if scoring['score_mode'] != 'correctness' or scoring['role_alpha'] != 0:
        raise ValueError('2x2 comparison requires correctness-only scoring')
    vote = dict(docs[0]['selection'], method='weighted_vote')
    if vote != dict(docs[1]['selection'], method='weighted_vote'):
        raise ValueError('Voting beta/temperature differ')
    selection_config(**vote)
    if docs[0]['split'] != docs[1]['split']:
        raise ValueError('Splits differ')
    manifests, datasets = [], []
    for doc in docs:
        if not verified(doc['ranked']) or not verified(doc['raw_scores']):
            raise ValueError('Both runs require valid ranked/raw completion markers')
        meta = source_manifest(doc['raw_scores']); check_protocol(meta)
        manifests.append(meta)
        ranked_meta = read_json(doc['ranked']+'.meta.json')
        if (ranked_meta.get('scoring') != scoring or
                ranked_meta.get('source_sha256') != file_hash(doc['raw_scores'])):
            raise ValueError('Run record disagrees with ranked provenance')
        rows = read_jsonl(doc['ranked']); validate_ranked(rows, require_steps=True)
        # Recompute scores from step probabilities to detect incompatible legacy aggregation.
        for row in rows:
            for c in row['ranking']:
                expected_score = aggregate([step_score(t,scoring) for t in c['step_predictions']],scoring)
                if not math.isclose(expected_score,c['prm_score'],abs_tol=1e-10,rel_tol=1e-9):
                    raise ValueError('Stored scores do not match the shared arithmetic aggregation')
        datasets.append(rows)
    for field in ('candidates_sha256','max_length','separator','dtype','batch_size','dependencies'):
        if manifests[0].get(field) != manifests[1].get(field):
            raise ValueError(f'Inference contract differs: {field}')
    left,right = datasets
    if any(r.get('prm_model_type')!='three_class_prm' for r in left):
        raise ValueError('First run must be three-class PRM')
    if any(r.get('prm_model_type')!='hierarchical_multitask_prm' for r in right):
        raise ValueError('Second run must be hierarchical PRM')
    expected = [dict(id=r['id'],problem=r['problem'],answer=r['answer'],candidates=[c['candidate'] for c in sorted(r['ranking'],key=lambda c:c['candidate_id'])]) for r in left]
    validate_ranked(right,expected,require_steps=True)
    ri=index_rows(right); right=[ri[key(r)] for r in left]
    if len({problem_hash(r['problem']) for r in left}) != len(left):
        raise ValueError('Duplicate question text: problem-level bootstrap requires one row per problem')
    unverified = 0
    for l,r in zip(left,right):
        lf,rf=l.get('tokenizer_fingerprint'),r.get('tokenizer_fingerprint')
        if lf and rf and lf!=rf: raise ValueError('Tokenizer vocabulary/special-token semantics differ')
        if not lf or not rf: unverified+=1
        for lc,rc in zip(sorted(l['ranking'],key=lambda c:c['candidate_id']),sorted(r['ranking'],key=lambda c:c['candidate_id'])):
            for ls,rs in zip(lc['step_predictions'],rc['step_predictions']):
                lh,rh=ls.get('input_ids_sha256'),rs.get('input_ids_sha256')
                if lh and rh and lh!=rh: raise ValueError('Actual step input token IDs differ')
                if not lh or not rh: unverified+=1
    if unverified and not a.allow_unverified_inputs:
        raise ValueError('Old cache has no token fingerprints. Rerun hierarchy with this bundle for strict alignment, or explicitly use --allow-unverified-inputs for a qualified historical comparison.')
    ns=[int(n) for n in a.n_values.split(',')]
    if not ns or min(ns)<1 or len(set(ns))!=len(ns) or any(len(r['ranking'])<max(ns) for r in left):
        raise ValueError('Invalid N values')
    results=[]
    detail_all={}
    for n in ns:
        cells={}
        for model,rows in [('Three-class PRM',left),('Hierarchical PRM',right)]:
            for method in ('prm','weighted_vote'):
                choices=[]
                for row in rows:
                    truth=extract_gt_answer(row['answer'])
                    if truth is None: raise ValueError('Unparseable reference answer')
                    subset=sorted(row['ranking'],key=lambda c:c['candidate_id'])[:n]
                    chosen,_=select_candidate(subset,dict(vote,method=method))
                    choices.append(dict(id=row['id'],prm_correct=is_correct(extract_pred_answer(chosen['candidate']),truth),candidate_id=chosen['candidate_id']))
                cells[model+' / '+method]=choices
        accuracies={k:sum(c['prm_correct'] for c in v)/len(v) for k,v in cells.items()}
        deltas={}
        for method in ('prm','weighted_vote'):
            deltas['hierarchical_minus_threeclass_'+method]=paired_comparison(cells['Hierarchical PRM / '+method],cells['Three-class PRM / '+method],a.bootstrap_samples,a.seed)
        for model in ('Three-class PRM','Hierarchical PRM'):
            deltas[model+'_vote_minus_prm']=paired_comparison(cells[model+' / weighted_vote'],cells[model+' / prm'],a.bootstrap_samples,a.seed)
        results.append(dict(N=n,accuracies=accuracies,paired_differences=deltas))
        detail_all[str(n)]=cells
    output=Path(a.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    summary=dict(schema='prm-2x2-aligned-v1',questions=len(left),split=docs[0]['split'],
                 scoring=scoring,voting=vote,input_packing_verified=(unverified==0),
                 protocol=ALIGNMENT_PROTOCOL,bootstrap_samples=a.bootstrap_samples,seed=a.seed,
                 source_runs=[dict(path=str(Path(x).resolve()),sha256=file_hash(x)) for x in (a.threeclass_run,a.hierarchical_run)],
                 source_ranked_sha256=[file_hash(d['ranked']) for d in docs],results=results)
    atomic_write(output/'comparison_2x2.json',summary)
    atomic_write(output/'comparison_details.json',detail_all)
    lines=['# Same-candidate PRM comparison','',f'Questions: {len(left)}; input packing verified: {unverified==0}.',
           '', '| N | Three-class: highest score | Three-class: weighted vote | Hierarchical: highest score | Hierarchical: weighted vote |',
           '|---|---:|---:|---:|---:|']
    for result in results:
        vals=result['accuracies']
        lines.append('| '+str(result['N'])+' | '+' | '.join(f"{100*vals[k]:.2f}%" for k in ('Three-class PRM / prm','Three-class PRM / weighted_vote','Hierarchical PRM / prm','Hierarchical PRM / weighted_vote'))+' |')
    if unverified: lines+=['','Historical inputs lack token fingerprints: this is not a fully verified token-identical comparison.']
    lines+=['','Paired differences and 95% intervals are in comparison_2x2.json. No parameter search is performed.']
    text='\n'.join(lines)+'\n'
    (output/'comparison_2x2.md').write_text(text,encoding='utf-8')
    print(text)


# Use --ranked for direct evaluation or a subcommand for other tasks.
if __name__ == '__main__':
    arguments = sys.argv[1:]
    commands = {'run': pipeline_cli, 'rerank': rerank_cli, 'tune': tune_cli, 'report': evaluation_cli, 'compare': compare_cli}
    if arguments and arguments[0] in commands:
        commands[arguments[0]](arguments[1:])
    else:
        evaluation_cli(arguments)
