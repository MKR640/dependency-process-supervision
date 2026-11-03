#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Generate one candidate pool shared by all PRM comparisons.
# Single-device output uses the *_rank0.jsonl filename convention.

import os
import re
import json
import random
import argparse
from typing import List, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Data
# ============================================================

def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_no}: {e}"
                ) from e
    return data


def get_problem(item):
    if "problem" in item:
        return item["problem"]
    if "question" in item:
        return item["question"]
    raise ValueError(
        f"Unknown data format: {list(item.keys())}"
    )


# ============================================================
# Prompt
# ============================================================

def build_prompt(problem: str) -> str:
    """
    Prompt aligned with the Format-SFT generator.

    Expected raw SFT output:
        <think>
        Step1: ...

        Step2: ...
        </think>
        <answer>...</answer>

    Downstream we normalize it into the PRM trajectory format:
        Step1: ...

        Step2: ...
        #### final_number
    """
    return (
        "Solve exactly ONE math problem.\n\n"
        "Return the solution using exactly this structure:\n"
        "<think>\n"
        "Step1: <first reasoning step>\n\n"
        "Step2: <second reasoning step>\n\n"
        "Step3: <continue with as many steps as needed>\n"
        "</think>\n"
        "<answer><final numerical answer only></answer>\n\n"
        "Rules:\n"
        "- Number the reasoning steps consecutively as Step1, Step2, ..., StepN.\n"
        "- Put exactly one blank line between reasoning steps.\n"
        "- Keep one logical calculation/reasoning action in each step.\n"
        "- Do not repeat the problem.\n"
        "- Do not start another problem or add unrelated text.\n"
        "- Put only the final numerical answer inside <answer>...</answer>.\n"
        "- End immediately after </answer>.\n\n"
        f"Problem:\n{problem}\n"
    )


# ============================================================
# Candidate cleanup / format normalization
# ============================================================

# Matches Step1:, Step 1:, Step1), Step 1. ...
STEP_HEADER_RE = re.compile(
    r"(?im)^\s*step\s*\d+\s*[:.)-]\s*"
)

NUMBER_RE = re.compile(
    r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
)

THINK_RE = re.compile(
    r"<think>\s*(.*?)\s*</think>",
    flags=re.IGNORECASE | re.DOTALL,
)

ANSWER_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)


def truncate_extra_problem(text: str) -> str:
    """
    Cut off common drift into another task/problem.

    For SFT output, anything after </answer> is also discarded.
    """
    text = (text or "").strip()

    # SFT output should end at </answer>.
    answer_end = re.search(
        r"</answer>",
        text,
        flags=re.IGNORECASE,
    )
    if answer_end:
        text = text[:answer_end.end()]

    markers = [
        "\nProblem:",
        "\nQuestion:",
        "\nNew Problem:",
        "\nNext Problem:",
        "\nSolve exactly ONE math problem",
    ]

    positions = []
    for marker in markers:
        pos = text.find(marker)
        if pos > 0:
            positions.append(pos)

    if positions:
        text = text[: min(positions)]

    return text.strip()


def extract_sft_sections(text: str) -> Tuple[str, Optional[str]]:
    """
    Return:
        reasoning_text,
        answer_text

    Handles both complete and partially truncated SFT tags.
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    if not text:
        return "", None

    # -----------------------------
    # Answer
    # -----------------------------
    answer_text = None

    answer_match = ANSWER_RE.search(text)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        # Partial <answer> without closing tag.
        partial_answer = re.search(
            r"<answer>\s*(.*)$",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if partial_answer:
            answer_text = partial_answer.group(1).strip()

    # -----------------------------
    # Reasoning
    # -----------------------------
    think_match = THINK_RE.search(text)

    if think_match:
        reasoning_text = think_match.group(1).strip()
    else:
        # Partial <think> without closing tag.
        partial_think = re.search(
            r"<think>\s*(.*)$",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if partial_think:
            reasoning_text = partial_think.group(1)

            # Do not include answer section in PRM reasoning.
            reasoning_text = re.split(
                r"<answer>",
                reasoning_text,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
        else:
            # Fallback for older/base-style outputs.
            reasoning_text = re.split(
                r"<answer>",
                text,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()

    # Remove any leftover tags from reasoning.
    reasoning_text = re.sub(
        r"</?think>",
        "",
        reasoning_text,
        flags=re.IGNORECASE,
    ).strip()

    return reasoning_text, answer_text


def extract_final_number(text: str) -> Optional[str]:
    """
    Best-effort final numeric answer extraction for GSM8K-like data.

    Priority:
      1) <answer>...</answer>
      2) #### ...
      3) \\boxed{...}
      4) explicit answer phrases
      5) last number
    """
    if not text:
        return None

    text = str(text).strip()

    # --------------------------------------------------------
    # 1. Format-SFT answer tag
    # --------------------------------------------------------
    _, answer_text = extract_sft_sections(text)

    if answer_text:
        nums = NUMBER_RE.findall(answer_text)
        if nums:
            return nums[-1].replace(",", "")

    # --------------------------------------------------------
    # 2. GSM8K explicit final answer
    # --------------------------------------------------------
    if "####" in text:
        tail = text.rsplit("####", 1)[-1]
        nums = NUMBER_RE.findall(tail)
        if nums:
            return nums[-1].replace(",", "")

    # --------------------------------------------------------
    # 3. LaTeX boxed answer
    # --------------------------------------------------------
    boxed_matches = re.findall(
        r"\\boxed\s*\{([^{}]+)\}",
        text,
    )
    if boxed_matches:
        nums = NUMBER_RE.findall(boxed_matches[-1])
        if nums:
            return nums[-1].replace(",", "")

    # --------------------------------------------------------
    # 4. Explicit answer phrases
    # --------------------------------------------------------
    patterns = [
        r"(?i)final\s+answer\s*(?:is|:|=)\s*([^\n]+)",
        r"(?i)the\s+answer\s*(?:is|:|=)\s*([^\n]+)",
        r"(?i)answer\s*(?:is|:|=)\s*([^\n]+)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            nums = NUMBER_RE.findall(matches[-1])
            if nums:
                return nums[-1].replace(",", "")

    # --------------------------------------------------------
    # 5. Last-number fallback
    # --------------------------------------------------------
    nums = NUMBER_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "")

    return None


def split_reasoning_chunks(
    raw_text: str,
) -> Tuple[List[str], Optional[str]]:
    """
    Parse an SFT/base completion into reasoning chunks.

    The <think> and <answer> wrappers are NOT passed into the PRM.
    The answer is converted to a trailing GSM8K marker:
        #### <number>

    Reasoning priority:
      1) explicit StepN headers
      2) blank-line blocks
      3) non-empty lines
      4) one whole paragraph
    """
    raw_text = (
        raw_text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )

    reasoning_text, answer_text = extract_sft_sections(raw_text)

    final_line = None

    # Prefer the explicit SFT <answer>.
    if answer_text:
        nums = NUMBER_RE.findall(answer_text)
        if nums:
            final_line = f"#### {nums[-1].replace(',', '')}"

    # Backward compatibility with old #### outputs.
    if final_line is None and "####" in raw_text:
        _, after = raw_text.rsplit("####", 1)
        nums = NUMBER_RE.findall(after)
        if nums:
            final_line = f"#### {nums[-1].replace(',', '')}"

    # If no <think> was present, strip a trailing #### section
    # from the fallback reasoning text.
    if "####" in reasoning_text:
        reasoning_text = reasoning_text.rsplit("####", 1)[0].strip()

    if not reasoning_text:
        return [], final_line

    # --------------------------------------------------------
    # Explicit StepN headers
    # --------------------------------------------------------
    matches = list(
        STEP_HEADER_RE.finditer(reasoning_text)
    )

    if matches:
        chunks = []

        for i, match in enumerate(matches):
            start = match.end()
            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(reasoning_text)
            )

            chunk = reasoning_text[start:end].strip()

            if chunk:
                chunks.append(chunk)

        if chunks:
            return chunks, final_line

    # --------------------------------------------------------
    # Blank-line blocks
    # --------------------------------------------------------
    blocks = [
        x.strip()
        for x in re.split(
            r"\n\s*\n+",
            reasoning_text,
        )
        if x.strip()
    ]

    if len(blocks) >= 2:
        return blocks, final_line

    # --------------------------------------------------------
    # Non-empty lines
    # --------------------------------------------------------
    lines = [
        x.strip()
        for x in reasoning_text.split("\n")
        if x.strip()
    ]

    if len(lines) >= 2:
        return lines, final_line

    return (
        [reasoning_text] if reasoning_text else [],
        final_line,
    )


def normalize_candidate(
    raw_completion: str,
    append_final_marker: bool = True,
):
    """
    Normalize Format-SFT output into the downstream PRM format.

    Raw:
        <think>
        Step1: ...
        Step2: ...
        </think>
        <answer>18</answer>

    Normalized:
        Step1: ...

        Step2: ...
        #### 18

    This intentionally removes <think>/<answer> before PRM ranking.
    """
    text = (raw_completion or "").strip()

    if not text:
        return "", 0, False

    text = truncate_extra_problem(text)

    chunks, final_line = split_reasoning_chunks(text)

    # If the model did not produce a usable explicit answer,
    # recover the number from its own completion.
    if final_line is None and append_final_marker:
        final_number = extract_final_number(text)

        if final_number is not None:
            final_line = f"#### {final_number}"

    normalized_steps = []

    for i, chunk in enumerate(chunks, start=1):
        # Remove a possible duplicated StepN prefix inside the chunk.
        chunk = STEP_HEADER_RE.sub(
            "",
            chunk,
            count=1,
        ).strip()

        # Remove accidental wrapper remnants.
        chunk = re.sub(
            r"</?(?:think|answer)>",
            "",
            chunk,
            flags=re.IGNORECASE,
        ).strip()

        if chunk:
            # IMPORTANT: match the user's PRM training format: Step1, Step2, ...
            normalized_steps.append(
                f"Step{i}: {chunk}"
            )

    normalized = "\n\n".join(
        normalized_steps
    )

    if final_line:
        # Keep answer attached to the last reasoning block by a single newline,
        # so prm_rank.py's blank-line split does not create a fake answer step.
        if normalized:
            normalized = (
                normalized
                + "\n"
                + final_line
            )
        else:
            normalized = final_line

    step_count = len(normalized_steps)

    # One-step GSM8K solutions are still valid; require a recoverable final answer.
    format_ok = (
        step_count >= 1
        and final_line is not None
        and len(normalized) >= 20
    )

    return (
        normalized.strip(),
        step_count,
        format_ok,
    )


# ============================================================
# Generation
# ============================================================

def decode_batch_outputs(
    tokenizer,
    outputs,
    padded_prompt_len,
    batch_size,
    num_samples,
):
    grouped = [
        []
        for _ in range(batch_size)
    ]

    for i in range(batch_size):
        for j in range(num_samples):
            out_idx = (
                i * num_samples + j
            )

            completion_ids = (
                outputs[out_idx][padded_prompt_len:]
            )

            text = tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
            )

            grouped[i].append(
                text.strip()
            )

    return grouped


def generate_once(
    model,
    tokenizer,
    prompts,
    device,
    args,
    num_return_sequences,
):
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=args.max_prompt_tokens,
        return_tensors="pt",
    )

    input_ids = (
        inputs["input_ids"].to(device)
    )

    attention_mask = (
        inputs["attention_mask"].to(device)
    )

    padded_prompt_len = (
        input_ids.shape[1]
    )

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=num_return_sequences,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return decode_batch_outputs(
        tokenizer,
        outputs,
        padded_prompt_len=padded_prompt_len,
        batch_size=len(prompts),
        num_samples=num_return_sequences,
    )


def regenerate_invalid_slots(
    model,
    tokenizer,
    prompts,
    candidate_groups,
    device,
    args,
):
    """
    Re-sample only malformed/empty candidates.
    """
    total_regenerated = 0

    for _attempt in range(
        args.max_regen_attempts
    ):
        jobs = []

        for problem_idx, group in enumerate(
            candidate_groups
        ):
            for candidate_idx, candidate_info in enumerate(
                group
            ):
                if not candidate_info[2]:
                    jobs.append(
                        (
                            problem_idx,
                            candidate_idx,
                            prompts[problem_idx],
                        )
                    )

        if not jobs:
            break

        for start in range(
            0,
            len(jobs),
            args.regen_batch_size,
        ):
            chunk_jobs = jobs[
                start:
                start + args.regen_batch_size
            ]

            chunk_prompts = [
                job[2]
                for job in chunk_jobs
            ]

            raw_groups = generate_once(
                model,
                tokenizer,
                chunk_prompts,
                device,
                args,
                num_return_sequences=1,
            )

            for job, raw_group in zip(
                chunk_jobs,
                raw_groups,
            ):
                problem_idx, candidate_idx, _ = job

                raw = (
                    raw_group[0]
                    if raw_group
                    else ""
                )

                replacement = normalize_candidate(
                    raw,
                    append_final_marker=args.append_final_marker,
                )

                old = (
                    candidate_groups
                    [problem_idx]
                    [candidate_idx]
                )

                if (
                    replacement[2]
                    or replacement[1] > old[1]
                    or (
                        replacement[1] == old[1]
                        and len(replacement[0]) > len(old[0])
                    )
                ):
                    candidate_groups[
                        problem_idx
                    ][
                        candidate_idx
                    ] = replacement

                total_regenerated += 1

    return (
        candidate_groups,
        total_regenerated,
    )


# ============================================================
# Main
# ============================================================

def main(args):
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            "0",
        )
    )

    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )

    torch.cuda.set_device(
        local_rank
    )

    device = f"cuda:{local_rank}"

    seed = (
        args.seed
        + local_rank
    )

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("=" * 88)
    print("Best-of-N Candidate Generation (Format-SFT)")
    print("=" * 88)
    print(f"Rank       : {local_rank}/{world_size}")
    print(f"Device     : {device}")
    print(f"Model      : {args.model_path}")
    print(f"Data       : {args.data}")
    print(f"N          : {args.num_samples}")
    print(f"Batch size : {args.batch_size}")
    print(f"Max new tok: {args.max_new_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )

    # Decoder-only batched generation should use left padding.
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print(
        f"eos_token={tokenizer.eos_token!r}, "
        f"eos_token_id={tokenizer.eos_token_id}, "
        f"pad_token={tokenizer.pad_token!r}, "
        f"pad_token_id={tokenizer.pad_token_id}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map={"": local_rank},
        trust_remote_code=True,
    )

    model.eval()

    full_data = load_jsonl(
        args.data
    )

    if args.num_questions is not None:
        full_data = full_data[
            :args.num_questions
        ]

    indexed_data = list(
        enumerate(full_data)
    )

    rank_data = indexed_data[
        local_rank::world_size
    ]

    print(
        f"Total selected problems : "
        f"{len(full_data)}"
    )

    print(
        f"Rank {local_rank} problems   : "
        f"{len(rank_data)}"
    )

    output_dir = os.path.dirname(
        args.output
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    root, ext = os.path.splitext(
        args.output
    )

    ext = ext or ".jsonl"

    output_file = (
        f"{root}_rank{local_rank}{ext}"
    )

    stats = {
        "problems": 0,
        "candidates": 0,
        "format_ok": 0,
        "format_bad": 0,
        "empty": 0,
        "regen_calls": 0,
        "step_count_sum": 0,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as fout:

        for start in tqdm(
            range(
                0,
                len(rank_data),
                args.batch_size,
            ),
            desc=f"GPU-{local_rank}",
        ):
            batch_pairs = rank_data[
                start:
                start + args.batch_size
            ]

            global_indices = [
                x[0]
                for x in batch_pairs
            ]

            batch = [
                x[1]
                for x in batch_pairs
            ]

            problems = [
                get_problem(x)
                for x in batch
            ]

            prompts = [
                build_prompt(x)
                for x in problems
            ]

            raw_groups = generate_once(
                model,
                tokenizer,
                prompts,
                device,
                args,
                num_return_sequences=args.num_samples,
            )

            candidate_groups = []

            for raw_candidates in raw_groups:
                normalized_group = [
                    normalize_candidate(
                        raw,
                        append_final_marker=args.append_final_marker,
                    )
                    for raw in raw_candidates
                ]

                candidate_groups.append(
                    normalized_group
                )

            if args.max_regen_attempts > 0:
                candidate_groups, regen_count = (
                    regenerate_invalid_slots(
                        model,
                        tokenizer,
                        prompts,
                        candidate_groups,
                        device,
                        args,
                    )
                )

                stats["regen_calls"] += (
                    regen_count
                )

            for (
                item,
                source_index,
                problem,
                normalized_group,
            ) in zip(
                batch,
                global_indices,
                problems,
                candidate_groups,
            ):
                candidates = [
                    x[0]
                    for x in normalized_group
                ]

                step_counts = [
                    x[1]
                    for x in normalized_group
                ]

                format_flags = [
                    x[2]
                    for x in normalized_group
                ]

                row_id = item.get(
                    "id",
                    source_index,
                )

                record = {
                    "id": row_id,
                    "source_index": source_index,
                    "generator_model": args.model_path,
                    "problem": problem,
                    "answer": item.get("answer"),
                    "num_candidates": len(candidates),
                    "candidates": candidates,
                    "candidate_step_counts": step_counts,
                    "candidate_format_ok": format_flags,
                }

                fout.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                stats["problems"] += 1
                stats["candidates"] += len(candidates)
                stats["format_ok"] += sum(format_flags)
                stats["format_bad"] += (
                    len(format_flags)
                    - sum(format_flags)
                )
                stats["empty"] += sum(
                    1
                    for x in candidates
                    if not x.strip()
                )
                stats["step_count_sum"] += sum(
                    step_counts
                )

            fout.flush()

    print("\n" + "=" * 88)
    print(f"Rank {local_rank} finished")
    print(f"Saved         : {output_file}")
    print(f"Problems      : {stats['problems']}")
    print(f"Candidates    : {stats['candidates']}")
    print(
        f"Format OK     : "
        f"{stats['format_ok']} / "
        f"{stats['candidates']}"
    )
    print(f"Format bad    : {stats['format_bad']}")
    print(f"Empty         : {stats['empty']}")
    print(f"Regen samples : {stats['regen_calls']}")

    if stats["candidates"]:
        print(
            f"Avg steps     : "
            f"{stats['step_count_sum'] / stats['candidates']:.2f}"
        )

    print("=" * 88)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        default=(
            "models/"
            "Qwen3-4B-SFT-generator"
        ),
    )

    parser.add_argument(
        "--data",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--num_questions",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
    )

    # 1024 can truncate some of the SFT model's long-tail responses.
    # 2048 is a safer Best-of-N default while EOS still stops normal outputs early.
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max_regen_attempts",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--regen_batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--append_final_marker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append '#### <number>' using the completion's own "
            "extracted final number when absent."
        ),
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
    )

    args = parser.parse_args()
    main(args)
