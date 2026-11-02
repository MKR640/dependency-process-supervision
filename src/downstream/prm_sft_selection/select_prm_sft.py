#!/usr/bin/env python3
"""Quizlet SFT selection. CPU prepare/select; one token-classification PRM per GPU.

Only problem, options, solution supply model/SFT text. No gold-answer verifier.
"""
import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys

VERSION = 1


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rows(path):
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError("expected object")
                except (ValueError, TypeError) as e:
                    raise ValueError(f"{path}:{line_no}: {e}") from e
                yield line_no, obj


def dump(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")


@contextlib.contextmanager
def atomic(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with open(temp, "w", encoding="utf-8") as f:
            yield f
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, obj):
    with atomic(path) as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def options_text(value):
    """Deterministic serialization; reject unknown structures instead of guessing."""
    if value is None or value == [] or value == {} or value == "":
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if not all(isinstance(v, str) for v in value.values()):
            raise ValueError("options dict values must be strings")
        return "\n".join(f"{k}. {v}" for k, v in value.items())
    if isinstance(value, list):
        result = []
        for i, v in enumerate(value):
            if isinstance(v, str):
                # Preserve existing A/B labels. For unlabeled strings add a label.
                labeled = re.match(r"^\s*(?:[A-Za-z]|\d+)[.、:：)）]\s*", v)
                label = chr(65 + i) if i < 26 else str(i + 1)
                result.append(v if labeled else f"{label}. {v}")
            elif isinstance(v, dict) and set(v) <= {"label", "text"} and isinstance(v.get("text"), str):
                label = v.get("label", chr(65 + i) if i < 26 else str(i + 1))
                result.append(f"{label}. {v['text']}")
            else:
                raise ValueError("unsupported options item; expected str or {label,text}")
        return "\n".join(result)
    raise ValueError("unsupported options type")


def make_pair(obj):
    problem, solution = obj.get("problem"), obj.get("solution")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("missing_or_empty_problem")
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError("missing_or_empty_solution")
    opts = options_text(obj.get("options"))
    # Do not strip/rewrite the solution, including HTML, Step markers or Answer.
    return problem + ("\n\nOptions:\n" + opts if opts else ""), solution


def split_steps(solution):
    # Match only line-start markers. Never split on Step numbers inside an equation.
    marker = re.compile(r"(?im)^[ \t]*(?:Step\s*\d+\s*[:：.)、]|Answer\s*[:：])")
    positions = [m.start() for m in marker.finditer(solution)]
    if positions:
        cuts = sorted(set([0] + positions + [len(solution)]))
        parts = [solution[a:b] for a, b in zip(cuts, cuts[1:])]
        mode = "step_and_answer_markers"
    else:
        parts = re.split(r"\n[ \t]*\n+", solution)
        mode = "paragraphs" if len(parts) > 1 else "single_block"
    # Whitespace-only blocks carry no content. All other text remains for PRM.
    return [s.strip() for s in parts if s.strip()], mode


def prepare(args):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    source_hash = digest(args.data)
    manifest_path = root / "prepare_manifest.json"
    config = {"version": VERSION, "source_sha256": source_hash, "source": str(Path(args.data).resolve())}
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old["config"] != config:
            raise ValueError("Input changed. Use a NEW output directory; cached scores cannot be reused.")
        if old["all_sha256"] != digest(root / "all_sft.jsonl"):
            raise ValueError("all_sft.jsonl changed; use a new output directory")
        print("[prepare] reuse", old["counts"], flush=True)
        return
    counts = collections.Counter()
    with atomic(root / "all_sft.jsonl") as out, atomic(root / "invalid_records.jsonl") as bad:
        for line_no, obj in rows(args.data):
            counts["source_records"] += 1
            try:
                prompt, completion = make_pair(obj)
            except ValueError as e:
                counts["invalid_records"] += 1
                dump(bad, {"source_line": line_no, "id": obj.get("id"), "reason": str(e)})
                continue
            dump(out, {"source_line": line_no, "id": obj.get("id"),
                       "prompt": prompt, "completion": completion})
            counts["usable_records"] += 1
    if digest(args.data) != source_hash:
        raise ValueError("Source changed during preparation; retry in a new output directory")
    if not counts["usable_records"]:
        raise ValueError("No usable records. Inspect invalid_records.jsonl")
    write_json(manifest_path, {"config": config, "counts": dict(counts),
                              "all_sha256": digest(root / "all_sft.jsonl")})
    print("[prepare]", dict(counts), flush=True)


def model_signature(model_path):
    path = Path(model_path).resolve()
    if not path.is_dir():
        raise ValueError("--model must be a local full token-classification checkpoint directory")
    # Weight identity uses path + size + nanosecond mtime, to avoid reading GBs on every GPU.
    files = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}:
            st = p.stat()
            files.append([str(p.relative_to(path)), st.st_size, st.st_mtime_ns,
                          digest(p) if p.suffix in {".json", ".txt"} else None])
    if not files:
        raise ValueError("Empty model directory")
    return {"path": str(path), "files": files}


def encode_inputs(tokenizer, prompt, steps, max_length):
    """Match uploaded trainer: BOS + problem SEP + contiguous recent prefix + step SEP.

    Unlike training's extreme fallback, refuse to cut the problem/current step.
    This marks an incomplete scoring attempt as unscorable, never high quality.
    """
    sep = tokenizer.encode("\n\n", add_special_tokens=False)
    if not sep:
        raise ValueError("Empty separator encoding")
    bos = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    problem = tokenizer.encode(prompt, add_special_tokens=False) + sep
    pieces = [tokenizer.encode(s, add_special_tokens=False) + sep for s in steps]
    inputs, dropped = [], []
    for i, piece in enumerate(pieces):
        available = max_length - len(bos) - len(problem) - len(piece)
        if available < 0:
            return None, {"reason": "problem_plus_step_exceeds_max_length", "step_index": i,
                          "required_tokens": len(bos) + len(problem) + len(piece)}
        start = i
        for j in range(i - 1, -1, -1):
            if len(pieces[j]) > available:
                break
            available -= len(pieces[j])
            start = j
        prefix = [token for p in pieces[start:i] for token in p]
        inputs.append(bos + problem + prefix + piece)
        dropped.append(start)
    return inputs, {"dropped_prefix_steps": dropped}


def infer_steps(model, tokenizer, inputs, device, batch_size, torch, class_ids):
    result, cursor = [], 0
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
    while cursor < len(inputs):
        batch = inputs[cursor:cursor + batch_size]
        ids = mask = output = logits = None
        try:
            width = max(map(len, batch))
            ids = torch.tensor([x + [pad] * (width - len(x)) for x in batch], device=device)
            mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in batch], device=device)
            with torch.inference_mode():
                output = model(input_ids=ids, attention_mask=mask, use_cache=False)
                positions = torch.tensor([len(x) - 1 for x in batch], device=device)
                logits = output.logits[torch.arange(len(batch), device=device), positions].float()
                if logits.shape[-1] != 3 or not torch.isfinite(logits).all():
                    raise ValueError("Invalid/nonfinite PRM logits")
                probs = logits.softmax(-1).cpu().tolist()
                q = logits[:, [class_ids["ERROR"], class_ids["CORRECT"]]].softmax(-1)[:, 1].cpu().tolist()
            for p, conditional in zip(probs, q):
                result.append({"p_error": p[class_ids["ERROR"]], "p_neutral": p[class_ids["NEUTRAL"]],
                               "p_correct": p[class_ids["CORRECT"]], "q_correct_vs_error": conditional})
            cursor += len(batch)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"[OOM retry] batch_size={batch_size}", flush=True)
        finally:
            del ids, mask, output, logits
            # Empty cache only after OOM retries would be faster, but do not retain activations.
        if cursor < len(inputs) and device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def score(args):
    import torch
    import transformers
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    root = Path(args.output_dir)
    manifest = read_json(root / "prepare_manifest.json")
    if digest(root / "all_sft.jsonl") != manifest["all_sha256"]:
        raise ValueError("all_sft hash mismatch")
    config = {"version": VERSION, "code_sha256": digest(__file__), "all_sha256": manifest["all_sha256"], "model": model_signature(args.model),
              "max_length": args.max_length, "dtype": args.dtype, "class_order": args.class_order,
              "world_size": world, "rank": rank, "torch": torch.__version__, "transformers": transformers.__version__}
    shard = root / f"scores.rank{rank:04d}.jsonl"
    done = root / f"scores.rank{rank:04d}.done.json"
    if done.exists():
        meta = read_json(done)
        if meta["config"] != config:
            raise ValueError("Scoring configuration changed. Use a NEW output directory.")
        if digest(shard) != meta["sha256"]:
            raise ValueError(f"Corrupt completed score shard: {shard}")
        print(f"[rank {rank}] reuse completed shard: {meta['count']}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("GPU scoring requires CUDA in the current Python environment")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, fix_mistral_regex=True)
    model, loading = AutoModelForTokenClassification.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa", output_loading_info=True)
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError(f"Incomplete PRM checkpoint; refusing randomly initialized weights: {loading}")
    model = model.to(device).eval()
    if model.config.num_labels != 3:
        raise ValueError("Requires a three-class TOKEN-classification PRM; do not resize its head")
    mapping = {str(v).upper(): int(k) for k, v in model.config.id2label.items()}
    if args.class_order:
        names = args.class_order.upper().split(",")
        if len(names) != 3 or set(names) != {"ERROR", "NEUTRAL", "CORRECT"}:
            raise ValueError("class-order must contain ERROR,NEUTRAL,CORRECT exactly once")
        override = {name: i for i, name in enumerate(names)}
        if set(mapping) == set(override) and mapping != override:
            raise ValueError("Explicit class-order conflicts with checkpoint label metadata")
        mapping = override
    if set(mapping) != {"ERROR", "NEUTRAL", "CORRECT"}:
        raise ValueError("Unknown label names; explicitly supply --class-order ERROR,NEUTRAL,CORRECT only after confirming")
    if sorted(mapping.values()) != [0, 1, 2]:
        raise ValueError(f"Invalid class indices: {mapping}")
    count = 0
    with atomic(shard) as out:
        for index, (_, item) in enumerate(rows(root / "all_sft.jsonl")):
            if index % world != rank:
                continue
            steps, mode = split_steps(item["completion"])
            inputs, info = encode_inputs(tokenizer, item["prompt"], steps, args.max_length)
            result = {"index": index, "source_line": item["source_line"], "id": item["id"],
                      "num_steps": len(steps), "split_mode": mode, **info}
            if inputs is None:
                result["status"] = "unscorable"
            else:
                result["status"] = "ok"
                result["step_scores"] = infer_steps(model, tokenizer, inputs, device, args.batch_size, torch, mapping)
            dump(out, result)
            count += 1
            if count % 100 == 0:
                print(f"[rank {rank}] scored {count}", flush=True)
    write_json(done, {"config": config, "count": count, "sha256": digest(shard), "class_ids": mapping})
    print(f"[rank {rank}] completed {count}", flush=True)


def aggregate(step_scores, mix_weight):
    q = [s["q_correct_vs_error"] for s in step_scores]
    if not q or any(not isinstance(x, (float, int)) or not math.isfinite(x) or not 0 <= x <= 1 for x in q):
        raise ValueError("Missing/invalid step probabilities")
    minimum = min(q)
    geometric = 0.0 if minimum == 0 else math.exp(sum(math.log(x) for x in q) / len(q))
    mixed = geometric if mix_weight == 0 else minimum if mix_weight == 1 else minimum ** mix_weight * geometric ** (1 - mix_weight)
    return {"min_q": minimum, "geomean_q": geometric, "mixed_q": mixed,
            "max_p_error": max(s["p_error"] for s in step_scores),
            "mean_p_neutral": sum(s["p_neutral"] for s in step_scores) / len(q)}


def quantiles(values):
    s = sorted(values)
    return {str(p): s[round((len(s) - 1) * p)] if s else None for p in (0, .1, .25, .5, .75, .9, 1)}


def select(args):
    root, dest = Path(args.output_dir), Path(args.selection_dir)
    if dest.resolve() == root.resolve():
        raise ValueError("selection-dir must differ from output-dir")
    prepared = read_json(root / "prepare_manifest.json")
    if digest(root / "all_sft.jsonl") != prepared["all_sha256"]:
        raise ValueError("all_sft hash mismatch")
    first = read_json(root / "scores.rank0000.done.json")
    world = first["config"]["world_size"]
    n = prepared["counts"]["usable_records"]
    summaries, failures, config = {}, collections.Counter(), dict(first["config"])
    config.pop("rank")
    with atomic(dest / "all_scores.jsonl") as merged:
        for rank in range(world):
            shard = root / f"scores.rank{rank:04d}.jsonl"
            meta = read_json(root / f"scores.rank{rank:04d}.done.json")
            check = dict(meta["config"])
            actual_rank = check.pop("rank")
            if actual_rank != rank or check != config or digest(shard) != meta["sha256"]:
                raise ValueError(f"Shard configuration/hash mismatch: {shard}")
            if check["all_sha256"] != prepared["all_sha256"]:
                raise ValueError("Score shard does not match prepared data")
            count = 0
            for _, row in rows(shard):
                i = row["index"]
                if not isinstance(i, int) or i in summaries or not 0 <= i < n or i % world != rank:
                    raise ValueError(f"Invalid/duplicate score index: {i}")
                reasons = []
                stats = {}
                if row["status"] == "ok":
                    if len(row["step_scores"]) != row["num_steps"]:
                        raise ValueError("Incomplete step scores")
                    stats = aggregate(row["step_scores"], args.mix_weight)
                    if stats["min_q"] < args.min_step_q:
                        reasons.append("min_step_q")
                    if stats["mixed_q"] < args.min_score:
                        reasons.append("mixed_score")
                    if args.require_full_prefix and any(row["dropped_prefix_steps"]):
                        reasons.append("prefix_trimmed")
                elif row["status"] == "unscorable":
                    reasons.append(row["reason"])
                else:
                    raise ValueError(f"Unknown score status: {row['status']}")
                failures.update(reasons)
                small = {k: row[k] for k in ("index", "id", "source_line", "status", "num_steps")}
                summaries[i] = {**small, **stats, "reasons": reasons}
                dump(merged, row)
                count += 1
            if count != meta["count"]:
                raise ValueError("Shard row count mismatch")
    if len(summaries) != n:
        raise ValueError(f"Incomplete scores: {len(summaries)} of {n}")
    eligible = [i for i, s in summaries.items() if not s["reasons"]]
    if args.top_fraction is not None:
        keep = math.floor(n * args.top_fraction)
        eligible.sort(key=lambda i: (-summaries[i]["mixed_q"], i))
        for i in eligible[keep:]:
            summaries[i]["reasons"].append("outside_top_fraction")
        eligible = eligible[:keep]
    selected = set(eligible)
    random_selected = set(random.Random(args.seed).sample(range(n), len(selected)))
    with atomic(dest / "selected_sft.jsonl") as good, atomic(dest / "random_matched_sft.jsonl") as rand, \
            atomic(dest / "selection_audit.jsonl") as audit:
        for i, (_, item) in enumerate(rows(root / "all_sft.jsonl")):
            if summaries[i]["source_line"] != item["source_line"] or summaries[i]["id"] != item["id"]:
                raise ValueError("Scores and SFT records misaligned")
            if i in selected:
                dump(good, item)
            if i in random_selected:
                dump(rand, item)
            dump(audit, {**summaries[i], "selected": i in selected})
    # Reproducible random audit samples for manual review. Selection does not use these labels.
    rng = random.Random(args.seed)
    review_ids = set()
    groups = {"selected": sorted(selected),
              "rejected": [i for i in range(n) if i not in selected]}
    for group in groups.values():
        review_ids.update(rng.sample(group, min(args.review_size, len(group))))
    with atomic(dest / "manual_review.jsonl") as review:
        for i, (_, item) in enumerate(rows(root / "all_sft.jsonl")):
            if i in review_ids:
                dump(review, {**item, "prm": summaries[i], "selected": i in selected,
                              "human_correct": None, "human_self_contained": None})
    report = {"source": prepared, "scoring_config": config,
              "selection_config": {k: v for k, v in vars(args).items() if k != "func"},
              "usable_records": n, "selected_records": len(selected), "selection_rate": len(selected) / n,
              "threshold_failure_counts_overlapping": dict(failures),
              "unscorable_records": sum(s["status"] != "ok" for s in summaries.values()),
              "scorable_mixed_q_quantiles": quantiles([s["mixed_q"] for s in summaries.values() if s["status"] == "ok"]),
              "selected_step_count_quantiles": quantiles([summaries[i]["num_steps"] for i in selected]),
              "all_step_count_quantiles": quantiles([s["num_steps"] for s in summaries.values()]),
              "outputs": {"all_sft": str((root / "all_sft.jsonl").resolve()),
                          "selected_sft": str((dest / "selected_sft.jsonl").resolve()),
                          "random_matched_sft": str((dest / "random_matched_sft.jsonl").resolve())},
              "note": "PRM preference scores, not ground-truth correctness. Defaults are uncalibrated on Quizlet."}
    write_json(dest / "selection_report.json", report)
    print(json.dumps({k: report[k] for k in ("usable_records", "selected_records", "selection_rate", "unscorable_records")}, indent=2))
    if not selected:
        raise SystemExit("No rows selected. Inspect scores/review samples before choosing new thresholds.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--data", required=True)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=prepare)
    p = subs.add_parser("score")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--class-order", default="")
    p.set_defaults(func=score)
    p = subs.add_parser("select")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--selection-dir", required=True)
    p.add_argument("--mix-weight", type=float, default=.5)
    p.add_argument("--min-step-q", type=float, default=.5)
    p.add_argument("--min-score", type=float, default=.8)
    p.add_argument("--top-fraction", type=float, default=None,
                   help="Optional cap as fraction of ALL usable records, after quality gates; may retain fewer")
    p.add_argument("--require-full-prefix", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--review-size", type=int, default=50, help="Manual review samples PER selected/rejected group")
    p.set_defaults(func=select)
    args = parser.parse_args()
    for name in ("mix_weight", "min_step_q", "min_score", "top_fraction"):
        value = getattr(args, name, None)
        if value is not None and not 0 <= value <= 1:
            parser.error(f"{name} must be in [0,1]")
    for name in ("batch_size", "max_length"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if getattr(args, "review_size", 0) < 0:
        parser.error("review-size must be nonnegative")
    args.func(args)


if __name__ == "__main__":
    main()
