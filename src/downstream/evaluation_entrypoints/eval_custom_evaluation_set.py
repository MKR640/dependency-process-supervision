#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vLLM 多 GPU + 批量推理 + PPL 批处理脚本
适配场景：Classification (chat/base) + Generation
保存全部结果与 summary
"""

import argparse
import os
import json
import math
import re
import string
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from vllm import LLM, SamplingParams

# ---------------------------
# 命令行参数
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="base", help="base or chat")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--output_path", type=str, default="result.jsonl")
parser.add_argument("--dataset_name", type=str, default=None)
parser.add_argument("--metrics", type=str, default="acc,f1,bleu,rouge")
parser.add_argument("--tq_gpu_num", type=int, default=None, help="tensor parallel size; 如果为空会自动使用可见 GPU 数")
parser.add_argument("--batch_size", type=int, default=128, help="一次 generate 的 prompt 数（根据显存与 GPU 数调整）")
parser.add_argument("--max_tokens", type=int, default=2048)
args = parser.parse_args()

MODEL_TYPE = args.model_type.lower()
MODEL_PATH = args.model_path
DATASET_PATH = args.dataset_path
OUTPUT_PATH = args.output_path
METRICS = args.metrics.split(",")
BATCH_SIZE = args.batch_size
MAX_TOKENS = args.max_tokens
TQ_GPU_NUM = args.tq_gpu_num

# 输出文件夹
dir_path = os.path.dirname(OUTPUT_PATH)
if dir_path and not os.path.exists(dir_path):
    os.makedirs(dir_path, exist_ok=True)

# 数据集名
if args.dataset_name:
    DATASET_NAME = args.dataset_name
else:
    DATASET_NAME = os.path.splitext(os.path.basename(DATASET_PATH))[0]


sampling_params = SamplingParams(
    temperature=0.0,
    n=1,
    max_tokens=MAX_TOKENS,
    logprobs=1,
)

llm = LLM(
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    tensor_parallel_size=TQ_GPU_NUM,
    gpu_memory_utilization=0.9,
)

# ---------------------------
# 工具函数：去重 / 映射 / ppl 解析
# ---------------------------
def dedup_paragraphs(text: str) -> str:
    paragraphs = text.split("\n")
    seen = set()
    deduped = []
    for p in paragraphs:
        key = re.sub(r'\s+', '', p).lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(p)
    return "\n".join(deduped)

def dedup_sentences(text: str) -> str:
    # 更稳健的中文/英文句子分割，保留标点
    parts = re.split(r'([。！？]|[!?]|\.{1,3}|\n+)', text)
    sentences = []
    buf = ""
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r'[。！？]|[!?]|\.{1,3}|\n+', part):
            buf += part
            sentences.append(buf)
            buf = ""
        else:
            buf += part
    if buf.strip():
        sentences.append(buf)
    seen = set()
    res = []
    for s in sentences:
        key = re.sub(r'\s+', '', s).lower()
        if key not in seen:
            seen.add(key)
            res.append(s)
    # 保留原有分隔符，合并返回
    return "".join(res)

def build_choice_mapping(choices):
    letters = list(string.ascii_uppercase)
    mapping = {}
    reverse_mapping = {}
    for idx, ch in enumerate(choices):
        letter = letters[idx]
        ch = str(ch).strip()
        mapping[letter] = ch
        reverse_mapping[ch] = letter
    return mapping, reverse_mapping

def compute_ppl_from_output(output):
    """
    output 是 vLLM generate 的单个 item（outputs[i]）
    根据该结构计算 ppl：exp(-avg_log_prob)
    """
    try:
        inner = output.outputs[0]
        cumulative_logprob = inner.cumulative_logprob
        token_ids = inner.token_ids
        if token_ids is None or len(token_ids) == 0:
            return float("inf")
        N = len(token_ids)
        avg_log_prob = cumulative_logprob / N
        ppl = math.exp(-avg_log_prob)
        return ppl
    except Exception as e:
        # 出错返回很大值
        return float("inf")

# ---------------------------
# 读取数据集
# ---------------------------
dataset = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        dataset.append(json.loads(line))

# ---------------------------
# 构建 prompts（两种并行队列）
# - gen_prompts: 用于 chat classification 与 generation（直接返回文本）
# - ppl_prompts: 用于 base 分类的 PPL 计算（对每个选项生成一个 prompt）
# 保存足够元信息以还原
# ---------------------------
gen_prompts = []         # list of prompt strings
gen_meta = []            # list of metadata dict: {"idx": sample_idx, "mode": "classification"/"generation", ...}

ppl_prompts = []         # 每个选项一个 prompt（用于计算 ppl）
ppl_meta = []            # list of metadata dict: {"sample_idx": idx, "option_text": text, "letter": letter}

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

for idx, sample in enumerate(dataset):
    s_type = sample.get("type", "").strip()
    if MODEL_TYPE == "chat" and s_type == "Classification":
        # 直接让模型输出字母（batch）
        question = sample["question"]
        choices = [str(c) for c in sample["choices"]]
        hint = "You are a helpful assistant. This is a single choice question. Answer by replying the letter corresponding to your choice (A, B, C...)."
        choice_lines = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(len(choices)))
        prompt = f"{hint}\nQuestion: {question}\n{choice_lines}\nAnswer: "
        gen_prompts.append(prompt)
        gen_meta.append({"idx": idx, "mode": "classification", "choices": choices})
    elif MODEL_TYPE == "base" and s_type == "Classification":
        # 对每个选项计算 ppl（batch all options）
        question = sample["question"]
        choices = [str(c) for c in sample["choices"]]
        mapping, reverse_mapping = build_choice_mapping(choices)
        for letter, text in mapping.items():
            prompt = f"You are a helpful assistant. The following are multiple choice questions (with answers).\nQuestion: {question}\nAnswer: {text}"
            ppl_prompts.append(prompt)
            ppl_meta.append({"sample_idx": idx, "option_text": text, "letter": letter})
    else:
        # generation（batch）
        question = sample["question"]
        prompt = f"You are a helpful assistant. Please answer the following question:\nQuestion: {question}"
        gen_prompts.append(prompt)
        gen_meta.append({"idx": idx, "mode": "generation"})

# ---------------------------
# 批量 generate 的工具（分批执行以防 OOM）
# 返回输出列表（与传入 prompts 顺序一致）
# ---------------------------
def batch_generate(prompts_list, batch_size=BATCH_SIZE, sampling_params=sampling_params):
    outputs_all = []
    n = len(prompts_list)
    for i in tqdm(range(0, n, batch_size), desc="Batch generating"):
        batch = prompts_list[i:i+batch_size]
        try:
            out = llm.generate(batch, sampling_params)
            outputs_all.extend(out)
        except Exception as e:
            # 出错时 fallback：逐条 generate，保证不中断（但会慢）
            print(f"Batch generate error at chunk {i}-{i+batch_size}: {e}. Falling back to single-item generate.")
            for p in tqdm(batch, desc="Fallback single generate"):
                try:
                    out = llm.generate([p], sampling_params)
                    outputs_all.extend(out)
                except Exception as e2:
                    print("Single generate failed:", e2)
                    outputs_all.append(None)
    return outputs_all

# ---------------------------
# 对 gen_prompts 做 batch generate 并解析结果
# ---------------------------
gen_outputs = []
if gen_prompts:
    print(f"Total gen_prompts: {len(gen_prompts)} -> batching with size {BATCH_SIZE}")
    gen_outputs = batch_generate(gen_prompts, batch_size=BATCH_SIZE)

# 解析 gen outputs（还原到 dataset idx）
preds = {}   # sample_idx -> pred (string for generation, letter for chat classification)
for out, meta in zip(gen_outputs, gen_meta):
    idx = meta["idx"]
    mode = meta["mode"]

    if out is None:
        text = ""
    else:
        try:
            text = out.outputs[0].text.strip()
        except Exception:
            # vLLM 版本差异
            text = getattr(out, "text", "")
            if text is None:
                text = ""

        # 后处理去重
        text = dedup_sentences(text)

    if mode == "classification":
        # 使用正则抽取首个大写字母 A-Z
        match = re.search(r'[A-Z]', text.upper())
        preds[idx] = match.group(0) if match else ""
    else:
        preds[idx] = text

# ---------------------------
# 对基础分类（PPL）做 batch generate 并解析 PPL
# ---------------------------
ppl_outputs = []
if ppl_prompts:
    print(f"Total ppl_prompts (options): {len(ppl_prompts)} -> batching with size {BATCH_SIZE}")
    ppl_outputs = batch_generate(ppl_prompts, batch_size=BATCH_SIZE)

# 将 ppl_outputs 与 ppl_meta 对齐，计算每个样本的最佳选项
ppl_scores_per_sample = {}  # sample_idx -> {letter: ppl}
for out, meta in zip(ppl_outputs, ppl_meta):
    sample_idx = meta["sample_idx"]
    letter = meta["letter"]
    option_text = meta["option_text"]
    if out is None:
        ppl_val = float("inf")
    else:
        ppl_val = compute_ppl_from_output(out)
    if sample_idx not in ppl_scores_per_sample:
        ppl_scores_per_sample[sample_idx] = {}
    ppl_scores_per_sample[sample_idx][letter] = ppl_val

# 将 ppl 最低的选项写入 preds（以字母形式）
for sample_idx, score_dict in ppl_scores_per_sample.items():
    best_letter = min(score_dict, key=score_dict.get)
    preds[sample_idx] = best_letter

# ---------------------------
# 现在 preds 包含了所有样本的预测（classification & generation）
# 计算指标并写入结果
# ---------------------------
results = []
cls_acc_list, cls_f1_list = [], []
gen_bleu_list, gen_rouge1_list, gen_rouge2_list, gen_rougeL_list = [], [], [], []

smoothing = SmoothingFunction().method1
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

for idx, sample in enumerate(tqdm(dataset, desc="Metrics calc")):
    gold = sample.get("answer", "")
    pred = preds.get(idx, "")
    s_type = sample.get("type", "")
    metrics = {}
    if s_type == "Classification":
        acc = 1 if str(pred).strip() == str(gold).strip() else 0
        f1 = acc
        cls_acc_list.append(acc)
        cls_f1_list.append(f1)
        metrics = {"acc": acc, "f1": f1}
    else:
        # generation
        # 使用 BLEU+ROUGE
        try:
            bleu4 = sentence_bleu([str(gold).split()], str(pred).split(),
                                  weights=(0.25,0.25,0.25,0.25),
                                  smoothing_function=smoothing)
        except Exception:
            bleu4 = 0.0
        scores = scorer.score(str(gold), str(pred))
        r1 = scores['rouge1'].fmeasure
        r2 = scores['rouge2'].fmeasure
        rL = scores['rougeL'].fmeasure
        gen_bleu_list.append(bleu4)
        gen_rouge1_list.append(r1)
        gen_rouge2_list.append(r2)
        gen_rougeL_list.append(rL)
        metrics = {"bleu-4": bleu4, "rouge1": r1, "rouge2": r2, "rougeL": rL}

    results.append({
        "question": sample.get("question"),
        "choices": sample.get("choices"),
        "gold_answer": gold,
        "pred_answer": pred,
        "type": s_type,
        "metrics": metrics
    })

# Summary
summary = {
    "dataset_name": DATASET_NAME,
    "avg_metrics": {
        "classification": {
            "acc": sum(cls_acc_list)/len(cls_acc_list) if cls_acc_list else None,
            "f1": sum(cls_f1_list)/len(cls_f1_list) if cls_f1_list else None
        },
        "generation": {
            "bleu-4": sum(gen_bleu_list)/len(gen_bleu_list) if gen_bleu_list else None,
            "rouge1": sum(gen_rouge1_list)/len(gen_rouge1_list) if gen_rouge1_list else None,
            "rouge2": sum(gen_rouge2_list)/len(gen_rouge2_list) if gen_rouge2_list else None,
            "rougeL": sum(gen_rougeL_list)/len(gen_rougeL_list) if gen_rougeL_list else None
        }
    }
}

# 写结果（全部）
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for r in results[0:100]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

summary_path = OUTPUT_PATH.replace(".jsonl", "_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(json.dumps(summary, ensure_ascii=False) + "\n")

print(f"Evaluation finished.\nResults saved to {OUTPUT_PATH}\nSummary saved to {summary_path}")
