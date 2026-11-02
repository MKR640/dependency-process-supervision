#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

from opencompass.openicl.icl_inferencer import GenInferencer, PPLInferencer
# from opencompass.models.vllm import VLLM  
from vllm import LLM, SamplingParams

# -----------------------------
# 命令行参数解析
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="base", help="base or chat")
parser.add_argument("--model_path", type=str, required=True, help="模型路径")
parser.add_argument("--dataset_path", type=str, required=True, help="jsonl 数据集路径")
parser.add_argument("--output_path", type=str, default="result.jsonl", help="输出 JSONL 文件路径")
parser.add_argument("--dataset_name", type=str, default=None, help="数据集名称")
parser.add_argument("--metrics", type=str, default="acc,f1,bleu,rouge", help="评测指标，用逗号分隔")
args = parser.parse_args()

MODEL_TYPE = args.model_type.lower()
MODEL_PATH = args.model_path
DATASET_PATH = args.dataset_path
OUTPUT_PATH = args.output_path
METRICS = args.metrics.split(",")

# -----------------------------
# 数据集名称自动获取
# -----------------------------
if args.dataset_name:
    DATASET_NAME = args.dataset_name
else:
    DATASET_NAME = os.path.splitext(os.path.basename(DATASET_PATH))[0]

# -----------------------------
# 初始化模型
# -----------------------------
model = LLM(model_kwargs)

if MODEL_TYPE == "base":
    ppl_inferencer = PPLInferencer(model)
    gen_inferencer = GenInferencer(model)
else:
    gen_inferencer = GenInferencer(model)

# -----------------------------
# 读取数据集
# -----------------------------
dataset = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            dataset.append(json.loads(line))

# -----------------------------
# 推理函数
# -----------------------------
def predict(sample):
    task_type = "classification" if sample["type"] == "分类任务" else "generation"
    if MODEL_TYPE == "base" and task_type == "classification":
        pred = ppl_inferencer.inference(sample)
    else:
        pred = gen_inferencer.inference(sample)
    return pred, task_type

# -----------------------------
# 指标计算
# -----------------------------
def compute_class_metrics(pred, gold):
    acc = 1 if pred == gold else 0
    f1 = acc
    return {"acc": acc, "f1": f1}

def compute_gen_metrics(pred, gold):
    result = {}
    if "bleu" in METRICS:
        result["bleu"] = sentence_bleu([gold.split()], pred.split())
    if "rouge" in METRICS:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        result["rougeL"] = scorer.score(gold, pred)['rougeL'].fmeasure
    return result

# -----------------------------
# 评测主循环
# -----------------------------
results = []
cls_acc_list, cls_f1_list = [], []
gen_bleu_list, gen_rouge_list = [], []

for sample in tqdm(dataset, desc="Evaluating"):
    gold = sample["answer"]
    pred, task_type = predict(sample)

    if task_type == "classification":
        metrics = compute_class_metrics(pred, gold)
        cls_acc_list.append(metrics["acc"])
        cls_f1_list.append(metrics["f1"])
    else:
        metrics = compute_gen_metrics(pred, gold)
        if "bleu" in metrics:
            gen_bleu_list.append(metrics["bleu"])
        if "rougeL" in metrics:
            gen_rouge_list.append(metrics["rougeL"])

    results.append({
        "question": sample["question"],
        "choices": sample.get("choices"),
        "gold_answer": gold,
        "pred_answer": pred,
        "type": sample["type"],
        "metrics": metrics
    })

# -----------------------------
# summary
# -----------------------------
summary = {
    "dataset_name": DATASET_NAME,
    "avg_metrics": {
        "classification": {
            "acc": sum(cls_acc_list)/len(cls_acc_list) if cls_acc_list else None,
            "f1": sum(cls_f1_list)/len(cls_f1_list) if cls_f1_list else None
        },
        "generation": {
            "bleu": sum(gen_bleu_list)/len(gen_bleu_list) if gen_bleu_list else None,
            "rougeL": sum(gen_rouge_list)/len(gen_rouge_list) if gen_rouge_list else None
        }
    }
}

# -----------------------------
# 写入 results.jsonl
# -----------------------------
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -----------------------------
# 写入 summary.jsonl
# -----------------------------
summary_path = OUTPUT_PATH.replace(".jsonl", "_summary.jsonl")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write(json.dumps(summary, ensure_ascii=False) + "\n")

print(f"Evaluation finished.\nResults saved to {OUTPUT_PATH}\nSummary saved to {summary_path}")
