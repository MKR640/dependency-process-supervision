#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overall-only comparison on one merged MRBEN evaluation file.

This evaluator intentionally does NOT require a train/validation/test split and
performs no train-vs-test overlap check. It evaluates every trajectory in the
single frozen reference, using exactly the same metric definitions as
``evaluate_complexity.py``.

The intended input is:
    mrben_phase2_hierarchical_prm_flat_all.jsonl
which is train + validation + test concatenated together.

Dataset split names do not establish model exposure. Training-history and
problem-overlap evidence must be recorded separately before claiming transfer.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_complexity import (
    METRICS,
    Metrics,
    ci,
    difference,
    fmt,
    load_prediction,
    trace_features,
)
from reference_io import file_hash, read_jsonl, validate_rows, write_json


TABLE_METRICS = [
    ("error_ap", "Error AP"),
    ("error_f1", "Error F1"),
    ("balanced_accuracy", "Bal. acc."),
    ("new_error_ap", "New AP"),
    ("role_macro_f1", "Role F1"),
    ("first_error_accuracy", "FE"),
]


def load_all_reference(path: str | Path):
    path = Path(path)
    rows = list(read_jsonl(path))
    validate_rows(rows)

    meta_path = Path(str(path) + ".meta.json")
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("split") != "all":
        raise ValueError(
            f"Expected merged all-data reference with split='all'; "
            f"got {meta.get('split')!r}"
        )
    if meta.get("reference_sha256") != file_hash(path):
        raise ValueError("Reference hash does not match .meta.json")

    return rows, meta


def dataset_counts(rows, traces):
    indices = [i for t in traces for i in t["inds"]]
    return {
        "problems": len({t["problem_id"] for t in traces}),
        "traces": len(traces),
        "steps": len(indices),
        "error_steps": sum(t["errors"] for t in traces),
        "error_traces": sum(t["errors"] > 0 for t in traces),
        "correct_traces": sum(t["errors"] == 0 for t in traces),
        "new_error_steps": sum(rows[i]["role"] == "new_error" for i in indices),
        "propagated_error_steps": sum(
            rows[i]["role"] == "propagated_error" for i in indices
        ),
        "truncated_traces": sum(t["truncated"] for t in traces),
        "low_support": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True)
    p.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="NAME=merged_prediction.jsonl; repeatable",
    )
    p.add_argument("--baseline", default="Three_class_v2")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260924)
    a = p.parse_args()

    if a.threshold != 0.5:
        raise ValueError("This fixed-threshold bundle requires --threshold 0.5")
    if a.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be >= 0")

    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            f"Output directory is nonempty: {out}. Use a fresh output directory."
        )

    rows, ref = load_all_reference(a.reference)
    pids = sorted({r["problem_id"] for r in rows})
    if not pids:
        raise ValueError("No problems in reference")
    problem_index = {pid: i for i, pid in enumerate(pids)}

    preds = {}
    for spec in a.prediction:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path or name in preds:
            raise ValueError("--prediction must be unique NAME=PATH")
        preds[name] = load_prediction(path, rows, ref, name)

    if a.baseline not in preds or len(preds) < 2:
        raise ValueError("At least two models and a valid baseline are required")

    # Apples-to-apples guard: all three models must see identical packed input.
    baseline = preds[a.baseline]
    for name, pred in preds.items():
        if (
            pred["hashes"] != baseline["hashes"]
            or pred["meta"]["packing"] != baseline["meta"]["packing"]
            or pred["meta"]["tokenizer_sha256"]
            != baseline["meta"]["tokenizer_sha256"]
        ):
            raise ValueError(
                name + ": input hashes/tokenizer/packing differ across models"
            )

    traces = trace_features(rows, preds)
    counts = dataset_counts(rows, traces)

    evaluators = {
        name: Metrics(rows, traces, pred, a.threshold, problem_index)
        for name, pred in preds.items()
    }
    unit_weights = np.ones(len(pids), dtype=float)
    estimates = {
        name: evaluator.compute(unit_weights)
        for name, evaluator in evaluators.items()
    }

    draws = {
        name: {metric: [] for metric in METRICS}
        for name in preds
    }

    rng = np.random.default_rng(a.seed)
    for b in range(a.bootstrap_samples):
        # Paired problem-cluster bootstrap: one common multiplicity vector for
        # every model in each replicate.
        w = np.bincount(
            rng.integers(0, len(pids), size=len(pids)),
            minlength=len(pids),
        )
        for name, evaluator in evaluators.items():
            metrics = evaluator.compute(w)
            for metric in METRICS:
                draws[name][metric].append(metrics[metric])

        if (b + 1) % 100 == 0 or b + 1 == a.bootstrap_samples:
            print(f"bootstrap {b + 1}/{a.bootstrap_samples}", flush=True)

    records = []
    base_est = estimates[a.baseline]
    for name in preds:
        metric_payload = {}
        for metric in METRICS:
            value = estimates[name][metric]
            delta_draws = [
                difference(x, y)
                for x, y in zip(
                    draws[name][metric],
                    draws[a.baseline][metric],
                )
            ]
            metric_payload[metric] = {
                "value": value,
                "ci": ci(draws[name][metric]),
                "delta_vs_baseline": difference(value, base_est[metric]),
                "paired_delta_ci": ci(delta_draws),
            }

        records.append(
            {
                "axis": "overall",
                "group": "all",
                "model": name,
                "counts": counts,
                "metrics": metric_payload,
            }
        )

    result = {
        "schema": "mrben_merged_all_comparison_v1",
        "baseline": a.baseline,
        "evaluation_scope": (
            "Full supplied physics evaluation corpus; model exposure and "
            "training-history overlap require separate verification"
        ),
        "policy": {
            "threshold": a.threshold,
            "decision": "ERROR iff p_error >= threshold",
            "bootstrap_unit": "problem",
        },
        "reference": ref,
        "models": {
            name: {
                "path": pred["path"],
                "metadata": pred["meta"],
            }
            for name, pred in preds.items()
        },
        "bootstrap": {
            "samples": a.bootstrap_samples,
            "seed": a.seed,
            "unit": "problem",
            "interval": "paired percentile 95%",
        },
        "results": records,
    }

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", result)

    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "model",
            "problems",
            "traces",
            "steps",
            "metric",
            "value",
            "ci_low",
            "ci_high",
            "delta_vs_baseline",
            "delta_ci_low",
            "delta_ci_high",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for metric, payload in record["metrics"].items():
                writer.writerow(
                    {
                        "model": record["model"],
                        "problems": counts["problems"],
                        "traces": counts["traces"],
                        "steps": counts["steps"],
                        "metric": metric,
                        "value": payload["value"],
                        "ci_low": payload["ci"]["low"],
                        "ci_high": payload["ci"]["high"],
                        "delta_vs_baseline": payload["delta_vs_baseline"],
                        "delta_ci_low": payload["paired_delta_ci"]["low"],
                        "delta_ci_high": payload["paired_delta_ci"]["high"],
                    }
                )

    display_names = {"Binary_PRM": "Binary PRM", "Three_class_v2": "Three-class PRM",
                     "Full_hierarchy": "Hierarchical PRM"}
    # Markdown overall table.
    report = [
        "# MRBEN merged-all three-model comparison",
        "",
        (
            f"Evaluation data: {counts['problems']} problems, "
            f"{counts['traces']} trajectories, {counts['steps']} steps."
        ),
        (
            "Full supplied physics evaluation corpus. Dataset split names do not "
            "establish model exposure; see reference overlap checks and training records."
        ),
        f"Fixed ERROR threshold: {a.threshold}.",
        "",
        "| Model | Error AP | Error F1 | Bal. acc. | New AP | Role F1 | FE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        vals = [
            fmt(record["metrics"][metric]["value"])
            for metric, _ in TABLE_METRICS
        ]
        report.append(
            "| " + display_names.get(record["model"], record["model"]) + " | " + " | ".join(vals) + " |"
        )

    report += [
        "",
        "## Dataset counts",
        "",
        f"- problems: {counts['problems']}",
        f"- trajectories: {counts['traces']}",
        f"- steps: {counts['steps']}",
        f"- error steps: {counts['error_steps']}",
        f"- error trajectories: {counts['error_traces']}",
        f"- correct trajectories: {counts['correct_traces']}",
        f"- new-error steps: {counts['new_error_steps']}",
        f"- propagated-error steps: {counts['propagated_error_steps']}",
        "",
        "95% paired problem-cluster bootstrap intervals and paired deltas are in metrics.json/metrics.csv.",
    ]

    # [NEW] Show FE denominators separately, using the same cached predictions.
    diag = ["# Fixed-threshold diagnostics", "",
            "| Model | ERROR Recall | Correct-step FPR | Error-only FE | Correct-trace pass |",
            "|---|---:|---:|---:|---:|"]
    keys = ['error_recall', 'false_positive_rate', 'first_error_accuracy_error_traces',
            'no_error_trace_accuracy']
    for r in records:
        diag.append('| ' + display_names.get(r['model'], r['model']) + ' | ' +
                    ' | '.join(fmt(r['metrics'][k]['value']) for k in keys) + ' |')
    diag += ['', f"Error traces: {counts['error_traces']}; correct traces: {counts['correct_traces']}.",
             'Values are percentages. Error-only FE excludes fully correct traces.',
             'AP is step-level; FE includes no-error decisions on fully correct traces.']
    (out / 'diagnostics.md').write_text('\n'.join(diag)+'\n', encoding='utf-8')
    report += ['', *diag]

    (out / "overall_comparison.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    # LaTeX table matching the requested paper-style columns.
    tex = [
        "% Requires booktabs.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        (
            "\\caption{Three-model comparison on the merged MRBEN all-data "
            f"set ({counts['problems']} problems, {counts['traces']} trajectories, "
            f"and {counts['steps']} steps; \\%). Fixed error threshold 0.5. "
            "Training exposure requires separate verification.}"
        ),
        "\\label{tab:mrben_all_models}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Model & Error AP & Error F1 & Bal. acc. & New AP & Role F1 & FE \\\\",
        "\\midrule",
    ]
    display_names = {
        "Binary_PRM": "Binary PRM",
        "Three_class_v2": "Three-class PRM",
        "Full_hierarchy": "Hierarchical PRM",
    }
    for record in records:
        values = []
        for metric, _ in TABLE_METRICS:
            value = record["metrics"][metric]["value"]
            values.append("--" if value is None else f"{100 * value:.2f}")
        tex.append(
            " & ".join(
                [display_names.get(record["model"], record["model"])] + values
            )
            + " \\\\"
        )
    tex += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    (out / "table_overall.tex").write_text(
        "\n".join(tex) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("MRBEN MERGED-ALL THREE-MODEL COMPARISON")
    print("=" * 100)
    print(
        f"Data: problems={counts['problems']}, traces={counts['traces']}, "
        f"steps={counts['steps']}"
    )
    for record in records:
        print(
            record["model"],
            {
                label: fmt(record["metrics"][metric]["value"])
                for metric, label in TABLE_METRICS
            },
        )
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
