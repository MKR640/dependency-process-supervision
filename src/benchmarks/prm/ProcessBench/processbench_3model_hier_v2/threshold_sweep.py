#!/usr/bin/env python3

import os
import json
import csv
import argparse
import numpy as np


CONFIGS = [
    "gsm8k",
    "math",
    "olympiadbench",
    "omnimath",
]


def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                data.append(
                    json.loads(line)
                )

    return data


def get_prediction(
    step_scores,
    threshold,
):
    """
    step_scores:
        P(NON_ERROR) for every reasoning step.

    score >= threshold:
        NON_ERROR

    score < threshold:
        ERROR

    Return:
        first error index;
        -1 if all steps are predicted NON_ERROR.
    """

    for i, score in enumerate(step_scores):

        if score < threshold:
            return i

    return -1


def calculate_metrics(
    data,
    threshold,
):

    error_total = 0
    error_correct = 0

    correct_total = 0
    correct_correct = 0

    total_correct = 0

    for item in data:

        label = int(
            item["label"]
        )

        prediction = get_prediction(
            item["step_scores"],
            threshold,
        )

        match = (
            prediction == label
        )

        total_correct += int(
            match
        )

        # ProcessBench:
        # label == -1 means the whole trajectory is correct.
        if label == -1:

            correct_total += 1

            if match:
                correct_correct += 1

        else:

            error_total += 1

            if match:
                error_correct += 1

    error_acc = (
        error_correct / error_total
        if error_total > 0
        else 0.0
    )

    correct_acc = (
        correct_correct / correct_total
        if correct_total > 0
        else 0.0
    )

    overall_acc = (
        total_correct / len(data)
        if len(data) > 0
        else 0.0
    )

    # Keep exactly the same definition as ProcessBench:
    # harmonic mean of Error Acc and Correct Acc.
    if error_acc + correct_acc > 0:

        f1 = (
            2
            * error_acc
            * correct_acc
            / (
                error_acc
                + correct_acc
            )
        )

    else:

        f1 = 0.0

    return {
        "error_acc":
            error_acc * 100,

        "correct_acc":
            correct_acc * 100,

        "overall_acc":
            overall_acc * 100,

        "f1":
            f1 * 100,

        "num_error":
            error_total,

        "num_correct":
            correct_total,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result-dir",
        required=True,
        help=(
            "ProcessBench result directory "
            "containing gsm8k/math/olympiadbench/omnimath"
        ),
    )

    parser.add_argument(
        "--threshold-start",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--threshold-end",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load all four datasets only once
    # ---------------------------------------------------------

    datasets = {}

    for config in CONFIGS:

        path = os.path.join(
            args.result_dir,
            config,
            "all_results.jsonl",
        )

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Cannot find: {path}"
            )

        datasets[config] = (
            load_jsonl(path)
        )

        print(
            f"Loaded {config}: "
            f"{len(datasets[config])} samples"
        )

    # Avoid floating-point accumulation problems.
    thresholds = np.arange(
        args.threshold_start,
        args.threshold_end
        + args.threshold_step / 2,
        args.threshold_step,
    )

    rows = []

    print()
    print("=" * 110)
    print(
        f"{'Threshold':>10} | "
        f"{'GSM8K':>8} | "
        f"{'MATH':>8} | "
        f"{'Olympiad':>10} | "
        f"{'OmniMath':>8} | "
        f"{'Avg F1':>8}"
    )
    print("=" * 110)

    best_threshold = None
    best_average_f1 = -1

    for threshold in thresholds:

        threshold = float(
            round(threshold, 6)
        )

        config_metrics = {}

        f1_values = []

        for config in CONFIGS:

            metrics = calculate_metrics(
                datasets[config],
                threshold,
            )

            config_metrics[
                config
            ] = metrics

            f1_values.append(
                metrics["f1"]
            )

        average_f1 = float(
            np.mean(f1_values)
        )

        row = {
            "threshold":
                threshold,

            "average_f1":
                average_f1,
        }

        for config in CONFIGS:

            metrics = config_metrics[
                config
            ]

            row[
                f"{config}_error_acc"
            ] = metrics[
                "error_acc"
            ]

            row[
                f"{config}_correct_acc"
            ] = metrics[
                "correct_acc"
            ]

            row[
                f"{config}_overall_acc"
            ] = metrics[
                "overall_acc"
            ]

            row[
                f"{config}_f1"
            ] = metrics[
                "f1"
            ]

        rows.append(row)

        print(
            f"{threshold:10.2f} | "
            f"{config_metrics['gsm8k']['f1']:8.2f} | "
            f"{config_metrics['math']['f1']:8.2f} | "
            f"{config_metrics['olympiadbench']['f1']:10.2f} | "
            f"{config_metrics['omnimath']['f1']:8.2f} | "
            f"{average_f1:8.2f}"
        )

        if average_f1 > best_average_f1:

            best_average_f1 = (
                average_f1
            )

            best_threshold = (
                threshold
            )

    print("=" * 110)

    print()
    print(
        f"Best threshold: "
        f"{best_threshold:.4f}"
    )

    print(
        f"Best Average F1: "
        f"{best_average_f1:.2f}"
    )

    # ---------------------------------------------------------
    # Detailed result at best threshold
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print(
        f"Best threshold = "
        f"{best_threshold:.4f}"
    )
    print("=" * 80)

    for config in CONFIGS:

        metrics = calculate_metrics(
            datasets[config],
            best_threshold,
        )

        print(
            f"{config:<15} "
            f"error acc: {metrics['error_acc']:5.1f}  "
            f"correct acc: {metrics['correct_acc']:5.1f}  "
            f"f1: {metrics['f1']:5.1f}"
        )

    print(
        f"\nAverage F1: "
        f"{best_average_f1:.1f}"
    )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    if args.output is None:

        args.output = os.path.join(
            args.result_dir,
            "threshold_sweep.csv",
        )

    output_dir = os.path.dirname(
        args.output
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        args.output,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print(
        f"Threshold sweep saved to:"
    )

    print(
        args.output
    )


if __name__ == "__main__":
    main()