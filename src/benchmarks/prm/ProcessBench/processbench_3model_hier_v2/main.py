#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from datasets import DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm

from prm_eval import PRM_REGISTRY


DEFAULT_CONFIGS = [
    "gsm8k",
    "math",
    "olympiadbench",
    "omnimath",
]



def load_processbench_split(dataset_name: str, config: str):
    """Load one ProcessBench split from a local directory or a HF repo id."""

    if not os.path.exists(dataset_name):
        return load_dataset(
            dataset_name,
            split=config,
        )

    root = os.path.abspath(dataset_name)

    if not os.path.isdir(root):
        raise ValueError(
            f"Local --dataset must be a directory, got: {root}"
        )

    # Whole DatasetDict produced by datasets.DatasetDict.save_to_disk().
    if os.path.isfile(os.path.join(root, "dataset_dict.json")):
        ds = load_from_disk(root)
        if not isinstance(ds, DatasetDict):
            raise ValueError(
                f"{root} has dataset_dict.json but did not load as DatasetDict."
            )
        if config not in ds:
            raise KeyError(
                f"Split '{config}' not found in {root}. "
                f"Available splits: {list(ds.keys())}"
            )
        print(f"Loading local DatasetDict split: {config}")
        return ds[config]

    # Split independently saved with Dataset.save_to_disk().
    split_dir = os.path.join(root, config)
    if os.path.isdir(split_dir):
        if (
            os.path.isfile(os.path.join(split_dir, "state.json"))
            or os.path.isfile(os.path.join(split_dir, "dataset_info.json"))
        ):
            try:
                ds = load_from_disk(split_dir)
                print(f"Loading local saved split: {split_dir}")
                return ds
            except Exception:
                pass

    # Plain files directly under root.
    formats = [
        ("parquet", [f"{config}.parquet"]),
        ("json", [f"{config}.jsonl", f"{config}.json"]),
        ("csv", [f"{config}.csv"]),
    ]

    for loader_name, names in formats:
        files = [
            os.path.join(root, name)
            for name in names
            if os.path.isfile(os.path.join(root, name))
        ]
        if files:
            print(
                f"Loading local ProcessBench split '{config}' from: {files}"
            )
            return load_dataset(
                loader_name,
                data_files=files,
                split="train",
            )

    # Plain files inside <root>/<config>/.
    if os.path.isdir(split_dir):
        for loader_name, pattern in [
            ("parquet", "*.parquet"),
            ("json", "*.jsonl"),
            ("json", "*.json"),
            ("csv", "*.csv"),
        ]:
            files = sorted(str(p) for p in Path(split_dir).glob(pattern))
            if files:
                print(
                    f"Loading local ProcessBench split '{config}' "
                    f"from {len(files)} file(s) under {split_dir}"
                )
                return load_dataset(
                    loader_name,
                    data_files=files,
                    split="train",
                )

    raise FileNotFoundError(
        f"Cannot find local ProcessBench split '{config}' under {root}.\\n"
        "Supported layouts include:\\n"
        "  <root>/dataset_dict.json + split directories (save_to_disk)\\n"
        f"  <root>/{config}/state.json (split saved with save_to_disk)\\n"
        f"  <root>/{config}.jsonl\\n"
        f"  <root>/{config}.json\\n"
        f"  <root>/{config}.parquet\\n"
        f"  <root>/{config}/*.jsonl or *.parquet"
    )


def validate_local_processbench(dataset_name: str, configs=None):
    """Validate all local ProcessBench splits before loading a large model."""

    if configs is None:
        configs = DEFAULT_CONFIGS

    if not os.path.exists(dataset_name):
        raise FileNotFoundError(
            f"Local ProcessBench path does not exist: {dataset_name}"
        )

    print("=" * 80)
    print("Preflight: validating local ProcessBench dataset")
    print(f"Dataset root: {os.path.abspath(dataset_name)}")
    print("=" * 80)

    for config in configs:
        ds = load_processbench_split(dataset_name, config)
        required = {"problem", "steps", "label"}
        missing = required - set(ds.column_names)
        if missing:
            raise ValueError(
                f"Split '{config}' is missing required columns: "
                f"{sorted(missing)}. Columns: {ds.column_names}"
            )
        print(
            f"[OK] {config:<15} samples={len(ds):>6} "
            f"columns={ds.column_names}"
        )

    print("=" * 80)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "ProcessBench evaluation for "
            "process reward models."
        )
    )

    parser.add_argument(
        "--prm-type",
        type=str,
        required=True,
        choices=list(
            PRM_REGISTRY.keys()
        ),
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help=(
            "Local checkpoint path or "
            "HuggingFace model name."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="Qwen/ProcessBench",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "NON_ERROR probability threshold. "
            "score < threshold => ERROR."
        ),
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--step-separator",
        type=str,
        default="\\n\\n",
        help=(
            "Step separator for Qwen3 PRM. "
            "Default is double newline."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--error-label-ids",
        nargs="+",
        type=int,
        default=None,
        help=(
            "For multi-class Qwen3 PRMs: label ids that mean ERROR. "
            "P(NON_ERROR) is computed as 1 - sum(P(error labels))."
        ),
    )

    parser.add_argument(
        "--non-error-label-ids",
        nargs="+",
        type=int,
        default=None,
        help=(
            "For multi-class Qwen3 PRMs: label ids that mean NON_ERROR. "
            "P(NON_ERROR) is computed as sum(P(non-error labels))."
        ),
    )

    parser.add_argument(
        "--hierarchical-score-mode",
        type=str,
        default="joint_first_error",
        choices=[
            "joint_first_error",
            "joint_first_error_raw",
            "logit_fusion",
            "new_error",
            "correctness",
            "correctness_role_product",
            "correctness_role_mean",
        ],
        help=(
            "How the hierarchical PRM is converted to a ProcessBench "
            "NON_ERROR score. 'new_error' uses 1-P(role=new_error) and "
            "directly targets first-error localization."
        ),
    )

    parser.add_argument(
        "--hierarchical-alpha",
        type=float,
        default=0.5,
        help=(
            "Alpha for hierarchical logit_fusion. "
            "0 uses role only; 1 uses correctness only."
        ),
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        choices=DEFAULT_CONFIGS,
    )

    return parser.parse_args()


def decode_separator(
    separator: str,
):

    """
    Convert shell-friendly '\\n\\n'
    into actual double-newline characters.
    """

    return separator.encode(
        "utf-8"
    ).decode(
        "unicode_escape"
    )


def save_jsonl(
    data: List[Dict],
    path: str,
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for item in data:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_json(
    data: Dict,
    path: str,
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def calculate_metrics(
    results: List[Dict],
):

    error_data = [
        item
        for item in results
        if int(item["label"]) != -1
    ]

    correct_data = [
        item
        for item in results
        if int(item["label"]) == -1
    ]

    if error_data:

        error_acc = (
            sum(
                item["match"]
                for item in error_data
            )
            / len(error_data)
        )

    else:
        error_acc = 0.0

    if correct_data:

        correct_acc = (
            sum(
                item["match"]
                for item in correct_data
            )
            / len(correct_data)
        )

    else:
        correct_acc = 0.0

    overall_acc = (
        sum(
            item["match"]
            for item in results
        )
        / max(
            1,
            len(results),
        )
    )

    if (
        error_acc
        + correct_acc
        > 0
    ):

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
        "num_samples":
            len(results),

        "num_error_samples":
            len(error_data),

        "num_correct_samples":
            len(correct_data),

        "error_acc":
            error_acc,

        "correct_acc":
            correct_acc,

        "overall_acc":
            overall_acc,

        "f1":
            f1,
    }


def evaluate_config(
    prm,
    dataset_name: str,
    config: str,
    output_dir: str,
):

    print()
    print("=" * 80)
    print(
        f"Evaluating ProcessBench: {config}"
    )
    print("=" * 80)

    dataset = load_processbench_split(
        dataset_name=dataset_name,
        config=config,
    )

    print(
        f"Number of samples: {len(dataset)}"
    )

    results = []

    for start in tqdm(
        range(
            0,
            len(dataset),
            prm.batch_size,
        ),
        desc=config,
    ):

        end = min(
            start
            + prm.batch_size,
            len(dataset),
        )

        batch_examples = [
            dict(dataset[i])
            for i in range(
                start,
                end,
            )
        ]

        batch_predictions = (
            prm.predict_batch(
                batch_examples
            )
        )

        for example, prediction_info in zip(
            batch_examples,
            batch_predictions,
        ):

            result = dict(
                example
            )

            result[
                "step_scores"
            ] = prediction_info[
                "step_scores"
            ]

            result[
                "step_predictions"
            ] = prediction_info[
                "step_predictions"
            ]

            result[
                "prediction"
            ] = prediction_info[
                "prediction"
            ]

            if (
                "step_aux"
                in prediction_info
            ):
                result[
                    "step_aux"
                ] = prediction_info[
                    "step_aux"
                ]

            result[
                "match"
            ] = (
                int(
                    prediction_info[
                        "prediction"
                    ]
                )
                == int(
                    example["label"]
                )
            )

            results.append(
                result
            )

    # ========================================================
    # Metrics
    # ========================================================

    metrics = calculate_metrics(
        results
    )

    print()
    print(
        f"{config} error acc   : "
        f"{metrics['error_acc'] * 100:.1f}"
    )

    print(
        f"{config} correct acc : "
        f"{metrics['correct_acc'] * 100:.1f}"
    )

    print(
        f"{config} overall acc : "
        f"{metrics['overall_acc'] * 100:.1f}"
    )

    print(
        f"{config} F1          : "
        f"{metrics['f1'] * 100:.1f}"
    )

    # ========================================================
    # Save
    # ========================================================

    config_dir = os.path.join(
        output_dir,
        config,
    )

    os.makedirs(
        config_dir,
        exist_ok=True,
    )

    save_jsonl(
        results,
        os.path.join(
            config_dir,
            "all_results.jsonl",
        ),
    )

    error_results = [
        item
        for item in results
        if int(item["label"]) != -1
    ]

    correct_results = [
        item
        for item in results
        if int(item["label"]) == -1
    ]

    save_jsonl(
        error_results,
        os.path.join(
            config_dir,
            "error.jsonl",
        ),
    )

    save_jsonl(
        correct_results,
        os.path.join(
            config_dir,
            "correct.jsonl",
        ),
    )

    save_json(
        metrics,
        os.path.join(
            config_dir,
            "metrics.json",
        ),
    )

    return metrics


def main():

    args = parse_args()

    if args.configs is None:
        args.configs = (
            DEFAULT_CONFIGS
        )

    if not (
        0.0
        <= args.threshold
        <= 1.0
    ):
        raise ValueError(
            "--threshold must be in [0, 1]"
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    print(
        f"Loading PRM: {args.prm_type}  "
        f"model: {args.model_name}"
    )

    # ========================================================
    # Build PRM
    # ========================================================

    prm_class = PRM_REGISTRY[
        args.prm_type
    ]

    common_kwargs = {
        "model_name":
            args.model_name,

        "batch_size":
            args.batch_size,

        "threshold":
            args.threshold,

        "max_length":
            args.max_length,
    }

    if (
        args.prm_type
        == "qwen3-math-prm-4b"
    ):

        common_kwargs[
            "step_separator"
        ] = decode_separator(
            args.step_separator
        )

        common_kwargs[
            "error_label_ids"
        ] = args.error_label_ids

        common_kwargs[
            "non_error_label_ids"
        ] = args.non_error_label_ids

        common_kwargs[
            "hierarchical_score_mode"
        ] = args.hierarchical_score_mode

        common_kwargs[
            "hierarchical_alpha"
        ] = args.hierarchical_alpha

    prm = prm_class(
        **common_kwargs
    )

    prm.load()

    # ========================================================
    # Evaluate
    # ========================================================

    summary = {
        "prm_type":
            args.prm_type,

        "model_name":
            args.model_name,

        "dataset":
            args.dataset,

        "batch_size":
            args.batch_size,

        "threshold":
            args.threshold,

        "max_length":
            args.max_length,

        "hierarchical_score_mode":
            args.hierarchical_score_mode,

        "configs": {},
    }

    f1_scores = []

    for config in args.configs:

        metrics = evaluate_config(
            prm=prm,
            dataset_name=args.dataset,
            config=config,
            output_dir=args.output_dir,
        )

        summary[
            "configs"
        ][config] = metrics

        f1_scores.append(
            metrics["f1"]
        )

    if f1_scores:

        average_f1 = (
            sum(f1_scores)
            / len(f1_scores)
        )

    else:

        average_f1 = 0.0

    summary[
        "average_f1"
    ] = average_f1

    # ========================================================
    # Save summary
    # ========================================================

    summary_path = os.path.join(
        args.output_dir,
        "summary.json",
    )

    save_json(
        summary,
        summary_path,
    )

    print()
    print("=" * 80)

    print(
        "ProcessBench Average F1: "
        f"{average_f1 * 100:.1f}"
    )

    print(
        f"Summary saved to: "
        f"{summary_path}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()