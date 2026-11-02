#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

CONFIGS = ["gsm8k", "math", "olympiadbench", "omnimath"]


def pct(x):
    return float(x) * 100.0


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def best_sweep_row(rows):
    if not rows:
        return None
    return max(rows, key=lambda r: float(r["average_f1"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model directory names under output-root.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
    )
    parser.add_argument(
        "--json-output",
        default=None,
    )
    args = parser.parse_args()

    root = Path(args.output_root)
    rows = []
    detailed = {}

    for model in args.models:
        model_dir = root / model
        summary_path = model_dir / "summary.json"
        sweep_path = model_dir / "threshold_sweep.csv"

        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing {summary_path}")
        if not sweep_path.is_file():
            raise FileNotFoundError(f"Missing {sweep_path}")

        summary = load_json(summary_path)
        sweep_rows = load_csv(sweep_path)
        best = best_sweep_row(sweep_rows)

        fixed_threshold = float(summary["threshold"])
        fixed = {
            cfg: {
                "error_acc": pct(summary["configs"][cfg]["error_acc"]),
                "correct_acc": pct(summary["configs"][cfg]["correct_acc"]),
                "overall_acc": pct(summary["configs"][cfg]["overall_acc"]),
                "f1": pct(summary["configs"][cfg]["f1"]),
            }
            for cfg in CONFIGS
        }
        fixed_avg = pct(summary["average_f1"])

        best_threshold = float(best["threshold"])
        tuned = {
            cfg: {
                "error_acc": float(best[f"{cfg}_error_acc"]),
                "correct_acc": float(best[f"{cfg}_correct_acc"]),
                "overall_acc": float(best[f"{cfg}_overall_acc"]),
                "f1": float(best[f"{cfg}_f1"]),
            }
            for cfg in CONFIGS
        }
        tuned_avg = float(best["average_f1"])

        row = {
            "model": model,
            "fixed_threshold": fixed_threshold,
            **{f"fixed_{cfg}_f1": fixed[cfg]["f1"] for cfg in CONFIGS},
            "fixed_avg_f1": fixed_avg,
            "best_threshold": best_threshold,
            **{f"best_{cfg}_f1": tuned[cfg]["f1"] for cfg in CONFIGS},
            "best_avg_f1": tuned_avg,
        }
        rows.append(row)

        detailed[model] = {
            "model_name": summary.get("model_name"),
            "fixed_threshold": fixed_threshold,
            "fixed": fixed,
            "fixed_average_f1": fixed_avg,
            "best_threshold_on_processbench": best_threshold,
            "best": tuned,
            "best_average_f1": tuned_avg,
        }

    csv_output = Path(args.csv_output) if args.csv_output else root / "processbench_3models_summary.csv"
    json_output = Path(args.json_output) if args.json_output else root / "processbench_3models_detailed.json"
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)

    with csv_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with json_output.open("w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)

    print("\nFixed threshold results (F1, %)")
    print("=" * 104)
    print(f"{'model':54} {'thr':>6} {'gsm8k':>8} {'math':>8} {'olymp':>8} {'omni':>8} {'avg':>8}")
    print("-" * 104)
    for r in rows:
        print(
            f"{r['model'][:54]:54} {r['fixed_threshold']:6.2f} "
            f"{r['fixed_gsm8k_f1']:8.2f} {r['fixed_math_f1']:8.2f} "
            f"{r['fixed_olympiadbench_f1']:8.2f} {r['fixed_omnimath_f1']:8.2f} "
            f"{r['fixed_avg_f1']:8.2f}"
        )

    print("\nBest threshold on ProcessBench sweep (F1, %)")
    print("=" * 104)
    print(f"{'model':54} {'thr':>6} {'gsm8k':>8} {'math':>8} {'olymp':>8} {'omni':>8} {'avg':>8}")
    print("-" * 104)
    for r in rows:
        print(
            f"{r['model'][:54]:54} {r['best_threshold']:6.2f} "
            f"{r['best_gsm8k_f1']:8.2f} {r['best_math_f1']:8.2f} "
            f"{r['best_olympiadbench_f1']:8.2f} {r['best_omnimath_f1']:8.2f} "
            f"{r['best_avg_f1']:8.2f}"
        )

    print(f"\nCSV:  {csv_output}")
    print(f"JSON: {json_output}")


if __name__ == "__main__":
    main()
