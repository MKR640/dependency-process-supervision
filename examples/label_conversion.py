"""Run the actual target converter on an authored synthetic example (stdlib only).

Labels are supplied explicitly, not inferred by this demonstration. No model,
API, research dataset or stored prediction is loaded.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New directory for generated example files")
    args = parser.parse_args()
    output = args.output_dir or ROOT / "outputs" / ("example_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))

    source = ROOT / "src/annotation/prepare_prm800k_phase2_hierarchical.py"
    spec = importlib.util.spec_from_file_location("target_converter", source)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)

    record = {
        "id": "synthetic-ideal-gas",
        "context": {
            "problem": "For one mole of ideal gas at T=300 K and p=100000 Pa, use R=8.314 J/(mol K) to compute V and pV.",
            "subject": "physics",
        },
        "first_error": "s2",
        "steps": [
            {"id": "s1", "text": "nRT = 1 * 8.314 * 300 = 2494.2 J.",
             "prm_label": "C", "role": "correct"},
            {"id": "s2", "text": "V = nRT/p = 2494.2/100000 = 0.24942 cubic metres.",
             "prm_label": "I", "role": "new_error", "error_type": "calculation"},
            {"id": "s3", "text": "Using that V, pV = 100000 * 0.24942 = 24942 J.",
             "prm_label": "I", "role": "propagated_error", "caused_by": ["s2"]},
            {"id": "s4", "text": "Independently, the value from s1 gives nRT/T = 2494.2/300 = 8.314 J/K.",
             "prm_label": "C", "role": "correct"},
            {"id": "s5", "text": "Using pV from s3, (pV)/2 = 24942/2 = 1247.1 J.",
             "prm_label": "I", "role": "new_error", "error_type": "calculation",
             "caused_by": ["s3"], "mixed_error": True},
            {"id": "s6", "text": "Correcting the division: V = 0.024942 cubic metres, hence pV = 2494.2 J.",
             "prm_label": "C", "role": "cessation"},
        ],
    }
    trajectory, rows = converter.convert_record(record, "demo", "demo", Path("synthetic_example.json"))
    # The historical converter stamps a dataset name. This fixture is authored
    # for the demonstration and is not a PRM800K example.
    for item in [trajectory, *rows]:
        item["source"] = "synthetic_example"
        item["phase"] = "demo"

    expected = {
        "correctness_ids": [1, 0, 0, 1, 0, 1],
        "role_ids": [0, 1, 2, 0, 1, 0],
        "error_type_mask": [False, True, False, False, True, False],
        "mixed_error": [False, False, False, False, True, False],
    }
    for key, value in expected.items():
        if trajectory[key] != value:
            raise RuntimeError(f"Unexpected {key}: {trajectory[key]}")
    for index, row in enumerate(rows):
        if row["prefix"] != trajectory["completions"][:index]:
            raise RuntimeError(f"Incorrect input prefix at step {index + 1}")
        if not row["error_type_mask"] and row["error_type_id"] != converter.IGNORE_INDEX:
            raise RuntimeError("Inactive type targets must use IGNORE_INDEX")

    output.mkdir(parents=True, exist_ok=False)
    for name, value in [("synthetic_input.json", record), ("trajectory.json", trajectory), ("flat_rows.json", rows)]:
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print("Synthetic example: supplied labels -> training targets (no model inference).")
    print("step  correctness  role               type target     mixed")
    for row in rows:
        print(f"{row['step_id']:4}  {row['prm_label']:11}  {row['role']:17}  {str(row['error_type'] or 'masked'):14}  {row['mixed_error']}")
    print(f"Checks passed. Example files: {output}")


if __name__ == "__main__":
    main()
