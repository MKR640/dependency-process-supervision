# Table 2 reproduction package

Companion materials for **Tracing Error Origins and Propagation:
Dependency-Grounded Process Supervision for Mathematics and Physics**.

This standalone CPU package recomputes Table 2(a), Table 2(b), and paired
contrasts from frozen predictions for the five shared-label conditions on
PRM800K. It includes all inputs needed for these calculations.

## Run

Use Python 3.12. From the extracted package directory:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then:

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

Only NumPy is required. No GPU, model weights, API key, or external dataset
is needed. After installing NumPy, the calculation runs offline.

The program verifies package hashes, computes the results from the numeric
data, checks them against the archived reference at absolute tolerance
`1e-12`, and prints both tables. A successful run exits with code `0` and writes:

| Output | Contents |
|---|---|
| `results/table2.md` | Both paper tables and the role macro-F1 difference with its 95% CI |
| `results/table2a.csv`, `results/table2b.csv` | Unrounded table values, expressed as percentages |
| `results/contrasts.csv` | Paired supervision contrasts and 95% CIs, in percentage points |
| `results/metrics.json` | Full metrics, per-class scores, confusion matrices, thresholds and intervals |
| `results/verification.json` | Execution environment, data hashes and successful reference comparison |

Existing output directories are never overwritten. To run again:

```bash
python reproduce.py --output results_second_run
```

## What is reproduced

- Table 2(a): Error AP, Error F1, balanced accuracy, first-error accuracy on
  all/error-containing trajectories, role macro-F1 and type macro-F1.
- Table 2(b): the validation-calibrated sequential role rule and the learned
  hierarchical role head. The macro-F1 difference is approximately **+4.56
  points**, with paired 95% CI **[3.37, 5.72]**.
- The five conditions are first-error masking, full-trace binary,
  correctness + type, correctness + role, and full hierarchy. They share the
  dependency-derived reference labels. The binary condition corresponds to
  the paper's *BFE-style binary* condition, not a separate relabeling experiment.

Decision thresholds are selected on the validation split. Type F1 averages
the six supported types on reference new errors with known types; role F1
averages all three role classes. The two paired bootstrap procedures retain
their original group ordering and seeds (2,000 replicates each).

This is metric reproduction from fixed model outputs. It does not rerun
annotation, model inference or training; its intervals quantify test-problem
sampling uncertainty, not variation across training runs.

## Data format

`data/validation.npz` and `data/test.npz` contain numeric arrays only, loaded
with `allow_pickle=False`. Both contain row-aligned step labels and error
probabilities; the test file also contains the required role/type probabilities
and bootstrap groups. See [SCHEMA.md](SCHEMA.md) and [protocol.json](protocol.json).

Original trajectory IDs and per-question hashes have been replaced by integer
codes. Group membership and ordering are preserved, so calibration, alignment,
and paired confidence intervals remain reproducible. There are **no problem
statements, answers, worked solutions or reasoning text** in this package;
no private mathematics/physics collection is included.

`expected.json` is used only to check already-computed results. It does not
supply thresholds, predictions or metric outputs to the calculation.

## Source and scope

The numerical functions are retained from the accompanying code repository at
commit `39ee6d3b3a3837b88767a359b72bde3f91df6107`. The standalone entry adapts
loading to the text-free numeric files and restricts reporting to Table 2's
five conditions. `manifest.json` records the hashes of all package inputs and
code. See [NOTICES.md](NOTICES.md).
