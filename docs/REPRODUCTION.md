# Reproduction guide

The repository supports target construction, training preparation, model evaluation,
and metric replay. Choose the entry that matches the resources available to you.

| Task | Entry | Required resources |
|---|---|---|
| Inspect label construction | `python examples/label_conversion.py` | Python standard library |
| Generate annotations | `src/annotation/pipeline_code/`, `configs/annotation/prompts/` | Input trajectories and the configured segmentation/annotation models |
| Convert annotations to training examples | `src/annotation/prepare_prm800k_phase2_hierarchical.py` | Materialized step labels |
| Prepare a shared-label training run | `scripts/prepare_training.py` | Frozen train/validation/test data and common trained initialization |
| Recompute experiment metrics | `scripts/reproduce.py` | CPU environment and corresponding stored predictions/labels |
| Produce new predictions | `src/existing_eval/`, `src/benchmarks/`, `src/downstream/` | Model weights, evaluation inputs, and task-specific environment |

## Table 2: self-contained replay

Download the [standalone ZIP](https://github.com/MKR640/dependency-process-supervision/raw/refs/heads/main/artifacts/table2-role-replay-v1.zip)
and follow its [README](../reproduction/table2/README.md). Alternatively, from a
repository clone:

```bash
cd reproduction/table2
python -m pip install -r requirements.txt
python reproduce.py
```

This package includes the numeric inputs for Table 2(a), Table 2(b), validation
threshold selection and paired bootstrap intervals. It uses a compact NPZ format;
it is not an input to the original `scripts/reproduce.py --table2-bundle` command.
All values match the historical reference within absolute tolerance `1e-12`.

## CPU environment

```bash
python -m venv .venv-cpu
# Linux/macOS: source .venv-cpu/bin/activate
# Windows PowerShell: .\.venv-cpu\Scripts\Activate.ps1
python -m pip install -r environments/cpu/requirements.txt
python -m unittest discover -s tests -v
```

The tested CPU environment is Python 3.12.14 and NumPy 2.3.5 on Windows.
Training and external evaluation use separate environments; see
[environment records](../environments/README.md).

## Replay stored experiment outputs

The following commands use the original external bundle format. For the included
Table 2 package, use the self-contained command above.

The external asset-root layout is specified in
[`configs/resources.json`](../configs/resources.json). Those assets are not
included in the source repository. The replay commands consume predictions and
labels; they do not train models or call annotation APIs.

```bash
python scripts/reproduce.py --module existing --resource-root /path/to/assets --verify
python scripts/reproduce.py --module all --resource-root /path/to/assets --verify --strict
```

Shared-label detection and role diagnosis also accept a separate prediction bundle:

```bash
python scripts/reproduce.py --module table2 --table2-bundle /path/to/formal_bundle --verify
python scripts/reproduce.py --module roles --table2-bundle /path/to/formal_bundle --verify
```

| Module | Computation |
|---|---|
| `annotation` | Offline annotation/target-construction replay |
| `existing` | Existing-model metric replay |
| `benchmarks` | Benchmark result verification |
| `table2` | Shared-label experiment metrics and comparisons |
| `roles` | Sequential-rule and learned-role comparison |
| `voting` | Stored answer-selection results |
| `sft` | Stored SFT benchmark results |

Each command writes results to a new directory under `outputs/`. Exit code `0`
means all requested modules passed; `1` indicates a failed computation and `2`
indicates missing resources. Missing resources also produce a nonzero exit when
running `--module all`.

## Shared-label PRM training

| Condition | Configuration |
|---|---|
| First-error masking | [`M0.json`](../configs/table2/M0.json) |
| Full-trace binary | [`M1.json`](../configs/table2/M1.json) |
| Correctness + type | [`M3.json`](../configs/table2/M3.json) |
| Correctness + role | [`M4.json`](../configs/table2/M4.json) |
| Full hierarchy | [`M5.json`](../configs/table2/M5.json) |

To prepare the full-hierarchy run:

```bash
python scripts/prepare_training.py --arm M5 \
  --data-dir /path/to/frozen_data \
  --init-model /path/to/common_three_class_prm \
  --plan-dir outputs/new_plan \
  --output-root outputs/new_training
```

This checks the frozen data manifest and writes `config.json` and `command.sh`
under the new plan directory. It does not start training. The shell command uses
`torchrun` and the corresponding source entry; run it in a suitable Linux GPU
environment after reviewing the [training record](../environments/training/recorded.json).
The common initialization is a trained three-class PRM, not a fresh base model.

The original and supplement training directories contain the implementations used
for their respective experimental conditions. Their objective settings and batch
normalization are described in the [experiment map](EXPERIMENTS.md).

## Data and model access

See [Data access](DATA_ACCESS.md) and the root [`resources.json`](../resources.json)
for required external assets. The text-free Table 2 package is downloadable;
model weights and other prediction bundles remain external. Private-data experiments require the original
collection, which cannot be redistributed. The synthetic example runs independently
of these resources.
