# Tracing Error Origins and Propagation

**Dependency-Grounded Process Supervision for Mathematics and Physics**

Code and reproducibility materials for the paper, including training and
evaluation scripts and a runnable reproduction package for Table 2.

*Anonymous submission to ICLR 2027.*

[Overview](#overview) · [Method](#method) · [Results](#results) ·
[Quick start](#quick-start) · [Experiments](docs/EXPERIMENTS.md) ·
[Reproduction guide](docs/REPRODUCTION.md)

## Overview

Mathematical and physical reasoning is central to AI for Science. Process reward
models (PRMs) support this reasoning by evaluating intermediate steps, but an
incorrect step can either introduce a mistake or inherit one from an earlier
step. These cases call for different feedback.

Derivations reuse quantities, equations, and assumptions through identifiable
input-output dependencies. We use these dependencies to construct supervision
for **step correctness, error origin or propagation, and intrinsic error type**.
Attribution follows the content actually consumed: a valid output from a
partially incorrect step can still support correct reasoning, while a step can
both inherit an error and introduce a new one.

The study connects this annotation design to PRM training, evaluates detection
and role diagnosis in mathematics and physics, and examines process scores in
answer voting and training-data selection.

## Method

![Method overview: consumed-input attribution, hierarchical supervision, and process scoring](assets/method_overview.svg)

1. **Extract inputs and outputs.** Preserve the solution text, establish step
   boundaries, and identify the quantities and premises each step uses and produces.
2. **Attribute errors through consumed content.** Separate locally introduced
   mistakes from invalid consequences inherited through the step's inputs.
3. **Learn hierarchical supervision.** Train a Qwen3-4B PRM with correctness,
   role, and intrinsic-type heads. At inference, the model evaluates the problem,
   preceding reasoning, and current step.

| Step situation | Correctness target | Role target | Intrinsic-type supervision |
|---|---|---|---|
| Contextually correct | Correct | Correct | Inactive |
| Introduces a mistake | Incorrect | New error | Active when the type is known |
| Inherits an error without a new mistake | Incorrect | Propagated error | Inactive |
| Inherits an error and introduces another | Incorrect | New error; inherited causes retained in annotations | Type of the newly introduced mistake |
| Recovers valid reasoning | Correct | Correct | Inactive |

Correctness and role supervision cover all evaluable steps. Type supervision
applies to known new-error types, including mixed errors. The full objective is
`L = L_correctness + 0.5 L_role + 0.5 L_type`.

The [annotation prompts](configs/annotation/prompts/),
[target converter](src/annotation/prepare_prm800k_phase2_hierarchical.py), and
[training configurations](configs/table2/) implement these stages.

## Results

### Error detection in mathematics and physics

Existing-model comparison from Table 1; Error AP is reported in percent.

| Model | PRM800K | Annotated physics subset |
|---|---:|---:|
| Binary PRM | 72.85 | 70.66 |
| Three-class PRM | 75.77 | 75.83 |
| Hierarchical PRM | **86.33** | **84.23** |

These are comparisons of models with different training histories. The separate
shared-label study below fixes initialization and dependency-derived labels to
compare supervision objectives.

### Error-role diagnosis under shared labels

Table 2(b) compares learned role predictions with a sequential rule applied to
full-trace binary scores. The rule marks the first predicted error as new and
subsequent predicted errors as propagated.

| Predictor | Role macro-F1 (%) | New-error F1 (%) | Propagated-error F1 (%) |
|---|---:|---:|---:|
| Full-trace binary scores + sequential rule | 66.24 | 46.81 | 69.61 |
| Hierarchical role head | **70.80** | **58.90** | **71.81** |

The role macro-F1 improvement is **4.56 points** (paired 95% CI [3.37, 5.72]).
Detection AP is similar for full hierarchy and full-trace binary supervision
(86.58% vs. 86.53%). This comparison measures the diagnostic information added by
role supervision while holding attribution labels fixed.

The paper also reports reasoning-length and first-error-position analyses,
ProcessBench evaluation, answer voting, and SFT data selection. The
[experiment map](docs/EXPERIMENTS.md) connects these studies to their code and
configurations.

## Quick start

### Run the label-construction example

Requires Python 3.12; no additional packages, model, API key, or dataset are needed.
From the repository root:

```bash
python examples/label_conversion.py
```

The example converts supplied labels for a synthetic ideal-gas derivation into
training targets, covering new errors, propagation, mixed errors and recovery.
Outputs are saved under `outputs/example_*`.

### Reproduce Table 2 and role diagnosis

[Download the standalone replay package (about 2 MB)](https://github.com/MKR640/dependency-process-supervision/raw/refs/heads/main/artifacts/table2-role-replay-v1.zip)
· [Instructions and source](reproduction/table2/README.md)

The package includes text-free numeric predictions, reference labels, and anonymous
group codes for all five shared-label conditions. No GPU, model weights, API key,
or external dataset is needed. After extracting the ZIP, run from its directory:

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

With Python 3.12, this recomputes both Table 2 panels and paired confidence
intervals, verifies them against archived results, and writes Markdown, CSV and
JSON outputs under `results/`. From a repository clone, use the root entry instead:

```bash
python -m pip install -r environments/cpu/requirements.txt
python scripts/reproduce.py --module table2
```

This writes the same results under `outputs/<run>/table2/`. The `roles` module
also uses the included package. No problem statements or reasoning text are included.
This replays fixed model outputs; it does not retrain the models.

### Training and other experiments

See the [reproduction guide](docs/REPRODUCTION.md) for annotation, training,
other evaluation commands and their required data or model weights.

## Repository structure

| Directory | Contents |
|---|---|
| [`src/annotation/`](src/annotation/) | Step processing, dependency attribution, and training-target construction |
| [`src/table2/`](src/table2/) | Shared-label training objectives and result reporting |
| [`src/existing_eval/`](src/existing_eval/) and [`src/benchmarks/`](src/benchmarks/) | Mathematics, physics, and ProcessBench evaluation |
| [`src/downstream/`](src/downstream/) | Answer voting, SFT selection, training, and task evaluation |
| [`configs/`](configs/) | Annotation prompts, training settings, evaluation protocols, and reference metrics |
| [`reproduction/table2/`](reproduction/table2/) | Standalone Table 2 and role-diagnosis replay with numeric inputs |
| [`scripts/`](scripts/) | Original-bundle reproduction and training-preparation entry points |
| [`examples/`](examples/) | Self-contained synthetic example |
| [`environments/`](environments/) | CPU dependencies and training/evaluation environment records |
| [`docs/`](docs/) | Experiment mapping, reproduction instructions, and resource availability |

## Resources and availability

This release contains source code, prompts, configurations, reference metrics,
the synthetic example, and the text-free Table 2 reproduction package. Model
weights, original datasets, and other prediction caches are not bundled. Public dataset sources and private-data restrictions are listed
in [Data access](docs/DATA_ACCESS.md). The provider-supplied problem collection
cannot be redistributed.

Table 2 replay has been tested on Windows with Python 3.12 and NumPy 2.3.5.
See [execution coverage](docs/RELEASE_STATUS.md) for other evaluation and training entries.

Author-code licensing is pending. Third-party notices and component-specific
terms are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
