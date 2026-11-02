# Dependency-grounded process supervision

Code for extracting step inputs and outputs, constructing error-role supervision,
training hierarchical process reward models, and evaluating mathematical and physical reasoning.

The repository contains annotation prompts, target construction, five training
configurations, and evaluation scripts. Model weights and evaluation caches are
external resources. Author-code licensing is pending.

## Start here

| Task | Entry | Resources |
|---|---|---|
| Try target construction | `examples/label_conversion.py` | Python standard library only |
| Recompute stored results | `scripts/reproduce.py` | Python 3.12, CPU requirements, corresponding external caches |
| Prepare a training run | `scripts/prepare_training.py` | Frozen data and the common trained initialization |
| Recreate annotations | `src/annotation/` and `configs/annotation/prompts/` | SaT and Teacher models, input trajectories |
| Generate fresh model predictions | `src/benchmarks/`, `src/existing_eval/`, `src/downstream/` | Model weights, benchmark inputs, task environment |

### Run a self-contained example

From the repository root:

```bash
python examples/label_conversion.py
```

This runs the actual target converter on a synthetic ideal-gas derivation with
supplied labels. It shows a new calculation error, propagation, a correct step
after an error, a mixed error, and recovery. Correctness and role targets cover
all steps; type targets are active only for supported new-error types. A mixed
error retains its inherited cause while receiving a new-error target, and recovery
maps to the correct training role.

The script checks targets and input prefixes and writes the input, trajectory,
and per-step training rows to a new `outputs/example_*` directory. It requires no
GPU, model, API key or dataset. It demonstrates label conversion, not automatic
attribution, model predictions or a paper benchmark result.

### CPU result replay

Use a separate environment:

```bash
python -m venv .venv-cpu
# Activate .venv-cpu using your platform's activation command.
python -m pip install -r environments/cpu/requirements.txt
python scripts/reproduce.py --module existing --resource-root /path/to/assets --verify
python scripts/reproduce.py --module all --resource-root /path/to/assets --verify --strict
```

The external asset root follows `configs/resources.json`. No original data or recovery
archives are bundled in this source candidate. Missing assets produce `blocked` and
exit code **2**, including with `--module all`. A failed computation returns **1**;
all requested modules passing returns **0**. Results are written to a fresh `outputs/` directory.

Replay uses stored predictions and judgments. It does not retrain models or call a
generation/judging API. See [experiment mapping](docs/EXPERIMENTS.md) for what each module establishes.

To run the small CPU checks after installing the CPU requirements:

```bash
python -m unittest discover -s tests -v
```

### Table 2

```bash
python scripts/reproduce.py --module table2 --table2-bundle /path/to/formal_bundle --verify
python scripts/reproduce.py --module roles --table2-bundle /path/to/formal_bundle --verify
python scripts/prepare_training.py --arm M5 \
  --data-dir /path/to/frozen_data --init-model /path/to/common_three_class_prm \
  --plan-dir outputs/new_plan --output-root outputs/new_training
```

The training command prepares a configuration and a Linux shell command; it does not
start training. Inputs are checked against the formal data manifest. A freshly
downloaded base model is not a substitute for the trained common initialization.

## Layout

- `examples/`: a self-contained target-conversion example.
- `src/`: recovered method, training and evaluation implementations.
- `configs/`: prompts, five supervision conditions, numerical protocols and expected results.
- `resources.json`: external-resource requirements and access status.
- `environments/`: separate CPU, training and external-evaluation records.
- `THIRD_PARTY_NOTICES.md`: available third-party notices and outstanding attribution checks.
- `docs/`: experiment mapping, data boundaries and release status.
- `manifest.json`: hashes of the source candidate; excludes outputs, private inputs and environments.

The two Table 2 source views correspond to different formal runs. Their training
logic is retained; they have not been merged into an unvalidated replacement.
Path-default changes are recorded in `configs/source_provenance.json`.

For current completion status, see [release status](docs/RELEASE_STATUS.md).

The seven CPU replay modules have been checked internally against recovered
experiment outputs. Those caches are not included here. Fresh GPU training and
inference, and execution on Linux, have not been validated for this release.

Before adding any data or caches, consult the [data release whitelist](docs/DATA_RELEASE_WHITELIST.md)
and [third-party notices](THIRD_PARTY_NOTICES.md). The local replay inputs are not
approved release archives. Author-code licensing remains undecided.
