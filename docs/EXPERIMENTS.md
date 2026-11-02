# Experiment-to-code mapping

This guide maps the experiments in *Tracing Error Origins and Propagation:
Dependency-Grounded Process Supervision for Mathematics and Physics* to the
implementations and configurations in this repository.

| Paper location | Study | Main code entry |
|---|---|---|
| Methods; data, attribution, and training appendices | Annotation and supervision construction | `src/annotation/`, `configs/annotation/prompts/`, `configs/table2/` |
| Table 1 | Existing-model detection and diagnosis in mathematics and physics | `src/existing_eval/`, `src/benchmarks/trl/mrben_reeval_fixed/` |
| Table 2(a) | Shared-label supervision objectives | `src/table2/`; `scripts/reproduce.py --module table2` |
| Table 2(b) | Learned roles versus correctness-based sequential inference | `scripts/replay_role_baseline.py`; `--module roles` |
| Table 3 | ProcessBench first-error localization and correct-trace acceptance | `src/benchmarks/prm/ProcessBench/processbench_3model_hier_v2/` |
| Figure 4(a) | Shared-pool answer voting | `src/downstream/prm_3_cls_vote_aligned/` |
| Figure 4(b); SFT appendix | Training-data selection and downstream evaluation | `src/downstream/prm_sft_selection/`, `sft_training/`, `evaluation_entrypoints/` |

See the [reproduction guide](REPRODUCTION.md) for commands and external inputs.

| Experiment | Implementation / configuration | Reproduction level |
|---|---|---|
| Dependency annotation | `src/annotation/pipeline_code/`, prompts in `configs/annotation/prompts/` | Online annotation requires external models. Offline replay fixes their outputs. |
| Hierarchical training data | `src/annotation/prepare_prm800k_phase2_hierarchical.py` | Materialized labels to trajectory/flat formats; shared-label runs use the separately frozen grouped split. |
| Shared-label M0/M5 | `src/table2/original/train.py`, `configs/table2/M0.json`, `M5.json` | First-error masking / full hierarchy; common initialization and frozen labels. |
| Shared-label M1/M3/M4 | `src/table2/supplement/train_expanded.py`, matching configurations | Full-trace binary / correctness+type / correctness+role. |
| Formal shared-label metrics | `src/table2/reporting/report_supplement_results.py` via `scripts/reproduce.py --module table2` | Formal-cache replay verified against the complete historical report; external cache is not bundled in this source candidate. |
| Sequential role comparison (+4.56 macro-F1) | `scripts/replay_role_baseline.py`, `role_baseline_functions.py`; `--module roles` | Formal-cache replay verified: threshold 0.511; macro-F1 66.2449104% vs 70.8018613%; delta 4.55695094 points with paired 95% CI [3.36695527, 5.72121926]. |
| Existing mathematics/physics PRMs | `src/existing_eval/`, `src/benchmarks/trl/mrben_reeval_fixed/` | Cached metric replay; fresh inference requires model weights. |
| ProcessBench | `src/benchmarks/prm/ProcessBench/processbench_3model_hier_v2/` | Cached correctness-role decoding and metric replay. |
| Answer voting | `src/downstream/prm_3_cls_vote_aligned/` | Replay summarizes stored per-question selections; fresh reranking needs the candidate pool and probabilities. |
| SFT data selection | `src/downstream/prm_sft_selection/select_prm_sft.py` | Recorded scorer: **three-class PRM**. See `configs/downstream_protocol.json`. |
| SFT training / external benchmarks | `src/downstream/sft_training/`, `evaluation_entrypoints/`, `opencompass/` | Cached results available internally; private selection pool and model weights are external. |

## Objective and historical conditions

The shared-label loss is weighted CE summed over eligible targets, divided by the
number of input rows. Masked rows remain in that denominator. Role/type coefficients
and class weights are preserved in the five protocol files. Changing normalization
would define a new experiment.

The shared-label comparison holds dependency-derived labels fixed; it tests coverage
and auxiliary objectives, not an annotation-method ablation. Existing-model comparisons
use their recorded model histories. Voting and SFT selection consume correctness scores.

## SFT scorer provenance

Recovered selection records identify the common three-class PRM as the SFT scorer.
The archived submitted manuscript describes this scorer as hierarchical; this source
candidate follows the actual recovered run rather than substituting another model.
The cached result values are unchanged. Manuscript clarification is tracked separately
and is not an edit performed by this code release.
