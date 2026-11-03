# Data release whitelist

Updated 2026-11-03. The text-free Table 2 numeric export is included; other
external caches remain excluded. Author-owned code,
prompts and new annotations still require an author-selected license.

The machine-readable inventory is [data_release_policy.json](../configs/data_release_policy.json).
It separates the upstream license declaration, evidence of local use, and the
decision to include an asset. A current upstream revision is not necessarily the
revision used in the experiments.

## What may accompany this candidate

| Material | Decision | Conditions |
|---|---|---|
| Existing aggregate metrics and protocol metadata | Include the exact files and hashes listed in the policy | No problem text, individual responses, private paths or annotator identities; preserve experiment/model distinctions |
| Download/conversion code, public source links, label definitions | Include as source/configuration | Follow code licenses; a download helper does not grant rights to the downloaded data |
| Table 2 numeric labels, probabilities and integer grouping codes | Included in `reproduction/table2/` and the standalone ZIP | Explicit field whitelist; no free text, original IDs or per-question hashes; hash-checked and numerically equivalent |
| Other public-example IDs, split membership and derived annotations | Not included; assess separately | The Table 2 export does not authorize other caches or private collections |
| Original questions, solutions, model reasoning, judge explanations, retrieved input/output content | Not included in this candidate | Upstream permission may allow a future release, but text-containing derivatives need source-specific attribution and checks |
| Provider collection, unknown-origin SFT pool, internal recovery archives | Withhold | See dataset decisions below; internal bundles are not release packages |
| Model weights or tokenizers | External assets | Separate model-license and exact-version review; see third-party notices |

The approved-file list covers **data-like result/configuration material only**. It
is not a replacement for the source manifest or the unresolved author-code license.

## Actual data use and release route

| Dataset / collection | Use in this project | Verified public declaration | Current release decision |
|---|---|---|---|
| PRM800K, phase 2 | Annotation replay, shared-label training, mathematical evaluation | [Official repository: MIT](https://github.com/openai/prm800k/blob/main/LICENSE); includes problems from MATH | Obtain upstream; provide transformations and the included text-free Table 2 numeric export. Do not redistribute the recovered materialized JSONL or teacher bundles as-is. Preserve MATH provenance. |
| PRMBench Preview, physics subset | Existing-model physics evaluation | [Dataset card: Apache-2.0](https://huggingface.co/datasets/hitsmy/PRMBench_Preview) | Obtain upstream; identify the selected source rows and annotation conversion. No raw subset bundled. Local historical filename aliases do not identify a different public benchmark. |
| ProcessBench | First-error evaluation on its four subsets | [Dataset card: Apache-2.0](https://huggingface.co/datasets/Qwen/ProcessBench) | Obtain this benchmark from upstream; retain its constituent-source attribution. Its dataset license does not settle the license of recovered evaluator code. |
| GSM8K | Answer voting and SFT-model evaluation | [Official repository: MIT](https://github.com/openai/grade-school-math/blob/master/LICENSE) | Obtain upstream; any future repackaging must retain the notice and distinguish original data from generated candidate solutions. |
| MATH | SFT-model evaluation; source of PRM800K problems | [Official repository: MIT](https://github.com/hendrycks/math/blob/main/LICENSE); [author-linked data mirror: MIT](https://huggingface.co/datasets/qwedsacf/competition_math) | Obtain from the author-linked source. Local OpenCompass data/split identity remains to be matched; do not assume the mirror's current split matches the historical experiment. |
| Omni-MATH | SFT-model evaluation | [Dataset card: Apache-2.0](https://huggingface.co/datasets/KbsdJames/Omni-MATH) | Obtain upstream; retain source/contest attribution supplied with the dataset; no text cache bundled. |
| UGPhysics, English MCQ / true-false / numeric | Three SFT-model evaluation tasks | [Dataset card: CC-BY-NC-SA-4.0](https://huggingface.co/datasets/UGPhysics/ugphysics) | External download. Any shared licensed data/adaptation must retain attribution, noncommercial and ShareAlike conditions; do not place under the project code license. |
| PROST | SFT-model evaluation | [Dataset card: Apache-2.0](https://huggingface.co/datasets/corypaik/prost) | Obtain upstream; no raw evaluation records bundled. |
| IFEval | SFT-model evaluation | [Official dataset card: Apache-2.0](https://huggingface.co/datasets/google/IFEval) | Obtain upstream; preserve instructions/data attribution; no generated-response cache bundled. |
| ARC Easy / Challenge | Two SFT-model evaluation tasks | [Official dataset card: CC-BY-SA-4.0](https://huggingface.co/datasets/allenai/ai2_arc) | External download. Retain attribution and ShareAlike conditions for distributed adaptations; record the exact local split. |
| LAMBADA OpenAI English default test, OpenCompass format | SFT-model evaluation | [Matched EleutherAI distribution: MIT](https://huggingface.co/datasets/EleutherAI/lambada_openai) | All 5,153 contexts and reference words match in each of four recovered runs. Obtain the pinned public variant and apply the recorded formatting; no raw cache bundled. Older paper Base has matching configuration only; historical local file bytes/download route remain unverified. |
| Provider mathematics/physics collection | Private data preparation/annotation | Provider confirmed redistribution is prohibited | No raw text, worked solutions, text-bearing annotations, or excerpts without additional permission. Supply code, schema and permitted aggregate descriptions. |
| SFT selection pool (internal name “Quizlet”) | All / random-matched / PRM-selected SFT training | Origin and redistribution permission not established | Withhold all row-level data and derived subsets. The internal name is neither a license nor proof of identity with the provider collection or a public service. |

Public availability and a declared license are recorded facts; neither proves that
every incorporated third-party question is owned by the distributor. Keep source
attributions and any additional notices. Current-source metadata is recorded in the
policy without claiming an exact historical match.

The CC conditions are described by the licensors in
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). They apply to the
licensed material and relevant adaptations, not automatically to independent code
merely because it evaluates that material.

## Minimal future per-example export

A public-data-only export may contain dataset/revision IDs, public example ID,
opaque trajectory ID, split, step ordinal/offsets, numeric correctness/role/type
targets, validity masks, dependency indices, numeric predictions and model IDs.
These are candidate fields, not permission to export existing dictionaries verbatim.

Exclude question/prompt/solution text, consumed/produced content strings, rationales,
reference answer text, raw model responses, annotator IDs, API/service metadata,
timestamps tied to private runs, and absolute paths. Hashing a private problem does
not by itself make it publishable. Private collections have no per-example export
permission under this policy.

Before releasing such an export, verify its source and license, retain required
notices, choose the authors' annotation license where applicable, and verify that
the exported fields reproduce the intended metrics. No current replay command is a
sanitizer: runtime outputs and original recovery ZIP/TAR files remain excluded.

## Remaining release work

1. Resolve the SFT pool's origin/permissions. LAMBADA's content variant is confirmed
   for four recovered runs; obtain the older paper Base per-example records if exact
   historical coverage of that run is required.
2. Match public source revisions and split/example IDs to the historical local data.
3. Produce and verify a text-free public-data cache if offline replay is to be offered.
4. Select the author-code/annotation license and complete the per-file third-party
   provenance review described in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Optional dataset imports in the recovered evaluation framework are not evidence
that those datasets were used in the reported experiments. The inventory follows
the recovered experiment configurations and result records.

The LAMBADA comparison uses the ordered English default test file at revision
`900124bf3b8235c6daf21033af9948b3f07346c4`. Each source text is split at its last
ASCII space into context and target word, then wrapped with the recorded completion
prompt. All four runs match exactly, including order. This establishes content
identity, not the historical download date or ownership of underlying book excerpts.
Checksums, configuration and scope are in
[asset_provenance_checks.json](../configs/asset_provenance_checks.json).
