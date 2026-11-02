# Third-party notices and license inventory

Reviewed 2026-11-02. **Author-code licensing remains undecided.** This document
records upstream declarations; it does not relicense the project or clear every
recovered source file. The structured inventory is
[third_party_licenses.json](configs/third_party_licenses.json).

## Included source and integrations

| Component | Local material | Declaration and remaining action |
|---|---|---|
| OpenCompass | Selected files under `src/downstream/opencompass/` | [Apache-2.0](https://github.com/open-compass/opencompass/blob/main/LICENSE). The recovered [LICENSE](src/downstream/opencompass/LICENSE) matches the current upstream file byte-for-byte. Preserve it and any file-level notices. The exact historical upstream commit and local modification boundaries still need mapping; this is not a complete OpenCompass distribution. |
| Local ProcessBench integration | `src/benchmarks/prm/ProcessBench/` | [Official repository](https://github.com/QwenLM/ProcessBench) has no license notice identified across 12 inspected commits. The seven local release files have no exact file/AST match to its history; recovered interface specifications and repeated local versions indicate a local integration, not an identical official export. Local authorship/adaptation rights still need confirmation. Dataset Apache-2.0 is not a code-license substitute. See [provenance checks](configs/asset_provenance_checks.json). |
| PRM / SFT wrappers and recovered model modules | `src/table2/`, `src/downstream/sft_training/`, related evaluation modules | They import Transformers/TRL. Importing a library does not establish that all wrapper code originated there. Record any copied/adapted files and original notices before release; author-owned portions remain pending the authors' license. |
| Annotation clients and prompts | `src/annotation/`, `configs/annotation/prompts/` | Project-specific material; author rights/license pending. SaT runtime/model terms are separate from prompt and client ownership. |

Third-party attribution must remain even in an anonymous submission. Preserve
required notices; identify upstream authors as upstream authors rather than removing
their names. For changed Apache-licensed files, retain notices and identify changes.

The server follow-up found no Git metadata in the inspected local integration
locations or ancestors. All seven final source hashes match the earlier recovery;
three additional local READMEs supply development context but no explicit authorship
or license grant. Original patch delivery/generation records or author confirmation
remain the next provenance evidence; the integration has not been cleared for release.

## External runtime dependencies

These are direct or configured dependencies, **not bundled library distributions**.
This list is not a complete transitive SBOM or an installation lock.

| Dependency | Verified upstream license | Local context |
|---|---|---|
| [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE) | Apache-2.0 | Model/Trainer integration; historical record 4.57.6 |
| [TRL](https://github.com/huggingface/trl/blob/main/LICENSE) | Apache-2.0 | SFT integration; exact installed revision pending |
| [Datasets](https://github.com/huggingface/datasets/blob/main/LICENSE) | Apache-2.0 | Dataset loading; historical record 5.0.1 |
| [Accelerate](https://github.com/huggingface/accelerate/blob/main/LICENSE) | Apache-2.0 | Training runtime |
| [wtpsplit](https://github.com/segment-any-text/wtpsplit/blob/main/LICENSE) | MIT | SaT segmentation client |
| [vLLM](https://github.com/vllm-project/vllm/blob/main/LICENSE) | Apache-2.0 | Configured generation/evaluation backend |
| [DeepSpeed](https://github.com/deepspeedai/DeepSpeed/blob/master/LICENSE) | Apache-2.0 | Configured distributed training |
| [NumPy 2.3.5](https://github.com/numpy/numpy/blob/v2.3.5/LICENSE.txt) | BSD-3-Clause | Validated CPU replay dependency; installed wheel may carry additional bundled-library notices |
| [PyTorch 2.8.0](https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE) | BSD-style root terms plus included component notices | Historical runtime is a vendor-specific build; upstream license alone does not clear redistribution of that build |

Install dependencies from their distributors with their notices. A later container,
wheel bundle or binary release needs review of the actual packaged dependencies.

## External model assets

| Asset | Public declaration | Scope / release status |
|---|---|---|
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), [Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base) | Apache-2.0 | Public upstream families; matching exact historical base/judge revisions remains necessary. Author-trained PRMs/SFT weights are not bundled or independently cleared by this declaration. |
| [SaT sat-3l-sm](https://huggingface.co/segment-any-text/sat-3l-sm) | MIT | Named by the recovered segmentation configuration; no weights bundled |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | MIT | Named tokenizer source; no tokenizer payload bundled |
| Recorded annotation service/model identifier | Exact upstream revision and applicable terms unresolved | Do not infer its license from another Qwen model. Resolve provider/model identity before offering reproducible service access or redistributing outputs/weights. |

Data and generated caches have separate permissions; see the
[data release whitelist](docs/DATA_RELEASE_WHITELIST.md). Repository-level licenses
are not blanket grants over third-party benchmark text. Full prompt publication also
requires checking for embedded source examples rather than only the surrounding code.
