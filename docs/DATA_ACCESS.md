# Data and model access

No dataset, prediction archive, private problem text, model payload or internal recovery
ledger is bundled in this source candidate. `resources.json` lists external assets.

The [data release whitelist](DATA_RELEASE_WHITELIST.md) records 13 dataset/collection
groups, their source declarations and release decisions. Exact approved aggregate
files are listed with hashes in [data_release_policy.json](../configs/data_release_policy.json).
Upstream download is the current route for public datasets; it is not a claim that
their raw data could never be redistributed under their licenses.

For each future downloadable asset, record the source, version, license/terms, checksum,
and associated experiment before including it. A public benchmark or derived prediction
cache is not automatically cleared for redistribution. Where possible, distribute a
download/conversion script and split identifiers instead of repackaging source text.

The provider-supplied private problem/solution collection cannot be redistributed.
Experiments requiring that collection cannot be recreated from the public code alone.
Synthetic examples may illustrate formats, but must not be described as the original
training data or as reproducing the private-data results.

The recovered SFT selection pool has unresolved provenance and redistribution terms.
It is tracked separately from the provider collection; their identity has not been
established. LAMBADA's content matches the OpenAI English default test in all four
recovered per-example runs; the older paper Base run has matching configuration only.
Historical local file bytes/download provenance remain unverified, and raw caches
remain excluded. These boundaries do not prevent publishing aggregate metrics.

Asset paths are supplied by the user. `--resource-root` reads external cached resources;
it does not import or sanitize them for redistribution. Runtime outputs may inherit
metadata from those resources and are excluded from the source release manifest.

Model requirements include the trained common initialization, evaluation PRMs, SFT
models, annotation models and semantic judges. Exact access routes and weight payload
checksums are not yet complete. Config and tokenizer recovery alone does not verify
all model weight bytes.
