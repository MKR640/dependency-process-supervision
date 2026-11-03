# Data and model access

The [Table 2 package](../reproduction/table2/README.md) includes numeric labels,
probabilities and anonymous grouping codes. It contains no question, answer or
reasoning text. Public dataset sources and included materials are listed in
the [data release inventory](DATA_RELEASE_WHITELIST.md).

Other replay modules require external saved predictions and labels, supplied
through `--resource-root`; see [reproduction instructions](REPRODUCTION.md).
Fresh inference and training also require model weights, including the trained
common three-class initialization for the shared-label study. These model
downloads are not currently provided by this repository.

The provider-supplied mathematics/physics collection cannot be redistributed.
The SFT selection pool is also excluded: its origin and redistribution terms
have not been established, and it is not assumed to be the same collection.
Experiments using these inputs cannot be recreated from the repository alone.

Model, dataset and third-party terms are documented in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and
[resources.json](../resources.json). Author-code licensing remains pending.
