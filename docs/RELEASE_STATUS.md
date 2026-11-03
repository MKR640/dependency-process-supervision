# Execution coverage

| Entry | Inputs | Verification |
|---|---|---|
| Label-construction example | Included synthetic example | Runs with the Python standard library |
| Table 2 and role diagnosis | Included numeric prediction package | Both tables, validation thresholds and paired intervals verified after ZIP extraction |
| Other CPU replay modules | External saved predictions and labels | Metrics checked against the corresponding experiment records |
| Fresh annotation, model inference and training | External data and models | Not executed as part of this release validation |

CPU checks used Windows, Python 3.12.14 and NumPy 2.3.5. Linux and fresh GPU
execution have not been validated for this release. Training and external
evaluation require the separate [environments](../environments/README.md).

See [data access](DATA_ACCESS.md) for available inputs and
[third-party notices](../THIRD_PARTY_NOTICES.md) for applicable terms.
