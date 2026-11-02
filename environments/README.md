# Environments

Use separate environments; these dependency sets are not intended to be merged.

| Purpose | Record | Verification |
|---|---|---|
| CPU cache replay | Python 3.12; `cpu/requirements.txt` | Clean Windows/Python 3.12.14 venv: seven cached modules verified across recorded runs, including restored Table 2/roles; four guard/semantic tests passed; Linux untested |
| Shared-label PRM training | `training/recorded.json` | Historical version record, not an installation lock or tested port |
| External benchmark evaluation | `../src/downstream/opencompass/requirements/` | Recovered framework declarations, not a fully locked historical environment |

The training record uses a vendor PyTorch build. Installing a generic PyPI torch
package does not reproduce that environment. A future compatible GPU environment
must be tested and described separately.

The external evaluator declares NumPy <2 and datasets <4, while CPU replay uses NumPy
2.3.5 and historical training records datasets 5.0.1. Its dependency list also contains
a mutable upstream Git branch. Pin exact resolved revisions before describing that
environment as reproducible. Selected evaluator files are not a complete installable
OpenCompass distribution; the archived framework requires its own validated setup.
