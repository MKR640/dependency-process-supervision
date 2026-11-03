# Numeric input schema

Each array's first dimension is the number of steps: 13,930 validation steps
and 9,073 test steps. Rows are ordered by `trace`, then `step`. Probabilities
are stored as float64 without rounding; targets/group codes use integer arrays.

| Field | Split | Shape / meaning |
|---|---|---|
| `trace` | Both | `(N,)` anonymous trajectory code; disjoint across splits |
| `step` | Both | `(N,)` zero-based position; contiguous within each trajectory |
| `correctness` | Both | `(N,)` 0 = incorrect, 1 = correct |
| `role` | Both | `(N,)` 0 = correct, 1 = new error, 2 = propagated error |
| `M0_p_error`, `M1_p_error`, `M3_p_error`, `M4_p_error`, `M5_p_error` | Both | `(N,)` error probabilities for each condition |
| `detection_group` | Test | `(N,)` anonymous normalized-question group, preserving the detection bootstrap's ordering |
| `role_group` | Test | `(N,)` anonymous problem group, preserving the role bootstrap's ordering |
| `type_target` | Test | `(N,)` class 0–7 in `protocol.json` order; -100 means excluded from type evaluation |
| `M4_role_probabilities`, `M5_role_probabilities` | Test | `(N, 3)` correct / new / propagated probabilities |
| `M3_type_probabilities`, `M5_type_probabilities` | Test | `(N, 8)` intrinsic-type probabilities |

No reverse map from integer codes to original identifiers is included. Separate
group fields preserve the two historical evaluation conventions. All model
probabilities are aligned to the same reference rows before export. A mixed
error is included in the new-error role; recovery is included in correct.

For a quick inspection:

```python
import numpy as np
with np.load("data/test.npz", allow_pickle=False) as data:
    for key in data.files:
        print(key, data[key].shape, data[key].dtype)
```
