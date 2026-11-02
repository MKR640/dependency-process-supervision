#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

CONFIGS = [
    "gsm8k",
    "math",
    "olympiadbench",
    "omnimath",
]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def harmonic(a, b):
    if a + b <= 0:
        return 0.0
    return 2.0 * a * b / (a + b)


def metrics(data, predictions):
    err_total = err_ok = 0
    cor_total = cor_ok = 0

    for item, pred in zip(data, predictions):
        label = int(item["label"])
        ok = int(pred) == label

        if label == -1:
            cor_total += 1
            cor_ok += int(ok)
        else:
            err_total += 1
            err_ok += int(ok)

    ea = err_ok / err_total if err_total else 0.0
    ca = cor_ok / cor_total if cor_total else 0.0

    return {
        "error_acc": ea * 100,
        "correct_acc": ca * 100,
        "f1": harmonic(ea, ca) * 100,
    }


def clip(p):
    return min(max(float(p), 1e-6), 1.0 - 1e-6)


def logit(p):
    p = clip(p)
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def error_scores(item, mode, alpha=0.5):
    aux = item.get("step_aux")
    if not aux:
        raise ValueError(
            "Missing step_aux. Re-run the hierarchical model with "
            "the v2 evaluator before offline rescoring."
        )

    out = []

    for s in aux:
        p_err = float(s["p_error"])
        p_new = float(s["p_new_error"])
        p_prop = float(s["p_propagated_error"])
        denom = max(p_new + p_prop, 1e-8)
        p_new_cond = p_new / denom

        if mode == "correctness":
            q = p_err
        elif mode == "new_error":
            q = p_new
        elif mode == "joint_first_error":
            q = p_err * p_new_cond
        elif mode == "joint_first_error_raw":
            q = p_err * p_new
        elif mode == "logit_fusion":
            q = sigmoid(
                alpha * logit(p_err)
                + (1.0 - alpha) * logit(p_new_cond)
            )
        else:
            raise ValueError(mode)

        out.append(min(max(q, 0.0), 1.0))

    return out


def earliest_threshold(item, mode, threshold, alpha=0.5):
    qs = error_scores(item, mode, alpha)
    for i, q in enumerate(qs):
        if q >= threshold:
            return i
    return -1


def max_hazard(item, mode, threshold, alpha=0.5):
    qs = error_scores(item, mode, alpha)
    if not qs:
        return -1

    idx = max(range(len(qs)), key=lambda i: qs[i])

    if qs[idx] < threshold:
        return -1

    return idx


def viterbi(item, role_weight=0.5):
    """
    Structured path:
        correct* -> new_error -> propagated_error*
    or:
        correct*  (all-correct trajectory)

    Emissions fuse correctness + role heads in log space.
    """
    aux = item.get("step_aux")
    if not aux:
        raise ValueError("Missing step_aux.")

    if not aux:
        return -1

    w_role = float(role_weight)
    w_corr = 1.0 - w_role

    e_c = []
    e_n = []
    e_p = []

    for s in aux:
        p_err = clip(s["p_error"])
        p_non = clip(s["p_non_error"])
        p_rc = clip(s["p_role_correct"])
        p_new = clip(s["p_new_error"])
        p_prop = clip(s["p_propagated_error"])

        e_c.append(
            w_corr * math.log(p_non)
            + w_role * math.log(p_rc)
        )
        e_n.append(
            w_corr * math.log(p_err)
            + w_role * math.log(p_new)
        )
        e_p.append(
            w_corr * math.log(p_err)
            + w_role * math.log(p_prop)
        )

    n = len(aux)

    prefix_c = [0.0] * (n + 1)
    suffix_p = [0.0] * (n + 1)

    for i in range(n):
        prefix_c[i + 1] = prefix_c[i] + e_c[i]

    for i in range(n - 1, -1, -1):
        suffix_p[i] = suffix_p[i + 1] + e_p[i]

    best_pred = -1
    best_score = prefix_c[n]

    for k in range(n):
        score = prefix_c[k] + e_n[k] + suffix_p[k + 1]
        if score > best_score:
            best_score = score
            best_pred = k

    return best_pred


def avg_f1(config_metrics):
    return sum(m["f1"] for m in config_metrics.values()) / len(CONFIGS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument(
        "--output",
        default=None,
    )
    args = ap.parse_args()

    root = Path(args.result_dir)

    data = {
        cfg: load_jsonl(root / cfg / "all_results.jsonl")
        for cfg in CONFIGS
    }

    rows = []

    threshold_modes = [
        "correctness",
        "new_error",
        "joint_first_error",
        "joint_first_error_raw",
    ]

    # Same-benchmark exploratory sweep.
    # Do not present the selected row as a held-out estimate.
    for decoder in ["earliest", "max_hazard"]:
        for mode in threshold_modes:
            best = None

            for ti in range(1, 100):
                t = ti / 100.0
                cm = {}

                for cfg in CONFIGS:
                    if decoder == "earliest":
                        preds = [
                            earliest_threshold(x, mode, t)
                            for x in data[cfg]
                        ]
                    else:
                        preds = [
                            max_hazard(x, mode, t)
                            for x in data[cfg]
                        ]

                    cm[cfg] = metrics(
                        data[cfg],
                        preds,
                    )

                av = avg_f1(cm)

                if best is None or av > best["average_f1"]:
                    best = {
                        "decoder": decoder,
                        "mode": mode,
                        "alpha": "",
                        "threshold": t,
                        "average_f1": av,
                        **{
                            f"{cfg}_f1": cm[cfg]["f1"]
                            for cfg in CONFIGS
                        },
                    }

            rows.append(best)

    # Logit-fusion alpha grid.
    for decoder in ["earliest", "max_hazard"]:
        for ai in range(0, 11):
            alpha = ai / 10.0
            best = None

            for ti in range(1, 100):
                t = ti / 100.0
                cm = {}

                for cfg in CONFIGS:
                    fn = (
                        earliest_threshold
                        if decoder == "earliest"
                        else max_hazard
                    )

                    preds = [
                        fn(
                            x,
                            "logit_fusion",
                            t,
                            alpha,
                        )
                        for x in data[cfg]
                    ]

                    cm[cfg] = metrics(
                        data[cfg],
                        preds,
                    )

                av = avg_f1(cm)

                if best is None or av > best["average_f1"]:
                    best = {
                        "decoder": decoder,
                        "mode": "logit_fusion",
                        "alpha": alpha,
                        "threshold": t,
                        "average_f1": av,
                        **{
                            f"{cfg}_f1": cm[cfg]["f1"]
                            for cfg in CONFIGS
                        },
                    }

            rows.append(best)

    # Parameter-light structured Viterbi diagnostics.
    for rw in [0.25, 0.5, 0.75]:
        cm = {}

        for cfg in CONFIGS:
            preds = [
                viterbi(x, role_weight=rw)
                for x in data[cfg]
            ]

            cm[cfg] = metrics(
                data[cfg],
                preds,
            )

        rows.append({
            "decoder": "viterbi",
            "mode": "correct-new-propagated",
            "alpha": rw,
            "threshold": "",
            "average_f1": avg_f1(cm),
            **{
                f"{cfg}_f1": cm[cfg]["f1"]
                for cfg in CONFIGS
            },
        })

    rows.sort(
        key=lambda r: r["average_f1"],
        reverse=True,
    )

    output = (
        Path(args.output)
        if args.output
        else root / "hierarchical_decoder_comparison.csv"
    )

    fields = [
        "decoder",
        "mode",
        "alpha",
        "threshold",
        "gsm8k_f1",
        "math_f1",
        "olympiadbench_f1",
        "omnimath_f1",
        "average_f1",
    ]

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        "EXPLORATORY decoder comparison "
        "(selection performed on ProcessBench itself)"
    )
    print(
        "Do not treat the selected row as a held-out estimate."
    )
    print("=" * 118)
    print(
        f"{'decoder':12} {'mode':25} {'alpha':>7} "
        f"{'thr':>7} {'gsm8k':>8} {'math':>8} "
        f"{'olymp':>8} {'omni':>8} {'avg':>8}"
    )
    print("-" * 118)

    for r in rows[:20]:
        print(
            f"{r['decoder']:12} "
            f"{r['mode']:25} "
            f"{str(r['alpha']):>7} "
            f"{str(r['threshold']):>7} "
            f"{r['gsm8k_f1']:8.2f} "
            f"{r['math_f1']:8.2f} "
            f"{r['olympiadbench_f1']:8.2f} "
            f"{r['omnimath_f1']:8.2f} "
            f"{r['average_f1']:8.2f}"
        )

    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()
