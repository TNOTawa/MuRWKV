"""Cross-round paired comparison: R1 vs R2 free-running behavior (budget round).

    python scripts/compare_r1_r2.py --exp-a results/slakh_r1 --exp-b results/slakh_r2_carry \
        [--split test] [--out results/slakh_r2_carry/eval/r1_vs_r2.json]

Reads <exp>/eval/<split>/{continuous,reset}.json for both rounds (identical
track sets asserted) and reports, with the SAME conventions as
report_paired_eval.py (per-track paired delta, 10k-resample track-level
bootstrap percentile CI, seed 0):

  * per-model pooled metrics: onset/offset/inst F1, n_pred_ratio,
    instrument switches (flicker proxy), truncation, boundary errors;
  * within-model paired delta continuous - reset (the Level-4 question);
  * cross-model paired deltas per track: R2_cont - R1_cont and
    R2_reset - R1_reset (the "did R2 alleviate the R1 collapse" question).

Pure offline analysis on saved per-track rows; no GPU, fully deterministic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

MAIN_METRICS = ["onset_f1", "offset_f1", "inst_f1", "n_inst_switches", "n_pred_ratio"]


def load_rows(exp: str, split: str):
    d = os.path.join(exp, "eval", split)
    with open(os.path.join(d, "continuous.json")) as f:
        cont = {r["track"]: r for r in json.load(f)}
    with open(os.path.join(d, "reset.json")) as f:
        reset = {r["track"]: r for r in json.load(f)}
    assert set(cont) == set(reset), f"{exp}: arms must cover the identical track set (paired)"
    return cont, reset


def bootstrap_paired_ci(deltas: np.ndarray, n_boot=10000, seed=0, alpha=0.05):
    rng = np.random.RandomState(seed)
    n = len(deltas)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        means[b] = deltas[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def metric(row, name):
    if name == "n_inst_switches":
        return float(row.get("inst_switches", 0))
    if name == "n_pred_ratio":
        return row["n_pred"] / max(1, row["n_gt"])
    return float(row.get(name, 0.0))


def arm_stats(cont, reset, tracks):
    out = {"pooled": {}, "delta_cont_minus_reset": {}}
    for m in MAIN_METRICS:
        dc = np.array([metric(cont[t], m) for t in tracks])
        dr = np.array([metric(reset[t], m) for t in tracks])
        lo, hi = bootstrap_paired_ci(dc - dr)
        out["pooled"][m] = {
            "cont": float(dc.mean()), "reset": float(dr.mean()),
            "cont_total": float(dc.sum()), "reset_total": float(dr.sum()),
        }
        out["delta_cont_minus_reset"][m] = {
            "mean": float((dc - dr).mean()),
            "median": float(np.median(dc - dr)),
            "ci95": [round(lo, 4), round(hi, 4)],
            "n_tracks_positive": int(((dc - dr) > 1e-9).sum()),
            "lower_is_better": m == "n_inst_switches",
        }
    for key, arm in (("continuous", cont), ("reset", reset)):
        out[f"totals_{key}"] = {
            "truncated_chunks": int(sum(r.get("truncated", 0) for r in arm.values())),
            "boundary_errors": int(sum(r.get("boundary_errors", 0) for r in arm.values())),
            "inst_switches": int(sum(r.get("inst_switches", 0) for r in arm.values())),
            "n_pred": int(sum(r["n_pred"] for r in arm.values())),
            "n_gt": int(sum(r["n_gt"] for r in arm.values())),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-a", required=True, help="baseline round (R1)")
    ap.add_argument("--exp-b", required=True, help="treatment round (R2)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--label-a", default="R1")
    ap.add_argument("--label-b", default="R2")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ca, ra = load_rows(args.exp_a, args.split)
    cb, rb = load_rows(args.exp_b, args.split)
    assert set(ca) == set(cb), f"track sets differ: {len(ca)} vs {len(cb)}"
    tracks = sorted(ca)
    print(f"paired tracks: {len(tracks)} ({args.label_a} vs {args.label_b}, split {args.split})")

    report = {
        "n_tracks": len(tracks),
        "split": args.split,
        "exp_a": args.exp_a, "exp_b": args.exp_b,
        "label_a": args.label_a, "label_b": args.label_b,
        "note": "deterministic paired analysis of saved per-track rows; bootstrap 10k, seed 0",
    }
    report[args.label_a] = arm_stats(ca, ra, tracks)
    report[args.label_b] = arm_stats(cb, rb, tracks)

    # cross-model paired deltas (same tracks, per track)
    report["cross_model"] = {}
    for arm_name, arm_a, arm_b in (("continuous", ca, cb), ("reset", ra, rb)):
        report["cross_model"][arm_name] = {}
        for m in MAIN_METRICS:
            da = np.array([metric(arm_a[t], m) for t in tracks])
            db = np.array([metric(arm_b[t], m) for t in tracks])
            delta = db - da
            lo, hi = bootstrap_paired_ci(delta, args.n_boot, args.seed)
            report["cross_model"][arm_name][m] = {
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "ci95": [round(lo, 4), round(hi, 4)],
                "n_tracks_positive": int((delta > 1e-9).sum()),
                "lower_is_better": m == "n_inst_switches",
            }
            print(f"[{arm_name}] {m}: {args.label_b} {db.mean():.4f} vs {args.label_a} "
                  f"{da.mean():.4f}  delta {delta.mean():+.4f} [{lo:+.4f},{hi:+.4f}] "
                  f"pos {report['cross_model'][arm_name][m]['n_tracks_positive']}/{len(tracks)}")
        ta = report[args.label_a][f"totals_{arm_name}"]
        tb = report[args.label_b][f"totals_{arm_name}"]
        print(f"[{arm_name}] totals: trunc {ta['truncated_chunks']}->{tb['truncated_chunks']} "
              f"bnd {ta['boundary_errors']}->{tb['boundary_errors']} "
              f"switches {ta['inst_switches']}->{tb['inst_switches']}")

    out = args.out or os.path.join(args.exp_b, "eval", "r1_vs_r2.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
