"""Paired held-out evaluation report (Level 3/4 verdicts).

    python scripts/report_paired_eval.py --exp results/slakh_r1

Reads results/<exp>/eval/test/{continuous,reset}.json (identical per-track
row schema, both arms) and reports:

  * per-track paired table: delta = continuous - reset for every metric;
  * pooled micro averages per arm (reference only);
  * track-level paired bootstrap 95% CI (10k resamples, percentile) for the
    DELTA distribution of the main metrics (onset F1, onset+offset F1,
    instrument F1, note-count ratio, instrument switches);
  * duration-quartile split of the deltas (short/med/long by track duration);
  * Level 4 verdict strictly per the run protocol:
      mean delta > 0 AND CI excludes 0  -> "positive (Level-4 evidence)"
      mean delta > 0 but CI crosses 0   -> "positive trend, not established"
      else                              -> "no evidence"
    (Level 3 is assessed separately from the pooled held-out F1 magnitude.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

MAIN_METRICS = ["onset_f1", "offset_f1", "inst_f1", "n_inst_switches", "n_pred_ratio", "rtf"]


def load_rows(exp: str, split="test"):
    d = os.path.join(exp, "eval", split)
    with open(os.path.join(d, "continuous.json")) as f:
        cont = {r["track"]: r for r in json.load(f)}
    with open(os.path.join(d, "reset.json")) as f:
        reset = {r["track"]: r for r in json.load(f)}
    assert set(cont) == set(reset), "arms must cover the identical track set (paired)"
    return cont, reset


def bootstrap_paired_ci(deltas: np.ndarray, n_boot=10000, seed=0, alpha=0.05):
    rng = np.random.RandomState(seed)
    n = len(deltas)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        means[b] = deltas[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(means.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment dir (contains eval/test/*.json)")
    ap.add_argument("--out", default="", help="output json path (default: <exp>/eval/paired_report.json)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    exp = args.exp.rstrip("/")
    cont, reset = load_rows(exp)

    tracks = sorted(cont)
    durs = np.array([cont[t]["duration_s"] for t in tracks])
    qs = np.quantile(durs, [0.25, 0.5, 0.75])

    def metric(arm_row, name):
        if name == "n_inst_switches":
            return float(arm_row.get("inst_switches", 0))
        if name == "n_pred_ratio":
            return arm_row["n_pred"] / max(1, arm_row["n_gt"])
        if name == "rtf":
            return float(arm_row.get("rtf", 0.0))
        return float(arm_row.get(name, 0.0))

    per_track = []
    for t in tracks:
        row = {"track": t, "duration_s": durs[tracks.index(t)],
               "n_gt": cont[t]["n_gt"]}
        for m in MAIN_METRICS:
            row[f"{m}_cont"] = metric(cont[t], m)
            row[f"{m}_reset"] = metric(reset[t], m)
            row[f"{m}_delta"] = row[f"{m}_cont"] - row[f"{m}_reset"]
        row["truncation_cont"] = cont[t].get("truncated", 0)
        row["truncation_reset"] = reset[t].get("truncated", 0)
        row["boundary_cont"] = cont[t].get("boundary_errors", 0)
        row["boundary_reset"] = reset[t].get("boundary_errors", 0)
        per_track.append(row)

    report = {
        "n_tracks": len(tracks),
        "duration_quartiles_s": [round(float(q), 1) for q in qs],
        "duration_bins": {"short": "<%.1fs" % qs[0], "mid": "%.1f-%.1fs" % (qs[0], qs[2]),
                          "long": ">=%.1fs" % qs[2]},
        "metrics": {},
        "verdict": {},
    }

    def bin_of(d):
        if d < qs[0]:
            return "short"
        if d >= qs[2]:
            return "long"
        return "mid"

    for m in MAIN_METRICS:
        dc = np.array([r[f"{m}_cont"] for r in per_track])
        dr = np.array([r[f"{m}_reset"] for r in per_track])
        delta = dc - dr
        lo, hi, _ = bootstrap_paired_ci(delta, args.n_boot, args.seed)
        bins = {}
        for bname in ("short", "mid", "long"):
            idx = [i for i, r in enumerate(per_track) if bin_of(r["duration_s"]) == bname]
            dsel = delta[idx]
            bins[bname] = {"n": len(idx),
                           "mean_cont": float(dc[idx].mean()) if idx else None,
                           "mean_reset": float(dr[idx].mean()) if idx else None,
                           "mean_delta": float(dsel.mean()) if len(idx) else None}
        npos = int((delta > 1e-9).sum())
        report["metrics"][m] = {
            "pooled_cont": float(dc.mean()), "pooled_reset": float(dr.mean()),
            "mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)),
            "ci95_delta": [round(lo, 4), round(hi, 4)],
            "n_tracks_delta_positive": npos,
            "duration_quartiles": bins,
        }
        if m in ("onset_f1", "offset_f1", "inst_f1"):
            if delta.mean() > 1e-9 and lo > 0:
                v = "positive (Level-4 evidence)"
            elif delta.mean() > 1e-9:
                v = "positive trend, not established"
            else:
                v = "no evidence"
            report["verdict"][m] = v
        print(f"[paired] {m}: cont {dc.mean():.4f} vs reset {dr.mean():.4f} "
              f"delta {delta.mean():+.4f} [{lo:+.4f},{hi:+.4f}] positive {npos}/{len(tracks)}")

    # protocol audits
    n_trunc = sum(r["truncation_cont"] + r["truncation_reset"] for r in per_track)
    n_bnd = sum(r["boundary_cont"] + r["boundary_reset"] for r in per_track)
    report["protocol"] = {
        "truncation_total_cont_reset": int(n_trunc),
        "boundary_errors_total_cont_reset": int(n_bnd),
        "paired": "identical track sets in both arms (asserted)",
    }
    report["verdict"]["level4"] = report["verdict"].get("onset_f1", "no evidence")
    report["per_track"] = per_track

    out = args.out or os.path.join(exp, "eval", "paired_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"[paired] wrote {out}")
    print(f"[paired] Level-4 verdict (onset F1): {report['verdict']['level4']}")


if __name__ == "__main__":
    main()