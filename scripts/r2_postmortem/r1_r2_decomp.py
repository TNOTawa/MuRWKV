"""Q1: quantitative decomposition of the R1 -> R2 continuous-arm improvement.

Uses only the official per-track rows (results/slakh_r1 and
results/slakh_r2_carry, eval/test/continuous.json). No re-inference.

Outputs:
  * improvement_per_track.csv - P/R/F1 for R1 and R2, paired deltas, Shapley
    P-effect / R-effect of the F1 delta, count recovery, hygiene columns
  * improvement_groups.csv    - mean deltas grouped by track features
    (R1 truncation severity, R1 note-ratio regime, GT duration tercile,
    GT density tercile, drum share tercile, n_gt tercile)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402


def f1_of(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def shapley_pr(p1, r1, p2, r2):
    """Attribution of dF1 to precision and recall (Shapley on the harmonic mean)."""
    base = f1_of(p1, r1)
    both = f1_of(p2, r2)
    p_eff = 0.5 * ((f1_of(p2, r1) - base) + (both - f1_of(p1, r2)))
    r_eff = 0.5 * ((f1_of(p1, r2) - base) + (both - f1_of(p2, r1)))
    return p_eff, r_eff, both - base


def gt_features(tid):
    gt = C.canonical_gt(tid)
    arr_on = np.array([n.onset for n in gt])
    dur = np.array([n.offset - n.onset for n in gt])
    drum = np.array([n.is_drum for n in gt])
    dens = len(gt) / max(1e-9, arr_on.max() - arr_on.min()) if len(gt) > 1 else 0.0
    poly = np.mean([(arr_on == t).sum() for t in np.unique(arr_on)]) if len(gt) else 0.0
    row = C.r2_rows()[tid]
    return {
        "duration_s": row["duration_s"],
        "n_gt": len(gt),
        "gt_density_per_s": dens,
        "gt_polyphony_mean": poly,
        "gt_drum_share": drum.mean(),
        "gt_median_duration_s": float(np.median(dur)),
    }


def tercile_labels(values):
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    return ["low" if v < q1 else ("mid" if v < q2 else "high") for v in values]


def main():
    r1, r2 = C.r1_rows(), C.r2_rows()
    tracks = sorted(r2)
    assert set(tracks) == set(r1)

    # per-track table
    rows = []
    feats = {}
    for tid in tracks:
        a, b = r1[tid], r2[tid]
        pe, re_, df1 = shapley_pr(a["onset_p"], a["onset_r"], b["onset_p"], b["onset_r"])
        f = gt_features(tid)
        feats[tid] = f
        rows.append({
            "track": tid,
            "r1_f1": a["onset_f1"], "r2_f1": b["onset_f1"], "d_f1": b["onset_f1"] - a["onset_f1"],
            "r1_p": a["onset_p"], "r2_p": b["onset_p"],
            "r1_r": a["onset_r"], "r2_r": b["onset_r"],
            "p_effect": pe, "r_effect": re_,
            "r1_offset_f1": a["offset_f1"], "r2_offset_f1": b["offset_f1"],
            "r1_n_pred": a["n_pred"], "r2_n_pred": b["n_pred"], "n_gt": b["n_gt"],
            "r1_ratio": a["n_pred"] / b["n_gt"], "r2_ratio": b["n_pred"] / b["n_gt"],
            "r1_trunc": a["truncated"], "r2_trunc": b["truncated"],
            "r1_bnd": a["boundary_errors"], "r2_bnd": b["boundary_errors"],
            "r1_switches": a["inst_switches"], "r2_switches": b["inst_switches"],
            "d_switches": b["inst_switches"] - a["inst_switches"],
            "r1_inst_f1": a["inst_f1"], "r2_inst_f1": b["inst_f1"],
            **f,
        })
    # pooled Shapley (micro P/R)
    tp1 = sum(r["n_matched"] for r in r1.values()); tp2 = sum(r["n_matched"] for r in r2.values())
    np1 = sum(r["n_pred"] for r in r1.values()); np2 = sum(r["n_pred"] for r in r2.values())
    ng = sum(r["n_gt"] for r in r1.values())
    P1, R1_, P2, R2_ = tp1 / np1, tp1 / ng, tp2 / np2, tp2 / ng
    pe, re_, df1 = shapley_pr(P1, R1_, P2, R2_)
    pooled = {
        "P1": P1, "P2": P2, "R1": R1_, "R2": R2_,
        "f1_1": f1_of(P1, R1_), "f1_2": f1_of(P2, R2_), "d_f1": df1,
        "p_effect": pe, "r_effect": re_,
        "n_pred_1": np1, "n_pred_2": np2, "n_gt": ng,
        "trunc_1": sum(r["truncated"] for r in r1.values()), "trunc_2": sum(r["truncated"] for r in r2.values()),
        "bnd_1": sum(r["boundary_errors"] for r in r1.values()), "bnd_2": sum(r["boundary_errors"] for r in r2.values()),
        "sw_1": sum(r["inst_switches"] for r in r1.values()), "sw_2": sum(r["inst_switches"] for r in r2.values()),
        "inst_match_1": sum(r["n_inst_match"] for r in r1.values()), "inst_match_2": sum(r["n_inst_match"] for r in r2.values()),
        "matched_1": tp1, "matched_2": tp2,
    }
    print("pooled:", json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in pooled.items()}, indent=1))

    os.makedirs(C.OUT, exist_ok=True)
    with open(os.path.join(C.OUT, "improvement_per_track.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # grouping by features
    def group_stats(keyfn, name):
        buckets = defaultdict(list)
        for r in rows:
            buckets[keyfn(r)].append(r)
        out = []
        for g, rs in sorted(buckets.items()):
            out.append({
                "grouping": name, "group": g, "n_tracks": len(rs),
                "mean_r1_f1": float(np.mean([r["r1_f1"] for r in rs])),
                "mean_r2_f1": float(np.mean([r["r2_f1"] for r in rs])),
                "mean_d_f1": float(np.mean([r["d_f1"] for r in rs])),
                "median_d_f1": float(np.median([r["d_f1"] for r in rs])),
                "improved": sum(1 for r in rs if r["d_f1"] > 0),
                "mean_d_switches": float(np.mean([r["d_switches"] for r in rs])),
                "mean_r1_trunc": float(np.mean([r["r1_trunc"] for r in rs])),
                "mean_r2_trunc": float(np.mean([r["r2_trunc"] for r in rs])),
                "mean_r1_ratio": float(np.mean([r["r1_ratio"] for r in rs])),
                "mean_r2_ratio": float(np.mean([r["r2_ratio"] for r in rs])),
            })
        return out

    groups = []
    groups += group_stats(lambda r: "r1_trunc_0" if r["r1_trunc"] == 0 else ("r1_trunc_1_10" if r["r1_trunc"] <= 10 else "r1_trunc_gt10"), "r1_truncation")
    groups += group_stats(lambda r: "ratio<=1.5" if r["r1_ratio"] <= 1.5 else "ratio>1.5", "r1_overproduction")
    for feat in ("duration_s", "n_gt", "gt_density_per_s", "gt_polyphony_mean", "gt_drum_share", "gt_median_duration_s"):
        labels = tercile_labels([r[feat] for r in rows])
        for r, lab in zip(rows, labels):
            r[f"_b_{feat}"] = lab
        groups += group_stats(lambda r: r[f"_b_{feat}"], f"tercile_{feat}")
    # instrument flicker vs correctness
    d_sw = np.array([r["d_switches"] for r in rows])
    d_if1 = np.array([r["r2_inst_f1"] - r["r1_inst_f1"] for r in rows])
    d_f1 = np.array([r["d_f1"] for r in rows])
    from scipy.stats import spearmanr
    flicker = {
        "spearman_dSwitches_vs_dInstF1": float(spearmanr(-d_sw, d_if1).statistic),
        "spearman_dSwitches_vs_dF1": float(spearmanr(-d_sw, d_f1).statistic),
        "tracks_switches_down": int((d_sw < 0).sum()),
        "tracks_switches_down_and_instF1_up": int(((d_sw < 0) & (d_if1 > 0)).sum()),
        "tracks_switches_down_and_F1_down": int(((d_sw < 0) & (d_f1 < 0)).sum()),
        "inst_f1_equals_onset_f1_r2": int(sum(1 for r in rows if abs(r["r2_inst_f1"] - r["r2_f1"]) < 1e-9)),
    }
    with open(os.path.join(C.OUT, "improvement_groups.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(groups[0]))
        w.writeheader()
        w.writerows(groups)
    with open(os.path.join(C.OUT, "improvement_pooled.json"), "w") as f:
        json.dump({"pooled": pooled, "flicker": flicker}, f, indent=2)
    print("flicker:", json.dumps(flicker, indent=1))


if __name__ == "__main__":
    main()
