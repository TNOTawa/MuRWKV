"""Postmortem figures (only decision-relevant ones).

Outputs in results/r2_postmortem/:
  * taxonomy_bottleneck.png - where the remaining errors live
  * improvement.png         - R1->R2 per-track deltas and their P/R attribution
  * timing_duration.png     - onset delta histogram + pred/GT duration profile
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_taxonomy():
    rows = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(C.OUT, "error_taxonomy.csv"))):
        rows[(r["track"], r["mode"])][r["category"]] = float(r["value"])
    return rows


def pooled(rows_by_track_mode, mode, cat):
    if cat in ("precision", "recall", "f1"):
        tp = sum(rows_by_track_mode[(t, mode)]["tp_onset"] for t in C.r2_rows())
        npred = sum(rows_by_track_mode[(t, mode)]["n_pred"] for t in C.r2_rows())
        ngt = sum(rows_by_track_mode[(t, mode)]["n_gt"] for t in C.r2_rows())
        return {"precision": tp / npred, "recall": tp / ngt,
                "f1": 2 * tp / (npred + ngt)}[cat]
    return sum(rows_by_track_mode[(t, mode)][cat] for t in C.r2_rows())


def taxonomy_figure():
    tax = load_taxonomy()
    tracks = sorted(C.r2_rows())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) GT miss composition
    ax = axes[0]
    cats = [("no plausible pred\n(content absent)", "miss_no_candidate"),
            ("right key\nwrong timing", "miss_timing_near"),
            ("same key\nfar away", "miss_same_pitch_far"),
            ("octave", "miss_octave"), ("program swap", "miss_program_swap")]
    x = np.arange(len(cats))
    for k, (mode, off, lbl) in enumerate((("continuous", 0, "R2 continuous"), ("reset", 0.32, "R2 reset"))):
        tot = pooled(tax, mode, "n_gt")
        vals = [100 * pooled(tax, mode, c) / tot for _, c in cats]
        ax.bar(x + off - 0.16, vals, width=0.32, label=lbl)
        for xi, v in zip(x + off - 0.16, vals):
            ax.text(xi, v + 0.4, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cats], fontsize=8)
    ax.set_ylabel("% of GT notes")
    ax.set_title("GT notes: why are they missed?")
    ax.legend(fontsize=8)

    # (b) FP composition
    ax = axes[1]
    cats = [("right key\nwrong timing", "fp_timing"),
            ("wrong pitch\nnear GT onset", "fp_spurious_near"),
            ("octave", "fp_octave"), ("program swap", "fp_program_swap"),
            ("nothing near\n(hallucination)", "fp_spurious_far")]
    x = np.arange(len(cats))
    for k, (mode, off, lbl) in enumerate((("continuous", 0, "R2 continuous"), ("reset", 0.32, "R2 reset"))):
        tot = pooled(tax, mode, "n_pred")
        vals = [100 * pooled(tax, mode, c) / tot for _, c in cats]
        ax.bar(x + off - 0.16, vals, width=0.32, label=lbl)
        for xi, v in zip(x + off - 0.16, vals):
            ax.text(xi, v + 0.4, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cats], fontsize=8)
    ax.set_ylabel("% of predicted notes")
    ax.set_title("Predicted notes: what kind of wrong?")
    ax.legend(fontsize=8)

    # (c) drums vs pitched recall/precision
    ax = axes[2]
    labels = ["drum recall", "pitched recall", "drum precision", "pitched precision"]
    vals = []
    strata = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(C.OUT, "strata_errors.csv"))):
        strata[(r["mode"], r["stratum_type"], r["stratum"])] = r
    for mode in ("continuous", "reset"):
        vals.append([
            float(strata[(mode, "drums", "drum")]["recall"]),
            float(strata[(mode, "drums", "pitched")]["recall"]),
            float(strata[(mode, "pred_drums", "drum")]["precision"]),
            float(strata[(mode, "pred_drums", "pitched")]["precision"]),
        ])
    x = np.arange(4)
    ax.bar(x - 0.16, vals[0], width=0.32, label="R2 continuous")
    ax.bar(x + 0.16, vals[1], width=0.32, label="R2 reset")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("rate")
    ax.set_title("Drums vs pitched instruments")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT, "taxonomy_bottleneck.png"), dpi=140)
    plt.close(fig)


def improvement_figure():
    imp = list(csv.DictReader(open(os.path.join(C.OUT, "improvement_per_track.csv"))))
    d = np.array([float(r["d_f1"]) for r in imp])
    order = np.argsort(d)
    pj = json.load(open(os.path.join(C.OUT, "improvement_pooled.json")))
    p = pj["pooled"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    colors = ["#c44" if d[i] < 0 else "#484" for i in order]
    ax.bar(range(len(d)), d[order], color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("tracks sorted by delta")
    ax.set_ylabel("dF1 (R2 - R1, continuous)")
    ax.set_title(f"Per-track improvement (pooled +{p['d_f1']:.4f});\n"
                 f"P-effect +{p['p_effect']:.4f}, R-effect +{p['r_effect']:.4f}")
    ax = axes[1]
    groups = list(csv.DictReader(open(os.path.join(C.OUT, "improvement_groups.csv"))))
    g = [r for r in groups if r["grouping"] == "r1_truncation"]
    x = np.arange(len(g))
    ax.bar(x - 0.16, [float(r["mean_r1_f1"]) for r in g], width=0.32, label="R1")
    ax.bar(x + 0.16, [float(r["mean_r2_f1"]) for r in g], width=0.32, label="R2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['group']}\nn={r['n_tracks']}" for r in g], fontsize=8)
    ax.set_ylabel("mean onset F1")
    ax.set_title("Mean F1 by R1 collapse severity (R1 truncations)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT, "improvement.png"), dpi=140)
    plt.close(fig)


def timing_figure():
    on = list(csv.DictReader(open(os.path.join(C.OUT, "onset_deltas.csv"))))
    ks = np.array([int(r["delta_ticks"]) for r in on])
    vs = np.array([int(r["count"]) for r in on])
    dur = list(csv.DictReader(open(os.path.join(C.OUT, "duration_profile.csv"))))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.bar(ks, vs, width=0.9)
    ax.set_xlabel("matched-pair onset delta (ticks, 10 ms)")
    ax.set_ylabel("notes")
    ax.set_title("Onset timing of matched notes (both arms pooled)\n"
                 "note the mass exactly at the 5-tick (50 ms) tolerance edge")
    bins = ["0.01(1 tick)", "(0.01,0.05]", "(0.05,0.1]", "(0.1,0.25]", "(0.25,0.5]", "(0.5,1.0]", ">1.0"]
    ax = axes[1]
    x = np.arange(len(bins))
    for k, (src, off, lbl) in enumerate((("gt", 0, "GT"), ("pred", 0.32, "predicted"))):
        sel = [r for r in dur if r["mode"] == "continuous" and r["kind"] == "pitched" and r["source"] == src]
        m = {r["duration_bin"]: int(r["count"]) for r in sel}
        tot = sum(m.values())
        vals = [100 * m.get(b, 0) / max(1, tot) for b in bins]
        ax.bar(x + off - 0.16, vals, width=0.32, label=lbl)
        for xi, v in zip(x + off - 0.16, vals):
            ax.text(xi, v + 0.5, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=7, rotation=30)
    ax.set_ylabel("% of pitched notes")
    ax.set_title("Pitched-note duration profile: GT vs predicted (continuous)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT, "timing_duration.png"), dpi=140)
    plt.close(fig)


def main():
    taxonomy_figure()
    improvement_figure()
    timing_figure()
    print("figures written")


if __name__ == "__main__":
    main()
