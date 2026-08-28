"""Q3: free-running error propagation from existing per-chunk artifacts.

No re-inference. Two chunk-level error definitions are available offline:

  * NOTE  (R2 only, both arms): chunk-level note F1 from the listening MIDIs
    (results/r2_postmortem/chunk_level.csv); a chunk is BAD if it has >= 20 GT
    notes and chunk F1 < 0.10.
  * TOKEN (R1 + R2, both arms): per-chunk token counts from the official eval
    rows; a chunk is DEGENERATE if it is cap-hugging (>= 95% of the 4096 cap)
    or near-empty (<= 20 tokens).

Outputs:
  * propagation_summary.json - persistence ratios, permutation-null p-values,
    hazard curves, run-length stats per (round, mode, definition)
  * hazard_curves.csv        - P(chunk bad | k consecutive bad) per arm
  * propagation_runs.png     - hazard curves + example per-track runs
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

RNG = np.random.default_rng(0)


def load_chunk_level():
    path = os.path.join(C.OUT, "chunk_level.csv")
    rows = list(csv.DictReader(open(path)))
    per = defaultdict(list)
    for r in rows:
        per[(r["track"], r["mode"])].append(r)
    return per


def bad_note_series(track_rows, min_gt=20, f1_thr=0.10, definition="zero"):
    """(bad, empty) boolean series ordered by chunk.

    definition:
      * "zero": chunk has >= min_gt GT notes and ZERO matched notes (absolute)
      * "rel":  chunk note-F1 below the track's own median chunk F1 (relative dip)
    """
    track_rows = sorted(track_rows, key=lambda r: int(r["chunk"]))
    f1s = []
    for r in track_rows:
        gt = int(r["gt"])
        f1s.append(float(r["chunk_f1"]) if (r["chunk_f1"] not in ("", "nan") and gt >= min_gt) else np.nan)
    f1s = np.array(f1s, dtype=float)
    med = np.nanmedian(f1s)
    bad, empty = [], []
    for i, r in enumerate(track_rows):
        gt, pred = int(r["gt"]), int(r["pred"])
        f1 = f1s[i]
        if definition == "zero":
            bad.append(bool(not np.isnan(f1) and f1 == 0.0))
        else:
            bad.append(bool(not np.isnan(f1) and f1 < med))
        empty.append(bool(gt >= min_gt and pred == 0))
    return np.array(bad), np.array(empty)


def bad_token_series(tokens, cap=4096):
    t = np.asarray(tokens, dtype=np.int64)
    return (t >= int(0.95 * cap)) | (t <= 20)


def transition_stats(series, n_perm=2000):
    """P(bad|prev bad) / P(bad|prev good), permutation null within track."""
    s = series.astype(bool)
    ab = s[:-1]
    ba = s[1:]
    p_bb = float((ba & ab).sum() / max(1, ab.sum()))
    p_bg = float((ba & ~ab).sum() / max(1, (~ab).sum()))
    obs = p_bb / p_bg if p_bg > 0 else float("inf")
    # permutation null: shuffle within track (destroys order, keeps marginal)
    null = []
    for _ in range(n_perm):
        sp = RNG.permutation(s)
        abp, bap = sp[:-1], sp[1:]
        q_bb = (bap & abp).sum() / max(1, abp.sum())
        q_bg = (bap & ~abp).sum() / max(1, (~bap).sum())
        null.append(q_bb / q_bg if q_bg > 0 else float("inf"))
    null = np.array(null)
    pval = float((np.sum(null >= obs) + 1) / (n_perm + 1)) if np.isfinite(obs) else 0.0
    return {"p_bad_given_bad": p_bb, "p_bad_given_good": p_bg,
            "persistence_ratio": obs, "null_mean": float(null.mean()),
            "null_p95": float(np.quantile(null, 0.95)), "perm_p_value": pval}


def hazard(series, max_k=8):
    """P(next bad | run of k bad so far)."""
    s = series.astype(bool)
    out = []
    n = len(s)
    i = 0
    runs = []
    while i < n:
        if s[i]:
            j = i
            while j < n and s[j]:
                j += 1
            runs.append((i, j - i))
            i = j
        else:
            i += 1
    for k in range(1, max_k + 1):
        starts = [st for st, L in runs if L >= k and st + k < n]
        if not starts:
            out.append({"k_consecutive_bad": k, "n_opportunities": 0, "p_next_bad": None})
            continue
        nxt = [bool(s[st + k]) for st in starts]
        out.append({"k_consecutive_bad": k, "n_opportunities": len(nxt),
                    "p_next_bad": float(np.mean(nxt))})
    # marginal + run length stats
    p_bad = float(s.mean())
    run_lens = [L for _, L in runs]
    exp_indep = 1.0 / (1 - p_bad) if p_bad < 1 else float("inf")
    return {
        "transitions": out,
        "p_bad": p_bad,
        "n_runs": len(runs),
        "mean_run_len": float(np.mean(run_lens)) if runs else 0.0,
        "max_run_len": int(np.max(run_lens)) if runs else 0,
        "expected_run_len_independence": exp_indep,
        "frac_chunks_in_runs_ge3": float(sum(L for _, L in runs if L >= 3) / max(1, len(s))),
    }


def main():
    os.makedirs(C.OUT, exist_ok=True)
    summary = {}

    # ---- R2 note-level (from taxonomy chunk table)
    per = load_chunk_level()
    tracks = sorted({t for t, _ in per})
    for definition in ("zero", "rel"):
        for mode in ("continuous", "reset"):
            bad_all, empty_all = [], []
            trans_list = []
            haz_acc = defaultdict(list)
            run_stats = []
            for tid in tracks:
                bad, empty = bad_note_series(per[(tid, mode)], definition=definition)
                if bad.sum() < 2:
                    continue
                tr = transition_stats(bad, n_perm=500)
                trans_list.append(tr)
                hz = hazard(bad)
                for e in hz["transitions"]:
                    if e["p_next_bad"] is not None:
                        haz_acc[e["k_consecutive_bad"]].append((e["n_opportunities"], e["p_next_bad"]))
                run_stats.append({k: hz[k] for k in ("p_bad", "n_runs", "mean_run_len", "max_run_len",
                                                     "expected_run_len_independence", "frac_chunks_in_runs_ge3")})
                bad_all.append(bad)
                empty_all.append(empty)
            if not trans_list:
                continue
            # pooled transitions: concatenate all series (within-track edges only)
            cat = np.concatenate(bad_all)
            pooled_trans = transition_stats(cat, n_perm=2000)
            haz = []
            for k in sorted(haz_acc):
                n = sum(x[0] for x in haz_acc[k])
                p = sum(x[0] * x[1] for x in haz_acc[k]) / n
                haz.append({"k": k, "n": n, "p_next_bad": p})
            rs = {k: float(np.mean([r[k] for r in run_stats])) for k in run_stats[0]}
            summary[f"r2_{mode}_note_{definition}"] = {
                "definition": ("chunk with >=20 GT notes and ZERO matched notes"
                               if definition == "zero" else
                               "chunk note-F1 below the track's own median chunk F1"),
                "n_tracks_with_signals": len(trans_list),
                "pooled_transitions": pooled_trans,
                "persistence_ratio_track_mean": float(np.mean([t["persistence_ratio"] for t in trans_list])),
                "persistence_ratio_track_median": float(np.median([t["persistence_ratio"] for t in trans_list])),
                "n_tracks_infinite_ratio": int(sum(1 for t in trans_list if not np.isfinite(t["persistence_ratio"]))),
                "hazard": haz,
                "run_stats_mean": rs,
                "p_bad_mean": float(np.mean([r["p_bad"] for r in run_stats])),
            }
            # empty-prediction chunks (once, from the zero-definition pass)
            if definition == "zero":
                emp = np.concatenate(empty_all)
                summary[f"r2_{mode}_note_zero"]["empty_pred_chunk_rate"] = float(emp.mean())

    # ---- token-level degenerate chunks (R1 + R2, both arms)
    for rnd, exp in (("r1", C.R1), ("r2", C.R2)):
        for mode in ("continuous", "reset"):
            rows = C.eval_rows(exp, mode)
            series = []
            for tid, row in rows.items():
                s = bad_token_series(row.get("tokens_per_chunk") or [])
                if s.sum() < 1 or len(s) < 4:
                    continue
                series.append(s)
            if not series:
                continue
            cat = np.concatenate(series)
            summary[f"{rnd}_{mode}_token"] = {
                "definition": "tokens >= 95% of 4096 cap or <= 20 tokens",
                "n_tracks": len(series),
                "rate": float(cat.mean()),
                "pooled_transitions": transition_stats(cat, n_perm=2000),
            }

    with open(os.path.join(C.OUT, "propagation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # hazard CSV + plot
    with open(os.path.join(C.OUT, "hazard_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "k_consecutive_bad", "n", "p_next_bad"])
        for arm in ("r2_continuous_note_zero", "r2_reset_note_zero"):
            if arm not in summary:
                continue
            for e in summary[arm]["hazard"]:
                w.writerow([arm, e["k"], e["n"], e["p_next_bad"]])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax = axes[0]
        for arm, lbl in (("r2_continuous_note_zero", "R2 continuous"), ("r2_reset_note_zero", "R2 reset")):
            if arm not in summary:
                continue
            ks = [e["k"] for e in summary[arm]["hazard"]]
            ps = [e["p_next_bad"] for e in summary[arm]["hazard"]]
            ax.plot(ks, ps, marker="o", label=lbl)
            base = summary[arm]["p_bad_mean"]
            ax.axhline(base, ls="--", alpha=0.4)
        ax.set_xlabel("k consecutive zero-match chunks")
        ax.set_ylabel("P(chunk k+1 bad | k bad)")
        ax.set_title("Error hazard: sticky vs independent")
        ax.legend()
        ax2 = axes[1]
        arms = [a for a in summary if "pooled_transitions" in summary[a]]
        vals = [summary[a]["pooled_transitions"]["persistence_ratio"] for a in arms]
        ax2.bar(range(len(arms)), vals, color=["C0", "C1", "C2", "C3", "C4", "C5"][:len(arms)])
        ax2.axhline(1.0, color="k", ls="--", lw=1)
        ax2.set_xticks(range(len(arms)))
        ax2.set_xticklabels([a.replace("_token", "\n(token)").replace("_note", "\n(note)") for a in arms],
                            fontsize=6, rotation=20)
        ax2.set_ylabel("persistence ratio  P(bad|bad)/P(bad|good)")
        ax2.set_title("Persistence ratio (1 = independent)")
        fig.tight_layout()
        fig.savefig(os.path.join(C.OUT, "propagation_runs.png"), dpi=140)
    except Exception as e:  # matplotlib missing should not kill the analysis
        print("plot skipped:", e)

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("rate", "p_bad_mean", "empty_pred_chunk_rate", "persistence_ratio_track_mean")} for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
