"""Summarize the R2 val-only free-running diagnostic (budget round).

    PYTHONPATH=src python scripts/summarize_valdiag.py

Reads results/slakh_r2_valdiag/{best,final}/eval/valid/{continuous,reset}.json
(fixed deterministic subset: sorted(valid)[:6], frozen protocol observation
item) and prints the best_val vs final/latest comparison. F1s are **pooled
micro** (recomputed from per-track tp = f1*(n_pred+n_gt)/2), matching
eval_heldout's *_agg.json convention; per-track mean (macro) is also shown
as reference. Truncation/boundary/switches are totals.
Pure offline analysis; no GPU.
"""
from __future__ import annotations

import json
import os

ROOT = "results/slakh_r2_valdiag"
TRACKS = ["Track00038", "Track00050", "Track00075", "Track00111", "Track00148", "Track00224"]
CKPTS = ("best", "final")


def load(ckpt, mode):
    p = os.path.join(ROOT, ckpt, "eval", "valid", f"{mode}.json")
    if not os.path.exists(p):
        return None
    rows = {r["track"]: r for r in json.load(open(p))}
    return rows


def micro_f1(rows, key):
    tp = sum(r[key] * (r["n_pred"] + r["n_gt"]) / 2.0 for r in rows.values())
    npred = sum(r["n_pred"] for r in rows.values())
    ngt = sum(r["n_gt"] for r in rows.values())
    return 2 * tp / max(1e-9, npred + ngt)


def pool(rows, key):
    return sum(r[key] for r in rows.values()) / max(1, len(rows))


def main():
    out = {}
    for ckpt in CKPTS:
        for mode in ("continuous", "reset"):
            rows = load(ckpt, mode)
            if rows is None:
                print(f"[missing] {ckpt}/{mode}")
                continue
            assert set(rows) == set(TRACKS), f"{ckpt}/{mode}: track set mismatch"
            stat = {
                "n_tracks": len(rows),
                "onset_f1_micro": round(micro_f1(rows, "onset_f1"), 4),
                "onset_f1_macro": round(pool(rows, "onset_f1"), 4),
                "offset_f1_micro": round(micro_f1(rows, "offset_f1"), 4),
                "inst_f1_micro": round(micro_f1(rows, "inst_f1"), 4),
                "n_pred": int(sum(r["n_pred"] for r in rows.values())),
                "n_gt": int(sum(r["n_gt"] for r in rows.values())),
                "truncated": int(sum(r.get("truncated", 0) for r in rows.values())),
                "boundary_errors": int(sum(r.get("boundary_errors", 0) for r in rows.values())),
                "inst_switches": int(sum(r.get("inst_switches", 0) for r in rows.values())),
            }
            out[f"{ckpt}_{mode}"] = stat
            print(f"[{ckpt:6s} {mode:10s}] onsetF1 {stat['onset_f1_micro']:.4f} (macro {stat['onset_f1_macro']:.4f}) "
                  f"offF1 {stat['offset_f1_micro']:.4f} instF1 {stat['inst_f1_micro']:.4f} "
                  f"pred/GT {stat['n_pred']}/{stat['n_gt']} "
                  f"trunc {stat['truncated']} bnd {stat['boundary_errors']} sw {stat['inst_switches']}")
    with open(os.path.join(ROOT, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[written] {ROOT}/summary.json")


if __name__ == "__main__":
    main()
