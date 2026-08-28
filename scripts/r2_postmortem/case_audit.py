"""Q8: fixed-rule track selection + structured audit of the listening MIDIs.

Selection is mechanical (no cherry-picking), from the official per-track rows:
  * IMPROVED_TOP3 : largest dF1 = R2 - R1 (continuous arm)
  * DEGRADED_TOP3 : most negative dF1
  * CHANGED_LEAST3: smallest |dF1| among tracks not in the two groups above

For each selected track this script produces a STRUCTURED AUDIT of the note
arrays (GT from the data disk; predictions from the committed listening
MIDIs). IMPORTANT HONESTY NOTE, also recorded in the CSV and report: this is
a scripted piano-roll/structural audit computed from note arrays -- no human
listening happened in this offline round. Every observation is derived
mechanically from the numbers.

Output: results/r2_postmortem/track_case_audit.csv
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

HONESTY = "scripted structural audit of note arrays (no human listening in this offline round)"


def note_runs(pred, max_ioi=0.35, min_run=8):
    """Longest run of consecutive same-(program,pitch) notes with IOI <= max_ioi,
    and the share of notes inside runs >= min_run (a 'stuck key' signature)."""
    runs = []
    cur = []
    last = None
    for n in sorted(pred, key=lambda x: (x.program, x.pitch, x.onset)):
        pass
    # runs must be computed in TIME order per key
    from collections import defaultdict
    by_key = defaultdict(list)
    for n in pred:
        by_key[(n.is_drum, n.program, n.pitch)].append(n.onset)
    longest, in_run = 0, 0
    for key, ons in by_key.items():
        ons = sorted(ons)
        run = 1
        for a, b in zip(ons, ons[1:]):
            if b - a <= max_ioi:
                run += 1
            else:
                longest = max(longest, run)
                if run >= min_run:
                    in_run += run
                run = 1
        longest = max(longest, run)
        if run >= min_run:
            in_run += run
    return longest, in_run / max(1, len(pred))


def coverage(pred, duration, chunk_sec=5.0):
    if not pred:
        return 0.0, float(duration)
    n = int(duration // chunk_sec) + 1
    has = np.zeros(max(1, n), dtype=bool)
    for nte in pred:
        c = min(n - 1, int(nte.onset // chunk_sec))
        has[c] = True
    # longest run of empty chunks
    longest = cur = 0
    for h in has:
        cur = 0 if h else cur + 1
        longest = max(longest, cur)
    return float(has.mean()), longest * chunk_sec


def pitch_hist_overlap(gt, pred):
    if not gt or not pred:
        return 0.0
    g = np.bincount([n.pitch for n in gt], minlength=128).astype(float)
    p = np.bincount([n.pitch for n in pred], minlength=128).astype(float)
    g /= g.sum()
    p /= p.sum()
    return float(np.minimum(g, p).sum())


def audit_track(tid):
    gt = C.canonical_gt(tid)
    r2c = C.r2_rows()[tid]
    r1 = C.r1_rows()[tid]
    duration = r2c["duration_s"]
    gt_groups = Counter()
    for n in gt:
        gt_groups[C.group_name(n.program, n.is_drum)] += 1
    out = []
    for mode in ("continuous", "reset"):
        pred = C.pred_notes(tid, mode)
        pred_groups = Counter()
        for n in pred:
            pred_groups[C.group_name(n.program, n.is_drum)] += 1
        cov, longest_gap = coverage(pred, duration)
        longest_run, stuck_share = note_runs(pred)
        dens_gt = len(gt) / max(1e-9, duration)
        dens_pred = len(pred) / max(1e-9, duration)
        missing = [g for g, _ in gt_groups.most_common(6) if pred_groups.get(g, 0) < 0.1 * gt_groups[g]]
        extra = [g for g, c in pred_groups.most_common(6) if gt_groups.get(g, 0) < 0.1 * c]
        t = C.classify_track(tid, mode, gt, pred)
        row = C.eval_rows(C.R2, mode)[tid]
        out.append({
            "track": tid, "mode": mode, "audit_kind": HONESTY,
            "duration_s": duration,
            "r1_f1": r1["onset_f1"], "r2_f1": row["onset_f1"], "d_f1": row["onset_f1"] - r1["onset_f1"],
            "n_gt": len(gt), "n_pred": len(pred), "pred_gt_ratio": round(len(pred) / max(1, len(gt)), 3),
            "f1": row["onset_f1"], "precision": row["onset_p"], "recall": row["onset_r"],
            "inst_switches": row["inst_switches"], "truncated": row["truncated"],
            "gt_groups_top": ";".join(f"{g}:{c}" for g, c in gt_groups.most_common(6)),
            "pred_groups_top": ";".join(f"{g}:{c}" for g, c in pred_groups.most_common(6)),
            "gt_groups_missing_in_pred": ";".join(missing) or "(none)",
            "pred_groups_extra_vs_gt": ";".join(extra) or "(none)",
            "gt_drum_share": round(np.mean([n.is_drum for n in gt]), 3),
            "pred_drum_share": round(np.mean([n.is_drum for n in pred]), 3) if pred else 0.0,
            "chunk_coverage_share": round(cov, 3),
            "longest_no_pred_gap_s": round(longest_gap, 1),
            "density_ratio": round(dens_pred / max(1e-9, dens_gt), 3),
            "pitch_hist_overlap": round(pitch_hist_overlap(gt, pred), 3),
            "longest_repeat_run": longest_run,
            "share_notes_in_repeat_runs": round(stuck_share, 3),
            "miss_no_candidate": t.miss_other,
            "miss_timing_near": t.miss_timing_near,
            "fp_timing_near": t.fp_timing_near,
            "mean_pitch_gt": round(float(np.mean([n.pitch for n in gt])), 1),
            "mean_pitch_pred": round(float(np.mean([n.pitch for n in pred])), 1) if pred else 0.0,
        })
    return out


def main():
    imp = list(csv.DictReader(open(os.path.join(C.OUT, "improvement_per_track.csv"))))
    for r in imp:
        r["d_f1"] = float(r["d_f1"])
    sorted_by_d = sorted(imp, key=lambda r: r["d_f1"])
    improved = [r["track"] for r in sorted_by_d[-3:]][::-1]
    degraded = [r["track"] for r in sorted_by_d[:3]]
    rest = [r for r in sorted_by_d[3:-3]]
    least = [r["track"] for r in sorted(rest, key=lambda r: abs(r["d_f1"]))[:3]]
    selection = [(t, "improved_top3") for t in improved] + [(t, "degraded_top3") for t in degraded] + [(t, "changed_least3") for t in least]
    print("selection:", selection, flush=True)

    rows = []
    for tid, group in selection:
        for r in audit_track(tid):
            r["selection_rule"] = group
            rows.append(r)
        print("audited", tid, flush=True)
    fields = ["track", "selection_rule", "mode", "audit_kind", "duration_s", "r1_f1", "r2_f1", "d_f1",
              "n_gt", "n_pred", "pred_gt_ratio", "f1", "precision", "recall", "inst_switches", "truncated",
              "gt_groups_top", "pred_groups_top", "gt_groups_missing_in_pred", "pred_groups_extra_vs_gt",
              "gt_drum_share", "pred_drum_share", "chunk_coverage_share", "longest_no_pred_gap_s",
              "density_ratio", "pitch_hist_overlap", "longest_repeat_run", "share_notes_in_repeat_runs",
              "miss_no_candidate", "miss_timing_near", "fp_timing_near", "mean_pitch_gt", "mean_pitch_pred"]
    with open(os.path.join(C.OUT, "track_case_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote track_case_audit.csv", len(rows), "rows")


if __name__ == "__main__":
    main()
