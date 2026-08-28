"""Q2 + Q1-support: note-level error taxonomy over the 60 official test tracks.

Inputs (all committed, CPU-only):
  * canonical GT recomputed from the data disk (same object the official eval used)
  * predicted notes from artifacts/listening/<track>/murwkv_{continuous,reset}.mid
    (R2 best_val test-pool outputs, provenance-verified)
  * official per-track rows from results/{slakh_r1,slakh_r2_carry}/eval/test/*.json

Outputs (results/r2_postmortem/):
  * provenance_check.csv        - per (track, mode): official vs recomputed metrics
  * error_taxonomy.csv          - long-format per (track, mode, category)
  * taxonomy_pooled.csv         - pooled per mode with rates
  * strata_errors.csv           - pooled error rates by stratum (drums, duration,
                                  density, polyphony, chunk position, register)
  * onset_deltas.csv            - matched-pair onset delta histogram (ticks)
  * offset_deltas.csv           - matched-pair offset delta histogram (ticks)
  * chunk_level.csv             - per (track, mode, chunk) counts (feeds Q3)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

DURATION_BINS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1e9)]
DENSITY_BINS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]  # GT notes/s in +/-1 s
POLY_BINS = [(1, 1), (2, 3), (4, 6), (7, 1e9)]                 # concurrent GT notes
REGISTER_BINS = [(0, 47), (48, 59), (60, 71), (72, 108)]       # low/mid/mid-high/high


def f1(tp, pred, gt):
    return 2 * tp / (pred + gt) if (pred + gt) else float("nan")


def main():
    os.makedirs(C.OUT, exist_ok=True)
    rows_c = C.r2_rows()
    rows_r = C.eval_rows(C.R2, "reset")
    tracks = sorted(rows_c)

    tax_rows: list[C.TrackTaxonomy] = []
    prov_rows = []
    chunk_rows = []
    onset_hist: Counter = Counter()
    offset_hist: Counter = Counter()
    # strata accumulators: (mode, stratum_type, stratum) -> [gt, tp, fp, miss]
    strata: defaultdict = defaultdict(lambda: [0, 0, 0, 0])
    gt_dur_hist: Counter = Counter()
    pred_dur_hist: Counter = Counter()

    def dur_bin(d: float) -> str:
        if d <= 0.0101:
            return "0.01(1 tick)"
        if d <= 0.05:
            return "(0.01,0.05]"
        if d <= 0.1:
            return "(0.05,0.1]"
        if d <= 0.25:
            return "(0.1,0.25]"
        if d <= 0.5:
            return "(0.25,0.5]"
        if d <= 1.0:
            return "(0.5,1.0]"
        return ">1.0"

    for k, tid in enumerate(tracks):
        gt = C.canonical_gt(tid)
        for mode, row in (("continuous", rows_c[tid]), ("reset", rows_r[tid])):
            pred = C.pred_notes(tid, mode)
            t = C.classify_track(tid, mode, gt, pred)
            tax_rows.append(t)
            # provenance: recompute official-style metrics from reloaded MIDIs
            m = C.match_notes(gt, pred)
            inst_ok = sum(
                1 for a, b in m.pairs
                if gt[a].program == pred[b].program and gt[a].is_drum == pred[b].is_drum
            )
            off_ok = sum(
                1 for a, b in m.pairs
                if abs(round((pred[b].offset - gt[a].offset) * C.FRAME_RATE)) <= C.OFFSET_TOL_TICKS
            )
            prov_rows.append({
                "track": tid, "mode": mode,
                "official_matched": row["n_matched"], "recomputed_matched": m.n_matched,
                "official_f1": row["onset_f1"], "recomputed_f1": f1(m.n_matched, len(pred), len(gt)),
                "official_n_pred": row["n_pred"], "recomputed_n_pred": len(pred),
                "official_n_gt": row["n_gt"], "recomputed_n_gt": len(gt),
                "official_offset_matched": row["n_offset_matched"], "recomputed_offset_matched": off_ok,
                "official_inst_matched": row["n_inst_match"], "recomputed_inst_matched": inst_ok,
            })
            # histograms
            gtp = np.array([round(n.onset * C.FRAME_RATE) for n in gt])
            prp = np.array([round(n.onset * C.FRAME_RATE) for n in pred])
            gto = np.array([round(n.offset * C.FRAME_RATE) for n in gt])
            pro = np.array([round(n.offset * C.FRAME_RATE) for n in pred])
            for a, b in m.pairs:
                onset_hist[int(prp[b] - gtp[a])] += 1
                d = int(pro[b] - gto[a])
                if abs(d) <= 500:
                    offset_hist[d] += 1
            # strata
            matched_pred_set = set(b for _, b in m.pairs)
            matched_gt_set = set(a for a, _ in m.pairs)
            gt_chunk = (gtp // 500)
            for i, n in enumerate(gt):
                def add(stype, strat, is_tp):
                    s = strata[(mode, stype, strat)]
                    s[0] += 1
                    s[1] += 1 if is_tp else 0
                    if not is_tp:
                        pass
                dur = n.offset - n.onset
                dens = float(((np.abs(gtp - gtp[i]) <= 100) & (gtp != gtp[i])).sum()) / 2.0
                poly = int((gtp == gtp[i]).sum())
                reg = n.pitch
                is_dr = n.is_drum
                pos = int(gtp[i] % 500)
                matched = i in matched_gt_set
                add("drums", "drum" if is_dr else "pitched", matched)
                for lo, hi in DURATION_BINS:
                    if lo <= dur < hi:
                        add("gt_duration", f"{lo}-{hi}" if hi < 1e8 else ">=1.0s", matched)
                        break
                for lo, hi in DENSITY_BINS:
                    if lo <= dens < hi:
                        add("gt_density_per_s", f"{lo}-{hi}" if hi < 1e8 else ">=20", matched)
                        break
                for lo, hi in POLY_BINS:
                    if lo <= poly <= (hi if hi < 1e8 else 10 ** 9):
                        add("gt_polyphony", f"{lo}-{hi}" if hi < 1e8 else ">=7", matched)
                        break
                for lo, hi in REGISTER_BINS:
                    if lo <= reg < hi:
                        add("register", f"{lo}-{hi}", matched)
                        break
                add("chunk_pos", "boundary" if (pos < 25 or pos >= 475) else "interior", matched)
                add("program_group", C.group_name(n.program, n.is_drum), matched)
                gt_dur_hist[(mode, "drum" if is_dr else "pitched", dur_bin(dur))] += 1
            for b, n in enumerate(pred):
                is_dr = n.is_drum
                s = strata[(mode, "pred_drums", "drum" if is_dr else "pitched")]
                s[0] += 1
                s[1] += 1 if b in matched_pred_set else 0
                pg = strata[(mode, "pred_program_group", C.group_name(n.program, is_dr))]
                pg[0] += 1
                pg[1] += 1 if b in matched_pred_set else 0
                pred_dur_hist[(mode, "drum" if is_dr else "pitched", dur_bin(n.offset - n.onset))] += 1
            # chunk table
            chunk_rows.extend(C.chunk_level(tid, mode, gt, pred, row))
        print(f"[{k + 1}/60] {tid} done", flush=True)

    # ---------------- write outputs ----------------
    with open(os.path.join(C.OUT, "provenance_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prov_rows[0]))
        w.writeheader()
        w.writerows(prov_rows)

    cats = [
        ("n_gt", lambda t: t.n_gt), ("n_pred", lambda t: t.n_pred),
        ("tp_onset", lambda t: t.tp),
        ("tp_offset_ok", lambda t: t.tp_offset_ok),
        ("tp_offset_short", lambda t: t.tp_offset_short),
        ("tp_offset_long", lambda t: t.tp_offset_long),
        ("miss_timing_near", lambda t: t.miss_timing_near),
        ("miss_same_pitch_far", lambda t: t.miss_same_pitch_far),
        ("miss_octave", lambda t: t.miss_octave),
        ("miss_program_swap", lambda t: t.miss_program_swap),
        ("miss_no_candidate", lambda t: t.miss_other),
        ("fp_timing", lambda t: t.fp_timing_near),
        ("fp_octave", lambda t: t.fp_octave),
        ("fp_program_swap", lambda t: t.fp_program_swap),
        ("fp_spurious_near", lambda t: t.fp_spurious_near),
        ("fp_spurious_far", lambda t: t.fp_spurious_far),
        ("gt_drum", lambda t: t.gt_drum), ("pred_drum", lambda t: t.pred_drum),
        ("tp_drum", lambda t: t.tp_drum), ("miss_drum", lambda t: t.miss_drum),
        ("fp_drum", lambda t: t.fp_drum),
        ("tp_pitched", lambda t: t.tp_pitched),
        ("tp_pitched_offset_ok", lambda t: t.tp_pitched_offset_ok),
        ("tp_pitched_offset_short", lambda t: t.tp_pitched_offset_short),
        ("tp_pitched_offset_long", lambda t: t.tp_pitched_offset_long),
        ("mean_offset_delta_ticks_pitched", lambda t: (t.offset_delta_sum_pitched / t.tp_pitched) if t.tp_pitched else np.nan),
        ("mean_duration_ratio_pitched", lambda t: (t.duration_ratio_sum_pitched / t.tp_pitched) if t.tp_pitched else np.nan),
        ("precision", lambda t: t.tp / t.n_pred if t.n_pred else np.nan),
        ("recall", lambda t: t.tp / t.n_gt if t.n_gt else np.nan),
        ("f1", lambda t: f1(t.tp, t.n_pred, t.n_gt)),
        ("mean_onset_delta_ticks", lambda t: t.onset_delta_sum / t.tp if t.tp else np.nan),
        ("mean_offset_delta_ticks", lambda t: t.offset_delta_sum / t.tp if t.tp else np.nan),
        ("mean_duration_ratio", lambda t: t.duration_ratio_sum / t.tp if t.tp else np.nan),
    ]
    with open(os.path.join(C.OUT, "error_taxonomy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "mode", "category", "value"])
        for t in tax_rows:
            for name, fn in cats:
                v = fn(t)
                w.writerow([t.track, t.mode, name, f"{v:.6g}" if isinstance(v, float) else v])

    with open(os.path.join(C.OUT, "taxonomy_pooled.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "category", "value"])
        for mode in ("continuous", "reset"):
            sel = [t for t in tax_rows if t.mode == mode]
            for name, fn in cats:
                if name in ("mean_onset_delta_ticks", "mean_offset_delta_ticks", "mean_duration_ratio",
                            "mean_offset_delta_ticks_pitched", "mean_duration_ratio_pitched"):
                    num = sum(getattr(t, {"mean_onset_delta_ticks": "onset_delta_sum",
                                          "mean_offset_delta_ticks": "offset_delta_sum",
                                          "mean_duration_ratio": "duration_ratio_sum",
                                          "mean_offset_delta_ticks_pitched": "offset_delta_sum_pitched",
                                          "mean_duration_ratio_pitched": "duration_ratio_sum_pitched"}[name]) for t in sel)
                    tp = sum(t.tp_pitched if "pitched" in name else t.tp for t in sel)
                    v = num / tp if tp else np.nan
                elif name in ("precision", "recall", "f1"):
                    tp = sum(t.tp for t in sel)
                    npred = sum(t.n_pred for t in sel)
                    ngt = sum(t.n_gt for t in sel)
                    v = {"precision": tp / npred, "recall": tp / ngt, "f1": f1(tp, npred, ngt)}[name]
                else:
                    v = sum(fn(t) for t in sel)
                w.writerow([mode, name, f"{v:.6g}" if isinstance(v, float) else v])

    with open(os.path.join(C.OUT, "strata_errors.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "stratum_type", "stratum", "n_gt", "matched", "recall", "n_pred", "precision", "f1"])
        for (mode, stype, strat), v in sorted(strata.items()):
            ngt, matched = v[0], v[1]
            if stype.startswith("pred"):
                prec = matched / ngt if ngt else np.nan
                w.writerow([mode, stype, strat, "", "", "", ngt, f"{prec:.6g}", ""])
            else:
                w.writerow([mode, stype, strat, ngt, matched,
                            f"{matched / ngt:.6g}" if ngt else "", "", "", ""])

    with open(os.path.join(C.OUT, "duration_profile.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "kind", "source", "duration_bin", "count"])
        bins = ["0.01(1 tick)", "(0.01,0.05]", "(0.05,0.1]", "(0.1,0.25]", "(0.25,0.5]", "(0.5,1.0]", ">1.0"]
        for mode in ("continuous", "reset"):
            for kind in ("pitched", "drum"):
                for src, hist in (("gt", gt_dur_hist), ("pred", pred_dur_hist)):
                    for b in bins:
                        w.writerow([mode, kind, src, b, hist.get((mode, kind, b), 0)])

    with open(os.path.join(C.OUT, "onset_deltas.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta_ticks", "count"])
        for k in sorted(onset_hist):
            w.writerow([k, onset_hist[k]])

    with open(os.path.join(C.OUT, "offset_deltas.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta_ticks", "count"])
        for k in sorted(offset_hist):
            w.writerow([k, offset_hist[k]])

    with open(os.path.join(C.OUT, "chunk_level.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(chunk_rows[0]))
        w.writeheader()
        w.writerows(chunk_rows)

    # quick console summary
    for mode in ("continuous", "reset"):
        sel = [t for t in tax_rows if t.mode == mode]
        tp = sum(t.tp for t in sel)
        npred = sum(t.n_pred for t in sel)
        ngt = sum(t.n_gt for t in sel)
        print(f"== {mode}: pooled tp {tp} / pred {npred} / gt {ngt} -> F1 {f1(tp, npred, ngt):.5f}")
        for name in ("miss_timing_near", "miss_other", "fp_timing_near", "fp_spurious_far",
                     "fp_spurious_near", "miss_octave", "miss_program_swap", "fp_octave",
                     "fp_program_swap", "tp_offset_short", "tp_offset_long"):
            print(f"   {name}: {sum(getattr(t, name) for t in sel)}")


if __name__ == "__main__":
    main()
