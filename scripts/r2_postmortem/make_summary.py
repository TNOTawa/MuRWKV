"""Assemble results/r2_postmortem/summary.json from all postmortem outputs.

Run AFTER: taxonomy.py, r1_r2_decomp.py, propagation.py, data_audit.py,
case_audit.py, token_probe.py, ckpt_proxy.py, make_figures.py.
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

F1 = lambda tp, p, g: 2 * tp / (p + g) if (p + g) else None


def load_json(name):
    path = os.path.join(C.OUT, name)
    return json.load(open(path)) if os.path.exists(path) else None


def load_csv(name):
    path = os.path.join(C.OUT, name)
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def pooled_taxonomy():
    rows = defaultdict(dict)
    for r in load_csv("error_taxonomy.csv"):
        rows[(r["track"], r["mode"])][r["category"]] = float(r["value"])
    out = {}
    tracks = sorted(C.r2_rows())
    for mode in ("continuous", "reset"):
        sel = [rows[(t, mode)] for t in tracks]
        tp = sum(r["tp_onset"] for r in sel)
        npred = sum(r["n_pred"] for r in sel)
        ngt = sum(r["n_gt"] for r in sel)
        tp_p = sum(r["tp_pitched"] for r in sel)
        gt_p = sum(r["n_gt"] - r["gt_drum"] for r in sel)
        pr_p = sum(r["n_pred"] - r["pred_drum"] for r in sel)
        tp_d = sum(r["tp_drum"] for r in sel)
        gt_d = sum(r["gt_drum"] for r in sel)
        pr_d = sum(r["pred_drum"] for r in sel)
        out[mode] = {
            "note": "tick-exact matching (|dt|<=5 ticks); higher than the official float-boundary numbers, see provenance_check.csv",
            "n_gt": ngt, "n_pred": npred, "matched": tp,
            "precision": tp / npred, "recall": tp / ngt, "f1": F1(tp, npred, ngt),
            "miss": {
                "no_candidate_content_absent": sum(r["miss_no_candidate"] for r in sel),
                "right_key_wrong_timing_near": sum(r["miss_timing_near"] for r in sel),
                "same_key_far": sum(r["miss_same_pitch_far"] for r in sel),
                "octave": sum(r["miss_octave"] for r in sel),
                "program_swap": sum(r["miss_program_swap"] for r in sel),
            },
            "fp": {
                "right_key_wrong_timing": sum(r["fp_timing"] for r in sel),
                "wrong_pitch_near_onset": sum(r["fp_spurious_near"] for r in sel),
                "octave": sum(r["fp_octave"] for r in sel),
                "program_swap": sum(r["fp_program_swap"] for r in sel),
                "nothing_near": sum(r["fp_spurious_far"] for r in sel),
            },
            "drums": {"gt": gt_d, "pred": pr_d, "tp": tp_d,
                      "recall": tp_d / gt_d, "precision": tp_d / pr_d},
            "pitched": {"gt": gt_p, "pred": pr_p, "tp": tp_p,
                        "recall": tp_p / gt_p, "precision": tp_p / pr_p},
            "pitched_matched_offset": {
                "ok": sum(r["tp_pitched_offset_ok"] for r in sel),
                "too_short": sum(r["tp_pitched_offset_short"] for r in sel),
                "too_long": sum(r["tp_pitched_offset_long"] for r in sel),
            },
        }
    return out


def main():
    s = {}
    s["meta"] = {
        "round": "R2 postmortem (offline, CPU-only)",
        "question": "why did R2 eliminate the R1 free-running collapse yet keep very low held-out F1?",
        "inputs": [
            "results/slakh_r1/eval/test/*.json (official R1 per-track rows)",
            "results/slakh_r2_carry/eval/test/*.json (official R2 per-track rows)",
            "artifacts/listening/<track>/*.mid (R2 best_val note-level predictions, provenance-verified)",
            "results/slakh_r2_valdiag/ (free-running 4000-vs-5000 on 6 fixed val tracks)",
            "results/slakh_r{1,2}_carry/metrics.{csv,json} (training logs)",
            "results/slakh_r2_carry/split.json (frozen 120/20/60 manifest)",
        ],
        "not_modified": "no R1/R2 original result files were touched; all outputs are new under results/r2_postmortem/",
        "honesty_notes": [
            "no human listening occurred; the case audit is a scripted structural audit of note arrays",
            "matching recomputation uses integer ticks; the official float comparison drops pairs sitting exactly at the 50 ms boundary (quantified in provenance_check.csv)",
        ],
    }

    # ---- official headline (untouched)
    r1c, r2c = C.r1_rows(), C.r2_rows()
    tp1 = sum(r["n_matched"] for r in r1c.values()); tp2 = sum(r["n_matched"] for r in r2c.values())
    np1 = sum(r["n_pred"] for r in r1c.values()); np2 = sum(r["n_pred"] for r in r2c.values())
    ng = sum(r["n_gt"] for r in r1c.values())
    s["official_offline_recomputed"] = {
        "r1_continuous": {"pooled_f1": F1(tp1, np1, ng), "p": tp1 / np1, "r": tp1 / ng,
                          "n_pred": np1, "matched": tp1,
                          "truncated": sum(r["truncated"] for r in r1c.values()),
                          "boundary_errors": sum(r["boundary_errors"] for r in r1c.values())},
        "r2_continuous": {"pooled_f1": F1(tp2, np2, ng), "p": tp2 / np2, "r": tp2 / ng,
                          "n_pred": np2, "matched": tp2,
                          "truncated": sum(r["truncated"] for r in r2c.values()),
                          "boundary_errors": sum(r["boundary_errors"] for r in r2c.values())},
        "shapley": load_json("improvement_pooled.json")["pooled"],
        "flicker": load_json("improvement_pooled.json")["flicker"],
    }

    # ---- taxonomy
    s["error_taxonomy"] = pooled_taxonomy()

    # ---- provenance
    prov = load_csv("provenance_check.csv")
    om = np.array([int(r["official_matched"]) for r in prov])
    rm = np.array([int(r["recomputed_matched"]) for r in prov])
    s["provenance"] = {
        "rows_verified": len(prov),
        "official_matched_total": int(om.sum()),
        "tick_matched_total": int(rm.sum()),
        "ratio": float(rm.sum() / om.sum()),
        "explanation": "canonical GT and decoder output live on the 10 ms grid; 14% of matched pairs sit at exactly |dt|=5 ticks where float subtraction gives cost slightly > 0.05 and the official path drops them",
        "deltas_onset": {str(k): v for k, v in
                         (lambda h: h)(defaultdict(int, {k: sum(int(r["count"]) for r in load_csv("onset_deltas.csv") if abs(int(r["delta_ticks"])) == k) for k in (0, 5)})).items()},
    }

    # ---- propagation
    s["error_propagation"] = load_json("propagation_summary.json")

    # ---- data audit
    ks = load_json("train_test_ks.json")
    tracks = load_csv("data_audit_tracks.csv")
    split_stats = {}
    for sp in ("train", "valid", "test"):
        rs = [r for r in tracks if r["split"] == sp]
        split_stats[sp] = {
            "n": len(rs),
            "median_notes_per_track": float(np.median([float(r["n_notes"]) for r in rs])),
            "median_density_notes_per_s": float(np.median([float(r["density_notes_per_s"]) for r in rs])),
            "median_drum_share": float(np.median([float(r["drum_share"]) for r in rs])),
            "median_polyphony": float(np.median([float(r["polyphony_mean"]) for r in rs])),
            "median_groups_per_track": float(np.median([float(r["n_groups"]) for r in rs])),
            "total_tokens": int(sum(int(r["tokens_total"]) for r in rs)),
        }
    s["data_audit"] = {"split_stats": split_stats,
                       "train_vs_test_ks": ks,
                       "ks_note": "all complexity features match across splits; only duration_s differs (test tracks are longer by corpus design, KS p=0.0009)"}

    # ---- case audit
    cases = load_csv("track_case_audit.csv")
    s["case_audit"] = {
        "kind": cases[0]["audit_kind"] if cases else "missing",
        "selection": sorted({(r["track"], r["selection_rule"]) for r in cases}),
        "csv": "results/r2_postmortem/track_case_audit.csv",
    }

    # ---- token probe
    s["token_probe"] = load_json("token_probe_summary.json")
    s["cascade"] = load_json("cascade_summary.json")

    # ---- ckpt proxy
    s["ckpt_proxy"] = load_json("ckpt_proxy.json")

    # ---- four verdict sentences live in the report; mirror one-line verdicts here
    s["verdicts"] = {
        "fixed_catastrophic_free_running_collapse": "SUPPORTED",
        "improved_held_out_transcription": "SUPPORTED (continuous arm; reset slightly worse)",
        "proved_persistent_state_itself_beneficial": "NOT SUPPORTED (A+B confounded, no ablation)",
        "near_practical_AMT": "NOT SUPPORTED (pitched-note recall ~0.7-2.4%; ~97% of GT notes unmatched)",
    }
    json.dump(s, open(os.path.join(C.OUT, "summary.json"), "w"), indent=2)
    print("summary.json written")
    print(json.dumps(s["verdicts"], indent=1))


if __name__ == "__main__":
    main()
