"""Q6: data scale / coverage audit over the frozen 120/20/60 split.

Per track (all 200): GT note stats, program-group membership, tokenized
chunk stats (tokenizer.encode_song, the training representation), density,
polyphony, duration. Then train vs test comparisons.

Outputs (results/r2_postmortem/):
  * data_audit_tracks.csv      - per-track features
  * group_coverage.csv         - per program group x split: tracks containing
                                 it, notes, share
  * split_compare.csv          - train vs val vs test distribution summary
  * data_audit_*.png           - key distribution plots
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pretty_midi

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

from murwkv.tokenizer import encode_song, notes_from_pretty_midi, prepare_gt  # noqa: E402

DUR_BINS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)]


def dur_bin(d):
    for lo, hi in DUR_BINS:
        if lo <= d < hi:
            return f"{lo}-{hi}" if hi < 1e8 else ">=2.0"
    return ">=2.0"


def audit_track(tid):
    pm = pretty_midi.PrettyMIDI(C._bs().tracks[tid].midi_path)
    gt = prepare_gt(notes_from_pretty_midi(pm))
    on = np.array([n.onset for n in gt])
    off = np.array([n.offset for n in gt])
    drum = np.array([n.is_drum for n in gt])
    pitch = np.array([n.pitch for n in gt])
    dur = off - on
    span = max(1e-9, float(on.max() - on.min())) if len(on) > 1 else 1.0
    # polyphony: notes concurrent with each onset (excluding itself)
    poly = np.array([max(1, int((on == t).sum())) for t in on]) if len(on) else np.array([0])
    groups = Counter()
    for n in gt:
        groups[C.group_name(n.program, n.is_drum)] += 1
    chunks, stats = encode_song(gt, max_tokens_per_chunk=4096)
    tok_lens = [c.token_count for c in chunks]
    row = {
        "track": tid,
        "duration_s": round(float(on.max()) if len(on) else 0.0, 2),
        "n_notes": len(gt),
        "density_notes_per_s": round(len(gt) / span, 3),
        "polyphony_mean": round(float(np.mean(poly)), 3) if len(poly) else 0.0,
        "polyphony_p95": float(np.quantile(poly, 0.95)) if len(poly) else 0.0,
        "drum_share": round(float(drum.mean()), 4) if len(drum) else 0.0,
        "median_dur_s": round(float(np.median(dur)), 4) if len(dur) else 0.0,
        "mean_pitch": round(float(np.mean(pitch)), 2) if len(pitch) else 0.0,
        "n_groups": len(groups),
        "groups": ";".join(f"{g}:{c}" for g, c in groups.most_common()),
        "n_chunks": len(chunks),
        "tokens_total": int(sum(tok_lens)),
        "tokens_per_note": round(sum(tok_lens) / max(1, len(gt)), 3),
        "chunk_tokens_max": int(max(tok_lens)),
        "chunk_tokens_mean": round(float(np.mean(tok_lens)), 1),
        "empty_chunks": sum(1 for c in chunks if c.token_count <= 1),
    }
    hist = Counter(dur_bin(d) for d in dur)
    for b in ("0.0-0.1", "0.1-0.25", "0.25-0.5", "0.5-1.0", "1.0-2.0", ">=2.0"):
        row[f"durfrac_{b}"] = round(hist.get(b, 0) / max(1, len(dur)), 4)
    return row, groups


def main():
    os.makedirs(C.OUT, exist_ok=True)
    split = json.load(open(os.path.join(C.R2, "split.json")))
    all_rows = []
    group_rows = []
    for sp in ("train", "valid", "test"):
        tracks = split[sp]
        gcount = Counter()      # tracks containing group
        gnotes = Counter()      # notes per group
        gtrack_notes = Counter()
        dens = []
        for k, tid in enumerate(sorted(tracks)):
            row, groups = audit_track(tid)
            row["split"] = sp
            all_rows.append(row)
            dens.append(row["density_notes_per_s"])
            for g, c in groups.items():
                gcount[g] += 1
                gnotes[g] += c
                gtrack_notes[tid] = gtrack_notes.get(tid, 0) + c
            print(f"[{sp} {k + 1}/{len(tracks)}] {tid} notes={row['n_notes']} dens={row['density_notes_per_s']}", flush=True)
        for g in sorted(gnotes, key=lambda g: -gnotes[g]):
            group_rows.append({
                "split": sp, "group": g,
                "tracks_with_group": gcount[g], "track_share": round(gcount[g] / len(tracks), 3),
                "notes": gnotes[g], "note_share": round(gnotes[g] / sum(gnotes.values()), 4),
            })

    with open(os.path.join(C.OUT, "data_audit_tracks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    with open(os.path.join(C.OUT, "group_coverage.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "group", "tracks_with_group", "track_share", "notes", "note_share"])
        w.writeheader()
        w.writerows(group_rows)

    # split comparison
    feats = ["n_notes", "density_notes_per_s", "polyphony_mean", "drum_share",
             "median_dur_s", "n_groups", "tokens_total", "tokens_per_note", "duration_s"]
    comp = []
    for sp in ("train", "valid", "test"):
        rs = [r for r in all_rows if r["split"] == sp]
        for ft in feats:
            v = np.array([r[ft] for r in rs], dtype=float)
            comp.append({
                "split": sp, "feature": ft, "n": len(rs),
                "mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
                "p25": round(float(np.quantile(v, .25)), 4),
                "median": round(float(np.median(v)), 4),
                "p75": round(float(np.quantile(v, .75)), 4),
                "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
            })
    with open(os.path.join(C.OUT, "split_compare.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0]))
        w.writeheader()
        w.writerows(comp)

    # train vs test KS-style comparison per feature
    from scipy.stats import ks_2samp
    ks = {}
    for ft in feats:
        a = np.array([r[ft] for r in all_rows if r["split"] == "train"], dtype=float)
        b = np.array([r[ft] for r in all_rows if r["split"] == "test"], dtype=float)
        ks[ft] = {"ks_stat": round(float(ks_2samp(a, b).statistic), 4),
                  "ks_p": round(float(ks_2samp(a, b).pvalue), 5)}
    json.dump(ks, open(os.path.join(C.OUT, "train_test_ks.json"), "w"), indent=2)

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        for ax, ft, lbl in zip(axes.flat, feats[:6],
                               ["notes/track", "density (notes/s)", "polyphony (mean)",
                                "drum share", "median duration (s)", "instrument groups/track"]):
            for sp, c in (("train", "C0"), ("valid", "C1"), ("test", "C2")):
                v = np.array([r[ft] for r in all_rows if r["split"] == sp], dtype=float)
                q = np.quantile(v, np.linspace(0, 1, 51))
                ax.plot(np.linspace(0, 1, 51), q, color=c, label=sp)
            ax.set_title(lbl)
            ax.legend(fontsize=7)
        fig.suptitle("Split distributions (quantile curves)")
        fig.tight_layout()
        fig.savefig(os.path.join(C.OUT, "data_audit_quantiles.png"), dpi=140)

        # group coverage: train vs test note share
        tr = {r["group"]: r for r in group_rows if r["split"] == "train"}
        te = {r["group"]: r for r in group_rows if r["split"] == "test"}
        groups = sorted(set(tr) | set(te), key=lambda g: -(tr.get(g, te[g])["notes"]))
        top = groups[:22]
        x = np.arange(len(top))
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(x - 0.2, [tr.get(g, {}).get("note_share", 0) for g in top], width=0.4, label="train")
        ax.bar(x + 0.2, [te.get(g, {}).get("note_share", 0) for g in top], width=0.4, label="test")
        ax.set_xticks(x)
        ax.set_xticklabels(top, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("note share")
        ax.set_title("Program-group note share: train vs test")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(C.OUT, "data_audit_groups.png"), dpi=140)
    except Exception as e:
        print("plot skipped:", e)

    print("KS train vs test:", json.dumps(ks, indent=1))


if __name__ == "__main__":
    main()
