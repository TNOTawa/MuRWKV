"""Held-out evaluation (Gate 6): continuous vs reset on split tracks.

    python -m murwkv.eval.eval_heldout --exp results/<exp> --ckpt results/<exp>/final.pt \
        --split valid --mode both --out results/<exp>/eval/

Writes per-track metrics JSON (+ aggregated), listening MIDI files under
artifacts/listening/<track>/ and a metrics.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch

from ..data.babyslakh import BabySlakh
from ..model.murwkv_model import MuRWKVConfig, count_params
from ..model.rwkv7 import KERNEL_AVAILABLE
from ..tokenizer import notes_from_pretty_midi, program_rep
from .infer import Transcriber
from .metrics import aggregate, evaluate_track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="/root/autodl-tmp/data/babyslakh/babyslakh_16k")
    ap.add_argument("--split", default="valid", choices=["valid", "test", "train"])
    ap.add_argument("--mode", default="both", choices=["continuous", "reset", "both"])
    ap.add_argument("--max-tokens", type=int, default=600)
    args = ap.parse_args()

    bs = BabySlakh(args.data_root)
    split_path = os.path.join(args.exp, "split.json")
    splits = json.load(open(split_path))
    track_ids = splits[args.split]
    print("tracks:", track_ids)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_args = ckpt.get("args", {}) or {}
    cfg = MuRWKVConfig(n_layer=cfg_args.get("n_layer", 6), n_embd=cfg_args.get("n_embd", 512))
    from ..model.murwkv_model import MuRWKV

    model = MuRWKV(cfg).cuda().bfloat16()
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {count_params(model)/1e6:.2f}M params from {args.ckpt}")

    tr = Transcriber(model, device="cuda", max_tokens_per_chunk=args.max_tokens)
    out_dir = os.path.join(args.exp, "eval", args.split)
    os.makedirs(out_dir, exist_ok=True)
    all_rows = {"continuous": [], "reset": []}
    for tid in track_ids:
        wav, sr = sf.read(bs.tracks[tid].mix_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            raise ValueError(f"{tid} sr {sr}")
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        gt_notes = [program_rep(n) for n in notes_from_pretty_midi(__import__("pretty_midi").PrettyMIDI(bs.tracks[tid].midi_path))]
        dur = len(wav) / sr
        for mode in ("continuous", "reset"):
            if args.mode != "both" and args.mode != mode:
                continue
            t0 = time.time()
            tokens, notes, stats = tr.transcribe_wav(wav_t, mode=mode)
            dt = time.time() - t0
            m = evaluate_track(tid, gt_notes, notes, duration_s=dur, extra=stats)
            m.tokens_per_chunk = [len(t) for t in tokens]
            row = {
                "track": tid, "mode": mode,
                "n_gt": m.n_gt, "n_pred": m.n_pred,
                "onset_p": m.onset_p, "onset_r": m.onset_r, "onset_f1": m.onset_f1,
                "offset_f1": m.offset_f1, "inst_f1": m.inst_f1,
                "boundary_errors": m.boundary_errors, "truncated": m.truncated_chunks,
                "tokens_per_chunk": m.tokens_per_chunk, "decode_s": round(dt, 1),
            }
            all_rows[mode].append(row)
            print(f"[{tid} {mode}] notes {m.n_gt}->{m.n_pred} onsetF1 {m.onset_f1:.3f} offF1 {m.offset_f1:.3f} instF1 {m.inst_f1:.3f} bnd {m.boundary_errors} trunc {m.truncated_chunks} {dt:.0f}s")
            # listening artifacts
            art = os.path.join("artifacts", "listening", tid)
            os.makedirs(art, exist_ok=True)
            from ..tokenizer import tokens_to_midi

            tokens_to_midi(gt_notes, os.path.join(art, "gt.mid"))
            tokens_to_midi(notes, os.path.join(art, f"murwkv_{mode}.mid"))
            meta = {
                "track": tid, "checkpoint": args.ckpt, "mode": mode,
                "git": os.popen("git rev-parse HEAD").read().strip(),
                "metrics": row,
            }
            with open(os.path.join(art, f"metadata_{mode}.json"), "w") as f:
                json.dump(meta, f, indent=2)
    # write per-mode rows + aggregates
    for mode, rows in all_rows.items():
        if not rows:
            continue
        with open(os.path.join(out_dir, f"{mode}.json"), "w") as f:
            json.dump(rows, f, indent=2)
        ms = []
        for r in rows:
            from .metrics import TrackMetrics

            m = TrackMetrics(track=r["track"])
            for k, v in r.items():
                if hasattr(m, k):
                    setattr(m, k, v)
            ms.append(m)
        agg = aggregate(ms)
        with open(os.path.join(out_dir, f"{mode}_agg.json"), "w") as f:
            json.dump(agg, f, indent=2)
        print(f"[{mode}] AGG:", {k: round(v, 4) if isinstance(v, float) else v for k, v in agg.items()})


if __name__ == "__main__":
    main()