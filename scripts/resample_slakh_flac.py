"""Resample the Slakh2100 FLAC-redux corpus to 16 kHz mono WAV (mix only).

    python scripts/resample_slakh_flac.py \
        --src /root/autodl-tmp/data/slakh2100_flac_redux \
        --out /root/autodl-tmp/data/slakh2100_16k_from_flac

Produces: <out>/{train,validation,test}/Track*/mix.wav (+ copies all_src.mid)
Uses scipy.signal.resample_poly (no GPU needed).
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SRC_SR = 44100
DST_SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="*", default=["train", "validation", "test"])
    a = ap.parse_args()

    total_tracks = 0
    done = 0
    for sp in a.splits:
        sp_dir = os.path.join(a.src, sp)
        if not os.path.isdir(sp_dir):
            print("missing split", sp, file=sys.stderr)
            continue
        tracks = sorted(os.listdir(sp_dir))
        total_tracks += len(tracks)
        for tid in tracks:
            src_t = os.path.join(sp_dir, tid)
            dst_t = os.path.join(a.out, sp, tid)
            os.makedirs(dst_t, exist_ok=True)
            dst_wav = os.path.join(dst_t, "mix.wav")
            if os.path.exists(dst_wav):
                done += 1
                continue
            flac = os.path.join(src_t, "mix.flac")
            try:
                wav, sr = sf.read(flac, dtype="float32", always_2d=True)
            except Exception as e:
                print(f"SKIP (bad/corrupt file): {flac}: {e}", file=sys.stderr, flush=True)
                with open(os.path.join(dst_t, ".badflac"), "w") as f:
                    f.write(str(e))
                done += 1
                continue
            wav = wav.mean(axis=1)
            # 44100 -> 16000: ratio 160/441 (integer polyphase factors)
            wav16 = resample_poly(wav, up=160, down=441, axis=0)
            sf.write(dst_wav, wav16, DST_SR)
            mid_src = os.path.join(src_t, "all_src.mid")
            if os.path.exists(mid_src):
                subprocess.run(["cp", mid_src, os.path.join(dst_t, "all_src.mid")])
            done += 1
            if done % 100 == 0:
                print(f"{done}/{total_tracks}", flush=True)
    print(f"DONE {done}/{total_tracks} tracks -> {a.out}")


if __name__ == "__main__":
    main()