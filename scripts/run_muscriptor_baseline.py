"""MuScriptor baseline transcription (isolated venv, inference only).

    /root/autodl-tmp/venvs/muscriptor/bin/python scripts/run_muscriptor_baseline.py

Writes artifacts/listening/<track>/muscriptor.mid (+ metadata).
"""
import json
import os
import sys
import time

DATA = "/root/autodl-tmp/data/babyslakh/babyslakh_16k"
WEIGHTS = "/root/autodl-tmp/data/muscriptor_medium/model.safetensors"
TRACKS = ["Track00005", "Track00015"]

sys.path.insert(0, "/root/autodl-tmp/refs/muscriptor")
from muscriptor.transcription_model import TranscriptionModel  # noqa: E402

if __name__ == "__main__":
    m = TranscriptionModel.load_model(weights_path=WEIGHTS, device="cuda", dtype="float32")
    for tid in TRACKS:
        wav_path = os.path.join(DATA, tid, "mix.wav")
        t0 = time.time()
        midi_bytes, grid = m.transcribe_and_postprocess(
            wav_path,
            use_sampling=False,
            beam_size=1,
            prelude_forcing=True,
            no_eos_is_ok=True,
            detect_tempo=False,
        )
        dt = time.time() - t0
        art = os.path.join("artifacts", "listening", tid)
        os.makedirs(art, exist_ok=True)
        out = os.path.join(art, "muscriptor.mid")
        with open(out, "wb") as f:
            f.write(midi_bytes)
        meta = {
            "track": tid,
            "model": "MuScriptor/muscriptor-medium",
            "checkpoint": WEIGHTS,
            "git": os.popen("git -C /root/MuRWKV rev-parse HEAD").read().strip(),
            "decode_s": round(dt, 1),
            "grid": None if grid is None else {"bpm": grid.bpm},
        }
        with open(os.path.join(art, "metadata_muscriptor.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[{tid}] muscriptor baseline done in {dt:.0f}s -> {out}", flush=True)
    print("MUSCRIPTOR BASELINE DONE")