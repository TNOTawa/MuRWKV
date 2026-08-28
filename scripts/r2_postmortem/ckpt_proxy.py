"""Q4: post-hoc checkpoint-selection proxy, computed entirely on CPU.

The R2 selection criterion (teacher-forced val loss, R1 protocol: clean
inputs, fresh state, 4-unit windows, 16-window sample) ranked step 4000
above step 5000, while the free-running val diagnostic ranked 5000 clearly
above 4000 (macro onset F1 0.0551 vs 0.0399, continuous; micro 0.0493 vs
0.0323) -- a documented rank inversion.

This script recomputes two *training-matched* variants of the val criterion
on CPU for checkpoints {3000, 4000, 5000}:

  carry_clean : forward_gpt_carry (seg 2048, detached state carry + shift
                lead = lever B protocol) over 2-unit windows of the 6 fixed
                val-diagnostic tracks, clean GT history.
  carry_noisy : same, with train.py's history noise at p=0.15 (lever A).

Both are cheap offline diagnostics; nothing here redefines the official R2
result (best_val remains step 4000 by the frozen criterion).

Outputs:
  * ckpt_proxy.csv        - per (checkpoint, variant) loss/acc
  * ckpt_proxy.json       - same + ranking summary vs the recorded criteria
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import pretty_midi
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

from murwkv.audio.mel import LogMelFrontend  # noqa: E402
from murwkv.model.murwkv_model import CHUNK_FRAMES, MuRWKV, MuRWKVConfig  # noqa: E402
from murwkv.tokenizer import encode_song, notes_from_pretty_midi, prepare_gt  # noqa: E402

CKPTS = ["ckpt_003000.pt", "ckpt_004000.pt", "ckpt_005000.pt"]
UNITS = 2   # 2-unit windows: fits the 2 GB cgroup budget (L <= ~2600)
SEG_TOKENS = 2048
NOISE_P = 0.15
SEED = 42


def fixed_val_tracks():
    split = json.load(open(os.path.join(C.R2, "split.json")))
    return sorted(split["valid"])[:6]  # exactly the pre-registered val-diag subset


def track_plan(tid, rng):
    gt = C.canonical_gt(tid)
    chunks, _ = encode_song(gt, max_tokens_per_chunk=4096)
    sel = chunks[1:1 + UNITS] if len(chunks) >= 1 + UNITS else chunks[:UNITS]
    wav, sr = sf.read(C._bs().tracks[tid].mix_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    fm = LogMelFrontend()
    with torch.no_grad():
        mel = fm(torch.from_numpy(wav).unsqueeze(0)).squeeze(0)
    mel = torch.from_numpy(mel.numpy().astype(np.float16).astype(np.float32))
    need = len(chunks) * CHUNK_FRAMES
    if mel.shape[0] < need:
        mel = torch.nn.functional.pad(mel, (0, 0, 0, need - mel.shape[0]))
    is_audio, midi = [], []
    for c in sel:
        is_audio += [True] * CHUNK_FRAMES
        midi += [0] * CHUNK_FRAMES + list(c.tokens)
        is_audio += [False] * len(c.tokens)
    return (mel[: len(sel) * CHUNK_FRAMES].unsqueeze(0), torch.tensor([is_audio]),
            torch.tensor([midi]), len(sel))


def eval_ckpt(model, plans, rng):
    out = {}
    for variant in ("carry_clean", "carry_noisy", "fresh_clean"):
        tot_loss, tot_tok, tot_acc, n_units = 0.0, 0, 0.0, 0
        for melU, is_audio, midi_t, n in plans:
            melU = melU.to(next(model.parameters()).dtype)
            with torch.no_grad():
                if variant == "carry_noisy":
                    from murwkv.training.train import apply_history_noise

                    midi_in = apply_history_noise(midi_t, is_audio, NOISE_P)
                else:
                    midi_in = midi_t
                if variant == "fresh_clean":
                    logits = model.forward_gpt(melU, is_audio, midi_in, use_cuda_kernel=False)
                else:
                    logits, _ = model.forward_gpt_carry(
                        melU, is_audio, midi_in, seg_tokens=SEG_TOKENS, use_cuda_kernel=False)
            # targets: last audio position of each unit predicts midi[1..M]
            targets = torch.zeros_like(midi_t)
            mask = torch.zeros_like(is_audio)
            L = is_audio.shape[1]
            arr = is_audio[0].tolist()
            i = 0
            spans = []
            while i < L:
                if arr[i]:
                    j = i
                    while j < L and arr[j]:
                        j += 1
                    k = j
                    while k < L and not arr[k]:
                        k += 1
                    M = k - j
                    if M > 0:
                        spans.append((j - 1, M))  # E = last audio position
                    i = k
                else:
                    i += 1
            for E, M in spans:
                targets[0, E:E + M] = midi_t[0, E + 1:E + M + 1]
                mask[0, E:E + M] = True
            sel = mask.bool()
            idx = sel[0].nonzero(as_tuple=True)[0]
            # memory-lean: gather head outputs only at masked positions
            logits_sel = logits[0, idx].float()
            del logits
            ce = torch.nn.functional.cross_entropy(logits_sel, targets[0, idx], reduction="none")
            ntok = int(idx.numel())
            tot_loss += float(ce.sum())
            tot_tok += ntok
            tot_acc += int((logits_sel.argmax(-1) == targets[0, idx]).sum())
            n_units += n
            del logits_sel, ce
        out[variant] = {"loss": tot_loss / max(1, tot_tok), "tok": tot_tok,
                        "acc": tot_acc / max(1, tot_tok), "units": n_units}
    return out


def main():
    torch.set_num_threads(4)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    tracks = fixed_val_tracks()
    print("fixed val tracks:", tracks, flush=True)
    plans = []
    for tid in tracks:
        plans.append(track_plan(tid, rng))
        print("plan", tid, "built", flush=True)
    rows = []
    for ck in CKPTS:
        ckpt = torch.load(os.path.join(C.R2, ck), map_location="cpu", weights_only=False)
        cfg = MuRWKVConfig(n_layer=ckpt["args"]["n_layer"], n_embd=ckpt["args"]["n_embd"])
        model = MuRWKV(cfg)
        model.load_state_dict(ckpt["model"])
        model.eval().bfloat16()  # training dtype; 2 GB cgroup budget
        del ckpt
        r = eval_ckpt(model, plans, rng)
        step = int(ck.split("_")[1].split(".")[0])
        for variant, v in r.items():
            rows.append({"checkpoint": ck, "step": step, "variant": variant, **{k: round(x, 6) if isinstance(x, float) else x for k, x in v.items()}})
        print(ck, json.dumps(r), flush=True)
        del model
    os.makedirs(C.OUT, exist_ok=True)
    with open(os.path.join(C.OUT, "ckpt_proxy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    best_official = {"criterion": "TF val loss (R1 protocol, train.py)", "best": "step 4000 (1.3994)",
                     "runner_up": "step 5000 (1.4010)"}
    free_running = {"source": "results/slakh_r2_valdiag/summary.json",
                    "best": "step 5000 (macro onset F1 0.0551 continuous)",
                    "runner_up": "step 4000 (0.0399)"}
    summary = {"rows": rows, "official_criterion": best_official, "free_running_reference": free_running}
    for variant in ("carry_clean", "carry_noisy", "fresh_clean"):
        sel = {r["step"]: r["loss"] for r in rows if r["variant"] == variant}
        if sel:
            summary[variant + "_ranking"] = sorted(sel, key=sel.get)
    json.dump(summary, open(os.path.join(C.OUT, "ckpt_proxy.json"), "w"), indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
