"""Q7: token-level vs note-level gap (CPU teacher-forced probe + cascade sim).

Part A - teacher-forced token probe (model forward on CPU, float32):
  For a fixed, F1-stratified sample of 6 official-test tracks x units 1..4:
  run the exact training plan (audio frames + GT MIDI tokens) through
  results/slakh_r2_carry/best_val.pt on CPU and record, at every loss
  position, the target token class, CE, and argmax match. Two input
  conditions: clean GT history and noisy history (train.py's
  apply_history_noise at p=0.15).

Part B - single-token corruption cascade (no model, pure decode):
  For each sampled chunk, replace exactly ONE token with a same-class random
  valid token and decode the chunk; measure note-level damage vs the
  uncorrupted decode. Answers: how much note damage does one token error of
  each class cause?

Outputs:
  * token_probe.csv        - per (track, unit, condition, class) stats (Part A)
  * token_probe_summary.json
  * cascade.csv            - per (class) damage stats (Part B)
  * cascade_examples.csv   - a few concrete before/after examples
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict

import gc

import numpy as np
import pretty_midi
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

from murwkv.audio.mel import LogMelFrontend  # noqa: E402
from murwkv.model.murwkv_model import CHUNK_FRAMES, MuRWKV, MuRWKVConfig  # noqa: E402
from murwkv.tokenizer import (  # noqa: E402
    EOS_ID,
    VOCAB,
    VOCAB_SIZE,
    decode_chunks,
    encode_song,
    notes_from_pretty_midi,
    prepare_gt,
    token_id,
)

DEVICE = "cpu"
N_TRACKS_PER_STRATUM = 2
UNITS_PER_TRACK = 4
NOISE_P = 0.15
CASCADE_PER_CLASS = 24
SEED = 42


def select_tracks():
    rows = C.r2_rows()
    tracks = sorted(rows, key=lambda t: rows[t]["onset_f1"])
    n = len(tracks)
    third = n // 3
    idx = [0, 1, third, third + 1, n - 2, n - 1]
    return [tracks[i] for i in idx], [f"bottom{i%2+1}" if i < third else ("mid" if n // 3 <= i < 2 * third else "top") for i in idx]


def load_model():
    ckpt = torch.load(os.path.join(C.R2, "best_val.pt"), map_location="cpu", weights_only=False)
    cfg = MuRWKVConfig(n_layer=ckpt["args"]["n_layer"], n_embd=ckpt["args"]["n_embd"])
    model = MuRWKV(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().bfloat16()  # training dtype; halves memory under the 2 GB cgroup cap
    del ckpt
    return model


def track_mel(tid, max_seconds=30.0):
    """Mel for the probe units only.

    Units 1..4 use mel frames [500, 2500) => samples [0.5 s, 25.1 s). STFT
    frames are local, and the left reflect-pad is the file's own start, so
    computing the STFT on the first 30 s gives mel frames bit-identical to
    the full-track computation for every frame the probe uses, while keeping
    the transient STFT intermediates (and the page cache) ~8x smaller under
    the 2 GB cgroup budget.
    """
    path = C._bs().tracks[tid].mix_path
    info = sf.info(path)
    stop = min(info.frames, int(max_seconds * info.samplerate))
    wav, sr = sf.read(path, dtype="float32", start=0, stop=stop)
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except OSError:
        pass
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert sr == 16000
    fm = LogMelFrontend()
    with torch.no_grad():
        mel = fm(torch.from_numpy(wav).unsqueeze(0)).squeeze(0)
    del wav
    # training/infer protocol: fp32 CPU STFT -> fp16 round-trip
    mel = torch.from_numpy(mel.numpy().astype(np.float16).astype(np.float32))
    return mel


def class_of(tok: int) -> str:
    return VOCAB[tok].type


def apply_history_noise(midi_id: torch.Tensor, is_audio: torch.Tensor, p: float, rng) -> torch.Tensor:
    """Mirror of training's apply_history_noise (uniform over event ids >= 3)."""
    out = midi_id.clone()
    mask = (~is_audio) & (out >= 3)
    r = torch.rand(out.shape)
    flip = mask & (r < p)
    n = int(flip.sum())
    if n:
        out[flip] = torch.randint(3, VOCAB_SIZE, (n,))
    return out


def probe_track(model, tid, rng):
    """Teacher-forced probe, ONE 5 s unit per forward (2 GB cgroup budget).

    The exact training layout is kept within the unit (audio frames + GT MIDI
    tokens, loss at the training positions). Cross-unit context is dropped:
    this measures per-unit conditional token accuracy from a fresh state,
    which matches how the val criterion sees units modulo window length.
    Clean and noisy-history conditions are run sequentially and intermediates
    are freed between forwards.
    """
    gt = C.canonical_gt(tid)
    chunks, _ = encode_song(gt, max_tokens_per_chunk=4096)
    sel = chunks[1:UNITS_PER_TRACK + 1] if len(chunks) >= UNITS_PER_TRACK + 1 else chunks[:UNITS_PER_TRACK]
    mel = track_mel(tid)

    out_rows = []
    tokens_by_unit = []
    for u, c in enumerate(sel):
        tokens_by_unit.append(list(c.tokens))
        M = len(c.tokens)
        is_audio = [True] * CHUNK_FRAMES + [False] * M
        midi_clean = [0] * CHUNK_FRAMES + list(c.tokens)
        is_audio_t = torch.tensor([is_audio])
        midi_t = torch.tensor([midi_clean])
        lo, hi = CHUNK_FRAMES - 1, CHUNK_FRAMES - 1 + M  # loss positions
        tgt_ids = midi_t[0, lo + 1:hi + 1].clone()
        keep = tgt_ids != 0
        los_idx = (torch.arange(lo, hi))[keep]
        tgt_ids = tgt_ids[keep]

        with torch.no_grad():
            melU = mel[u * CHUNK_FRAMES:(u + 1) * CHUNK_FRAMES].unsqueeze(0).bfloat16()
            audio_emb = model.audio_front(melU)
            audio_cum = torch.cumsum(is_audio_t.long(), dim=1) - 1
            audio_emb_g = torch.gather(
                audio_emb, 1,
                audio_cum.clamp(min=0).unsqueeze(-1).expand(
                    audio_emb.shape[0], is_audio_t.shape[1], audio_emb.shape[-1]))
            midi_emb = model.emb(midi_t)
            x_clean = torch.where(is_audio_t.unsqueeze(-1), audio_emb_g, midi_emb)
            midi_noisy = apply_history_noise(midi_t, is_audio_t, NOISE_P, rng)
            x_noisy = torch.where(is_audio_t.unsqueeze(-1), audio_emb_g, model.emb(midi_noisy))
            del audio_emb, audio_emb_g, midi_emb

        for cond, x in (("clean", x_clean), ("noisy", x_noisy)):
            with torch.no_grad():
                v_first = torch.empty_like(x)
                for blk in model.blocks:
                    x, v_first = blk.forward_parallel(x, v_first, use_cuda_kernel=False)
                h = model.ln_out(x)
                del x, v_first
                logits_sel = model.head(h[0, los_idx]).float()
                del h
                ce = torch.nn.functional.cross_entropy(logits_sel, tgt_ids, reduction="none")
                am = logits_sel.argmax(-1)
                del logits_sel
            for j in range(len(tgt_ids)):
                out_rows.append({
                    "track": tid, "unit": u, "cond": cond,
                    "tclass": class_of(int(tgt_ids[j])),
                    "ce": float(ce[j]),
                    "correct": int(int(am[j]) == int(tgt_ids[j])),
                    "pred_class": class_of(int(am[j])),
                    "pos_in_unit": j,
                    "unit_len": M,
                })
    return out_rows, tokens_by_unit


def cascade(tokens_by_unit, rng, tid):
    """Single-token corruption -> note damage vs uncorrupted decode."""
    rows = []
    for u, toks in enumerate(tokens_by_unit):
        if len(toks) < 6:
            continue
        base_notes = decode_chunks([list(toks)])
        by_class = defaultdict(list)
        for i, t in enumerate(toks):
            if t != EOS_ID:
                by_class[class_of(t)].append(i)
        for cls, positions in by_class.items():
            if not positions:
                continue
            for _ in range(min(CASCADE_PER_CLASS // max(1, UNITS_PER_TRACK), len(positions))):
                i = int(rng.choice(positions))
                choices = [v for v in range(VOCAB_SIZE) if VOCAB[v].type == cls and v != toks[i]]
                if not choices:
                    continue
                new_tok = int(rng.choice(choices))
                corrupt = list(toks)
                corrupt[i] = new_tok
                dmg_notes = decode_chunks([corrupt])
                # damage: note-level mismatch vs base decode
                keys_base = Counter((n.is_drum, n.program, n.pitch, round(n.onset, 2)) for n in base_notes)
                keys_cor = Counter((n.is_drum, n.program, n.pitch, round(n.onset, 2)) for n in dmg_notes)
                lost = sum((keys_base - keys_cor).values())
                added = sum((keys_cor - keys_base).values())
                # how far do surviving changed notes move?
                rows.append({
                    "track": tid, "unit": u, "class": cls,
                    "chunk_len": len(toks), "pos": i,
                    "orig_tok": toks[i], "new_tok": new_tok,
                    "notes_before": len(base_notes), "notes_after": len(dmg_notes),
                    "notes_lost": lost, "notes_added": added,
                    "damage": lost + added,
                })
    return rows


def main():
    os.makedirs(C.OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    tracks, strata = select_tracks()
    print("selected tracks:", list(zip(tracks, strata)), flush=True)
    model = load_model()
    print("model loaded", flush=True)

    all_probe = []
    all_cascade = []
    for k, tid in enumerate(tracks):
        rows, tokens_by_unit = probe_track(model, tid, rng)
        all_probe.extend(rows)
        all_cascade.extend(cascade(tokens_by_unit, rng, tid))
        C.canonical_gt.cache_clear()
        gc.collect()
        print(f"[{k + 1}/6] {tid} probe rows {len(rows)} cascade rows {len(all_cascade)}", flush=True)

    with open(os.path.join(C.OUT, "token_probe.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_probe[0]))
        w.writeheader()
        w.writerows(all_probe)

    # summaries
    summ = {}
    for cond in ("clean", "noisy"):
        rows = [r for r in all_probe if r["cond"] == cond]
        per_class = {}
        for cls in sorted({r["tclass"] for r in rows}):
            rs = [r for r in rows if r["tclass"] == cls]
            per_class[cls] = {
                "n_positions": len(rs),
                "mean_ce": float(np.mean([r["ce"] for r in rs])),
                "token_acc": float(np.mean([r["correct"] for r in rs])),
                "argmax_class_share": {
                    c: sum(1 for r in rs if r["pred_class"] == c) / len(rs)
                    for c in sorted({r["pred_class"] for r in rs})},
            }
        summ[cond] = per_class
    # diverged-by position: first wrong token per unit (clean)
    fd = []
    by_unit = defaultdict(list)
    for r in all_probe:
        if r["cond"] == "clean":
            by_unit[(r["track"], r["unit"])].append(r)
    for key, rs in by_unit.items():
        rs.sort(key=lambda r: r["pos_in_unit"])
        first = next((r["pos_in_unit"] for r in rs if not r["correct"]), None)
        n_wrong = sum(1 for r in rs if not r["correct"])
        fd.append({"track": key[0], "unit": key[1], "unit_len": rs[0]["unit_len"],
                   "first_error_at": first, "n_wrong": n_wrong})
    summ["first_error"] = {
        "units": len(fd),
        "units_with_zero_error": sum(1 for r in fd if r["n_wrong"] == 0),
        "median_first_error_pos": float(np.median([r["first_error_at"] for r in fd if r["first_error_at"] is not None])) if any(r["first_error_at"] is not None for r in fd) else None,
        "median_unit_len": float(np.median([r["unit_len"] for r in fd])),
        "mean_token_error_rate": float(np.mean([r["n_wrong"] / max(1, r["unit_len"]) for r in fd])),
    }
    json.dump(summ, open(os.path.join(C.OUT, "token_probe_summary.json"), "w"), indent=2)

    with open(os.path.join(C.OUT, "cascade.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_cascade[0]))
        w.writeheader()
        w.writerows(all_cascade)
    cs = {}
    for cls in sorted({r["class"] for r in all_cascade}):
        rs = [r for r in all_cascade if r["class"] == cls]
        cs[cls] = {
            "n": len(rs),
            "damage_mean": float(np.mean([r["damage"] for r in rs])),
            "damage_p50": float(np.median([r["damage"] for r in rs])),
            "damage_p90": float(np.quantile([r["damage"] for r in rs], 0.9)),
            "damage_zero_share": float(np.mean([float(r["damage"] == 0) for r in rs])),
            "notes_lost_mean": float(np.mean([r["notes_lost"] for r in rs])),
            "notes_added_mean": float(np.mean([r["notes_added"] for r in rs])),
        }
    json.dump(cs, open(os.path.join(C.OUT, "cascade_summary.json"), "w"), indent=2)
    print(json.dumps({"probe": {c: {k: v for k, v in d.items() if k != "argmax_class_share"} for c, d in s.items()} for c, s in summ.items() if c in ("clean", "noisy")}, indent=1))
    print(json.dumps(cs, indent=1))


if __name__ == "__main__":
    main()
