"""GATE 5-v2 — leak-free audio recurrent-memory probe on the FROZEN AMT model.

Post-review replacement for `memory_probe.py` (G5, `results/gate5_probe`).
What changed, and why (review findings, recorded here as the design contract):

1. G5 trained a SEPARATE small RWKV probe (2 layers, n_embd 256) with an
   artificial slow-decay init (`w0 -= 6`), NOT the trained 21.9M AMT model.
   -> v2 FREEZES the G4 AMT checkpoint (`--ckpt`) and trains only a tiny
   ridge (exact linear) probe on its recurrent state. Official learned decay,
   exactly as trained — no `w0` bias anywhere.

2. G5 used ONE song per class and took `stems[0]` blindly (never verifying the
   stem's instrument against Slakh metadata), and split train/val at the
   SAMPLE level (crops of the same 2 songs in both).
   -> v2 selects stems via `metadata.yaml` `inst_class` (asserted per stem),
   uses ALL tracks of the corpus (every BabySlakh 16k track has both a Guitar
   and a Piano stem), and splits at TRACK level: train/val/test tracks are
   disjoint; the probe TEST tracks are forced to be the tracks the G4 AMT
   model NEVER trained on (so test-split accuracy cannot ride on AMT
   memorization of those songs).

3. Claim discipline (review): the old continuous 1.0 vs reset 0.483 showed
   remote acoustic SOURCE identity retention, NOT cross-track instrument
   generalization. The v2 test split (instrument classes seen only through
   held-out tracks) is what establishes instrument-level generalization; if
   test accuracy is chance, v2 reports exactly that.

Task (controlled, no MIDI/text cues):
    sample = [history: `hist_chunks` x 5s of ONE instrument stem] + [N] + [N]
    N      = fixed neutral log-mel chunk, bit-identical across ALL samples.
    target = instrument class of the history (Guitar vs Piano).

Four observation arms (single parallel forwards, all from ONE joint mel):
    continuous: blocks over [H H H H | N N]  — wkv state + shift carry flow.
    lead1:      blocks over [..H | N N], only the last 3 history frames in the
                batch and a FRESH state — bounds the short-horizon direct
                channel (1-frame shift carry + 3 frames of wkv content).
    reset:      blocks over [N N] fresh — pure neutral content; chance unless
                memory is real. (The shift-carry sensitivity is what lead1
                bounds.)
    history:    blocks over [H H H H] — identity immediately available at the
                end of history (upper bound, no neutral decay).

Features (per sample, frozen model, no grad): h_last (C,) after the last
frame + last-layer wkv state S flat (H*N*N, fp32)  ->  C + H*N*N dims.
Probes: exact ridge (kernel-trick, deterministic) on (a) full features and
(b) per-head S statistics (mean/std/max, robust low-dim). Optional small MLP
(--mlp) as a capacity sensitivity check on (b).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from ..audio.mel import LogMelFrontend
from ..data.babyslakh import BabySlakh
from ..model.murwkv_model import CHUNK_FRAMES, MuRWKV, MuRWKVConfig

NEUTRAL_FIXED_SEED = 999  # same neutral as the original G5 (continuity)
INST_CLASSES = {"guitar": "Guitar", "piano": "Piano"}

# The 4 BabySlakh tracks the G4 AMT model never trained on (its fixed
# valid/test split). Probe TEST is forced to these so test-split accuracy
# cannot come from the AMT model having memorized the songs.
G4_HELDOUT_TRACKS = ["Track00005", "Track00015", "Track00006", "Track00020"]


def make_neutral_chunk(n_frames: int = CHUNK_FRAMES) -> np.ndarray:
    """Fixed neutral log-mel chunk (same seed/recipe as G5: bit-identical
    across all samples and both classes)."""
    rng = np.random.RandomState(NEUTRAL_FIXED_SEED)
    base = rng.randn(n_frames, 512).astype(np.float32)
    return base * 0.05 + 0.1


def load_amt_model(ckpt_path: str, device: str, dtype=torch.bfloat16) -> MuRWKV:
    """Load the trained AMT checkpoint and FREEZE it (eval, no grad)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = MuRWKVConfig(**{k: v for k, v in ckpt["args"].items() if k in ("n_layer", "n_embd")})
    model = MuRWKV(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Instrument selection + track-level split (the leakage fixes)
# ---------------------------------------------------------------------------


def select_class_stems(bs: BabySlakh, tid: str, inst_class: str) -> list[str]:
    """Stem ids of `tid` whose metadata inst_class == target, VERIFIED.

    Returns the sorted stem ids that exist on disk AND carry the exact
    inst_class in metadata.yaml. Raises if a stem file exists whose metadata
    disagrees (catches the old `stems[0]`-without-verification bug class).
    """
    meta = bs.stem_metadata(tid)
    stems = bs.stems(tid)
    out, bad = [], []
    for p in stems:
        sid = os.path.splitext(os.path.basename(p))[0]
        info = meta.get(sid)
        if not info or not info.get("inst_class"):
            bad.append((sid, "no metadata"))
        elif info["inst_class"] == inst_class:
            out.append(sid)
    if bad:
        raise AssertionError(f"{tid}: stem files without matching metadata: {bad}")
    return sorted(out)


def build_probe_split(track_ids, seed: int = 42, n_test: int = 4, n_val: int = 2,
                      force_test: list[str] | None = None) -> dict:
    """Track-level tri-partition (deterministic).

    force_test: tracks that MUST land in test (G4-held-out tracks). Tracks in
    force_test that are not in the corpus are ignored. Guarantees: every track
    in exactly one split; test is as large as requested.
    """
    force_test = [t for t in (force_test or []) if t in track_ids]
    rest = [t for t in sorted(track_ids) if t not in force_test]
    rng = random.Random(seed)
    rng.shuffle(rest)
    val = sorted(rest[:n_val])
    train = sorted(rest[n_val : len(rest) - max(0, n_test - len(force_test))])
    test = sorted(force_test + rest[len(rest) - max(0, n_test - len(force_test)) :])
    return {"train": train, "val": val, "test": test}


def assert_no_track_overlap(splits: dict) -> None:
    """Invariant: a track's crops must never appear in two splits."""
    all_ids = [t for v in splits.values() for t in v]
    assert len(all_ids) == len(set(all_ids)), f"track overlap in splits: {all_ids}"
    for name, ids in splits.items():
        assert len(ids) == len(set(ids)), f"duplicate track inside split {name}"


SPLIT_SEED_OFFSETS = {"train": 0, "val": 1, "test": 2}  # deterministic (never hash())


def build_samples(split_tracks: dict, stems_by_class: dict, stem_paths: dict, stem_frames: dict,
                  hist_chunks: int, samples_per_track: int, seed: int):
    """Per-split samples with a deterministic RNG.

    Per (track, class) exactly `samples_per_track` samples; the stem is chosen
    among the track's VERIFIED class stems; the start frame is random inside
    the stem such that the full history fits. Returns a list of dicts.
    """
    hist_frames = hist_chunks * CHUNK_FRAMES
    out = []
    for sp, tids in split_tracks.items():
        seed_sp = seed + SPLIT_SEED_OFFSETS[sp]
        rng = random.Random(seed_sp)
        for tid in sorted(tids):
            covered = False
            for label, cls in enumerate(INST_CLASSES.values()):
                stems = stems_by_class.get((tid, cls)) or []
                for _ in range(samples_per_track):
                    if not stems:
                        continue
                    sid = stems[rng.randrange(len(stems))]
                    n_mel = stem_frames[(tid, sid)]  # mel frames (100 fps)
                    if n_mel <= hist_frames:  # need at least one free window start
                        continue
                    s0 = rng.randint(0, n_mel - hist_frames)
                    out.append({"split": sp, "tid": tid, "stem": sid, "class": label,
                                "start": s0, "hist_frames": hist_frames})
                covered = covered or bool(stems)
            if not covered:
                raise AssertionError(f"no class-covered stems for {tid} (hist {hist_frames} frames)")
    return out


# ---------------------------------------------------------------------------
# Feature extraction (frozen model)
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_features(model: MuRWKV, mel_seq: torch.Tensor, arms: list[str],
                     device: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """mel_seq: (T, 512) fp32 full sequence [history | N | N] (or parts).

    Returns per arm: (h_last (C,), S_last_flat (H*N*N,)) fp32 — features at
    the LAST observed frame, with the wkv state the arm ended with.
    """
    out = {}
    dtype = next(model.parameters()).dtype
    x = model.audio_front(mel_seq.unsqueeze(0).to(device=device, dtype=dtype))  # (1, T, C)

    def run(slice_l, slice_r, label):
        seg = x[:, slice_l:slice_r]
        v_first = torch.empty_like(seg)
        s_last = None
        for blk in model.blocks:
            # fresh wkv state at the arm's sequence start (continuous starts at
            # history frame 0 like a streamed song; other arms start at their
            # observation window). Never the CUDA kernel: T is not % 16 here.
            seg, v_first = blk.forward_parallel(seg, v_first, use_cuda_kernel=False)
            s_last = blk._last_state  # (B, H, N, N) fp32 at this layer's seq end
        h = model.ln_out(seg)[:, -1].to(torch.float32).squeeze(0)  # (C,)
        out[label] = (h, s_last.to(torch.float32).reshape(-1))

    T = x.shape[1]
    n_hist_mel = mel_seq.shape[0] - 2 * CHUNK_FRAMES  # design: exactly 2 neutrals
    # continuous: whole sequence; lead1: last 3 history frames + neutrals
    # (frontend conv carry needs 2 past mel frames for the final embedding);
    # reset: neutrals only; history: history only.
    if "continuous" in arms:
        run(0, T, "continuous")
    if "lead1" in arms:
        run(max(0, n_hist_mel - 2), T, "lead1")
    if "reset" in arms:
        run(n_hist_mel, T, "reset")
    if "history" in arms:
        run(0, n_hist_mel, "history")
    return out


def feats_to_np(h: torch.Tensor, S: torch.Tensor) -> np.ndarray:
    return np.concatenate([h.numpy(), S.numpy()]).astype(np.float32)


def stats_feats(h: torch.Tensor, S: torch.Tensor, H: int, N: int) -> np.ndarray:
    """Per-head summary stats of the last-layer state + h_last (robust low-dim)."""
    S = S.reshape(H, N, N)
    stats = np.stack([S.mean(axis=(1, 2)), S.std(axis=(1, 2)), np.abs(S).max(axis=(1, 2))]).reshape(-1)
    return np.concatenate([h.numpy(), stats]).astype(np.float32)


# ---------------------------------------------------------------------------
# Probes: exact ridge (linear) + optional small MLP
# ---------------------------------------------------------------------------


def ridge_probe(Xtr, ytr, Xva, yva, Xte, yte, lam_list=(1e-4, 1e-2, 1.0, 1e2)):
    """Exact ridge with intercept on standardized features; λ chosen on the
    val split. Kernel-trick solve (n×n Gram) so it is exact and cheap even
    with ~33k-dim features. Returns metrics per λ + best-λ test result."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s = (Xtr - mu) / sd
    Xva_s = (Xva - mu) / sd
    Xte_s = (Xte - mu) / sd
    ytr = ytr.astype(np.float64)
    ymean = ytr.mean()
    ytr_c = ytr - ymean
    rows = []
    best = None
    for lam in lam_list:
        K = Xtr_s @ Xtr_s.T  # (n, n)
        alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), ytr_c)
        w = Xtr_s.T @ alpha
        f = lambda X: X @ w + ymean  # noqa: E731
        acc = lambda X, y: float(((f(X) > 0) == (y > 0)).mean())  # noqa: E731
        r = {"lam": lam, "train_acc": acc(Xtr_s, ytr), "val_acc": acc(Xva_s, yva), "test_acc": acc(Xte_s, yte)}
        rows.append(r)
        if best is None or r["val_acc"] > best["val_acc"]:
            best = r
    return rows, best, {"mu": mu, "sd": sd, "ymean": ymean}


def binomial_ci(k: int, n: int) -> tuple[float, float]:
    from scipy.stats import binomtest

    if n == 0:
        return (0.0, 0.0)
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return float(ci.low), float(ci.high)


class MLPProbe(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp_probe(Xtr, ytr, Xva, yva, Xte, yte, seed=0, epochs=60, lr=1e-3, wd=1e-2):
    """Small MLP sensitivity check (capacity control: strong weight decay)."""
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xva_s, Xte_s = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    model = MLPProbe(Xtr.shape[1]).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    t = torch.from_numpy(Xtr_s), torch.from_numpy(ytr)
    for _ in range(epochs):
        for (xb, yb) in [(t[0], t[1])]:
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(xb), (yb > 0).float())
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = lambda X, y: float(((model(torch.from_numpy(X)) > 0) == (y > 0)).float().mean())  # noqa: E731
    return {"train_acc": acc(Xtr_s, ytr), "val_acc": acc(Xva_s, yva), "test_acc": acc(Xte_s, yte)}


# ---------------------------------------------------------------------------
# State-distance analysis (frozen model, continuous arm)
# ---------------------------------------------------------------------------


@torch.no_grad()
def state_distance_after(model: MuRWKV, mel_hist: torch.Tensor, neutral: torch.Tensor,
                         k_neutral: int, device: str) -> torch.Tensor:
    """Per-layer state norms at the end of history + k neutrals (continuous)."""
    seq = torch.cat([mel_hist, *([neutral] * k_neutral)], 0)
    dtype = next(model.parameters()).dtype
    x = model.audio_front(seq.unsqueeze(0).to(device=device, dtype=dtype))
    st = model.initial_state(1, device)
    v_first = torch.empty_like(x)
    for blk in model.blocks:
        x, v_first = blk.forward_parallel(x, v_first, use_cuda_kernel=False, init_state=None)
        st.S[blk.layer_id] = blk._last_state
    return torch.stack([s.float().norm() for s in st.S])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_mel_cache(bs: BabySlakh, samples: list[dict], cache_dir: str, stem_paths: dict) -> dict:
    """Log-mel per (track, stem), cached fp16 npz on the data disk (dataset
    convention). Returns {(tid, stem): mel (F, 512) fp32}."""
    os.makedirs(cache_dir, exist_ok=True)
    frontend = LogMelFrontend(sample_rate=16000, n_fft=2048, hop_length=160, n_mels=512)
    cache: dict = {}
    keys = sorted({(s["tid"], s["stem"]) for s in samples})
    for tid, sid in keys:
        p = os.path.join(cache_dir, f"{tid}_{sid}.npz")
        if os.path.exists(p):
            cache[(tid, sid)] = np.load(p)["mel"].astype(np.float32)
            continue
        wav, sr = sf.read(stem_paths[(tid, sid)], dtype="float32")
        assert sr == 16000, f"{tid}/{sid}: sr {sr}"
        if wav.ndim > 1:
            wav = wav.mean(1)
        with torch.no_grad():
            mel = frontend(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()
        np.savez(p, mel=mel.astype(np.float16))
        cache[(tid, sid)] = mel
    return cache


def run(args):
    t0 = time.time()
    bs = BabySlakh(args.data_root, splits=args.splits)
    track_ids = bs.track_ids
    # ---- instrument stem selection (verification is inside) ----
    stems_by_class: dict = {}
    coverage = {}
    stem_paths: dict = {}
    stem_frames: dict = {}
    for tid in track_ids:
        for p in bs.stems(tid):
            sid = os.path.splitext(os.path.basename(p))[0]
            stem_paths[(tid, sid)] = p
            # stem length in MEL frames (16kHz audio, hop=160 -> 100 fps)
            stem_frames[(tid, sid)] = sf.info(p).frames // 160
    for cls in INST_CLASSES.values():
        n = 0
        for tid in track_ids:
            stems = select_class_stems(bs, tid, cls)
            if stems:
                stems_by_class[(tid, cls)] = stems
                n += 1
        coverage[cls] = n
        print(f"[g5v2] class {cls!r}: {n}/{len(track_ids)} tracks with verified stems", flush=True)
    # ---- track-level split ----
    force_test = [t for t in G4_HELDOUT_TRACKS if t in track_ids]
    splits = build_probe_split(track_ids, seed=args.split_seed, n_test=args.n_test,
                               n_val=args.n_val, force_test=force_test)
    assert_no_track_overlap(splits)
    print("[g5v2] split:", {k: len(v) for k, v in splits.items()}, flush=True)
    # ---- samples ----
    all_samples = []
    for sp, tids in splits.items():
        n = {"train": args.samples_train, "val": args.samples_val, "test": args.samples_test}[sp]
        all_samples += build_samples({sp: tids}, stems_by_class, stem_paths, stem_frames,
                                     args.hist_chunks, n, seed=args.seed_data)
    by_split = {}
    for s in all_samples:
        by_split.setdefault(s["split"], []).append(s)
    print("[g5v2] samples:", {k: len(v) for k, v in by_split.items()}, flush=True)
    # ---- artifacts ----
    os.makedirs(args.exp, exist_ok=True)
    neutral = make_neutral_chunk()
    np.save(os.path.join(args.exp, "neutral.npy"), neutral)
    manifest = {
        "split": splits,
        "g4_trained_tracks": args.g4_trained_tracks,
        "instrument_coverage": coverage,
        "stems_by_class": {f"{tid}/{cls}": stems for (tid, cls), stems in sorted(stems_by_class.items())},
    }
    with open(os.path.join(args.exp, "split.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(args.exp, "samples.json"), "w") as f:
        json.dump(all_samples, f, indent=2)

    if args.prepare_only:
        print("[g5v2] data artifacts written; --prepare-only, exiting")
        return

    # ---- frozen AMT model ----
    model = load_amt_model(args.ckpt, args.device, dtype=torch.bfloat16 if not args.fp32 else torch.float32)
    print(f"[g5v2] frozen AMT model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params "
          f"({model.cfg.n_layer}L × {model.cfg.n_embd}), official learned decay, no w0 bias", flush=True)

    # ---- feature extraction ----
    mel_cache = build_mel_cache(bs, all_samples, args.mel_cache, stem_paths)
    feats_dir = os.path.join(args.exp, "feats")
    os.makedirs(feats_dir, exist_ok=True)
    arms = ["continuous", "lead1", "reset", "history"]
    H = model.cfg.n_embd // model.cfg.head_size
    N = model.cfg.head_size
    hist_frames = args.hist_chunks * CHUNK_FRAMES
    records = {sp: {a: {"tid": [], "class": [], "feat_full": [], "feat_stats": []} for a in arms}
               for sp in by_split}
    n_done = 0
    for sp, samples in by_split.items():
        for s in samples:
            mel = mel_cache[(s["tid"], s["stem"])]
            hist = torch.from_numpy(mel[s["start"]: s["start"] + hist_frames].astype(np.float32))
            neutral_t = torch.from_numpy(neutral)
            seq = torch.cat([hist, neutral_t, neutral_t], 0)
            res = extract_features(model, seq, arms, args.device)
            for a in arms:
                h, S = res[a]
                rec = records[sp][a]
                rec["tid"].append(s["tid"])
                rec["class"].append(s["class"])
                rec["feat_full"].append(feats_to_np(h, S))
                rec["feat_stats"].append(stats_feats(h, S, H, N))
            n_done += 1
            if n_done % 25 == 0:
                print(f"[g5v2] features {n_done}/{sum(len(v) for v in by_split.values())} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    for sp in records:
        for a in arms:
            rec = records[sp][a]
            np.savez(os.path.join(feats_dir, f"{a}_{sp}.npz"),
                     tid=np.array(rec["tid"]), label=np.array(rec["class"]),
                     feat_full=np.stack(rec["feat_full"]), feat_stats=np.stack(rec["feat_stats"]))

    # ---- probes ----
    results = {"args": vars(args), "neutral_sha": neutral.tobytes().hex()[:16],
               "arms": arms, "split": splits, "n_samples": {k: len(v) for k, v in by_split.items()},
               "probes": {}, "state_distance": {}}
    label_of = lambda sp, a: np.load(os.path.join(feats_dir, f"{a}_{sp}.npz"))["label"].astype(np.float64) * 2 - 1  # noqa: E731
    for a in arms:
        Xtr = np.load(os.path.join(feats_dir, f"{a}_train.npz"))["feat_full"]
        Xva = np.load(os.path.join(feats_dir, f"{a}_val.npz"))["feat_full"]
        Xte = np.load(os.path.join(feats_dir, f"{a}_test.npz"))["feat_full"]
        ytr, yva, yte = label_of("train", a), label_of("val", a), label_of("test", a)
        rows, best, _ = ridge_probe(Xtr, ytr, Xva, yva, Xte, yte)
        entry = {"feature_set": "h_last+S_last", "dims": Xtr.shape[1], "ridge_rows": rows,
                 "best": best}
        if args.mlp:
            entry["mlp"] = mlp_probe(Xtr, ytr, Xva, yva, Xte, yte, seed=args.seed_data)
        # per-split accuracy + binomial 95% CI of the chosen ridge model
        lam = best["lam"]
        mu, sd, ymean = Xtr.mean(0), Xtr.std(0) + 1e-8, ytr.mean()
        Xtr_s = (Xtr - mu) / sd
        ytr_c = ytr.astype(np.float64) - ymean
        alpha = np.linalg.solve(Xtr_s @ Xtr_s.T + lam * np.eye(Xtr_s.shape[0]), ytr_c)
        w = Xtr_s.T @ alpha
        acc_with_ci = {}
        for sp_name, X, y in (("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)):
            Xs = (X - mu) / sd
            pred = (Xs @ w + ymean > 0)
            k = int((pred == (y > 0)).sum())
            n = len(y)
            lo, hi = binomial_ci(k, n)
            acc_with_ci[sp_name] = {"acc": k / n, "k": k, "n": n, "ci95": [round(lo, 4), round(hi, 4)]}
        entry["acc_ci"] = acc_with_ci
        results["probes"][a] = entry
        print(f"[g5v2] arm {a}: ridge best λ={lam} Test acc {acc_with_ci['test']['acc']:.4f} "
              f"({acc_with_ci['test']['ci95']}) | val {acc_with_ci['val']['acc']:.4f} | "
              f"train {acc_with_ci['train']['acc']:.4f}", flush=True)
        # ---- low-dim stats-feature ridge (robustness) ----
        Xtr_sm = np.load(os.path.join(feats_dir, f"{a}_train.npz"))["feat_stats"]
        Xva_sm = np.load(os.path.join(feats_dir, f"{a}_val.npz"))["feat_stats"]
        Xte_sm = np.load(os.path.join(feats_dir, f"{a}_test.npz"))["feat_stats"]
        rows2, best2, _ = ridge_probe(Xtr_sm, ytr, Xva_sm, yva, Xte_sm, yte)
        results["probes"][a + "_stats"] = {"feature_set": "per-head stats + h_last",
                                           "dims": Xtr_sm.shape[1], "ridge_rows": rows2, "best": best2}
        print(f"[g5v2] arm {a} (stats): best test acc {best2['test_acc']:.4f} (λ={best2['lam']})", flush=True)

    # ---- state distance across neutral count (continuous, test tracks) ----
    dist_rows = []
    test_samps = by_split["test"]
    by_class = {0: [], 1: []}
    for s in test_samps[:40]:
        by_class[s["class"]].append(s)
    for k in (0, 1, 2, 3, 4):
        dA, dB = [], []
        for s in by_class[0][:10]:
            mel = mel_cache[(s["tid"], s["stem"])]
            hist = torch.from_numpy(mel[s["start"]: s["start"] + hist_frames].astype(np.float32))
            dA.append(state_distance_after(model, hist, torch.from_numpy(neutral), k, args.device))
        for s in by_class[1][:10]:
            mel = mel_cache[(s["tid"], s["stem"])]
            hist = torch.from_numpy(mel[s["start"]: s["start"] + hist_frames].astype(np.float32))
            dB.append(state_distance_after(model, hist, torch.from_numpy(neutral), k, args.device))
        A = torch.stack(dA).mean(0)
        B = torch.stack(dB).mean(0)
        dist_rows.append({"n_neutral": k, "dist": float((A - B).abs().mean().item()),
                          "nA": float(A.mean().item()), "nB": float(B.mean().item())})
    results["state_distance"] = dist_rows
    print("[g5v2] state distance:", dist_rows, flush=True)

    with open(os.path.join(args.exp, "probe_v2_metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"[g5v2] DONE in {time.time()-t0:.0f}s -> {args.exp}")


def main():
    ap = argparse.ArgumentParser(description="GATE 5-v2 leak-free memory probe (frozen AMT model)")
    ap.add_argument("--exp", default="results/gate5_probe_v2")
    ap.add_argument("--data-root", default="/root/autodl-tmp/data/babyslakh/babyslakh_16k")
    ap.add_argument("--splits", action="store_true", help="Slakh2100-style <root>/train|validation|test layout")
    ap.add_argument("--ckpt", default="results/gate4_overfit_v2/final.pt")
    ap.add_argument("--g4-trained-tracks", nargs="*", default=["Track00001", "Track00002", "Track00003", "Track00004",
                                                               "Track00007", "Track00008", "Track00009", "Track00010",
                                                               "Track00011", "Track00012"])
    ap.add_argument("--hist-chunks", type=int, default=4, help="history length in 5s chunks (20s default)")
    ap.add_argument("--samples-train", type=int, default=16)
    ap.add_argument("--samples-val", type=int, default=8)
    ap.add_argument("--samples-test", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=4)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--seed-data", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true", help="run the model in fp32 (CPU smoke)")
    ap.add_argument("--mlp", action="store_true", help="also run the MLP sensitivity check")
    ap.add_argument("--prepare-only", action="store_true",
                    help="build/verify data artifacts (split, stems, samples) and exit — no model needed")
    ap.add_argument("--mel-cache", default="/root/autodl-tmp/data/probe_v2_mel")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()