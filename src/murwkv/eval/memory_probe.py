"""GATE 5 — audio recurrent-memory sanity probe (continuous vs reset).

Task (controlled, no MIDI/text cues): classify the instrument identity of
REMOTE audio history after TWO IDENTICAL neutral chunks.

  sample = [h1 h2 h3 h4] (history A/B) + [N] + [N]   (N bit-identical across classes)
  target = class A/B, depends ONLY on the remote history.

continuous: RWKV state flows from history through both neutrals.
reset:      state is zeroed before the neutral chunks (only neutral audio is
            visible) -> chance-level accuracy expected through the last chunk.

The probe trains a small RWKV backbone + state-classifier head from scratch
(random init). Uses BabySlakh stems as instrument sources.
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
from ..model.murwkv_model import CHUNK_FRAMES
from ..model.murwkv_model import MuRWKVConfig
from ..model.rwkv7 import RWKVState

NEUTRAL_FIXED_SEED = 999


def make_neutral_chunk(n_frames=CHUNK_FRAMES):
    """A fixed neutral log-mel chunk (same for every sample, both classes)."""
    rng = np.random.RandomState(NEUTRAL_FIXED_SEED)
    # pink-ish noise spectrogram-like pattern: deterministic, bit-identical everywhere
    base = rng.randn(n_frames, 512).astype(np.float32)
    base = base * 0.05 + 0.1
    return base


class ProbeModel(nn.Module):
    """Small RWKV-7 backbone (2 layers) + classifier on final states.

    NOTE (probe-specific init, documented deviation): with the official RWKV
    decay init, per-frame decay of 0.56-0.99 over 1000 frames of neutral audio
    zeroes both the retained signal AND its gradient (measured: probe features
    were bit-identical across samples). The probe therefore biases the decay
    slow (w0 -= 6 -> per-frame decay ~0.9985) so the remote-history signal and
    its gradient survive; whether the model then USES the retained state is
    exactly what the probe measures (continuous vs reset).
    """

    def __init__(self, n_embd=256, head_size=64, n_layer=2, n_classes=2, n_mels=512, decay_bias=-6.0):
        super().__init__()
        self.cfg = MuRWKVConfig(n_layer=n_layer, n_embd=n_embd, head_size=head_size, n_mels=n_mels)
        from ..model.murwkv_model import MuRWKV

        self.backbone = MuRWKV(self.cfg)
        h = n_embd // head_size
        self.head = nn.Sequential(
            nn.Linear(n_embd + h * head_size * head_size, 256),
            nn.GELU(),
            nn.Linear(256, n_classes),
        )
        if decay_bias:
            with torch.no_grad():
                for blk in self.backbone.blocks:
                    blk.att.w0.data.add_(decay_bias)

    def apply_decay_bias(self, decay_bias=-6.0):
        """(Re-)apply after any parameter randomization (randomization would
        otherwise overwrite w0)."""
        with torch.no_grad():
            for blk in self.backbone.blocks:
                blk.att.w0.data.add_(decay_bias)
        # backbone output/ffn are zero-init -> probe must train from real init;
        # we randomize backbone head/emb consistently with the train path.

    def forward_probe(self, mel, mode, hist_frames=None):
        """mel: (B, T, 512). Returns logits (B, n_classes).

        mode='continuous': RWKV state flows across the whole sequence.
        mode='reset':      state is zero before the first neutral chunk — the
                           history never enters the model's memory. The
                           classifier then sees only the (bit-identical)
                           neutral content -> chance unless memory is real.
        """
        B, T, C = mel.shape
        if mode == "reset":
            assert hist_frames is not None
            mel = mel[:, hist_frames:]
        x = self.backbone.audio_front(mel)
        v_first = torch.empty_like(x)
        state = self.backbone.initial_state(B, mel.device)
        for blk in self.backbone.blocks:
            x, v_first = blk.forward_parallel(x, v_first, use_cuda_kernel=True)
            state.S[blk.layer_id] = blk._last_state
        h_last = self.backbone.ln_out(x)[:, -1]
        S = state.S[-1]  # (B,H,N,N) fp32
        feat = torch.cat([h_last.view(B, -1), S.reshape(B, -1).to(h_last.dtype)], dim=-1)
        return self.head(feat)


def build_probe_data(bs: BabySlakh, mel_cache_dir: str, class_map: dict, n_samples=400, seed=0, hist_chunks=4):
    """class_map: {class_label: [track_ids]}. History = chunks from stems of a class track."""
    rng = random.Random(seed)
    frontend = LogMelFrontend(sample_rate=16000, n_fft=2048, hop_length=160, n_mels=512)
    os.makedirs(mel_cache_dir, exist_ok=True)

    cache = {}
    for label, tids in class_map.items():
        for tid in tids:
            p = os.path.join(mel_cache_dir, f"{tid}_stem0.npy")
            if os.path.exists(p):
                cache[(label, tid)] = np.load(p)
                continue
            stems = bs.stems(tid)
            assert stems, f"no stem for {tid}"
            wav, sr = sf.read(stems[0], dtype="float32")
            if sr != 16000:
                raise ValueError(sr)
            if wav.ndim > 1:
                wav = wav.mean(1)
            with torch.no_grad():
                mel = frontend(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()
            np.save(p, mel)
            cache[(label, tid)] = mel

    neutral = make_neutral_chunk()
    samples = []
    for _ in range(n_samples):
        label = rng.choice(list(class_map))
        tid = rng.choice(class_map[label])
        mel = cache[(label, tid)]
        n_frames = len(mel)
        n_hist = hist_chunks * CHUNK_FRAMES
        assert n_frames >= n_hist + CHUNK_FRAMES, f"{tid} too short"
        s0 = rng.randint(0, n_frames - n_hist - CHUNK_FRAMES + 1)
        hist = mel[s0 : s0 + n_hist]
        full = np.concatenate([hist, neutral, neutral], axis=0).astype(np.float32)
        samples.append((full, label))
    return samples, neutral


class ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        mel, label = self.samples[i]
        return torch.from_numpy(mel), label


def train_probe(args):
    bs = BabySlakh(args.data_root)
    class_map = {0: args.class_a, 1: args.class_b}
    n_hist = args.hist_chunks * CHUNK_FRAMES
    mel_cache = os.path.join(os.path.dirname(args.data_root), "probe_mel")
    samples, neutral = build_probe_data(bs, mel_cache, class_map, n_samples=args.samples, seed=args.seed, hist_chunks=args.hist_chunks)
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n_tr = int(len(samples) * 0.8)
    ds_tr, ds_va = ProbeDataset(samples[:n_tr]), ProbeDataset(samples[n_tr:])
    model = ProbeModel(n_embd=args.n_embd, n_layer=args.n_layer, n_classes=len(class_map)).cuda().bfloat16()
    # randomize the probe backbone (real-signal training)
    torch.manual_seed(args.seed)
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.normal_(p, 0, 0.05)
        else:
            nn.init.normal_(p, 0, 0.02)
    model.apply_decay_bias(-6.0)  # slow-decay probe init (after randomization)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    n_chunks_total = args.hist_chunks + 2
    rows = []
    for epoch in range(args.epochs):
        model.train()
        tl = 0
        for mel, lab in ds_tr:
            mel = mel.unsqueeze(0).cuda().bfloat16()
            lab = torch.tensor([lab], device="cuda")
            opt.zero_grad(set_to_none=True)
            lg = model.forward_probe(mel, mode="continuous")
            loss = torch.nn.functional.cross_entropy(lg, lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += float(loss)
        # validation: continuous vs reset, and neutral-only (classifier on state before history?)
        model.eval()
        acc_c = acc_r = 0
        with torch.no_grad():
            for mel, lab in ds_va:
                mel = mel.unsqueeze(0).cuda().bfloat16()
                lab = torch.tensor([lab], device="cuda")
                lg_c = model.forward_probe(mel, mode="continuous")
                lg_r = model.forward_probe(mel, mode="reset", hist_frames=n_hist)
                acc_c += (lg_c.argmax(-1) == lab).sum().item()
                acc_r += (lg_r.argmax(-1) == lab).sum().item()
        n = len(ds_va)
        row = {"epoch": epoch, "train_loss": round(tl / max(1, len(ds_tr)), 4), "cont_acc": acc_c / n, "reset_acc": acc_r / n}
        rows.append(row)
        print(row, flush=True)
    os.makedirs(args.exp, exist_ok=True)
    with open(os.path.join(args.exp, "probe_metrics.json"), "w") as f:
        json.dump({"rows": rows, "n_train": len(ds_tr), "n_val": len(ds_va), "neutral_hash": neutral.tobytes().hex()[:16], "args": vars(args)}, f, indent=2)
    # state-distance analysis: class-separability of the state vs #neutral chunks
    dist_rows = state_distance_analysis(model, samples, n_hist, neutral)
    with open(os.path.join(args.exp, "probe_state_distance.json"), "w") as f:
        json.dump(dist_rows, f, indent=2)
    print("PROBE DONE")


@torch.no_grad()
def state_distance_analysis(model, samples, hist_frames, neutral, reps=(0, 1, 2, 3, 4)):
    """Mean per-layer state-norm vectors per class after k neutrals; report the
    inter-class distance of those vectors (memory retention vs neutral decay)."""
    model.eval()
    by_class = {0: [], 1: []}
    for mel, label in samples[:60]:
        by_class[label].append(mel)
    out = []
    for k in reps:
        dA, dB = [], []
        for mel in by_class[0][:20]:
            dA.append(state_norm_after(model, mel, hist_frames, neutral, k))
        for mel in by_class[1][:20]:
            dB.append(state_norm_after(model, mel, hist_frames, neutral, k))
        A = torch.stack(dA).mean(0)
        B = torch.stack(dB).mean(0)
        out.append({"n_neutral": k, "dist": float((A - B).abs().mean().item()), "nA": float(A.mean().item()), "nB": float(B.mean().item())})
    return out


@torch.no_grad()
def state_norm_after(model, mel_np, hist_frames, neutral, k_neutral):
    from ..model.murwkv_model import CHUNK_FRAMES

    mel = torch.from_numpy(mel_np[:hist_frames]).unsqueeze(0).cuda().bfloat16()
    x = model.backbone.audio_front(mel)
    v_first = torch.empty_like(x)
    st = model.backbone.initial_state(1, mel.device)
    for blk in model.backbone.blocks:
        x, v_first = blk.forward_parallel(x, v_first)
        st.S[blk.layer_id] = blk._last_state
    for k in range(k_neutral):
        seg = torch.from_numpy(neutral).unsqueeze(0).cuda().bfloat16()
        x2 = model.backbone.audio_front(seg)
        v2 = torch.empty_like(x2)
        for blk in model.backbone.blocks:
            x2, v2 = blk.forward_parallel(x2, v2, init_state=st.S[blk.layer_id])
            st.S[blk.layer_id] = blk._last_state
    return torch.stack([s.norm() for s in st.S])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--data-root", default="/root/autodl-tmp/data/babyslakh/babyslakh_16k")
    ap.add_argument("--class-a", nargs="+", default=["Track00001"])
    ap.add_argument("--class-b", nargs="+", default=["Track00002"])
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--hist-chunks", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train_probe(args)


if __name__ == "__main__":
    main()