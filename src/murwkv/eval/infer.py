"""Streaming inference: continuous (state-carry) and reset (per-chunk) modes.

The transcribe() path is the exact streaming protocol verified in Gate 2:
  * frontend conv carry: last 2 mel frames,
  * time-mixing shift lead: last embedded frame,
  * recurrent state: RWKVState carried across chunks (continuous) or zeroed
    (reset), with the lead/carry zeroed as well in reset mode.

MIDI tokens are generated greedily (argmax) one at a time with true RNN steps
(no replay). The model emits its own tie prologue per chunk (as learned from
teacher-forced training); the decoder applies the open-note protocol.
"""
from __future__ import annotations

import os

import torch

from ..audio.mel import LogMelFrontend
from ..model.murwkv_model import (
    AUDIO_FRAME_RATE,
    CHUNK_FRAMES,
    MuRWKV,
    PAD_ID,
)
from ..model.rwkv7 import CHUNK_LEN
from ..tokenizer import EOS_ID, MT3Decoder, Note, notes_from_pretty_midi, program_rep, tokens_to_midi

DEFAULT_MAX_TOKENS_PER_CHUNK = 2048


def compile_stepwise_decode(model: MuRWKV):
    """torch.compile the per-token stepwise forward (Tensor-in/Tensor-out).

    Measured 2026-08-28 on the RTX 5090: 0.39 ms/token vs 3.65 ms eager
    (~9x) — the eager path is dominated by Python/launch overhead. Same
    greedy policy, both arms use it identically (continuous/reset remain
    paired). Returns fn(x_t, state, seed) -> (logits, state, v_first) with
    the same signature as MuRWKV.forward_rnn_step.
    """
    n_blocks = len(model.blocks)
    ids = list(range(n_blocks))

    def step_all(x_t, vf, *Sapf):
        Ss = list(Sapf[0:n_blocks])
        aps = list(Sapf[n_blocks:2 * n_blocks])
        fps = list(Sapf[2 * n_blocks:3 * n_blocks])
        for i in ids:
            x_t, aps[i], fps[i], Ss[i], vf = model.blocks[i].forward_step(
                x_t, vf, Ss[i], aps[i], fps[i])
        return (model.head(model.ln_out(x_t)), *Ss, vf, *aps, *fps)

    comp = torch.compile(step_all, mode="reduce-overhead")

    def forward(x_t: torch.Tensor, state, seed=None):
        if seed is not None:
            vf, seed_ap, seed_fp = seed
        else:
            vf = torch.zeros_like(x_t)
            seed_ap, seed_fp = state.att_prev, state.ffn_prev
        args = [x_t, vf]
        args += [state.S[i] for i in ids]
        args += [seed_ap[i] for i in ids]
        args += [seed_fp[i] for i in ids]
        outs = comp(*args)
        lg = outs[0]
        k = 1
        for i in ids:
            state.S[i] = outs[k]
            k += 1
        vf2 = outs[k]
        k += 1
        for i in ids:
            state.att_prev[i] = outs[k]
            k += 1
        for i in ids:
            state.ffn_prev[i] = outs[k]
            k += 1
        return lg, state, vf2

    return forward


class Transcriber:
    def __init__(self, model: MuRWKV, device="cuda", max_tokens_per_chunk: int = DEFAULT_MAX_TOKENS_PER_CHUNK,
                 use_compiled: bool = False):
        self.model = model.eval()
        self.device = device
        self.max_tokens_per_chunk = max_tokens_per_chunk
        if use_compiled:
            self.model.forward_rnn_step = compile_stepwise_decode(model)

    def _frontend(self, wav: torch.Tensor) -> torch.Tensor:
        """log-mel computed bit-identically to the training cache:
        CPU fp32 STFT -> float16 round-trip -> model dtype.

        (GPU STFT differs from CPU STFT in the last ulps; for a saturated
        model that is enough to flip argmaxes — see REPORT_10H bug log.)
        """
        import numpy as np

        fm = LogMelFrontend(sample_rate=16000, n_fft=2048, hop_length=160, n_mels=512)
        with torch.no_grad():
            mel = fm(wav.cpu()).squeeze(0)  # (F, 512) fp32
        mel = torch.from_numpy(mel.numpy().astype(np.float16).astype(np.float32))
        return mel.unsqueeze(0).to(self.device).to(next(self.model.parameters()).dtype)

    def _audio_chunk_emb(self, mel_chunk, conv_carry):
        """Frontend with causal carry. Returns (emb (1,500,C), new_carry)."""
        model = self.model
        if conv_carry is None:
            x = model.audio_front(mel_chunk)
        else:
            mel_ext = torch.cat([conv_carry, mel_chunk], dim=1)
            x = model.audio_front(mel_ext)[:, model.cfg.audio_conv_k - 1 :]
        new_carry = mel_chunk[:, -(model.cfg.audio_conv_k - 1) :]
        return x, new_carry

    def _blocks_parallel(self, x, state, lead=None):
        """DEPRECATED - use model.process_audio_parallel (kept for tests)."""
        return self.model.process_audio_parallel(x, state, lead=lead)[:2]

    @torch.no_grad()
    def transcribe_wav(self, wav: torch.Tensor, mode: str = "continuous", chunk_sec: float = 5.0):
        """wav: (1, N) float32 16k mono. Returns (per-chunk token lists, notes, stats).

        Streaming protocol (exact, Gate-2-verified):
          * frontend conv carry (2 mel frames),
          * per-layer shift-carry buffers (att/ffn x_prev) carried in RWKVState,
          * 1-frame shift lead fed to the chunk's block pass = the embedding
            of the previous token (last MIDI token of the previous chunk; for
            chunk 0 none), which is exactly the parallel-training semantics.
        """
        assert mode in ("continuous", "reset")
        mel = self._frontend(wav)
        mel = mel.to(next(self.model.parameters()).dtype).to(self.device)  # (1, F, 512)
        F = mel.shape[1]
        n_chunks = (F + CHUNK_FRAMES - 1) // CHUNK_FRAMES
        stats = {"chunks": n_chunks, "truncated": 0, "tokens": 0, "boundary_errors": 0}
        model = self.model
        B = 1
        state = model.initial_state(B, self.device)
        conv_carry = None
        prev_emb = None  # embedding of the previous token (lead-in frame)
        dec = MT3Decoder(frame_rate=AUDIO_FRAME_RATE)
        all_tokens = []
        for c in range(n_chunks):
            if mode == "reset":
                state = model.initial_state(B, self.device)
                conv_carry = None
                prev_emb = None
            seg = mel[:, c * CHUNK_FRAMES : (c + 1) * CHUNK_FRAMES]
            if seg.shape[1] < CHUNK_FRAMES:
                seg = torch.nn.functional.pad(seg, (0, 0, 0, CHUNK_FRAMES - seg.shape[1]))
            x, conv_carry = self._audio_chunk_emb(seg, conv_carry)
            h, state, vf_last, att_prevs, ffn_prevs = model.process_audio_parallel(x, state, lead=prev_emb)
            # greedy decode from the last audio hidden; the first step is
            # seeded with the per-layer shift-carry buffers at the last frame
            logits = model.head(h[:, -1])
            chunk_tokens = []
            seed = (vf_last, att_prevs, ffn_prevs)
            for _ in range(self.max_tokens_per_chunk):
                tok = int(logits.argmax(-1).item())
                chunk_tokens.append(tok)
                if tok == EOS_ID:
                    break
                xt = model.emb(torch.tensor([[tok]], device=self.device)).squeeze(1)
                logits, state, _ = model.forward_rnn_step(xt, state, seed)
                seed = None  # only the audio->MIDI transition is seeded
            else:
                stats["truncated"] += 1
                chunk_tokens = chunk_tokens[: self.max_tokens_per_chunk - 1] + [EOS_ID]
            all_tokens.append(chunk_tokens)
            stats["tokens"] += len(chunk_tokens)
            # the next chunk's shift lead = this chunk's last MIDI token
            last_tok = chunk_tokens[-1]
            prev_emb = model.emb(torch.tensor([[last_tok]], device=self.device))
            # decode
            dec.begin_chunk(c * chunk_sec, (c + 1) * chunk_sec)
            for t in chunk_tokens:
                if t == EOS_ID:
                    break
                dec.feed(t)
        dec.finish()
        stats["boundary_errors"] = dec.boundary_errors
        return all_tokens, dec.notes, stats


def load_model_ckpt(path: str, device="cuda") -> MuRWKV:
    from ..model.murwkv_model import MuRWKVConfig

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = MuRWKVConfig(**{k: v for k, v in ckpt["args"].items() if k in ("n_layer", "n_embd")})
    model = MuRWKV(cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device)


def transcript_to_midi(notes: list[Note], out_path: str, tid: str, model_path: str, mode: str, metrics: dict):
    tokens_to_midi(notes, out_path)
    meta = {"track": tid, "checkpoint": model_path, "mode": mode, "metrics": metrics}
    with open(out_path.replace(".mid", ".metadata.json"), "w") as f:
        import json

        json.dump(meta, f, indent=2)
    return out_path