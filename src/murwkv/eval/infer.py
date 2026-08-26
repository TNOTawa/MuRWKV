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


class Transcriber:
    def __init__(self, model: MuRWKV, device="cuda", max_tokens_per_chunk: int = DEFAULT_MAX_TOKENS_PER_CHUNK):
        self.model = model.eval()
        self.device = device
        self.max_tokens_per_chunk = max_tokens_per_chunk

    def _frontend(self, wav: torch.Tensor) -> torch.Tensor:
        fm = LogMelFrontend(sample_rate=16000, n_fft=2048, hop_length=160, n_mels=512).to(wav.device)
        return fm(wav)  # (1, F, 512)

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
        """Run blocks over x (1,T,C) with optional 1-frame shift lead.
        Returns (h (1,T,C) cropped, state updated, v_first_last (1,C))."""
        model = self.model
        if lead is not None:
            x_full = torch.cat([lead, x], dim=1)
            crop = 1
        else:
            x_full = x
            crop = 0
        v_first = None
        for i, blk in enumerate(model.blocks):
            init = state.S[i] if state is not None else None
            x_full, v_first = blk.forward_parallel(x_full, v_first, init_state=init)
            if state is not None:
                state.S[i] = blk._last_state
        h = model.ln_out(x_full)
        return h[:, crop:], state, v_first[:, -1] if v_first is not None else None

    @torch.no_grad()
    def transcribe_wav(self, wav: torch.Tensor, mode: str = "continuous", chunk_sec: float = 5.0):
        """wav: (1, N) float32 16k mono. Returns (per-chunk token lists, notes, stats)."""
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
        prev_emb = None
        dec = MT3Decoder(frame_rate=AUDIO_FRAME_RATE)
        all_tokens = []
        v_first = None
        for c in range(n_chunks):
            if mode == "reset":
                state = model.initial_state(B, self.device)
                conv_carry = None
                prev_emb = None
                v_first = None
            seg = mel[:, c * CHUNK_FRAMES : (c + 1) * CHUNK_FRAMES]
            if seg.shape[1] < CHUNK_FRAMES:
                seg = torch.nn.functional.pad(seg, (0, 0, 0, CHUNK_FRAMES - seg.shape[1]))
            x, conv_carry = self._audio_chunk_emb(seg, conv_carry)
            h, state, v_first = self._blocks_parallel(x, state, lead=prev_emb)
            prev_emb = x[:, -1:]
            # greedy decode from last audio hidden
            logits = model.head(h[:, -1])
            chunk_tokens = []
            vf = v_first
            for _ in range(self.max_tokens_per_chunk):
                tok = int(logits.argmax(-1).item())
                chunk_tokens.append(tok)
                if tok == EOS_ID:
                    break
                xt = model.emb(torch.tensor([[tok]], device=self.device)).squeeze(1)
                logits, state, vf = model.forward_rnn_step(xt, state, vf)
            else:
                stats["truncated"] += 1
                chunk_tokens = chunk_tokens[: self.max_tokens_per_chunk - 1] + [EOS_ID]
            all_tokens.append(chunk_tokens)
            stats["tokens"] += len(chunk_tokens)
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