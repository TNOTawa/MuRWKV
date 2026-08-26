"""MuRWKV: unified pure-RWKV Audio→MIDI sequence model.

Sequence layout per 5s chunk (unified recurrent stream, state never reset):

    [audio log-mel frames 1..500] [MIDI tokens 1..M (incl. EOS)] [next audio...]

Loss (CE) is computed ONLY at "MIDI prediction positions":
    * the last audio frame of each chunk predicts that chunk's MIDI token 0,
    * each MIDI position i predicts MIDI token i+1 (through and including EOS).
All other audio-frame positions and batch-PAD positions are loss-masked.

Architecture (R0, ~24M params):
    log-Mel(512) -> Linear(512) -> GELU -> causal Conv1d(k=3) -> GELU -> Linear -> C
    MIDI token -> Embedding(vocab 1393, C)
    6 x RWKV-7 x070 Blocks (PreLN), head_size 64, dim_att = dim_ffn = 512
    ln_out -> Linear(C, vocab)

State: RWKVState (per-layer S: (B,H,N,N) fp32). Both continuous inference
(state carried across chunks) and reset inference (state zeroed per chunk)
are first-class; see infer.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from .rwkv7 import CHUNK_LEN, Block, RWKVState

AUDIO_FRAME_RATE = 100
CHUNK_SECONDS = 5.0
CHUNK_FRAMES = int(CHUNK_SECONDS * AUDIO_FRAME_RATE)  # 500

PAD_ID = 0


@dataclass
class MuRWKVConfig:
    n_layer: int = 6
    n_embd: int = 512
    head_size: int = 64
    vocab_size: int = 1393
    n_mels: int = 512
    audio_conv_k: int = 3
    dropout: float = 0.0

    @property
    def dim_att(self):
        return self.n_embd

    @property
    def dim_ffn(self):
        # official RWKV default: 3.5x emb size, rounded to /32.
        # R0: 6x512 with 1x would be ~13.5M; 3.5x gives ~21M (task target 20-30M).
        return int((self.n_embd * 3.5) // 32 * 32)


class CausalConv1d(nn.Module):
    """Causal 1D conv over the time axis: no future leakage."""

    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=k - 1)

    def forward(self, x):  # x: (B, T, C)
        B, T, C = x.shape
        x = self.conv(x.transpose(1, 2))[:, :, :T]
        return x.transpose(1, 2)

    def stream(self, x_chunk, carry):
        """Causal streaming: x_chunk (B, Tchunk, C), carry (B, k-1, C).
        Returns (out (B,Tchunk,C), new_carry)."""
        B, T, C = x_chunk.shape
        x = torch.cat([carry, x_chunk], dim=1).transpose(1, 2)  # (B, C, T+k-1)
        out = self.conv(x)  # (B, C, T+k-1) valid start
        out = out.transpose(1, 2)
        out = out[:, self.k - 1 :, :]  # causal: first output needs k-1 past frames
        new_carry = x_chunk[:, -(self.k - 1) :, :]
        return out, new_carry


class AudioFrontend(nn.Module):
    """log-Mel -> Linear -> GELU -> causal conv -> GELU -> Linear -> n_embd."""

    def __init__(self, n_mels: int, n_embd: int, k: int = 3):
        super().__init__()
        self.k = k
        self.linear1 = nn.Linear(n_mels, n_embd)
        self.conv = CausalConv1d(n_embd, n_embd, k)
        self.linear2 = nn.Linear(n_embd, n_embd)
        with torch.no_grad():
            self.linear1.weight.data.uniform_(-0.5 / (n_mels**0.5), 0.5 / (n_mels**0.5))
            self.conv.conv.weight.data.uniform_(-0.5 / (n_embd**0.5), 0.5 / (n_embd**0.5))
            self.conv.conv.bias.data.zero_()
            self.linear2.weight.data.zero_()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:  # (B, F, n_mels) -> (B, F, n_embd)
        x = F.gelu(self.linear1(mel))
        x = self.conv(x)
        x = F.gelu(x)
        return self.linear2(x)


class MuRWKV(nn.Module):
    def __init__(self, cfg: MuRWKVConfig):
        super().__init__()
        self.cfg = cfg
        self.audio_front = AudioFrontend(cfg.n_mels, cfg.n_embd, cfg.audio_conv_k)
        self.emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.ln_out = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.uniform_(self.emb.weight, a=-1e-4, b=1e-4)
            if self.cfg.vocab_size > self.cfg.n_embd:
                scale = 0.5 * math.sqrt(self.cfg.vocab_size / self.cfg.n_embd)
            else:
                scale = 0.5
            nn.init.orthogonal_(self.head.weight, gain=scale)
            for i, block in enumerate(self.blocks):
                block.att.ln_x.weight.data.fill_(((1 + i) / self.cfg.n_layer) ** 0.7)

    # ------------------------------------------------------------------
    # GPT-mode: full teacher-forced sequence (training / eval)
    # ------------------------------------------------------------------

    def embed_plan(self, mel: torch.Tensor, is_audio: torch.Tensor, midi_id: torch.Tensor):
        """Assemble (B, L, C) from the flat plan.

        mel: (B, U*500, n_mels) full audio frames in song order.
        is_audio: (B, L) bool; audio_idx implicit = cumsum of is_audio.
        midi_id: (B, L) int (valid where not is_audio; PAD beyond real length).
        """
        B, L = is_audio.shape
        mel_emb = self.audio_front(mel)  # (B, U*500, C)
        # map flat positions to mel rows
        audio_cum = torch.cumsum(is_audio.long(), dim=1) - 1
        audio_emb = torch.gather(
            mel_emb, 1, audio_cum.clamp(min=0).unsqueeze(-1).expand(B, L, mel_emb.shape[-1])
        )
        midi_emb = self.emb(midi_id)
        x = torch.where(is_audio.unsqueeze(-1), audio_emb, midi_emb)
        return x

    def forward_gpt(self, mel, is_audio, midi_id, use_cuda_kernel=False):
        B, L = is_audio.shape
        x = self.embed_plan(mel, is_audio, midi_id)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block.forward_parallel(x, v_first, use_cuda_kernel)
        x = self.ln_out(x)
        return self.head(x)

    def build_targets(self, is_audio: torch.Tensor, midi_id: torch.Tensor, unit_midi_lens: list):
        """Loss plan: for each unit, positions [audio_end-1 .. audio_end+M-2]
        predict the next flat token (the unit's MIDI tokens, shifted by one).

        unit_midi_lens: per-row list of per-unit midi token counts, or a single
        list shared by all rows (B=1 convenience).

        Returns (targets (B, L) int64, mask (B, L) bool).
        """
        B, L = is_audio.shape
        per_row = unit_midi_lens
        if per_row and isinstance(per_row[0], int):
            per_row = [per_row] * B
        assert len(per_row) == B, f"{len(per_row)} != {B}"
        targets = torch.zeros(B, L, dtype=torch.long, device=is_audio.device)
        mask = torch.zeros(B, L, dtype=torch.bool, device=is_audio.device)
        for b in range(B):
            pos = 0
            for M in per_row[b]:
                if M <= 0:
                    continue
                # this unit's audio spans [pos, pos + CHUNK_FRAMES)
                E = pos + CHUNK_FRAMES - 1  # flat index of last audio frame
                lo, hi = E, E + M - 1  # loss positions [lo, hi)  (hi exclusive)
                seg = midi_id[b, lo + 1 : hi + 1]
                targets[b, lo:hi] = seg
                mask[b, lo:hi] = seg != PAD_ID
                pos += CHUNK_FRAMES + M
        return targets, mask

    def loss_and_metrics(self, logits, targets, mask):
        """Cross-entropy at masked positions + token accuracy."""
        logits = logits.float()
        flat_lg = logits.reshape(-1, logits.shape[-1])
        flat_tg = targets.reshape(-1)
        flat_ms = mask.reshape(-1)
        n = flat_ms.sum()
        if n == 0:
            return torch.tensor(0.0, device=logits.device), 0, torch.tensor(0.0, device=logits.device)
        nll = F.cross_entropy(flat_lg, flat_tg, reduction="none")
        loss = (nll * flat_ms).sum() / n
        acc = ((flat_lg.argmax(-1) == flat_tg) * flat_ms).sum() / n
        return loss, int(n), acc

    # ------------------------------------------------------------------
    # RNN-mode inference (true recurrent streaming)
    # ------------------------------------------------------------------

    def initial_state(self, B: int, device) -> RWKVState:
        return RWKVState.zeros(self.cfg.n_layer, B, self.cfg.n_embd // self.cfg.head_size, self.cfg.head_size, device)

    def forward_rnn_step(self, x_t: torch.Tensor, state: RWKVState, v_first: torch.Tensor | None = None):
        """One MIDI token: x_t (B, C) embedding. Returns (logits (B, vocab), new_state, v_first)."""
        v_first = v_first if v_first is not None else torch.zeros_like(x_t)
        for i, block in enumerate(self.blocks):
            x_t, state.S[i], v_first = block.forward_step(x_t, v_first, state.S[i])
        x_t = self.ln_out(x_t)
        return self.head(x_t), state, v_first


def build_model(cfg: MuRWKVConfig | None = None) -> MuRWKV:
    cfg = cfg or MuRWKVConfig()
    return MuRWKV(cfg)


def count_params(model: MuRWKV) -> int:
    return sum(p.numel() for p in model.parameters())