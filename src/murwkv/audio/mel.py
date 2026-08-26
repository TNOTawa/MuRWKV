"""Log-Mel spectrogram frontend (pure torch, MuScriptor-compatible).

Params (matching MuScriptor + task spec): mono 16 kHz, n_fft=2048,
hop=160 (100 fps), 512 HTK-mel bins, power=2, center=True reflect pad,
log(x + 1e-6) with natural log.
"""
from __future__ import annotations

import torch
from torch import nn


def _hz_to_mel_htk(freq):
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def _mel_to_hz_htk(mel):
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate):
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_min = _hz_to_mel_htk(torch.tensor(float(f_min)))
    m_max = _hz_to_mel_htk(torch.tensor(float(f_max)))
    m_pts = torch.linspace(m_min.item(), m_max.item(), n_mels + 2)
    f_pts = _mel_to_hz_htk(m_pts)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    fb = torch.maximum(torch.zeros(()), torch.minimum(down, up))
    return fb  # [n_freqs, n_mels]


class LogMelFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 160,
        n_mels: int = 512,
        eps: float = 1e-6,
        center: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.eps = eps
        self.center = center
        self.register_buffer("window", torch.hann_window(n_fft))
        fb = melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )
        self.register_buffer("fb", fb)

    def frames_per_second(self) -> float:
        return self.sample_rate / self.hop_length

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T_samples) float in [-1, 1] -> (B, F, n_mels) log-mel."""
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=self.center,
            pad_mode="reflect",
            return_complex=True,
            normalized=False,
            onesided=True,
        )  # (B, n_fft/2+1, F)
        power = spec.abs() ** 2
        mel = torch.einsum("btf,fm->btm", power.transpose(1, 2), self.fb)
        return torch.log(mel + self.eps)


def compute_log_mel(wav: torch.Tensor, frontend: LogMelFrontend) -> torch.Tensor:
    return frontend(wav)