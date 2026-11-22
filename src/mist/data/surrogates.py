"""Signal-level interventions used to measure what a model relies on.

All functions take and return float tensors [..., T] and are model-agnostic, so
the same audit runs on MIST-GRM and on every baseline.

  global_phase    keeps the epoch's amplitude spectrum, destroys waveform shape
                  and within-epoch timing
  local_phase     keeps the short-time amplitude spectrum (coarse time-frequency
                  structure), destroys waveform shape only
  flatten_tilt    keeps phase and narrow-band peaks, removes the broadband 1/f
                  tilt (a simplified aperiodic control, not a specparam fit)
"""
from __future__ import annotations

import math

import torch


def _rand_phase(shape, generator, device):
    theta = torch.rand(shape, generator=generator, device=device) * (2 * math.pi)
    return torch.polar(torch.ones_like(theta), theta)


def global_phase(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    n = x.shape[-1]
    X = torch.fft.rfft(x, dim=-1)
    ph = _rand_phase(X.shape, generator, x.device)
    ph[..., 0] = 1.0
    if n % 2 == 0:
        ph[..., -1] = 1.0
    return torch.fft.irfft(X.abs() * ph, n=n, dim=-1)


def local_phase(x: torch.Tensor, win: int = 200, generator: torch.Generator | None = None) -> torch.Tensor:
    """Phase-randomise each short-time frame (sqrt-Hann, 50% overlap) and overlap-add."""
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    w = torch.hann_window(win, periodic=True, device=x.device).sqrt()
    S = torch.stft(flat, n_fft=win, hop_length=win // 2, window=w, center=True, return_complex=True)
    ph = _rand_phase(S.shape, generator, x.device)
    ph[:, 0] = 1.0
    ph[:, -1] = 1.0
    y = torch.istft(S.abs() * ph, n_fft=win, hop_length=win // 2, window=w, center=True, length=shape[-1])
    return y.reshape(shape)


def flatten_tilt(x: torch.Tensor, fs: int = 100, f_lo: float = 1.0, f_hi: float = 35.0) -> torch.Tensor:
    """Divide out a power-law fit of the amplitude spectrum; phase is untouched."""
    n = x.shape[-1]
    X = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(n, d=1.0 / fs).to(x.device)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    lf = torch.log(freqs[band])
    la = torch.log(X.abs()[..., band] + 1e-8)
    lf_c = lf - lf.mean()
    slope = (la * lf_c).sum(-1, keepdim=True) / (lf_c ** 2).sum()
    gain = torch.ones_like(X.real)
    gain[..., band] = torch.exp(-slope * lf_c)
    y = torch.fft.irfft(X * gain, n=n, dim=-1)
    scale = torch.sqrt((x ** 2).mean(-1, keepdim=True) / ((y ** 2).mean(-1, keepdim=True) + 1e-12))
    return y * scale


CONDITIONS = {
    "real": lambda x, g=None: x,
    "global_phase": global_phase,
    "local_phase": lambda x, g=None: local_phase(x, generator=g),
    "flatten_tilt": lambda x, g=None: flatten_tilt(x),
}
