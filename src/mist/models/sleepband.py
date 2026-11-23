"""SleepBand baseline (T14): independent re-implementation from the paper text only.

Lu, Hu, Chen, "SleepBand: Single-Source Domain Generalization for Sleep Staging via Physiologically
Structured Spectral Modeling", arXiv:2607.04851 (2026). The authors' repository states no licence, so
nothing was copied from it and it was not read (DECISION_LOG D-023, OPEN_QUESTIONS Q11).

What the paper specifies and this module follows:
  * a learnable Gabor filterbank per channel; centre frequencies initialised on a log scale; constant-Q
    width sigma(eta) = q / (2 pi eta);
  * band responses z = |x * psi| (magnitude);
  * spectral integration: a learnable temporal convolution per band, linear mixing across bands, gates in (0, 1);
  * temporal averaging to a 512-d epoch feature from a lightweight epoch encoder;
  * one multi-head self-attention layer over the 20-epoch sequence with positional embeddings; linear classifier.

Choices the paper leaves open (made here without data; listed so they can be checked against the paper):
  * single EEG channel (this project's input), not EEG + EOG;
  * input scaled by the night's robust standard deviation, like every model here, instead of z-scoring;
  * 20 filters (the paper's multi-source default), kernel 257 samples (2.57 s), q learnable and shared,
    initialised at 5; centre frequencies initialised log-spaced over 0.5-35 Hz and kept inside it;
  * per-band temporal convolution kernel 25, average pooling by 4, mixing to 128 gated channels, then two
    stride-2 convolution blocks to 256 and 512 channels;
  * attention: 8 heads, model width 512, dropout 0.1.
The paper's training scheme (mean-teacher consistency on bandwise Fourier perturbations) is not part of the
model and is not implemented here (D-023).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FS = 100


class GaborBank(nn.Module):
    """Complex Gabor filters with learnable centre frequency and a shared constant-Q factor."""

    def __init__(self, n_filters: int = 20, kernel: int = 257, f_lo: float = 0.5, f_hi: float = 35.0, q: float = 5.0):
        super().__init__()
        self.f_lo, self.f_hi = f_lo, f_hi
        eta0 = torch.logspace(math.log10(f_lo), math.log10(f_hi), n_filters)
        self.eta_raw = nn.Parameter(torch.logit((eta0 - f_lo) / (f_hi - f_lo) * 0.98 + 0.01))
        self.log_q = nn.Parameter(torch.tensor(math.log(q)))
        self.register_buffer("t", (torch.arange(kernel) - kernel // 2).float() / FS, persistent=False)

    @property
    def eta(self) -> torch.Tensor:                              # Hz, kept inside [f_lo, f_hi]
        return self.f_lo + (self.f_hi - self.f_lo) * torch.sigmoid(self.eta_raw)

    def kernels(self) -> torch.Tensor:                          # [K, kernel] complex
        eta = self.eta[:, None]
        sigma = self.log_q.exp() / (2 * math.pi * eta)          # seconds; constant Q
        env = torch.exp(-0.5 * (self.t / sigma) ** 2)
        env = env / env.sum(-1, keepdim=True)
        return env * torch.exp(2j * math.pi * eta * self.t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # [N, T] -> |responses| [N, K, T]
        k = self.kernels()
        w = torch.cat([k.real, k.imag])[:, None]               # [2K, 1, kernel]
        y = F.conv1d(x[:, None], w, padding=w.shape[-1] // 2)
        re, im = y.chunk(2, 1)
        return torch.sqrt(re ** 2 + im ** 2 + 1e-8)


class SleepBand(nn.Module):
    def __init__(self, n_filters: int = 20, d_epoch: int = 512, seq_len: int = 20, n_heads: int = 8,
                 dropout: float = 0.1, n_classes: int = 5):
        super().__init__()
        K = n_filters
        self.bank = GaborBank(K)
        self.band_conv = nn.Conv1d(K, K, 25, padding=12, groups=K)          # one temporal filter per band
        self.mix = nn.Conv1d(K, 128, 1)                                      # linear mixing across bands
        self.gate_logit = nn.Parameter(torch.zeros(128))                     # gates in (0, 1)

        def block(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 7, 2, 3), nn.BatchNorm1d(o), nn.GELU())
        self.enc = nn.Sequential(block(128, 256), block(256, d_epoch))
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_epoch))
        self.attn = nn.TransformerEncoderLayer(d_epoch, n_heads, 2 * d_epoch, dropout, batch_first=True, norm_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_epoch, n_classes)

    def epoch_features(self, x: torch.Tensor) -> torch.Tensor:  # [N, T] -> [N, d_epoch]
        z = torch.log1p(self.bank(x))                           # magnitude, compressed
        z = F.avg_pool1d(F.gelu(self.band_conv(z)), 4)
        z = self.mix(z) * torch.sigmoid(self.gate_logit)[None, :, None]
        return self.enc(z).mean(-1)

    def forward(self, x, stats, **_):
        b, l, t = x.shape
        x = x / stats["night"]["scale"].view(b, 1, 1)
        h = self.epoch_features(x.reshape(b * l, t)).view(b, l, -1)
        h = self.attn(h + self.pos[:, :l])
        return self.head(self.drop(h)), {"losses": {}, "assign": []}

    def aux_loss(self, aux):
        return 0.0
