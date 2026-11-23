"""MIST-GRM model.

Modules (docs/ARCHITECTURE.md has the full specification):
  M0 night whitening, M1 tokeniser, M2 factoriser -> models/factorizer.py (no parameters)
  M3 stream encoders            -> GainEmbed, RhythmEncoder, WaveEncoder, ShapeCodebook
  M4 epoch composer             -> ScaleBranch (one per token scale)
  M5 sequence context           -> MistGRM.context
  M6 heads                      -> MistGRM.head, ScaleBranch.critic

Streams are selected with `streams`, a subset of {"G", "R", "M"}, or {"X"} for
the unfactorised raw-token control. The morphology dial has three settings:
  destroy_prob  probability that a token's phase is replaced by random phase
                (1.0 turns the M stream into a capacity-matched placebo);
  critic_weight weight of the real-vs-surrogate critic on the M embedding;
  allpass_rad   strength of the random phase-response augmentation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import N_CLASSES
from .factorizer import Factorizer, Factors, gain_rhythm, whiten


@dataclass
class MistConfig:
    streams: tuple = ("G", "R", "M")
    scales: tuple = ((1.0, 0.5), (4.0, 2.0))      # (window s, hop s)
    f_lo: float = 0.5
    f_hi: float = 35.0
    morph_input: str = "phase_only"               # or "unit_raw"
    night_relative: bool = True                   # False: no whitening, absolute log magnitudes
    n_slots: int = 32
    slot_temp: float = 0.1
    morph_residual: int = 8                       # capacity cap on non-prototype M features
    d_gain: int = 16
    d_stream: int = 64
    d_token: int = 96
    d_epoch: int = 128
    token_layers: int = 1
    context_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    stream_dropout: float = 0.1
    seq_len: int = 20
    epoch_samples: int = 3000
    fs: int = 100
    # morphology dial
    destroy_prob: float = 0.0
    critic_weight: float = 0.0
    critic_frac: float = 0.25
    allpass_rad: float = 0.0
    diversity_weight: float = 0.01
    critic_stages: tuple | None = None            # T9: critic only on tokens of epochs with these training labels
    # topology (ARCHITECTURE section 8): "parallel" = v1 (G, R, M each reach the composer);
    # "cascade" = rhythm gates per-band morphology, only G and gated M reach the composer
    topology: str = "parallel"
    cascade_channels: str = "all"                 # all = 5 bands + broadband | bands | broad
    band_slots: int = 16                          # shape slots per cascade channel
    # revision 3 (ARCHITECTURE section 9). Defaults reproduce the section 8 cascade exactly.
    band_rate: str = "full"                       # full: every channel at the token length | own: each at its own length
    band_scales: tuple | None = None              # token lengths (s) that carry band channels; None = every scale
    band_width: int | None = None                 # None: one WaveEncoder for all channels; int: second encoder of this width for the bands
    encoder_checkpoint: bool = True               # D-013 activation checkpointing of the shape encoder(s)
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------- encoders
class GainEmbed(nn.Module):
    def __init__(self, d: int, n_freq: int = 8):
        super().__init__()
        self.register_buffer("k", (2.0 ** torch.arange(n_freq)) * math.pi, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq + 1, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        u = torch.sigmoid(g)                      # G is a mean of robust z-scores; squash to (0, 1)
        ang = u.unsqueeze(-1) * self.k
        return self.mlp(torch.cat([u.unsqueeze(-1), ang.sin(), ang.cos()], -1))


class RhythmEncoder(nn.Module):
    """Position-specific MLP over frequency bins: alpha and sigma must stay distinct."""

    def __init__(self, n_freq: int, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_freq, 2 * d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(2 * d, d), nn.GELU())

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)


class WaveEncoder(nn.Module):
    """1-D CNN over one time-domain token with global max+mean pooling.

    Pooling makes the code insensitive to where inside the token an event sits.
    """

    def __init__(self, d: int, width: int = 32):
        super().__init__()
        def block(i, o, k, s):
            return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2), nn.GroupNorm(4, o), nn.GELU())
        self.conv = nn.Sequential(block(1, width, 9, 1), block(width, width, 9, 2),
                                  block(width, 2 * width, 7, 2), block(2 * width, 2 * width, 5, 2))
        self.out = nn.Linear(4 * width, d)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:  # [M, win] -> [M, d]
        h = self.conv(tok.unsqueeze(1))
        return self.out(torch.cat([h.amax(-1), h.mean(-1)], -1))


class ShapeCodebook(nn.Module):
    """Cosine prototypes over morphology embeddings.

    With phase-only input every prototype is, by construction, a shape prototype:
    it cannot encode band power or amplitude.
    """

    def __init__(self, n_slots: int, d: int, temp: float):
        super().__init__()
        self.proto = nn.Parameter(torch.randn(n_slots, d) * 0.1)
        self.temp = temp

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        sim = F.normalize(m, dim=-1) @ F.normalize(self.proto, dim=-1).t()
        return F.softmax(sim / self.temp, dim=-1)

    def diversity(self) -> torch.Tensor:
        p = F.normalize(self.proto, dim=-1)
        c = p @ p.t()
        off = c - torch.diag(torch.diag(c))
        return (off ** 2).sum() / (c.numel() - c.shape[0])


class AttnPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # [M, N, d]
        w = F.softmax(self.q(h), dim=1)
        return (w * h).sum(1)


# ------------------------------------------------------------ one token scale
class ScaleBranch(nn.Module):
    def __init__(self, cfg: MistConfig, win_s: float, hop_s: float):
        super().__init__()
        self.cfg = cfg
        self.fz = Factorizer(cfg.fs, win_s, hop_s, cfg.f_lo, cfg.f_hi)
        self.n_tok = self.fz.n_tokens(cfg.epoch_samples)
        s = set(cfg.streams)
        self.use = {k: (k in s) for k in "GRMX"}
        d_in = 0
        if self.use["G"]:
            self.gain = GainEmbed(cfg.d_gain); d_in += cfg.d_gain
        if self.use["R"]:
            self.rhythm = RhythmEncoder(self.fz.n_freq, cfg.d_stream, cfg.dropout); d_in += cfg.d_stream
        if self.use["M"]:
            self.morph = WaveEncoder(cfg.d_stream)
            self.codebook = ShapeCodebook(cfg.n_slots, cfg.d_stream, cfg.slot_temp)
            self.resid = nn.Linear(cfg.d_stream, cfg.morph_residual) if cfg.morph_residual > 0 else None
            self.critic = nn.Linear(cfg.d_stream, 1)
            d_in += cfg.n_slots + cfg.morph_residual
        if self.use["X"]:
            self.raw = WaveEncoder(cfg.d_stream); d_in += cfg.d_stream
        self.proj = nn.Linear(d_in, cfg.d_token)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tok, cfg.d_token))
        layer = nn.TransformerEncoderLayer(cfg.d_token, cfg.n_heads, 2 * cfg.d_token, cfg.dropout,
                                           batch_first=True, norm_first=True)
        self.tok_enc = nn.TransformerEncoder(layer, cfg.token_layers, enable_nested_tensor=False)
        self.pool = AttnPool(cfg.d_token)

    def forward(self, xw, xraw, stats, knock: frozenset, destroy_prob: float, gen=None,
                knock_slots: frozenset = frozenset(), y=None):
        """xw: whitened [B, L, T]; xraw: per-night scaled raw [B, L, T]; stats: mag_med, mag_iqr [B, F].

        knock_slots: shape-slot indices whose assignments are zeroed (the rest renormalised;
        knocking every slot leaves only the residual). y: training labels [B, L], used only to
        restrict the critic to `critic_stages`."""
        cfg, fz = self.cfg, self.fz
        B, L, _ = xw.shape
        f = fz.factorize(xw)                                  # leading dims [B, L, N]
        N = f.log_mag.shape[-2]
        parts, aux = [], {}

        def gate(name: str, h: torch.Tensor) -> torch.Tensor:
            if name in knock:
                return torch.zeros_like(h)
            if self.training and cfg.stream_dropout > 0 and sum(self.use.values()) > 1:
                keep = (torch.rand(B, 1, 1, 1, device=h.device) >= cfg.stream_dropout).to(h.dtype)
                return h * keep
            return h

        if self.use["G"] or self.use["R"]:
            rel = cfg.night_relative
            g, r = gain_rhythm(f.log_mag, stats["mag_med"] if rel else None, stats["mag_iqr"] if rel else None)
            if self.use["G"]:
                parts.append(gate("G", self.gain(g)))
            if self.use["R"]:
                parts.append(gate("R", self.rhythm(r)))
        if self.use["M"]:
            ph = f.phasor
            if self.training and cfg.allpass_rad > 0:
                ph = fz.allpass(ph, cfg.allpass_rad, gen)
            real = fz.morph_token(Factors(f.log_mag, ph), cfg.morph_input)
            tok = real
            if destroy_prob > 0:
                fake = fz.morph_token(Factors(f.log_mag, fz.random_phasor(ph, gen)), cfg.morph_input)
                kill = torch.rand(B, L, N, 1, device=xw.device, generator=gen) < destroy_prob
                tok = torch.where(kill, fake, real)
            m = self.morph(tok.reshape(-1, fz.win)).view(B, L, N, -1)
            a = self.codebook(m)
            if knock_slots:
                keep = torch.ones(a.shape[-1], device=a.device, dtype=a.dtype)
                keep[list(knock_slots)] = 0
                a = a * keep
                tot = a.sum(-1, keepdim=True)
                a = torch.where(tot > 0, a / tot.clamp_min(1e-12), torch.zeros_like(a))
            aux["assign"] = a
            mp = [a] + ([self.resid(m)] if self.resid is not None else [])
            parts.append(gate("M", torch.cat(mp, -1)))
            if self.training and cfg.critic_weight > 0:
                aux["critic"] = self._critic_loss(f, ph, real, gen, self._critic_mask(y, B, L, N))
        if self.use["X"]:
            raw = fz.tokens(xraw) * fz.window                 # conventional input: scaled raw waveform tokens
            parts.append(gate("X", self.raw(raw.reshape(-1, fz.win)).view(B, L, N, -1)))

        h = self.proj(torch.cat(parts, -1)).view(B * L, N, -1) + self.pos
        return self.pool(self.tok_enc(h)).view(B, L, -1), aux

    def _critic_mask(self, y, B, L, N):
        """Token mask [B*L*N] for critic_stages (None = all tokens)."""
        if self.cfg.critic_stages is None:
            return None
        if y is None:
            raise ValueError("critic_stages needs the training labels y")
        ok = torch.zeros_like(y, dtype=torch.bool)
        for st in self.cfg.critic_stages:
            ok |= y == st
        return ok.unsqueeze(-1).expand(B, L, N).reshape(-1)

    def _critic_loss(self, f, ph, real, gen, mask=None):
        """Tell real tokens from surrogates with the same magnitude spectrum."""
        fz, cfg = self.fz, self.cfg
        flat_real = real.reshape(-1, fz.win)
        pool = torch.arange(flat_real.shape[0], device=real.device)
        if mask is not None:
            pool = pool[mask.to(real.device)]
            if pool.numel() == 0:
                return real.sum() * 0.0                       # no eligible token in this batch
        n = min(pool.numel(), max(8, int(cfg.critic_frac * flat_real.shape[0])))
        idx = pool[torch.randperm(pool.numel(), device=real.device, generator=gen)[:n]]
        sel = lambda t: t.reshape(-1, t.shape[-1])[idx]
        fake_f = Factors(sel(f.log_mag), fz.random_phasor(sel(ph), gen))
        fake = fz.morph_token(fake_f, cfg.morph_input)
        logits = self.critic(self.morph(torch.cat([flat_real[idx], fake]))).squeeze(-1)
        target = torch.cat([torch.ones(n), torch.zeros(n)]).to(logits)
        return F.binary_cross_entropy_with_logits(logits, target)


# ------------------------------------------------------------------ full model
class MistGRM(nn.Module):
    def __init__(self, cfg: MistConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.topology == "cascade":
            from .cascade import CascadeBranch as Branch
        elif cfg.topology == "parallel":
            Branch = ScaleBranch
        else:
            raise ValueError(f"unknown topology {cfg.topology!r}")
        self.branches = nn.ModuleList([Branch(cfg, w, h) for w, h in cfg.scales])
        self.fuse = nn.Sequential(nn.Linear(cfg.d_token * len(cfg.scales), cfg.d_epoch), nn.GELU(),
                                  nn.Dropout(cfg.dropout))
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_epoch))
        if cfg.context_layers > 0:
            layer = nn.TransformerEncoderLayer(cfg.d_epoch, cfg.n_heads, 2 * cfg.d_epoch, cfg.dropout,
                                               batch_first=True, norm_first=True)
            self.context = nn.TransformerEncoder(layer, cfg.context_layers, enable_nested_tensor=False)
        else:
            self.context = None
        self.head = nn.Linear(cfg.d_epoch, N_CLASSES)

    def stat_keys(self):
        return [f"s{i}" for i in range(len(self.branches))]

    def forward(self, x, stats, knock=(), destroy_prob: float | None = None, gen=None, knock_slots=(), y=None,
                destroy_bands=()):
        """x: [B, L, T] in microvolt. stats: {"night": {...}, "s0": {...}, ...} from data.dataset.

        knock: streams zeroed at inference, e.g. ("M",). destroy_prob overrides
        the config value, e.g. 1.0 at test time measures morphology reliance.
        """
        knock, knock_slots = frozenset(knock), frozenset(knock_slots)
        extra = {}
        if destroy_bands:
            if self.cfg.topology != "cascade":
                raise ValueError("destroy_bands needs the cascade topology")
            extra["destroy_bands"] = tuple(destroy_bands)
        dp = self.cfg.destroy_prob if destroy_prob is None else destroy_prob
        outs, aux = [], {"assign": [], "losses": {}}
        scale = stats["night"]["scale"].view(-1, 1, 1)
        xraw = x / scale
        xw = whiten(x, stats["night"]["envelope"], self.cfg.fs) if self.cfg.night_relative else x / 50.0
        for i, br in enumerate(self.branches):
            h, a = br(xw, xraw, stats[f"s{i}"], knock, dp, gen, knock_slots=knock_slots, y=y, **extra)
            outs.append(h)
            if "assign" in a:
                aux["assign"].append(a["assign"])
            if "critic" in a:
                aux["losses"][f"critic_s{i}"] = a["critic"]
        h = self.fuse(torch.cat(outs, -1))
        if self.context is not None:
            h = self.context(h + self.pos[:, : h.shape[1]])
        aux["epoch_repr"] = h
        if "M" in self.cfg.streams and self.cfg.diversity_weight > 0:
            aux["losses"]["diversity"] = sum(b.codebook.diversity() for b in self.branches) / len(self.branches)
        return self.head(h), aux

    def aux_loss(self, aux) -> torch.Tensor:
        total = 0.0
        for k, v in aux["losses"].items():
            total = total + (self.cfg.critic_weight if k.startswith("critic") else self.cfg.diversity_weight) * v
        return total
