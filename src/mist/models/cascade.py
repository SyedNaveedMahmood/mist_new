"""Cascade topology (docs/ARCHITECTURE.md, section 8): rhythm feeds morphology.

    r     = RhythmEncoder(R)                      phase-blind rhythm code
    m_b   = irfft(exp(i*phi) on band b), unit RMS b = delta..beta, plus broadband m_0
    e_b   = WaveEncoder(m_b) + band_embedding_b   one shared encoder
    a_b   = ShapeCodebook_b(e_b)                  16 shape slots per band
    g_b   = softplus(Linear_b(r))                 positive gate per band, from rhythm only
    u     = sum_b g_b * (P_b a_b + W_b e_b)       W_b: residual capped at morph_residual dims
    token = proj([GainEmbed(G), u])

Only G and the gated morphology reach the composer; rhythm has no additive path.
Power-blindness holds for m_b, e_b and a_b (functions of arg X only), not for u.

Revision 3 (ARCHITECTURE section 9, `band_rate: own`): each channel's phase-only waveform is
synthesised at its own length n = 2*(k_top+1), rounded up to a multiple of 8, minimum 16
(k_top = the channel's highest FFT bin), i.e. the same band-limited waveform on a coarser grid;
band channels can be limited to some token scales (`band_scales`) and given their own, narrower
shared encoder (`band_width`). The defaults reproduce the section 8 model exactly.

Switches: destroy_prob replaces every band's phase by random phase (G and r are
untouched, parameters identical); destroy_bands randomises the phase of the named
bands' bins in every channel (the band channel and the broadband channel), or,
for "broad", the broadband channel's whole input; knock R sets the gates to
their running training mean; knock G / M zero the gain embedding / u.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .factorizer import Factors, gain_rhythm

BANDS = (("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 12.0), ("sigma", 12.0, 16.0),
         ("beta", 16.0, 35.0))
BAND_NAMES = tuple(b[0] for b in BANDS)
CHANNEL_SETS = {"all": BAND_NAMES + ("broad",), "bands": BAND_NAMES, "broad": ("broad",)}


def band_masks(freqs: torch.Tensor) -> torch.Tensor:
    """[5, F] bool. Lower edge inclusive; the last band also keeps its upper edge (35 Hz)."""
    rows = []
    for i, (_, lo, hi) in enumerate(BANDS):
        top = freqs <= hi if i == len(BANDS) - 1 else freqs < hi
        rows.append((freqs >= lo) & top)
    return torch.stack(rows)


def band_spectra(phasor: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """[..., F] phasor, [K, F] masks -> [..., K, F]: the phasor restricted to each band."""
    return phasor.unsqueeze(-2) * masks.to(phasor.real.dtype)


def unit_rms(tok: torch.Tensor) -> torch.Tensor:
    return tok / torch.sqrt((tok ** 2).mean(-1, keepdim=True) + 1e-8)


class BandCodebooks(nn.Module):
    """One cosine shape codebook per channel, stored as a single [C, slots, d] tensor."""

    def __init__(self, n_ch: int, n_slots: int, d: int, temp: float):
        super().__init__()
        self.proto = nn.Parameter(torch.randn(n_ch, n_slots, d) * 0.1)
        self.temp = temp

    def forward(self, e: torch.Tensor) -> torch.Tensor:          # [..., C, d] -> [..., C, slots]
        sim = torch.einsum("...cd,csd->...cs", F.normalize(e, dim=-1), F.normalize(self.proto, dim=-1))
        return F.softmax(sim / self.temp, dim=-1)

    def diversity(self) -> torch.Tensor:
        p = F.normalize(self.proto, dim=-1)
        c = p @ p.transpose(1, 2)
        eye = torch.eye(c.shape[-1], device=c.device, dtype=c.dtype)
        return ((c * (1 - eye)) ** 2).sum((1, 2)).mean() / (c.shape[-1] ** 2 - c.shape[-1])


class CascadeBranch(nn.Module):
    """One token scale of the cascade model. Same interface as mist_grm.ScaleBranch."""

    def __init__(self, cfg, win_s: float, hop_s: float):
        super().__init__()
        from .factorizer import Factorizer
        from .mist_grm import AttnPool, GainEmbed, RhythmEncoder, WaveEncoder

        if set(cfg.streams) != {"G", "R", "M"}:
            raise ValueError("the cascade needs streams G, R and M")
        if cfg.morph_input != "phase_only":
            raise ValueError("the cascade is defined for phase-only morphology")
        self.cfg = cfg
        self.fz = Factorizer(cfg.fs, win_s, hop_s, cfg.f_lo, cfg.f_hi)
        self.n_tok = self.fz.n_tokens(cfg.epoch_samples)
        if cfg.band_rate not in ("full", "own"):
            raise ValueError(f"band_rate must be full or own, not {cfg.band_rate!r}")
        bands_here = cfg.band_scales is None or any(abs(win_s - b) < 1e-9 for b in cfg.band_scales)
        self.channels = tuple(c for c in CHANNEL_SETS[cfg.cascade_channels] if bands_here or c == "broad")
        if not self.channels:
            raise ValueError(f"no morphology channel at token length {win_s} s")
        self.own = cfg.band_rate == "own"
        self.use = {"G": True, "R": True, "M": True, "X": False}
        masks = band_masks(self.fz.freqs)
        if not masks.any(-1).all():
            raise ValueError(f"a band has no bin at token length {win_s} s")
        self.register_buffer("all_masks", masks, persistent=False)
        self.sel = [BAND_NAMES.index(c) for c in self.channels if c != "broad"]
        self.broad = "broad" in self.channels
        C, d, du, ns, R = len(self.channels), cfg.d_stream, cfg.d_stream, cfg.band_slots, cfg.morph_residual
        self.gain = GainEmbed(cfg.d_gain)
        self.rhythm = RhythmEncoder(self.fz.n_freq, d, cfg.dropout)
        self.morph = WaveEncoder(d)                           # broadband (and, if band_width is None, every channel)
        self.morph_band = WaveEncoder(d, cfg.band_width) if cfg.band_width is not None and self.sel else None
        self.ch_len = []                                      # own-rate length of each channel
        for i, c in enumerate(self.channels):
            mask = torch.ones_like(masks[0]) if c == "broad" else masks[BAND_NAMES.index(c)]
            pos = torch.nonzero(mask).squeeze(-1)             # positions in the in-band spectrum
            k_top = int(self.fz.band_idx[pos].max())
            self.ch_len.append(max(16, -(-2 * (k_top + 1) // 8) * 8))
            self.register_buffer(f"ch_pos{i}", pos, persistent=False)
        self.band_emb = nn.Parameter(torch.randn(C, d) * 0.02)
        self.codebook = BandCodebooks(C, ns, d, cfg.slot_temp)
        self.slot_emb = nn.Parameter(torch.randn(C, ns, du) / math.sqrt(ns))
        if R > 0:
            self.res_in = nn.Parameter(torch.randn(C, d, R) / math.sqrt(d))
            self.res_out = nn.Parameter(torch.randn(C, R, du) / math.sqrt(R))
        self.gate = nn.Linear(d, C * du)
        self.register_buffer("gate_mean", torch.full((C, du), math.log(2.0)))   # softplus(0) until trained
        self.critic = nn.Linear(d, 1)
        self.proj = nn.Linear(cfg.d_gain + du, cfg.d_token)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tok, cfg.d_token))
        layer = nn.TransformerEncoderLayer(cfg.d_token, cfg.n_heads, 2 * cfg.d_token, cfg.dropout,
                                           batch_first=True, norm_first=True)
        self.tok_enc = nn.TransformerEncoder(layer, cfg.token_layers, enable_nested_tensor=False)
        self.pool = AttnPool(cfg.d_token)

    # ------------------------------------------------------------- inputs of M
    def own_synth(self, spec: torch.Tensor, i: int) -> torch.Tensor:
        """In-band spectrum [..., F] -> channel i's waveform [..., n_i] from that channel's bins only.

        irfft over bins 0..n_i/2 with n_i points: the band-limited waveform sampled every win/n_i
        samples of the token, times win/n_i (irfft's 1/n normalisation). n_i/2 > k_top, so the
        Nyquist bin is empty and rfft returns the channel's bins exactly."""
        n, pos = self.ch_len[i], getattr(self, f"ch_pos{i}")
        full = spec.new_zeros(*spec.shape[:-1], n // 2 + 1)
        full[..., self.fz.band_idx[pos]] = spec[..., pos]
        return torch.fft.irfft(full, n=n, dim=-1)

    def waveforms(self, phasor: torch.Tensor, destroy_bands=(), gen=None, normalise: bool = True):
        """Phasor [..., F] -> channel waveforms (bands in self.channels order, broad last).

        band_rate full: one tensor [..., C, win]. band_rate own: a list of C tensors [..., n_c]."""
        bad = set(destroy_bands) - set(BAND_NAMES) - {"broad"}
        if bad:
            raise ValueError(f"unknown bands {sorted(bad)}")
        if "broad" in destroy_bands and not self.broad:
            raise ValueError("this model has no broadband channel")
        ph = phasor
        hit = [BAND_NAMES.index(b) for b in destroy_bands if b != "broad"]
        if hit:
            bins = self.all_masks[hit].any(0)
            ph = torch.where(bins, self.fz.random_phasor(phasor, gen), phasor)
        if self.own:
            toks = []
            for i, c in enumerate(self.channels):
                src = self.fz.random_phasor(phasor, gen) if c == "broad" and "broad" in destroy_bands else ph
                toks.append(self.own_synth(src, i))
            return [unit_rms(t) for t in toks] if normalise else toks
        toks = []
        if self.sel:
            toks.append(self.fz.synth(band_spectra(ph, self.all_masks[self.sel])))
        if self.broad:
            bph = self.fz.random_phasor(phasor, gen) if "broad" in destroy_bands else ph
            toks.append(self.fz.synth(bph).unsqueeze(-2))
        m = torch.cat(toks, -2)
        return unit_rms(m) if normalise else m

    CKPT_CHUNK = 16384                                        # waveforms per checkpointed encoder call

    def encode(self, m) -> torch.Tensor:
        """[..., C, win] (or a list of C tensors [..., n_c]) -> e [..., C, d].

        While gradients are recorded, the encoder runs in checkpointed chunks: its activations are
        recomputed in the backward pass instead of stored. Six channels per token do not fit a 16 GB
        GPU at the frozen batch size otherwise. The arithmetic is unchanged (the encoder has no
        dropout and only per-sample GroupNorm), so this is a memory measure, not a recipe change."""
        if isinstance(m, torch.Tensor) and self.morph_band is None:
            return self._run(self.morph, m) + self.band_emb
        chans = m.unbind(-2) if isinstance(m, torch.Tensor) else m
        enc = [self.morph if c == "broad" or self.morph_band is None else self.morph_band for c in self.channels]
        return torch.stack([self._run(f, w) for f, w in zip(enc, chans)], -2) + self.band_emb

    def _run(self, enc: nn.Module, w: torch.Tensor) -> torch.Tensor:
        lead = w.shape[:-1]
        flat = w.reshape(-1, w.shape[-1])
        if self.cfg.encoder_checkpoint and torch.is_grad_enabled() and flat.shape[0] > self.CKPT_CHUNK:
            from torch.utils.checkpoint import checkpoint
            out = torch.cat([checkpoint(enc, c, use_reentrant=False) for c in flat.split(self.CKPT_CHUNK)])
        else:
            out = enc(flat)
        return out.view(*lead, -1)

    def codes(self, f: Factors, stats: dict, destroy_prob: float = 0.0, gen=None, destroy_bands=()) -> dict:
        """Every intermediate code of the cascade from the token factors (used by forward and the tests)."""
        cfg, fz = self.cfg, self.fz
        rel = cfg.night_relative
        g, R = gain_rhythm(f.log_mag, stats["mag_med"] if rel else None, stats["mag_iqr"] if rel else None)
        r = self.rhythm(R)
        lead = R.shape[:-1]
        gates = F.softplus(self.gate(r)).view(*lead, len(self.channels), -1)
        ph = f.phasor
        if self.training and cfg.allpass_rad > 0:
            ph = fz.allpass(ph, cfg.allpass_rad, gen)
        real_ph = ph
        if destroy_prob > 0:
            fake = fz.random_phasor(ph, gen)
            kill = torch.rand(*lead, 1, device=ph.device, generator=gen) < destroy_prob
            ph = torch.where(kill, fake, ph)
        m = self.waveforms(ph, destroy_bands, gen)
        e = self.encode(m)
        return {"g": g, "R": R, "r": r, "gates": gates, "phasor": ph, "real_phasor": real_ph,
                "m": m, "e": e, "a": self.codebook(e)}

    # ------------------------------------------------------------------ forward
    def forward(self, xw, xraw, stats, knock: frozenset, destroy_prob: float, gen=None,
                knock_slots: frozenset = frozenset(), y=None, destroy_bands=()):
        return self.from_factors(self.fz.factorize(xw), stats, knock, destroy_prob, gen, knock_slots, y,
                                 destroy_bands)

    def from_factors(self, f: Factors, stats, knock=frozenset(), destroy_prob: float = 0.0, gen=None,
                     knock_slots=frozenset(), y=None, destroy_bands=()):
        cfg = self.cfg
        c = self.codes(f, stats, destroy_prob, gen, destroy_bands)
        a, e, gates = c["a"], c["e"], c["gates"]
        B, L, N, C, ns = a.shape
        if knock_slots:                                       # flat slot index = channel * band_slots + slot
            keep = torch.ones(C * ns, device=a.device, dtype=a.dtype)
            keep[list(knock_slots)] = 0
            a = a * keep.view(C, ns)
            tot = a.sum(-1, keepdim=True)
            a = torch.where(tot > 0, a / tot.clamp_min(1e-12), torch.zeros_like(a))
        contrib = torch.einsum("blncs,csu->blncu", a, self.slot_emb)
        if cfg.morph_residual > 0:
            contrib = contrib + torch.einsum("blncd,cdr,cru->blncu", e, self.res_in, self.res_out)
        if self.training:
            with torch.no_grad():
                self.gate_mean.lerp_(gates.detach().mean((0, 1, 2)), 0.01)
        if "R" in knock:
            gates = self.gate_mean.expand_as(gates)
        u = (gates * contrib).sum(-2)
        gv = self.gain(c["g"])

        def drop(name, h):
            if name in knock:
                return torch.zeros_like(h)
            if self.training and cfg.stream_dropout > 0:
                return h * (torch.rand(B, 1, 1, 1, device=h.device) >= cfg.stream_dropout).to(h.dtype)
            return h

        aux = {"assign": a.reshape(B, L, N, C * ns)}
        if self.training and cfg.critic_weight > 0:
            from .mist_grm import ScaleBranch
            mask = ScaleBranch._critic_mask(self, y, B, L, N)
            aux["critic"] = self._critic_loss(c["real_phasor"], gen, mask)
        h = self.proj(torch.cat([drop("G", gv), drop("M", u)], -1)).view(B * L, N, -1) + self.pos
        return self.pool(self.tok_enc(h)).view(B, L, -1), aux

    def _critic_loss(self, real_ph, gen, mask=None):
        """Per band: real channel waveform vs the same band with random phase (same magnitude)."""
        flat = real_ph.reshape(-1, real_ph.shape[-1])
        pool = torch.arange(flat.shape[0], device=flat.device)
        if mask is not None:
            pool = pool[mask.to(flat.device)]
            if pool.numel() == 0:
                return flat.real.sum() * 0.0
        n = min(pool.numel(), max(8, int(self.cfg.critic_frac * flat.shape[0])))
        idx = pool[torch.randperm(pool.numel(), device=flat.device, generator=gen)[:n]]
        sel = flat[idx]
        real = self.waveforms(sel)
        fake = self.waveforms(self.fz.random_phasor(sel, gen))
        both = torch.cat([real, fake]) if self.own is False else [torch.cat([r, f]) for r, f in zip(real, fake)]
        logits = self.critic(self.encode(both)).squeeze(-1)      # [2n, C]
        target = torch.cat([torch.ones(n), torch.zeros(n)]).to(logits).unsqueeze(-1).expand_as(logits)
        return F.binary_cross_entropy_with_logits(logits, target)
