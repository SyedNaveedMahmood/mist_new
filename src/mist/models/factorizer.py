"""Night-relative, lossless gain / rhythm / morphology (GRM) factorisation.

Two parameter-free steps.

1. Night whitening (signal level). Each epoch is divided, in the frequency
   domain, by the smooth envelope of the night's own median amplitude spectrum.
   A recording-level gain or smooth zero-phase filter multiplies signal and
   envelope alike and cancels. Whitening also flattens the 1/f range, which keeps
   window leakage small in step 2.

2. Token factorisation. A Hann-windowed token of the whitened signal with
   in-band spectrum X(f) is coded as

       z(f)   = (log|X(f)| - median_night(f)) / (iqr_night(f) + 0.1)
       G      = mean_f z(f)                 gain:       one scalar per token
       R(f)   = z(f) - G                    rhythm:     spectral shape, phase-blind
       phi(f) = arg X(f)                    morphology: rendered as a time-domain
                                            "phase-only" token with flat magnitude,
                                            so it is power-blind

   (G, R, phi) plus the night statistics reconstruct the token exactly.

What holds by construction (tests/test_factorizer.py checks each numerically):
  * all three codes are invariant to any positive gain;
  * G and R cannot see waveform shape: they are functions of |X| only;
  * phase-only M cannot see power: it is a function of arg X only;
  * G, R and M are invariant to a per-night power-law tilt, and nearly so to
    any smooth zero-phase response.
What does not hold: a non-zero-phase response (group-delay distortion) changes
phi. That nuisance reaches only the M stream and is handled by augmentation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

EPS = 1e-8


@dataclass
class Factors:
    log_mag: torch.Tensor    # [..., N, F]   log |X(f)| of the (whitened) token
    phasor: torch.Tensor     # [..., N, F]   complex unit phasors exp(i*phi)


# ------------------------------------------------------------ night whitening
@torch.no_grad()
def night_envelope(x: torch.Tensor, fs: int = 100, octaves: float = 1 / 3, f_lo: float = 0.3,
                   f_hi: float = 36.0, chunk: int = 512) -> torch.Tensor:
    """Smooth envelope of the night's median amplitude spectrum. x: [n_epochs, T] -> [T//2+1].

    Inside [f_lo, f_hi] the log spectrum is averaged over a window of `octaves`
    centred in log frequency (weights 1/f), so a power-law tilt is reproduced
    exactly away from the band edges. Outside the band the envelope is 1 and
    `whiten` zeroes those bins, so filter skirts are never amplified.
    """
    amps = torch.cat([torch.fft.rfft(x[i:i + chunk].float(), dim=-1).abs() for i in range(0, x.shape[0], chunk)])
    if amps.shape[0] > 1500:                                  # median over a deterministic subsample
        amps = amps[:: math.ceil(amps.shape[0] / 1500)]
    log_s = torch.log(amps.median(0).values.double().clamp(min=1e-30))
    n = log_s.numel()
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / fs).double()
    inband = torch.nonzero((freqs >= f_lo) & (freqs <= f_hi)).squeeze(-1)
    b0, b1 = int(inband[0]), int(inband[-1]) + 1
    w = 1.0 / freqs.clamp(min=freqs[1])
    cw = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(w * log_s, 0)])
    c1 = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(w, 0)])
    a = 2.0 ** (octaves / 2)
    idx = torch.arange(n, dtype=torch.float64)
    lo = torch.floor(idx / a).long().clamp(b0, b1 - 1)
    hi = (torch.ceil(idx * a).long() + 1).clamp(b0 + 1, b1)
    hi = torch.maximum(hi, lo + 1)
    env = torch.exp((cw[hi] - cw[lo]) / (c1[hi] - c1[lo]))
    out = torch.ones(n, dtype=torch.float64)
    out[b0:b1] = env[b0:b1]
    return out.float()


def whiten(x: torch.Tensor, envelope: torch.Tensor, fs: int = 100, f_lo: float = 0.3, f_hi: float = 36.0) -> torch.Tensor:
    """x: [B, ..., T]; envelope: [B, T//2+1] (or [T//2+1]). Zero outside [f_lo, f_hi]."""
    n = x.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1.0 / fs).to(x.device)
    band = ((freqs >= f_lo) & (freqs <= f_hi)).to(x.dtype)
    gain = band / (envelope + EPS)
    if gain.dim() == 2:
        gain = gain.view(gain.shape[0], *([1] * (x.dim() - 2)), gain.shape[-1])
    return torch.fft.irfft(torch.fft.rfft(x, dim=-1) * gain, n=n, dim=-1)


# ------------------------------------------------------------ token factoriser
class Factorizer(nn.Module):
    def __init__(self, fs: int = 100, win_s: float = 1.0, hop_s: float = 0.5,
                 f_lo: float = 0.5, f_hi: float = 35.0):
        super().__init__()
        self.fs = fs
        self.win = int(round(win_s * fs))
        self.hop = int(round(hop_s * fs))
        freqs = torch.fft.rfftfreq(self.win, d=1.0 / fs)
        band = (freqs >= f_lo) & (freqs <= f_hi)
        if band[0] or band[-1]:
            raise ValueError("band must exclude DC and Nyquist so every bin has a free phase")
        self.register_buffer("window", torch.hann_window(self.win, periodic=True), persistent=False)
        self.register_buffer("band_idx", torch.nonzero(band).squeeze(-1), persistent=False)
        self.register_buffer("freqs", freqs[band].clone(), persistent=False)

    @property
    def n_freq(self) -> int:
        return int(self.band_idx.numel())

    def n_tokens(self, n_samples: int) -> int:
        return (n_samples - self.win) // self.hop + 1

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """[..., T] -> [..., N, win]; tokens never cross an epoch boundary."""
        return x.unfold(-1, self.win, self.hop)

    def spectra(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft(self.tokens(x) * self.window, dim=-1)[..., self.band_idx]

    def factorize(self, x: torch.Tensor) -> Factors:
        X = self.spectra(x)
        mag = X.abs()
        return Factors(log_mag=torch.log(mag + EPS), phasor=X / (mag + EPS))

    def synth(self, spec: torch.Tensor) -> torch.Tensor:
        """In-band complex spectrum [..., F] -> real token [..., win]."""
        full = spec.new_zeros(*spec.shape[:-1], self.win // 2 + 1)
        full[..., self.band_idx] = spec
        return torch.fft.irfft(full, n=self.win, dim=-1)

    def reconstruct(self, f: Factors) -> torch.Tensor:
        return self.synth(torch.exp(f.log_mag) * f.phasor)

    def morph_token(self, f: Factors, mode: str = "phase_only") -> torch.Tensor:
        """Time-domain input of the morphology stream, unit RMS.

        phase_only: flat magnitude, no power cue at all (default).
        unit_raw:   unit-RMS whitened waveform, spectrum included (ablation).
        """
        if mode == "phase_only":
            spec = f.phasor
        elif mode == "unit_raw":
            spec = torch.exp(f.log_mag) * f.phasor
        else:
            raise ValueError(mode)
        tok = self.synth(spec)
        return tok / torch.sqrt((tok ** 2).mean(-1, keepdim=True) + EPS)

    # ------------------------------------------------------------- interventions
    def random_phasor(self, like: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        theta = torch.rand(like.shape, generator=generator, device=like.device) * (2 * math.pi)
        return torch.polar(torch.ones_like(theta), theta)

    def allpass(self, phasor: torch.Tensor, max_rad: float, generator: torch.Generator | None = None) -> torch.Tensor:
        """Random smooth phase response, one per sequence: the residual nuisance for M."""
        b = phasor.shape[0]
        u = (self.freqs - self.freqs[0]) / (self.freqs[-1] - self.freqs[0])
        c = (torch.rand(b, 2, generator=generator, device=phasor.device) * 2 - 1) * max_rad
        theta = c[:, :1] * (u ** 2) + c[:, 1:] * torch.sin(math.pi * u)
        theta = theta.view(b, *([1] * (phasor.dim() - 2)), -1)
        return phasor * torch.polar(torch.ones_like(theta), theta)


def gain_rhythm(log_mag: torch.Tensor, med: torch.Tensor | None, iqr: torch.Tensor | None) -> tuple:
    """log|X| [B, ..., F] -> (G [B, ...], R [B, ..., F]). med/iqr: [B, F] or None (absolute coding)."""
    if med is None:
        z = log_mag
    else:
        shape = (log_mag.shape[0],) + (1,) * (log_mag.dim() - 2) + (log_mag.shape[-1],)
        z = (log_mag - med.view(shape)) / (iqr.view(shape) + 0.1)
    g = z.mean(-1)
    return g, z - g.unsqueeze(-1)


@torch.no_grad()
def token_stats(fz: Factorizer, x: torch.Tensor, chunk: int = 256) -> dict:
    """Per-frequency median and IQR of log|X| over one night. x: [n_epochs, T], already whitened."""
    mags = torch.cat([fz.factorize(x[i:i + chunk].float()).log_mag.reshape(-1, fz.n_freq)
                      for i in range(0, x.shape[0], chunk)])
    if mags.shape[0] > 200_000:
        mags = mags[:: math.ceil(mags.shape[0] / 200_000)]
    q25, q50, q75 = torch.quantile(mags, torch.tensor([0.25, 0.5, 0.75]), dim=0)
    return {"mag_med": q50, "mag_iqr": q75 - q25}


@torch.no_grad()
def night_reference(x: torch.Tensor, factorizers: list, fs: int = 100) -> dict:
    """Everything the model needs to know about a night, computed without labels.

    x: [n_epochs, T] of one channel. Returns
      night: envelope [T//2+1] for whitening, scale (robust std, for raw-input models)
      s<k>:  mag_med, mag_iqr for token scale k
    """
    x = x.float()
    env = night_envelope(x, fs)
    sub = x[:: max(1, x.shape[0] // 200)]
    scale = 1.4826 * (sub - sub.median()).abs().median()
    out = {"night": {"envelope": env, "scale": scale.clamp(min=1e-6)}}
    xw = torch.cat([whiten(x[i:i + 256], env, fs) for i in range(0, x.shape[0], 256)])
    for k, fz in enumerate(factorizers):
        out[f"s{k}"] = token_stats(fz, xw)
    return out


@torch.no_grad()
def night_codes(x: torch.Tensor, fz: Factorizer, fs: int = 100) -> tuple:
    """Analysis helper: the three stream inputs (G, R, phase-only M) for a whole night."""
    ref = night_reference(x, [fz], fs)
    f = fz.factorize(whiten(x.float(), ref["night"]["envelope"], fs))
    g, r = gain_rhythm(f.log_mag[None], ref["s0"]["mag_med"][None], ref["s0"]["mag_iqr"][None])
    return g[0], r[0], fz.morph_token(f)
