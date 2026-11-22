#!/usr/bin/env python
"""Synthetic sanity checks for the two premises of the factorisation. No real data.

A. Invariance: how far do the stream inputs move when a whole night is amplified,
   zero-phase filtered, or all-pass filtered at the signal level?
B. Learnability: can a small encoder, given only phase-only tokens, tell tokens that
   contain a transient waveform from tokens that do not? Pure 1/f noise should
   stay at chance; noise plus biphasic transients should not.

    python scripts/sanity_phase_only.py
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mist.models.factorizer import Factorizer, night_codes  # noqa: E402
from mist.models.mist_grm import WaveEncoder  # noqa: E402


def eeg_like(n, seed, T=3000):
    g = torch.Generator().manual_seed(seed)
    f = torch.fft.rfftfreq(T, 0.01)
    amp = torch.where((f >= 0.3) & (f <= 45), 1.0 / f.clamp(min=0.3), torch.zeros_like(f))
    X = amp * torch.polar(torch.ones(n, f.numel()), torch.rand(n, f.numel(), generator=g) * 2 * math.pi)
    x = torch.fft.irfft(X, n=T)
    return 20.0 * x / x.std()


def codes(fz, x):
    return (*night_codes(x, fz), fz.tokens(x / x.std()) * fz.window)


def part_a():
    fz = Factorizer(100, 1.0, 0.5)
    x = eeg_like(200, 0)
    X, f = torch.fft.rfft(x), torch.fft.rfftfreq(3000, 0.01)
    bump = 1 + 0.8 * torch.exp(-((f - 12) / 6) ** 2) - 0.5 * torch.exp(-((f - 3) / 2) ** 2)
    variants = {"gain x3": 3 * x,
                "zero-phase tilt f^-0.5": torch.fft.irfft(X * f.clamp(min=0.3) ** -0.5, n=3000),
                "zero-phase smooth response": torch.fft.irfft(X * bump, n=3000),
                "all-pass, quadratic phase": torch.fft.irfft(X * torch.polar(torch.ones_like(f), 1.5 * (f / 35) ** 2), n=3000)}
    base = codes(fz, x)
    rel = lambda a, b: float((a - b).norm() / a.norm())
    print("A. relative change of each stream input (0 = invariant); last column = z-scored raw tokens")
    for k, v in variants.items():
        c = codes(fz, v)
        print(f"   {k:28s} G {rel(base[0], c[0]):.3f}  R {rel(base[1], c[1]):.3f}  M {rel(base[2], c[2]):.3f}  raw {rel(base[3], c[3]):.3f}")


def part_b(with_transients: bool, steps=400, seed=0):
    torch.manual_seed(seed)
    fz = Factorizer(100, 4.0, 2.0)
    t = torch.arange(400) / 100.0

    def batch(n):
        x = eeg_like(n, int(torch.randint(0, 10**6, (1,))), T=400)
        y = (torch.rand(n) < 0.5).float()
        if with_transients:   # biphasic 0.8-s wave at a random position, amplitude comparable to background
            c = 1.0 + 2.0 * torch.rand(n, 1)
            w = -torch.exp(-((t - c) / 0.12) ** 2) + 0.6 * torch.exp(-((t - c - 0.35) / 0.18) ** 2)
            x = x + y[:, None] * 60.0 * w
        return fz.morph_token(fz.factorize(x), "phase_only")[:, 0], y    # 1/f input, not whitened: harder case

    enc, head = WaveEncoder(32), torch.nn.Linear(32, 1)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), 2e-3)
    for _ in range(steps):
        xb, yb = batch(64)
        loss = F.binary_cross_entropy_with_logits(head(enc(xb)).squeeze(-1), yb)
        opt.zero_grad(); loss.backward(); opt.step()
    enc.eval()
    with torch.no_grad():
        xb, yb = batch(2000)
        acc = ((head(enc(xb)).squeeze(-1) > 0).float() == yb).float().mean()
    return float(acc)


if __name__ == "__main__":
    part_a()
    print("B. accuracy of a phase-only encoder at detecting a transient (chance = 0.5)")
    print(f"   labels random, no transient added : {part_b(False):.3f}")
    print(f"   biphasic transient added          : {part_b(True):.3f}")
