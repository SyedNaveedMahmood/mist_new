#!/usr/bin/env python
"""E5a: measure, on real epochs, how far each stream's input moves under
acquisition nuisances. Expected: gain and smooth zero-phase filtering leave
G, R and phase-only M nearly unchanged; an all-pass filter moves mainly M.

    python scripts/check_invariance.py --n-nights 5
"""
import argparse

import torch

from mist.config import load_paths
from mist.data.store import load_cohort
from mist.models.factorizer import Factorizer, night_codes


def codes(fz, x):
    return night_codes(x, fz)


def rel(a, b):
    return float((a - b).norm() / (a.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nights", type=int, default=5)
    ap.add_argument("--channel", default="Fpz-Cz")
    a = ap.parse_args()
    fz = Factorizer(100, 1.0, 0.5)
    for n in load_cohort(load_paths()["store_root"], "SC")[: a.n_nights]:
        x = torch.from_numpy(n.x[n.view_slice("std"), n.channel_index(a.channel)].copy())
        X = torch.fft.rfft(x, dim=-1)
        f = torch.fft.rfftfreq(x.shape[-1], 0.01)
        variants = {
            "gain x3": x * 3.0,
            "zero-phase tilt f^-0.5": torch.fft.irfft(X * (f.clamp(min=0.3) ** -0.5), n=x.shape[-1]),
            "all-pass (quadratic phase)": torch.fft.irfft(X * torch.polar(torch.ones_like(f), 1.5 * (f / 35) ** 2),
                                                           n=x.shape[-1]),
        }
        base = codes(fz, x)
        for name, xv in variants.items():
            g, r, m = codes(fz, xv)
            print(f"{n.record_id} {name:28s} dG={rel(base[0], g):.4f} dR={rel(base[1], r):.4f} dM={rel(base[2], m):.4f}")


if __name__ == "__main__":
    main()
