#!/usr/bin/env python
"""Fidelity of the E0 surrogates on real epochs (no model, no test people).

    python scripts/audit_surrogates.py [--n-nights 8] [--channel Fpz-Cz]

For each surrogate, compares surrogate and real epochs on what the surrogate
claims to keep: RMS, band power (Welch, 4 s Hann, per epoch) in slow-osc
0.5-1, delta 1-4, theta 4-8, alpha 8-12, sigma 12-16, beta 16-35 Hz, and the
short-time amplitude spectrum seen through a window different from the
surrogate's own (1 s Hann, hop 0.5 s, which is also the MIST token scale).
Reports median |log10 ratio| (0.1 = 26% power change) and its stage breakdown.
Uses the first nights of fold-0 TRAINING people; the result informs how E0 is
read, never a model or analysis choice.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mist.config import load_paths
from mist.data import splits as S
from mist.data.store import load_cohort
from mist.data.surrogates import CONDITIONS

BANDS = {"so_0.5-1": (0.5, 1), "delta_1-4": (1, 4), "theta_4-8": (4, 8), "alpha_8-12": (8, 12),
         "sigma_12-16": (12, 16), "beta_16-35": (16, 35)}
STAGES = ["W", "N1", "N2", "N3", "REM"]


def welch_bandpower(x):                                        # x [n, 3000] -> {band: [n]}
    w = torch.hann_window(400, periodic=True)
    S_ = torch.stft(x, 400, 200, window=w, center=False, return_complex=True).abs() ** 2
    P = S_.mean(-1)                                            # [n, 201]
    f = torch.fft.rfftfreq(400, 0.01)
    return {b: P[:, (f >= lo) & (f < hi)].sum(-1) for b, (lo, hi) in BANDS.items()}


def short_spec(x):
    w = torch.hann_window(100, periodic=True)
    S_ = torch.stft(x, 100, 50, window=w, center=False, return_complex=True).abs()
    f = torch.fft.rfftfreq(100, 0.01)
    return S_[:, (f >= 1) & (f <= 35)]                         # [n, F, frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nights", type=int, default=8)
    ap.add_argument("--channel", default="Fpz-Cz")
    ap.add_argument("--out", default="runs/e0/surrogate_audit.json")
    a = ap.parse_args()
    sp = S.load("splits/sleepedf_v1.json")
    train = sorted(S.resolve(sp, "cv", 0)["train"])
    nights = [n for n in load_cohort(load_paths()["store_root"], "SC", set(train)) if n.night == 1][: a.n_nights]
    xs, ys = [], []
    for n in nights:
        sl = n.view_slice("std")
        y = n.y[sl]
        keep = y >= 0
        xs.append(np.asarray(n.x[sl, n.channel_index(a.channel)])[keep]); ys.append(y[keep])
    x = torch.from_numpy(np.concatenate(xs)).double(); y = np.concatenate(ys)
    bp_real, ss_real = welch_bandpower(x), short_spec(x)
    out = {"nights": [n.record_id for n in nights], "n_epochs": int(len(y)), "channel": a.channel,
           "metric": "median |log10(surrogate/real)| per epoch (bands: power; short_spec: amplitude, per bin and frame)",
           "conditions": {}}
    for cond in ("global_phase", "local_phase", "flatten_tilt"):
        g = torch.Generator().manual_seed(0)
        z = CONDITIONS[cond](x.float(), g).double()
        bp = welch_bandpower(z)
        rec = {"rms_log10_ratio_median": float(np.median(np.log10((z.pow(2).mean(-1) / x.pow(2).mean(-1)).sqrt())))}
        for b in BANDS:
            lr = torch.log10(bp[b] / bp_real[b]).numpy()
            rec[b] = {"all": float(np.median(np.abs(lr))), "signed_median": float(np.median(lr)),
                      **{s: float(np.median(np.abs(lr[y == k]))) for k, s in enumerate(STAGES) if (y == k).any()}}
        ss = torch.log10((short_spec(z) + 1e-9) / (ss_real + 1e-9)).abs()
        rec["short_spec_1s"] = {"all": float(ss.median()),
                                **{s: float(ss[torch.from_numpy(y == k)].median()) for k, s in enumerate(STAGES)
                                   if (y == k).any()}}
        out["conditions"][cond] = rec
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{len(nights)} nights, {len(y)} epochs, {a.channel}")
    hdr = ["rms"] + list(BANDS) + ["short_spec_1s"]
    print("condition".ljust(14), " ".join(h[:11].rjust(11) for h in hdr))
    for c, r in out["conditions"].items():
        vals = [r["rms_log10_ratio_median"]] + [r[b]["all"] for b in BANDS] + [r["short_spec_1s"]["all"]]
        print(c.ljust(14), " ".join(f"{v:11.3f}" for v in vals))
    print("signed median log10 band-power ratio (local_phase):",
          {b: round(out["conditions"]["local_phase"][b]["signed_median"], 3) for b in BANDS})
    print("local_phase |log10| by stage:", {b: {s: round(out["conditions"]["local_phase"][b].get(s, float("nan")), 3)
                                                for s in STAGES} for b in ("so_0.5-1", "delta_1-4", "alpha_8-12")})


if __name__ == "__main__":
    main()
