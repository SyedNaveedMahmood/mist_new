#!/usr/bin/env python
"""E5a: measure, on real epochs, how far each stream's input moves under
acquisition nuisances. Expected: gain and smooth zero-phase filtering leave
G, R and phase-only M nearly unchanged; an all-pass filter moves mainly M.

    python scripts/check_invariance.py --n-nights 5
"""
import argparse
import json
from pathlib import Path

import torch

from mist.config import load_paths
from mist.data import splits as S
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
    ap.add_argument("--split", default="splits/sleepedf_v1.json")
    ap.add_argument("--out", default="runs/e5a")
    a = ap.parse_args()
    fz = Factorizer(100, 1.0, 0.5)
    # dev people only: the lockbox is loaded by task T12 and nothing else
    dev = set(S.load(a.split)["dev"])
    rows = []
    for n in load_cohort(load_paths()["store_root"], "SC", dev)[: a.n_nights]:
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
            row = {"record": n.record_id, "variant": name, "dG": rel(base[0], g), "dR": rel(base[1], r),
                   "dM": rel(base[2], m)}
            rows.append(row)
            print(f"{n.record_id} {name:28s} dG={row['dG']:.4f} dR={row['dR']:.4f} dM={row['dM']:.4f}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with (out / f"invariance_{a.channel}.tsv").open("w") as f:
        f.write("record\tvariant\tdG\tdR\tdM\n")
        for r in rows:
            f.write(f"{r['record']}\t{r['variant']}\t{r['dG']:.6f}\t{r['dR']:.6f}\t{r['dM']:.6f}\n")
    summary = {}
    for name in dict.fromkeys(r["variant"] for r in rows):
        sub = [r for r in rows if r["variant"] == name]
        summary[name] = {k: {"mean": sum(r[k] for r in sub) / len(sub), "max": max(r[k] for r in sub)}
                         for k in ("dG", "dR", "dM")}
    summary["_meta"] = {"n_nights": len({r["record"] for r in rows}), "channel": a.channel, "view": "std",
                        "factorizer": "window 1.0 s, hop 0.5 s"}
    (out / f"invariance_{a.channel}_summary.json").write_text(json.dumps(summary, indent=1))
    for name, v in summary.items():
        if name != "_meta":
            print(f"{name:28s} " + " ".join(f"{k} mean={v[k]['mean']:.4f} max={v[k]['max']:.4f}" for k in v))


if __name__ == "__main__":
    main()
