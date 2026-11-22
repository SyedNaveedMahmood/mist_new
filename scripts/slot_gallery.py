#!/usr/bin/env python
"""T8: shape-slot gallery and after-the-fact naming for one MIST-GRM checkpoint.

    python scripts/slot_gallery.py --run runs/e1_grm_cv0_s0 [--top 24]

Uses the run's own VALIDATION people only (never test, never lockbox). For every
scale and slot: the top-activating tokens (highest assignment) drawn as raw
waveforms, and the lift of each night-relative detector (fraction of the slot's
top tokens inside detector events divided by the base rate over all tokens).
A slot is named after a detector only when lift >= 2 and the overlap >= 0.3;
otherwise it stays unnamed. Names are descriptive, not clinical claims.
Outputs runs/<run>/slots/gallery_s<k>.png and slots.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mist.config import load_paths
from mist.data import splits as S
from mist.data.dataset import split_stats
from mist.data.store import load_cohort
from mist.detectors import DETECTORS, name_slots
from mist.engine.loop import load_checkpoint, make_dataset


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--top", type=int, default=24)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(a.run, device)
    if "M" not in cfg["model"].get("streams", []):
        raise SystemExit("run has no M stream")
    roles = S.resolve(S.load(cfg["split"]["file"]), cfg["split"]["protocol"], cfg["split"].get("fold", 0))
    nights = load_cohort(load_paths()["store_root"], cfg["data"].get("cohort", "SC"), set(roles["val"]),
                         split_file=cfg["split"]["file"])
    ch = cfg["data"]["channel"]
    ds = make_dataset(nights, cfg, ch, train=False)
    masks = {}
    for vi, v in enumerate(ds.views):
        xv = np.asarray(v.night.x[v.start:v.stop, v.ch], dtype=np.float64)
        masks[vi] = {d: f(xv) for d, f in DETECTORS.items()}
    model.eval()
    per_scale = None
    for i in range(len(ds)):
        item = ds[i]
        batch = {k: (t.unsqueeze(0) if torch.is_tensor(t) else torch.tensor([t])) for k, t in item.items()}
        _, aux = model(batch["x"].float().to(device), split_stats(batch, device))
        if per_scale is None:
            per_scale = [{"assign": [], "where": []} for _ in aux["assign"]]
        keep = item["keep"].numpy()
        for k, (asg, br) in enumerate(zip(aux["assign"], model.branches)):
            A = asg[0].cpu().numpy()                                # [L, N, slots]
            for l in np.flatnonzero(keep):
                ep = item["start"] + l
                for n in range(A.shape[1]):
                    per_scale[k]["assign"].append(A[l, n])
                    per_scale[k]["where"].append((item["night"], ep, n * br.fz.hop, br.fz.win))
    out_dir = Path(a.run) / "slots"; out_dir.mkdir(exist_ok=True)
    report = {"run": a.run, "people": "validation only", "n_people": len(roles["val"]), "scales": []}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for k, ps in enumerate(per_scale):
        A = np.stack(ps["assign"]); W = ps["where"]
        ov = {d: np.array([masks[v][d][e, s:s + n].mean() for v, e, s, n in W]) for d in DETECTORS}
        base = {d: float(ov[d].mean()) for d in DETECTORS}
        top = np.argsort(-A, axis=0)[: a.top]                        # [top, slots]
        top_ov = {d: ov[d][top].mean(0) for d in DETECTORS}
        names = name_slots(top_ov, base)
        usage = A.mean(0)
        report["scales"].append({"scale": k, "base_rate": base, "usage": usage.tolist(),
                                 "slots": {str(s): v for s, v in names.items()}})
        n_slots = A.shape[1]
        fig, axes = plt.subplots(n_slots, 1, figsize=(6, 1.1 * n_slots), sharex=True)
        for s in range(n_slots):
            ax = axes[s]
            for j in top[:8, s]:
                v, e, st, n = W[j]
                vw = ds.views[v]
                w = np.asarray(vw.night.x[vw.start + e, vw.ch, st:st + n])
                ax.plot(w / (np.abs(w).max() + 1e-9), lw=0.6, alpha=0.7)
            ax.set_ylabel(f"{s}\n{names[s]['name'] or ''}", fontsize=6, rotation=0, labelpad=18)
            ax.set_yticks([])
        axes[-1].set_xlabel("sample (100 Hz)")
        fig.tight_layout(); fig.savefig(out_dir / f"gallery_s{k}.png", dpi=110); plt.close(fig)
    (out_dir / "slots.json").write_text(json.dumps(report, indent=1))
    for sc in report["scales"]:
        named = {s: v["name"] for s, v in sc["slots"].items() if v["name"]}
        print(f"scale {sc['scale']}: {len(named)} named slots {named}")


if __name__ == "__main__":
    main()
