#!/usr/bin/env python
"""Validation gate for revision 3 (owner, 2026-11-23). Validation only: reads best_val_mf1 from train_summary.json.

    python scripts/gate_cascade2.py --arm cascade2 [--ref cascade] [--seed 0]

Pass unless the mean best-validation macro-F1 of <arm> is more than 1.0 point BELOW that of <ref>, folds 0-4 of
the seed (one-sided; owner's correction of 2026-11-23: being above the reference passes). Exit 0 pass, 1 fail, 2 not all 10 runs trained. Writes runs/gates/.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--ref", default="cascade")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tol", type=float, default=0.010)
a = ap.parse_args()
vals = {}
for arm in (a.arm, a.ref):
    p = [Path(f"runs/e1_{arm}_cv{f}_s{a.seed}/train_summary.json") for f in range(5)]
    if not all(x.exists() for x in p):
        print(f"gate: {arm} seed {a.seed} not complete"); raise SystemExit(2)
    vals[arm] = [json.loads(x.read_text())["best_val_mf1"] for x in p]
m_arm, m_ref = (sum(vals[k]) / 5 for k in (a.arm, a.ref))
diff = m_arm - m_ref
ok = diff >= -a.tol - 1e-12
out = {"gate": "cascade2_validation", "arm": a.arm, "ref": a.ref, "seed": a.seed, "per_fold": vals,
       "mean_arm": m_arm, "mean_ref": m_ref, "diff_points": 100 * diff, "tol_points": 100 * a.tol, "rule": "one-sided: fail only if arm < ref - tol",
       "status": "PASSED" if ok else "FAILED", "note": "validation people only; no test score read"}
Path("runs/gates").mkdir(parents=True, exist_ok=True)
Path(f"runs/gates/val_gate_{a.arm}_s{a.seed}.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("arm", "mean_arm", "mean_ref", "diff_points", "status")}))
raise SystemExit(0 if ok else 1)
