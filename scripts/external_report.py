#!/usr/bin/env python
"""Phase-2 external development evaluation table (DECISION_LOG D-017 to D-020) -> reports/external_dev.md.

    python scripts/external_report.py [--seeds 0] [--arms raw,attnsleep,cascade2,cascade2_placebo]

Reads eval_test_ext-<dataset>-dev_rule.json written by `scripts/evaluate.py --dataset <d> --part dev` in the
Sleep-EDF cv runs (5 fold models per seed). Unit: the person; each person's confusion matrices are summed over
the fold models and seeds that tested them. Intervals: subject bootstrap (single arm) and paired subject
bootstrap (pairs), 10,000 resamples, 95% percentile. External holdouts are never read here (dev part only).
ISRUC (owner, D-020): rows ISRUC-M, ISRUC-A, all, in that order; claims per sub-domain; "all" only for comparison
with papers that pool ISRUC; *_absolute arms are evidence on ISRUC-M only.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from mist import STAGES
from mist.aggregate import paired_diff, scores_batch, summary
from mist.data.external import ISRUC_ROWS

DATASETS = ("hmc", "dodh", "cap", "isruc")
NAMES = {"hmc": "HMC", "dodh": "DOD-H", "cap": "CAP", "isruc": "ISRUC"}
# owner, 2026-11-24: each dataset leads with these two rows (seed 0, INTERIM, development-grade)
LEAD = [("cascade2_placebo", "raw", "factorisation without shape vs raw (H3 direction), INTERIM"),
        ("cascade2", "cascade2_placebo", "shape access (H1 direction), INTERIM")]
PAIRS = LEAD + [("raw_whitened", "raw", "night whitening alone, EXPLORATORY"),
                ("cascade2_placebo", "raw_whitened", "gain/rhythm split on top of whitening, EXPLORATORY"),
                ("attnsleep", "raw", "descriptive"),
                ("cascade", "cascade2_placebo", "full-rate shape access, EXPLORATORY")]   # owner 2026-11-24
INTERIM_ARMS = ("cascade2", "cascade2_placebo", "raw_whitened")
EXPLORATORY_ARMS = ("raw_whitened", "cascade")


def runs_of(arm, seeds):
    out = []
    for s in seeds:
        out += sorted(set(glob.glob(f"runs/e1_{arm}_cv[0-4]_s{s}") + glob.glob(f"runs/e0_{arm}_cv[0-4]_s{s}")))
    return out


def eval_file(run, dataset):
    return Path(run, f"eval_test_ext-{dataset}-dev_rule.json")


def run_key(f):
    """(fold, seed) of the Sleep-EDF cv run an external evaluation file belongs to."""
    name = Path(f).parent.name
    return int(name.split("_cv")[1].split("_")[0]), int(name.rsplit("_s", 1)[1])


def cms_by_row(files, dataset) -> dict:
    """{row: {subject: summed cm}}; rows are ISRUC_ROWS for isruc, else ("all",)."""
    rows = {}
    for f in files:
        d = json.loads(Path(f).read_text())
        meta = d["night_meta"]
        for rid, cm in d["cells"]["real"]["night_cm"].items():
            m = meta[rid]
            keys = ["all"] + ([m["subdomain"]] if dataset == "isruc" else [])
            for k in keys:
                rows.setdefault(k, {})
                rows[k][m["subject"]] = rows[k].get(m["subject"], 0) + np.asarray(cm, np.int64)
    order = ISRUC_ROWS if dataset == "isruc" else ("all",)
    return {k: rows[k] for k in order if k in rows}


def subject_ci(cms: dict, n_boot: int, seed: int = 0) -> tuple:
    subs = sorted(cms)
    A = np.stack([cms[s] for s in subs])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(subs), (n_boot, len(subs)))
    w = np.zeros((n_boot, len(subs))); np.add.at(w, (np.arange(n_boot)[:, None], idx), 1)
    lo, hi = np.nanpercentile(scores_batch(np.einsum("bs,sij->bij", w, A))["mf1"], [2.5, 97.5])
    return float(lo), float(hi)


def build(arms, seeds, n_boot):
    res = {}
    for ds in DATASETS:
        per_arm = {}
        for arm in arms:
            runs = runs_of(arm, seeds)
            files = [eval_file(r, ds) for r in runs if eval_file(r, ds).exists()]
            if not files:
                continue
            rows = cms_by_row(files, ds)
            per_arm[arm] = {"n_eval_files": len(files), "n_runs": len(runs), "rows": {
                k: summary(c) | dict(zip(("mf1_lo", "mf1_hi"), subject_ci(c, n_boot))) for k, c in rows.items()},
                "_files": {run_key(f): f for f in files}}
        pairs = []
        for a, b, tag in PAIRS:
            if a in per_arm and b in per_arm:
                fa, fb = per_arm[a]["_files"], per_arm[b]["_files"]
                common = sorted(set(fa) & set(fb))            # only the fold models both arms have
                if not common:
                    continue
                ra = cms_by_row([fa[c] for c in common], ds)
                rb = cms_by_row([fb[c] for c in common], ds)
                for k in ra:
                    if k not in rb or sorted(ra[k]) != sorted(rb[k]):
                        continue
                    pairs.append({"pair": f"{a} - {b}", "tag": tag, "row": k, "n_models": len(common),
                                  "lead": (a, b, tag) in LEAD, **paired_diff(ra[k], rb[k], n_boot=n_boot)})
        for v in per_arm.values():
            v.pop("_files")
        if per_arm:
            res[ds] = {"arms": per_arm, "pairs": pairs}
    return res


def shape_line(res) -> str:
    """One line for the session-2 report (owner 2026-11-24): is full-rate shape neutral or negative on the external
    development parts? Rule fixed before any cascade external number was read: per dataset row (ISRUC-M and
    ISRUC-A separately, not "all"), macro-F1 of cascade - cascade2_placebo is negative if the 95% interval lies
    below 0, positive if above 0, else neutral."""
    parts = []
    for ds, r in res.items():
        for p in r["pairs"]:
            if p["pair"] != "cascade - cascade2_placebo" or p["row"] == "all" and ds == "isruc":
                continue
            d = p["mf1"]
            v = "negative" if d["hi"] < 0 else "positive" if d["lo"] > 0 else "neutral"
            name = p["row"] if ds == "isruc" else NAMES[ds]
            parts.append(f"{name} {v} ({100 * d['diff']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}], "
                         f"{p['n_models']} fold models)")
    if not parts:
        return ("Full-rate shape on the external development parts (cascade - cascade2_placebo, seed 0, EXPLORATORY): "
                "NOT TESTED yet (no cascade external evaluation files).")
    return ("Full-rate shape on the external development parts (cascade - cascade2_placebo, macro-F1, seed 0, "
            "EXPLORATORY; negative/neutral = 95% interval below 0/includes 0): " + "; ".join(parts) + ".")


def role(arm, ds, row):
    if ds != "isruc":
        return ""
    if row == "all":
        return "comparison with pooled papers only"
    if arm.endswith("_absolute") and row == "ISRUC-A":
        return "descriptive only (calibration unknown; not evidence on H3)"
    return ""


def fmt(x):
    return f"{100 * x:.1f}"


def write(res, seeds, out_md, out_json):
    L = ["# External development evaluation (phase 2, development-grade)", "",
         f"Generated by `scripts/external_report.py`. Seeds {seeds}; each arm's 5 Sleep-EDF cv fold models (trained on "
         "Fpz-Cz, frozen recipe) evaluated on the **development part** (train + val people) of each external split, "
         "on the channel chosen per recording by the D-017 rule. External holdouts stay closed. These numbers change "
         "no phase-1 choice. Points (x100); intervals are subject bootstrap 95% percentile (10,000 resamples), a "
         "person's confusion matrices summed over the fold models that tested them.", "",
         "Caveats: DOD-H development part has 15 people, so its intervals are wide. ISRUC rows follow D-020: "
         "ISRUC-M (M1/M2 references, normal range), ISRUC-A (A1/A2 references, +/-25 uV header range), then all. "
         "The ISRUC-A difference is \"a different recording set-up or a different population; not separable from the "
         "files\" (CONDITIONAL). ISRUC group III lies entirely inside ISRUC-A, so any statement about it is "
         "confounded with the recording set-up; no healthy-versus-patient claim is made from ISRUC. "
         f"`{'`, `'.join(INTERIM_ARMS)}` rows are **INTERIM** (phase-1 design not complete).", ""]
    f = lambda d: f"{100 * d['diff']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"
    for ds, r in res.items():
        L += [f"## {NAMES[ds]}", ""]
        lead = [p for p in r["pairs"] if p["lead"]]
        if lead:
            L += ["**Lead rows** (seed 0, INTERIM, development-grade; paired subject bootstrap on the fold models both "
                  "arms have):", "", "| pair | role | row | fold models | people | macro-F1 | F1 W | F1 N1 |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
            L += [f"| {p['pair']} | {p['tag']} | {p['row']} | {p['n_models']} | {p['n_subjects']} | {f(p['mf1'])} | "
                  f"{f(p['f1_W'])} | {f(p['f1_N1'])} |" for p in lead]
            L.append("")
        L += ["| arm | row | fold models | people | macro-F1 [95%] | kappa | "
              + " | ".join(f"F1 {s}" for s in STAGES) + " | note |",
              "| --- | --- | ---: | ---: | ---: | ---: | " + " | ".join("---:" for _ in STAGES) + " | --- |"]
        for arm, v in r["arms"].items():
            for k, s in v["rows"].items():
                note = "; ".join(x for x in (role(arm, ds, k), "INTERIM" if arm in INTERIM_ARMS else "",
                                                 "EXPLORATORY" if arm in EXPLORATORY_ARMS else "") if x)
                L.append(f"| {arm} | {k} | {v['n_eval_files']}/{v['n_runs']} | {s['n_subjects']} | {fmt(s['mf1'])} "
                         f"[{fmt(s['mf1_lo'])}, {fmt(s['mf1_hi'])}] | {s['kappa']:.3f} | "
                         + " | ".join(fmt(s[f'f1_{x}']) for x in STAGES) + f" | {note} |")
        if r["pairs"]:
            L += ["", "All pairs:", "", "| pair | role | row | fold models | macro-F1 | F1 N1 | summed non-N1 F1 | people |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
            for p in r["pairs"]:
                L.append(f"| {p['pair']} | {p['tag']} | {p['row']} | {p['n_models']} | {f(p['mf1'])} | {f(p['f1_N1'])} | "
                         f"{f(p['f1_nonN1_sum'])} | {p['n_subjects']} |")
        L.append("")
    Path(out_md).write_text("\n".join(L) + "\n")
    Path(out_json).write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--arms", default="raw,attnsleep,cascade2,cascade2_placebo,raw_whitened,cascade")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default="reports/external_dev")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    res = build(a.arms.split(","), seeds, a.n_boot)
    write(res, seeds, f"{a.out}.md", f"{a.out}.json")
    Path(f"{a.out}_shape_line.txt").write_text(shape_line(res) + "\n")
    print(f"{a.out}.md")


if __name__ == "__main__":
    main()
