#!/usr/bin/env python
"""Build condition x stage tables with subject-bootstrap intervals.

    python scripts/aggregate.py e0            # writes reports/e0.md and reports/e0_tables.json

Inputs are runs/<arm-run>/eval_*.json. For each arm and protocol, every test
subject's confusion matrices are summed over the runs that tested them (seeds;
and folds for sc2st), then scored pooled. Intervals: paired subject bootstrap,
10,000 resamples, 95% percentile. Condition effects are reported as real minus
condition (positive = the intervention hurts = the model relies on what it
destroyed). ST is development-grade; all shifts are within-database.
"""
import argparse
import glob
import json
from pathlib import Path

from mist import STAGES
from mist.aggregate import collect, joint_shift_bootstrap, paired_diff, restrict, summary
from mist.data import splits as S
from mist.metrics import normalized_gap

CONDITIONS = ("real", "global_phase", "local_phase", "flatten_tilt")
COLS = ("mf1",) + tuple(f"f1_{s}" for s in STAGES)

E0_ARMS = ("attnsleep", "raw")
E0_PROTOCOLS = {  # key: (eval file name, night subset key, label)
    "cv": ("eval_test_cv_Fpz-Cz.json", None, "in-domain (cv, Fpz-Cz, std)"),
    "montage": ("eval_test_cv_Pz-Oz.json", None, "montage shift (train Fpz-Cz, test Pz-Oz, std)"),
    "sc2st": ("eval_test_sc2st_Fpz-Cz.json", None, "cohort shift sc2st, all ST nights (std)"),
    "sc2st_placebo": ("eval_test_sc2st_Fpz-Cz.json", "placebo", "sc2st, ST placebo nights (std)"),
    "sc2st_temazepam": ("eval_test_sc2st_Fpz-Cz.json", "temazepam", "sc2st, ST temazepam nights (std)"),
    "cv_lf": ("eval_test_cv_Fpz-Cz_lf.json", None, "in-domain, label-free lf window (sensitivity)"),
    "sc2st_lf": ("eval_test_sc2st_Fpz-Cz_lf.json", None, "sc2st, label-free lf window (sensitivity)"),
}


def night_subset(paths, condition):
    if condition is None:
        return None
    meta = json.loads(Path(paths[0]).read_text())["night_meta"]
    return {r for r, m in meta.items() if m.get("condition") == condition}


def fmt(x, pts=True):
    return f"{100 * x:.1f}" if pts else f"{x:.3f}"


def fmt_ci(d):
    return f"{100 * d['diff']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"


def build_e0(n_boot):
    tables = {}
    for arm in E0_ARMS:
        runs = sorted(glob.glob(f"runs/e0_{arm}_cv[0-4]_s[0-9]"))
        for key, (fname, sub, label) in E0_PROTOCOLS.items():
            paths = [f"{r}/{fname}" for r in runs if Path(f"{r}/{fname}").exists()]
            if not paths:
                continue
            nights = night_subset(paths, sub)
            cms = {c: collect(paths, c, nights) for c in CONDITIONS}
            t = {"label": label, "n_eval_files": len(paths), "runs": runs,
                 "scores": {c: summary(cms[c]) for c in CONDITIONS},
                 "real_minus": {c: paired_diff(cms["real"], cms[c], n_boot=n_boot) for c in CONDITIONS[1:]}}
            tables.setdefault(arm, {})[key] = t
    return tables


def g1_decision(tables):
    """D-005: G1 passes iff, in-domain, some model has a stage whose F1 drops by >= 1 point under
    local_phase with the 95% interval excluding 0. Shift protocols are reported, not used."""
    hits = []
    for arm, t in tables.items():
        if "cv" not in t:
            continue
        mr = t["cv"]["real_minus"]["local_phase"]
        for s in STAGES:
            d = mr[f"f1_{s}"]
            if d["diff"] >= 0.01 and d["lo"] > 0:
                hits.append(f"{arm} {s} {fmt_ci(d)}")
    return {"passed": bool(hits), "qualifying_cells": hits}


AUDIT = Path("runs/e0/surrogate_audit.json")


def probe_fidelity_section() -> list:
    """How well each surrogate keeps what it claims to keep (scripts/audit_surrogates.py; no model, no test people)."""
    L = ["## Probe fidelity and how to read these tables", "",
         "**Correction (2026-11-23).** The first E0 tables used a `local_phase` that lost about half the power in "
         "every band (overlap-add bug; registry `correction_e0_local_phase`). It was fixed with a regression test and "
         "every E0 evaluation was rerun; only the `local_phase` cells changed (real, global_phase and flatten_tilt "
         "are identical in all 150 files). The pre-fix report is kept as reports/e0_PREFIX_local_phase_bug.md.", ""]
    if AUDIT.exists():
        a = json.loads(AUDIT.read_text())
        bands = [k for k in next(iter(a["conditions"].values())) if k not in ("rms_log10_ratio_median", "short_spec_1s")]
        L += [f"Median |log10(surrogate / real)| per epoch on {a['n_epochs']} real {a['channel']} epochs "
              f"({len(a['nights'])} fold-0 training nights). 0.1 = 26% power change. Band power: Welch, 4 s; "
              "short-time spectrum: 1 s Hann, hop 0.5 s (the MIST token scale).", "",
              "| surrogate | RMS (signed) | " + " | ".join(bands) + " | 1 s short-time spectrum |",
              "| --- | ---: | " + " | ".join("---:" for _ in bands) + " | ---: |"]
        for c, r in a["conditions"].items():
            L.append(f"| {c} | {r['rms_log10_ratio_median']:+.3f} | " + " | ".join(f"{r[b]['all']:.3f}" for b in bands)
                     + f" | {r['short_spec_1s']['all']:.3f} |")
        L.append("")
    L += ["Reading the effects:",
          "- `local_phase` keeps band power within about 0.03-0.04 log10 from 1 Hz up (0.09 below 1 Hz, limited by "
          "its 2 s window). It does **not** keep the 1 s short-time spectrum, so its cost includes reliance on the "
          "timing of spectral events at sub-2 s scale (spindle and K-complex onsets), not only on waveform shape "
          "within a cycle.",
          "- Every surrogate also moves the input away from the training distribution. `flatten_tilt` keeps phase "
          "exactly and still collapses both models, so part of any surrogate's cost is fragility to unfamiliar "
          "input, not reliance on what it destroys. The effects are upper bounds on reliance.",
          "- G1 is decided by the pre-registered rule (D-005), which these caveats do not change; its margin "
          "(smallest qualifying effect several points, intervals far from 0) is large relative to the fidelity "
          "errors above.", ""]
    return L


def write_e0(tables, g1, out_md, out_json):
    L = ["# E0: do trained stagers use morphology? (gate G1, hypothesis H0)", "",
         "Generated by `scripts/aggregate.py e0`. Numbers in points (x100). Effects are real minus condition: "
         "positive means the surrogate hurts, i.e. the model relies on what it destroys. Intervals: paired subject "
         "bootstrap, 10,000 resamples, 95% percentile, subjects' confusion matrices summed over seeds (and folds for "
         "sc2st). All shifts are **within-database** (Sleep-EDF); ST is a **development-grade** target already "
         "exposed in earlier work; SC/ST person linkage is unknown.", "",
         "Surrogates: `global_phase` keeps the epoch amplitude spectrum and destroys shape and within-epoch timing; "
         "`local_phase` keeps the short-time (2 s) amplitude spectrum and destroys waveform shape only (this is the "
         "morphology-reliance probe, MR); `flatten_tilt` removes the broadband 1/f tilt and keeps phase.", "",
         f"## Gate G1: **{'PASSED' if g1['passed'] else 'FAILED'}**", "",
         "Rule (docs/DECISION_LOG.md D-005): in-domain, at least one model has at least one stage whose F1 drops "
         "by >= 1 point under `local_phase` with the 95% interval excluding 0.", ""]
    L += [f"- {h}" for h in g1["qualifying_cells"]] or ["- no qualifying cell"]
    L.append("")
    L += probe_fidelity_section()
    for arm, ts in tables.items():
        L += [f"## {arm}", ""]
        for key, t in ts.items():
            L += [f"### {t['label']}", "", f"{t['n_eval_files']} evaluation files, "
                  f"{t['scores']['real']['n_subjects']} test subjects.", "",
                  "| condition | macro-F1 | " + " | ".join(f"F1 {s}" for s in STAGES) + " | kappa |",
                  "| --- | ---: | " + " | ".join("---:" for _ in STAGES) + " | ---: |"]
            for c in CONDITIONS:
                s = t["scores"][c]
                L.append(f"| {c} | {fmt(s['mf1'])} | " + " | ".join(fmt(s[f'f1_{x}']) for x in STAGES)
                         + f" | {fmt(s['kappa'], False)} |")
            L += ["", "| real minus | macro-F1 | " + " | ".join(f"F1 {s}" for s in STAGES) + " |",
                  "| --- | ---: | " + " | ".join("---:" for _ in STAGES) + " |"]
            for c in CONDITIONS[1:]:
                d = t["real_minus"][c]
                L.append(f"| {c} | {fmt_ci(d['mf1'])} | " + " | ".join(fmt_ci(d[f'f1_{x}']) for x in STAGES) + " |")
            L.append("")
    Path(out_md).write_text("\n".join(L) + "\n")
    Path(out_json).write_text(json.dumps({"g1": g1, "tables": tables}, indent=1))


# ----------------------------------------------------------------------------- E1
E1_ARMS = ("raw", "attnsleep", "r", "gr", "grm_placebo", "grm_nocritic", "grm", "m", "grm_loose", "grm_absolute",
           "cascade", "cascade_placebo", "cascade_nocritic", "cascade_bands", "cascade_broad", "cascade_absolute",
           "cascade2", "cascade2_placebo", "cascade2_absolute", "cascade2w32", "cascade2w32_placebo", "cascade2w32_absolute")
SEEDS = None        # --seeds: restrict every arm to these seeds (interim seed-0 analysis, EXPERIMENT_PLAN section 12)
SHIFTS = ("montage", "sc2st", "young2old", "old2young")
# EXPERIMENT_PLAN section 11 (D7, cascade main model): the cascade pairs decide; the parallel pairs are the
# topology replicate; cascade - grm is exploratory and outside the Holm family.
# EXPERIMENT_PLAN section 12 (revision 3, 2026-11-23): cascade2 is the main model; if the validation gate
# fell back to band_width 32, the cascade2w32 pairs take that role.
PAIRS = [("cascade2", "cascade2_placebo", "H1"), ("cascade2_placebo", "raw", "H3"),
         ("cascade2", "cascade2_absolute", "H3"), ("cascade2", "raw", "G2"),
         ("cascade2", "cascade", "own-rate vs full-rate (descriptive)"), ("cascade2", "grm", "exploratory"),
         ("cascade2w32", "cascade2w32_placebo", "H1 (gate fallback)"), ("cascade2w32_placebo", "raw", "H3 (gate fallback)"),
         ("cascade2w32", "cascade2w32_absolute", "H3 (gate fallback)"), ("cascade2w32", "raw", "G2 (gate fallback)"),
         ("cascade", "cascade_placebo", "H1 full-rate (section 11)"), ("cascade_placebo", "raw", "H3"),
         ("cascade", "cascade_absolute", "H3"), ("cascade", "raw", "G2"), ("cascade", "grm", "G2 diagnostic / exploratory"),
         ("cascade_bands", "cascade", "broadband channel"), ("cascade_broad", "cascade", "per-band channels"),
         ("grm", "gr", "H1 parallel replicate"), ("grm_placebo", "gr", "G3 (parallel)"), ("gr", "raw", "H3 parallel"),
         ("grm", "grm_absolute", "H3 parallel"), ("grm", "raw", "G2 parallel")]


def e1_runs(arm, block):
    runs = _e1_runs(arm, block)
    skips = Path("runs/session_skips.json")
    if skips.exists():                                   # owner rule B3: a fold skipped for one arm is dropped for all
        bad = set(json.loads(skips.read_text()))
        runs = [r for r in runs if Path(r).name not in bad]
    if SEEDS is not None:
        runs = [r for r in runs if int(r.rsplit("_s", 1)[1]) in SEEDS]
    return runs


def _e1_runs(arm, block):
    if block == "cv":
        runs = glob.glob(f"runs/e1_{arm}_cv[0-4]_s[0-9]")
        if arm in ("raw", "attnsleep"):
            runs += glob.glob(f"runs/e0_{arm}_cv[0-4]_s[0-9]")      # identical recipe and config (D-006)
        return sorted(runs)
    if block == "cvpz":
        return sorted(glob.glob(f"runs/e1pz_{arm}_cv[0-4]_s[0-9]"))
    return sorted(glob.glob(f"runs/e1_{arm}_{block}_s[0-9]"))


def e1_cells(arm, sp):
    """{protocol: {"cms": subject cms (real), "oracle": cms or None, "label": str}} for one arm."""
    def get(block, fname, nights_cond=None):
        paths = [f"{r}/{fname}" for r in e1_runs(arm, block) if Path(f"{r}/{fname}").exists()]
        if not paths:
            return None, 0
        get.last = (paths, nights_cond)
        return collect(paths, "real", night_subset(paths, nights_cond)), len(paths)
    out = {}
    get.last = None
    cv, n = get("cv", "eval_test_cv_Fpz-Cz.json")
    if cv:
        out["cv"] = {"cms": cv, "src": get.last, "oracle": None, "n": n, "label": "in-domain (cv, Fpz-Cz, std)"}
    pz, n = get("cvpz", "eval_test_cv_Pz-Oz.json")
    if pz:
        out["cvpz"] = {"cms": pz, "src": get.last, "oracle": None, "n": n, "label": "in-domain, trained and tested on Pz-Oz"}
    m, n = get("cv", "eval_test_cv_Pz-Oz.json")
    if m:
        out["montage"] = {"cms": m, "src": get.last, "oracle": pz, "n": n, "label": "montage: train Fpz-Cz, test Pz-Oz (oracle: Pz-Oz cv)"}
    mr, n = get("cvpz", "eval_test_cv_Fpz-Cz.json")
    if mr:
        out["montage_rev"] = {"cms": mr, "src": get.last, "oracle": cv, "n": n, "label": "montage reverse: train Pz-Oz, test Fpz-Cz (oracle: Fpz-Cz cv)"}
    for key, cond, lab in (("sc2st", None, "all ST nights"), ("sc2st_placebo", "placebo", "ST placebo nights"),
                           ("sc2st_temazepam", "temazepam", "ST temazepam nights")):
        c, n = get("cv", "eval_test_sc2st_Fpz-Cz.json", cond)
        if c:
            out[key] = {"cms": c, "src": get.last, "oracle": None, "n": n, "label": f"cohort shift sc2st, {lab} (no oracle)"}
    for p, band in (("young2old", "old"), ("old2young", "young")):
        c, n = get(p, f"eval_test_{p}_Fpz-Cz.json")
        if c:
            orc = restrict(cv, sp["sets"][band]) if cv else None
            out[p] = {"cms": c, "src": get.last, "oracle": orc, "n": n, "label": f"age shift {p} (oracle: cv restricted to the {band} band)"}
    for key, fname, lab in (("cv_lf", "eval_test_cv_Fpz-Cz_lf.json", "in-domain, lf window"),
                            ("sc2st_lf", "eval_test_sc2st_Fpz-Cz_lf.json", "sc2st, lf window")):
        c, n = get("cv", fname)
        if c:
            out[key] = {"cms": c, "src": get.last, "oracle": None, "n": n, "label": lab + " (sensitivity)"}
    return out


def balanced_cms(src_a, src_b):
    """Pairs use only the (fold, seed) runs both arms have, so seeds and folds are balanced between the arms."""
    (pa, cond), (pb, _) = src_a, src_b
    ka = {_run_key(str(Path(p).parent)): p for p in pa}
    kb = {_run_key(str(Path(p).parent)): p for p in pb}
    common = sorted(set(ka) & set(kb))
    if not common:
        return None, None, 0
    fa, fb = [ka[k] for k in common], [kb[k] for k in common]
    return (collect(fa, "real", night_subset(fa, cond)), collect(fb, "real", night_subset(fb, cond)), len(common))


def build_e1(n_boot):
    sp = S.load("splits/sleepedf_v1.json")
    cells = {a: e1_cells(a, sp) for a in E1_ARMS}
    cells = {a: c for a, c in cells.items() if c}
    tables = {}
    for a, cs in cells.items():
        for key, c in cs.items():
            sm = summary(c["cms"])
            if c["oracle"]:
                orc = c["oracle"]
                if sorted(orc) == sorted(c["cms"]):
                    sm["oracle_mf1"] = summary(orc)["mf1"]
                    sm["normalized_gap"] = normalized_gap(sm["oracle_mf1"], sm["mf1"])
            sm["n_eval_files"] = c["n"]
            tables.setdefault(key, {"label": c["label"], "arms": {}})["arms"][a] = sm
    pairs = []
    for x, y, tag in PAIRS:
        if x not in cells or y not in cells:
            continue
        for key in cells[x]:
            if key in cells[y]:
                a, b, nk = balanced_cms(cells[x][key]["src"], cells[y][key]["src"])
                if a is None:
                    pairs.append({"pair": f"{x} - {y}", "tag": tag, "protocol": key, "skipped": "no common (fold, seed) runs"})
                    continue
                if sorted(a) != sorted(b):
                    pairs.append({"pair": f"{x} - {y}", "tag": tag, "protocol": key, "skipped": "subject sets differ"})
                    continue
                pairs.append({"pair": f"{x} - {y}", "tag": tag, "protocol": key, "n_runs_per_arm": nk,
                              **paired_diff(a, b, n_boot=n_boot)})
    return tables, pairs


def e1_gates(pairs):
    g = {}
    def find(pair, proto):
        return next((p for p in pairs if p["pair"] == pair and p["protocol"] == proto and "mf1" in p), None)
    for main in ("cascade2", "cascade2w32"):
        p = find(f"{main} - raw", "cv")
        if p:   # G2 (section 12): the main model within 1.5 macro-F1 points of raw in-domain (point estimate)
            g["G2"] = {"passed": p["mf1"]["diff"] >= -0.015, f"{main}_minus_raw_mf1": p["mf1"]}
    p = find("cascade - raw", "cv")
    if p:   # section 11 rule for the full-rate cascade (reported only)
        g["G2_fullrate"] = {"passed": p["mf1"]["diff"] >= -0.015, "cascade_minus_raw_mf1": p["mf1"]}
    p = find("grm - raw", "cv")
    if p:   # same rule for the parallel topology (reported; decides only the parallel arms)
        g["G2_parallel"] = {"passed": p["mf1"]["diff"] >= -0.015, "grm_minus_raw_mf1": p["mf1"]}
    p = find("grm_placebo - gr", "cv")
    if p:   # G3 (parallel arms only): fails only if the placebo beats gr with the interval excluding 0
        g["G3"] = {"passed": not (p["mf1"]["lo"] > 0), "placebo_minus_gr_mf1": p["mf1"]}
    return g


def write_e1(tables, pairs, gates, out_md, out_json, seeds=None):
    L = ["# E1: factor arms (H1, H3, H5; gates G2, G3)", "",
         "Main model: `cascade2`, the own-rate cascade (revision 3, DECISIONS D8; EXPERIMENT_PLAN section 12). "
         "H1 is decided by cascade2 - cascade2_placebo, pooled over the four shift protocols by the joint "
         "subject bootstrap of `aggregate.py h1` (EXPERIMENT_PLAN section 7, D-024); the pairs below are per protocol. The full-rate `cascade` (seed 0) and the parallel pairs are reported for "
         "reference; cascade2 - grm is exploratory (outside Holm). G3 applies to the parallel arms only.", "",
         "Generated by `scripts/aggregate.py e1`. Points (x100). Subject-level: each person's confusion matrices "
         "summed over seeds (and folds where a person is tested by several fold models), paired subject bootstrap "
         "10,000 resamples. All shifts are **within-database** (Sleep-EDF). ST is **development-grade**. "
         "Normalised gap = (oracle - transfer) / oracle, same test people; blank where no oracle exists or "
         "the oracle's subject set differs.", ""]
    if seeds is not None:
        L += [f"**INTERIM: seeds {sorted(seeds)} only.** Not the confirmatory analysis (3 seeds).", ""]
    L += ["## Gates", ""]
    for k, v in gates.items():
        key = [x for x in v if x != "passed"][0]
        L.append(f"- **{k}: {'PASSED' if v['passed'] else 'FAILED'}** ({key} {fmt_ci(v[key])})")
    if not gates:
        L.append("- not yet decidable (arms missing)")
    L.append("")
    for key, t in tables.items():
        L += [f"## {t['label']}", "", "| arm | runs | macro-F1 | kappa | " + " | ".join(f"F1 {s}" for s in STAGES)
              + " | oracle | gap |", "| --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in STAGES) + " | ---: | ---: |"]
        for a, s in t["arms"].items():
            orc = f"{fmt(s['oracle_mf1'])}" if "oracle_mf1" in s else ""
            gap = f"{s['normalized_gap']:.3f}" if "normalized_gap" in s else ""
            L.append(f"| {a} | {s['n_eval_files']} | {fmt(s['mf1'])} | {fmt(s['kappa'], False)} | "
                     + " | ".join(fmt(s[f'f1_{x}']) for x in STAGES) + f" | {orc} | {gap} |")
        L.append("")
    L += ["## Decisive pairs (paired subject bootstrap)", "",
          "Each pair uses only the (fold, seed) runs both arms have (runs per arm column).", "",
          "| pair | role | protocol | runs per arm | macro-F1 | F1 N1 | summed non-N1 F1 | n subjects |",
          "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for p in pairs:
        if "skipped" in p:
            L.append(f"| {p['pair']} | {p['tag']} | {p['protocol']} | | skipped: {p['skipped']} | | | |")
        else:
            L.append(f"| {p['pair']} | {p['tag']} | {p['protocol']} | {p['n_runs_per_arm']} | {fmt_ci(p['mf1'])} | {fmt_ci(p['f1_N1'])} | "
                     f"{fmt_ci(p['f1_nonN1_sum'])} | {p['n_subjects']} |")
    Path(out_md).write_text("\n".join(L) + "\n")
    Path(out_json).write_text(json.dumps({"gates": gates, "tables": tables, "pairs": pairs}, indent=1))



# ----------------------------------------------------------------------------- H1 and E3
# EXPERIMENT_PLAN section 7 (owner correction, 2026-11-24, D-024). Primary H1 statistic: per shift protocol the
# paired subject bootstrap of main - placebo; pooled = equal-weight mean of the four shift effects, interval from
# one joint bootstrap over people (SC development and ST as two strata). In-domain is reported beside, not pooled.
# The run-level mixed model is secondary (shift protocols only); no claim rests on it.
H1_SHIFTS = {"montage": ("cv", "eval_test_cv_Pz-Oz.json"), "sc2st": ("cv", "eval_test_sc2st_Fpz-Cz.json"),
             "young2old": ("young2old", "eval_test_young2old_Fpz-Cz.json"),
             "old2young": ("old2young", "eval_test_old2young_Fpz-Cz.json")}
H1_INDOMAIN = ("cv", "eval_test_cv_Fpz-Cz.json")
H1_MIN_GAIN = 0.02
H1_CONFIRMATORY = Path("reports/h1_confirmatory.json")


def _run_key(run):
    import re
    m = re.search(r"_cv(\d)_s(\d+)$", run)
    return (int(m.group(1)), int(m.group(2))) if m else (0, int(run.rsplit("_s", 1)[1]))


def h1_pairs(main, placebo, block, fname):
    """{(fold, seed): (main file, placebo file)} where both arms have the evaluation (balanced pairs only)."""
    have = []
    for arm in (main, placebo):
        have.append({_run_key(r): f"{r}/{fname}" for r in e1_runs(arm, block) if Path(r, fname).exists()})
    return {k: (have[0][k], have[1][k]) for k in sorted(set(have[0]) & set(have[1]))}


def h1_needed(proto):
    block = H1_SHIFTS[proto][0] if proto in H1_SHIFTS else H1_INDOMAIN[0]
    return {(f, s) for f in range(5) for s in range(3)} if block == "cv" else {(0, s) for s in range(3)}


def h1_rows(main, placebo):
    """Secondary analysis: run-level macro-F1 per (arm, shift protocol, fold, seed), balanced pairs only."""
    from mist import metrics
    rows = []
    for proto, (block, fname) in H1_SHIFTS.items():
        for (f, sd), files in h1_pairs(main, placebo, block, fname).items():
            for arm, fn in zip((main, placebo), files):
                rows.append((arm, proto, f, sd, metrics.scores(sum(collect([fn], "real").values()))["mf1"]))
    return rows


def h1_mixed(rows, main):
    import numpy as np
    from mist.mixed import fit_reml
    arm = np.array([1.0 if r[0] == main else 0.0 for r in rows])
    y = np.array([r[4] for r in rows])
    fit = fit_reml(y, np.column_stack([np.ones_like(arm), arm]),
                   {"protocol": [r[1] for r in rows], "fold": [r[2] for r in rows], "seed": [r[3] for r in rows]})
    return {"effect_mf1": float(fit["beta"][1]), "se": float(fit["se"][1]), "ci95": [float(v) for v in fit["ci95"][1]],
            "variances": fit["variances"], "converged": fit["converged"], "n_rows": len(rows),
            "description": "variability over seeds and folds on the same people (secondary; no claim rests on it). "
                           "Run-level macro-F1 ~ arm + (1|protocol) + (1|fold) + (1|seed), REML, shift protocols "
                           "only; normal 95% interval, approximate"}


def build_h1(main="cascade2", placebo="cascade2_placebo", confirmatory=False, n_boot=10000):
    if confirmatory and H1_CONFIRMATORY.exists():
        raise SystemExit(f"the confirmatory H1 analysis is run once only; {H1_CONFIRMATORY} exists")
    sp = S.load("splits/sleepedf_v1.json")
    cms_a, cms_b, keys = {}, {}, {}
    for proto, (block, fname) in H1_SHIFTS.items():
        pr = h1_pairs(main, placebo, block, fname)
        if pr:
            keys[proto] = set(pr)
            cms_a[proto] = collect([a for a, _ in pr.values()], "real")
            cms_b[proto] = collect([b for _, b in pr.values()], "real")
    if not cms_a:
        return None
    ind = h1_pairs(main, placebo, *H1_INDOMAIN)
    complete = (set(keys) == set(H1_SHIFTS) and all(h1_needed(p) <= keys[p] for p in H1_SHIFTS)
                and h1_needed("cv") <= set(ind))
    if confirmatory and not complete:
        raise SystemExit("confirmatory H1 needs 3 seeds x 5 folds of both arms and the age-shift block (owner B7)")
    joint = joint_shift_bootstrap(cms_a, cms_b, [list(sp["dev"]), list(sp["st"])], n_boot=n_boot)
    per = {p: paired_diff(cms_a[p], cms_b[p], n_boot=n_boot) | {"n_runs_per_arm": len(keys[p])} for p in cms_a}
    indomain = None
    if ind:
        indomain = paired_diff(collect([a for a, _ in ind.values()], "real"),
                               collect([b for _, b in ind.values()], "real"), n_boot=n_boot) | {"n_runs_per_arm": len(ind)}
    pooled = joint["pooled"]
    decision = None
    if complete:
        decision = {"supported": bool(pooled["diff"] >= H1_MIN_GAIN and pooled["lo"] > 0),
                    "rule": "pooled gain >= 2 macro-F1 points and the 95% interval excludes 0 (Holm across the "
                            "confirmatory hypotheses uses p_boot)"}
    seeds = sorted({s for k in keys.values() for _, s in k})
    rows = h1_rows(main, placebo)
    out = {"pair": f"{main} - {placebo}", "label": "CONFIRMATORY" if confirmatory else "INTERIM",
           "complete": complete, "seeds": seeds, "protocols": joint["protocols"],
           "pooled": pooled, "per_protocol": per, "joint_marginals": joint["per_protocol"],
           "in_domain_beside": indomain, "decision": decision,
           "secondary_mixed_model": h1_mixed(rows, main) if len({r[1] for r in rows}) and len(rows) >= 4 else None,
           "method": "per protocol: paired subject bootstrap, subject confusion matrices summed over folds and "
                     "seeds; pooled: equal-weight mean of the shift-protocol macro-F1 differences, one joint "
                     f"bootstrap ({n_boot} resamples) over people, strata SC development ({len(sp['dev'])}) and ST "
                     f"({len(sp['st'])}), 95% percentile interval (EXPERIMENT_PLAN section 7, D-024)"}
    if confirmatory:
        H1_CONFIRMATORY.write_text(json.dumps(out, indent=1))
    return out


def write_h1(r, out_md):
    lab = r["label"] + ("" if r["label"] == "CONFIRMATORY" else f", seeds {r['seeds']}")
    L = [f"# H1 ({lab}): {r['pair']}", "",
         "Within-database shifts (Sleep-EDF); ST is development-grade. Points (x100). " + r["method"] + ".", ""]
    if r["label"] != "CONFIRMATORY":
        L += [f"**INTERIM.** Seeds {r['seeds']}; pooled over {len(r['protocols'])} of 4 shift protocols "
              f"({', '.join(r['protocols'])}). Not the confirmatory analysis; changes no plan (owner B6).", ""]
    p = r["pooled"]
    L += ["## Pooled shift effect (primary statistic)", "",
          f"- macro-F1 {fmt_ci(p)}, bootstrap p {p['p_boot']:.4f}" +
          (f"; decision: **{'SUPPORTED' if r['decision']['supported'] else 'NOT SUPPORTED'}** ({r['decision']['rule']})"
           if r["decision"] else "; no decision (design incomplete)"), "",
          "## Per protocol (paired subject bootstrap)", "",
          "| protocol | runs per arm | n subjects | macro-F1 | F1 N1 | summed non-N1 F1 | " + " | ".join(f"F1 {s}" for s in STAGES if s != "N1") + " |",
          "| --- | ---: | ---: | ---: | ---: | ---: | " + " | ".join("---:" for s in STAGES if s != "N1") + " |"]
    rows = list(r["per_protocol"].items()) + ([("in-domain (beside, not pooled)", r["in_domain_beside"])] if r["in_domain_beside"] else [])
    for name, d in rows:
        L.append(f"| {name} | {d['n_runs_per_arm']} | {d['n_subjects']} | {fmt_ci(d['mf1'])} | {fmt_ci(d['f1_N1'])} | "
                 f"{fmt_ci(d['f1_nonN1_sum'])} | " + " | ".join(fmt_ci(d[f'f1_{s}']) for s in STAGES if s != "N1") + " |")
    m = r["secondary_mixed_model"]
    if m:
        L += ["", "## Secondary: mixed model", "",
              f"- {100 * m['effect_mf1']:+.2f} points, 95% interval [{100 * m['ci95'][0]:+.2f}, {100 * m['ci95'][1]:+.2f}], "
              f"{m['n_rows']} run-level rows. {m['description']}."]
    Path(out_md).write_text("\n".join(L) + "\n")


E3_FILES = {"cv": ("eval_test_cv_Fpz-Cz.json", "eval_e3_cv_Fpz-Cz.json"),
            "montage": ("eval_test_cv_Pz-Oz.json", "eval_e3_cv_Pz-Oz.json"),
            "sc2st": ("eval_test_sc2st_Fpz-Cz.json", "eval_e3_sc2st_Fpz-Cz.json")}


def build_e3(arm, n_boot):
    """Cost of every E3 intervention, real minus intervened, per protocol (paired subject bootstrap)."""
    out = {}
    for proto, (f_real, f_e3) in E3_FILES.items():
        runs = [r for r in e1_runs(arm, "cv") if Path(r, f_real).exists() and Path(r, f_e3).exists()]
        if not runs:
            continue
        real = collect([f"{r}/{f_real}" for r in runs], "real")
        cells = json.loads(Path(runs[0], f_e3).read_text())["cells"]
        out[proto] = {"n_runs": len(runs), "cost": {
            c: paired_diff(real, collect([f"{r}/{f_e3}" for r in runs], c), n_boot=n_boot) for c in cells}}
    return out


def write_e3(arm, res, out_md):
    bands = [c for c in next(iter(res.values()))["cost"] if "destroy_" in c and "destroy_M" not in c] if res else []
    L = [f"# E3: band x stage (cost of destroying each band's phase), {arm}", "",
         "Real minus intervened, points (x100), paired subject bootstrap 95% intervals. Positive = the model relies "
         "on that band's shape. Within-database shifts; ST development-grade.", ""]
    for proto, r in res.items():
        L += [f"## {proto} ({r['n_runs']} runs)", "", "| intervention | macro-F1 | " + " | ".join(f"F1 {s}" for s in STAGES) + " |",
              "| --- | ---: | " + " | ".join("---:" for _ in STAGES) + " |"]
        for c, d in r["cost"].items():
            L.append(f"| {c} | {fmt_ci(d['mf1'])} | " + " | ".join(fmt_ci(d[f'f1_{x}']) for x in STAGES) + " |")
        L.append("")
    Path(out_md).write_text("\n".join(L) + "\n")
    return bands

def build_pair(x, y, protocols=("cv", "montage", "sc2st"), n_boot=10000):
    """One paired comparison x - y on the (fold, seed) runs both arms have (e.g. EXPLORATORY pairs)."""
    sp = S.load("splits/sleepedf_v1.json")
    cx, cy = e1_cells(x, sp), e1_cells(y, sp)
    out = []
    for key in protocols:
        if key in cx and key in cy:
            a, b, nk = balanced_cms(cx[key]["src"], cy[key]["src"])
            if a is not None and sorted(a) == sorted(b):
                out.append({"protocol": key, "label": cx[key]["label"], "n_runs_per_arm": nk,
                            **paired_diff(a, b, n_boot=n_boot)})
    return out


def write_pair(x, y, rows, label, out_md):
    L = [f"## {x} - {y} ({label}; seeds {sorted(SEEDS) if SEEDS is not None else 'all'})", "",
         "Paired subject bootstrap, 10,000 resamples, 95% percentile; only the (fold, seed) runs both arms have. "
         "Points (x100). Within-database shifts; ST development-grade.", "",
         "| protocol | runs per arm | people | macro-F1 | " + " | ".join(f"F1 {s}" for s in STAGES) + " |",
         "| --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in STAGES) + " |"]
    for r in rows:
        L.append(f"| {r['label']} | {r['n_runs_per_arm']} | {r['n_subjects']} | {fmt_ci(r['mf1'])} | "
                 + " | ".join(fmt_ci(r[f'f1_{s}']) for s in STAGES) + " |")
    Path(out_md).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["e0", "e1", "h1", "e3", "pair"])
    ap.add_argument("--pair", default=None, help="pair: 'x,y' (x - y)")
    ap.add_argument("--label", default="EXPLORATORY", help="pair: label printed in the heading")
    ap.add_argument("--confirmatory", action="store_true", help="h1: the single confirmatory run (owner B7)")
    ap.add_argument("--arm", default="cascade2", help="e3: arm; h1: main arm (placebo = <arm>_placebo)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seeds", default=None, help="e1, h1: comma list, e.g. 0 for the interim seed-0 analysis")
    ap.add_argument("--out", default="reports/e1", help="e1: output stem (.md and _tables.json)")
    a = ap.parse_args()
    global SEEDS
    if a.seeds is not None:
        SEEDS = {int(v) for v in a.seeds.split(",")}
    Path("reports").mkdir(exist_ok=True)
    if a.experiment == "e0":
        tables = build_e0(a.n_boot)
        g1 = g1_decision(tables)
        write_e0(tables, g1, "reports/e0.md", "reports/e0_tables.json")
        print("G1", g1)
    elif a.experiment == "h1":
        r = build_h1(a.arm, f"{a.arm}_placebo", a.confirmatory, a.n_boot)
        if r is None:
            raise SystemExit("no paired runs yet")
        stem = ("reports/h1_confirmatory" if a.confirmatory
                else a.out if a.out != "reports/e1" else "reports/h1_interim" + (f"_s{a.seeds.replace(',', '')}" if a.seeds else ""))
        if not a.confirmatory:
            Path(f"{stem}.json").write_text(json.dumps(r, indent=1))
        write_h1(r, f"{stem}.md")
        print(stem)
    elif a.experiment == "pair":
        x, y = a.pair.split(",")
        rows = build_pair(x, y, n_boot=a.n_boot)
        out = a.out if a.out != "reports/e1" else f"reports/pair_{x}_minus_{y}" + (f"_s{a.seeds.replace(',', '')}" if a.seeds else "")
        write_pair(x, y, rows, a.label, f"{out}.md")
        Path(f"{out}.json").write_text(json.dumps(rows, indent=1))
        print(out)
    elif a.experiment == "e3":
        res = build_e3(a.arm, a.n_boot)
        write_e3(a.arm, res, f"reports/e3_{a.arm}.md")
        Path(f"reports/e3_{a.arm}.json").write_text(json.dumps(res, indent=1))
    else:
        tables, pairs = build_e1(a.n_boot)
        gates = e1_gates(pairs)
        write_e1(tables, pairs, gates, f"{a.out}.md", f"{a.out}_tables.json", SEEDS)
        print("E1 gates", gates)


if __name__ == "__main__":
    main()
