"""Aggregation of evaluation cells into condition x stage tables (E0 onward).

Unit of analysis: the test subject. For each subject the confusion matrices of
all runs that tested them (seeds, and folds for sc2st) are summed, so every seed
counts equally and the bootstrap resamples people, not epochs or runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import STAGES

METRICS = ("mf1", "acc", "kappa") + tuple(f"f1_{s}" for s in STAGES) + ("f1_nonN1_sum",)


def collect(paths, cell: str, nights: set | None = None) -> dict:
    """{subject: summed 5x5 cm} over eval JSON files; `nights` restricts to record ids."""
    out = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        c = d["cells"][cell]
        if nights is None:
            items = c["subject_cm"].items()
        else:
            meta = d["night_meta"]
            items = [(meta[r]["subject"], cm) for r, cm in c["night_cm"].items() if r in nights]
        for s, cm in items:
            out[s] = out.get(s, 0) + np.asarray(cm, dtype=np.int64)
    return out


def scores_batch(cm: np.ndarray) -> dict:
    """Vectorised scores for cm [..., 5, 5]; macro-F1 averages classes present (nan-mean)."""
    cm = cm.astype(float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    fp, fn = cm.sum(-2) - tp, cm.sum(-1) - tp
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), np.nan)
    n = cm.sum((-2, -1))
    po = tp.sum(-1) / np.maximum(n, 1)
    pe = (cm.sum(-2) * cm.sum(-1)).sum(-1) / np.maximum(n * n, 1)
    out = {"mf1": np.nanmean(f1, -1), "acc": po, "kappa": (po - pe) / np.maximum(1 - pe, 1e-12)}
    out.update({f"f1_{s}": f1[..., i] for i, s in enumerate(STAGES)})
    out["f1_nonN1_sum"] = f1[..., [0, 2, 3, 4]].sum(-1)          # H5: summed F1 of W, N2, N3, REM
    return out


def summary(cms: dict) -> dict:
    s = scores_batch(np.stack(list(cms.values())).sum(0))
    return {k: float(v) for k, v in s.items()} | {"n_subjects": len(cms)}


def paired_diff(cms_a: dict, cms_b: dict, n_boot: int = 10000, seed: int = 0) -> dict:
    """a - b for every metric, with a 95% percentile interval from a paired subject bootstrap."""
    subs = sorted(cms_a)
    if subs != sorted(cms_b):
        raise ValueError("paired comparison needs identical subject sets")
    A = np.stack([cms_a[s] for s in subs]); B = np.stack([cms_b[s] for s in subs])
    point = {k: float(scores_batch(A.sum(0))[k] - scores_batch(B.sum(0))[k]) for k in METRICS}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(subs), (n_boot, len(subs)))
    w = np.zeros((n_boot, len(subs))); np.add.at(w, (np.arange(n_boot)[:, None], idx), 1)
    sa, sb = scores_batch(np.einsum("bs,sij->bij", w, A)), scores_batch(np.einsum("bs,sij->bij", w, B))
    out = {}
    for k in METRICS:
        d = sa[k] - sb[k]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        out[k] = {"diff": point[k], "lo": float(lo), "hi": float(hi)}
    out["n_subjects"] = len(subs)
    return out


def restrict(cms: dict, subjects) -> dict:
    """Keep only the given subjects (e.g. the old age band inside cv results)."""
    subjects = set(subjects)
    return {k: v for k, v in cms.items() if k in subjects}


def joint_shift_bootstrap(cms_a: dict, cms_b: dict, strata, n_boot: int = 10000, seed: int = 0,
                          metric: str = "mf1") -> dict:
    """Primary H1 statistic (EXPERIMENT_PLAN section 7): equal-weight mean over protocols of a - b.

    cms_a, cms_b: {protocol: {subject: summed cm}}, the same subjects for both arms within a protocol.
    strata: lists of people; each bootstrap resample draws every stratum with replacement once and that
    one draw is applied to all protocols, so a person tested in several protocols moves together.
    Returns per-protocol differences (marginals of the joint resample) and the pooled mean, with 95%
    percentile intervals and a two-sided bootstrap p-value for the pooled effect (for Holm).
    """
    protos = list(cms_a)
    if sorted(protos) != sorted(cms_b) or not protos:
        raise ValueError("both arms need the same, non-empty set of protocols")
    people = [s for st in strata for s in st]
    if len(set(people)) != len(people):
        raise ValueError("strata overlap")
    col = {s: i for i, s in enumerate(people)}
    rng = np.random.default_rng(seed)
    W = np.zeros((n_boot, len(people)))
    off = 0
    for st in strata:
        idx = rng.integers(0, len(st), (n_boot, len(st)))
        np.add.at(W, (np.arange(n_boot)[:, None], off + idx), 1)
        off += len(st)
    per, boot = {}, []
    for p in protos:
        subs = sorted(cms_a[p])
        if subs != sorted(cms_b[p]):
            raise ValueError(f"{p}: paired comparison needs identical subject sets")
        missing = [s for s in subs if s not in col]
        if missing:
            raise ValueError(f"{p}: subjects outside every stratum: {missing[:3]}")
        A = np.stack([cms_a[p][s] for s in subs]); B = np.stack([cms_b[p][s] for s in subs])
        w = W[:, [col[s] for s in subs]]
        d = (scores_batch(np.einsum("bs,sij->bij", w, A))[metric]
             - scores_batch(np.einsum("bs,sij->bij", w, B))[metric])
        point = float(scores_batch(A.sum(0))[metric] - scores_batch(B.sum(0))[metric])
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        per[p] = {"diff": point, "lo": float(lo), "hi": float(hi), "n_subjects": len(subs)}
        boot.append(d)
    pooled_boot = np.mean(boot, 0)          # nan if a resample left a protocol empty (vanishingly rare)
    pooled_boot = pooled_boot[np.isfinite(pooled_boot)]
    point = float(np.mean([per[p]["diff"] for p in protos]))
    lo, hi = np.percentile(pooled_boot, [2.5, 97.5])
    p_boot = float(min(1.0, 2 * min((pooled_boot <= 0).mean(), (pooled_boot >= 0).mean())))
    return {"per_protocol": per, "pooled": {"diff": point, "lo": float(lo), "hi": float(hi), "p_boot": p_boot,
            "n_boot_used": int(pooled_boot.size)}, "protocols": protos, "strata_sizes": [len(s) for s in strata]}


def joint_contrast(cms: dict, contrast, strata, n_boot: int = 10000, seed: int = 0, metrics=METRICS) -> dict:
    """Linear contrast of pooled scores over several (protocol, cell) subject tables, with one joint bootstrap.

    cms: {key: {subject: cm}}; contrast: [(weight, key), ...], e.g. band cost shift minus in-domain =
    [(+1, "sc2st|real"), (-1, "sc2st|band"), (-1, "cv|real"), (+1, "cv|band")]. Each resample draws every
    stratum of people once and applies the draw to every table, so a person in several tables moves together.
    """
    people = [s for st in strata for s in st]
    col = {s: i for i, s in enumerate(people)}
    rng = np.random.default_rng(seed)
    W = np.zeros((n_boot, len(people)))
    off = 0
    for st in strata:
        idx = rng.integers(0, len(st), (n_boot, len(st)))
        np.add.at(W, (np.arange(n_boot)[:, None], off + idx), 1)
        off += len(st)
    point = {m: 0.0 for m in metrics}
    boot = {m: np.zeros(n_boot) for m in metrics}
    for w, key in contrast:
        subs = sorted(cms[key])
        A = np.stack([cms[key][s] for s in subs])
        full, res = scores_batch(A.sum(0)), scores_batch(np.einsum("bs,sij->bij", W[:, [col[s] for s in subs]], A))
        for m in metrics:
            point[m] += w * float(full[m])
            boot[m] += w * res[m]
    out = {}
    for m in metrics:
        lo, hi = np.nanpercentile(boot[m], [2.5, 97.5])
        out[m] = {"diff": point[m], "lo": float(lo), "hi": float(hi)}
    return out
