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
