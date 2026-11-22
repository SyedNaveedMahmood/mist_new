"""Metrics and subject-level statistics. The unit of analysis is the subject."""
from __future__ import annotations

import numpy as np

from . import N_CLASSES, STAGES


def confusion(y: np.ndarray, p: np.ndarray, n: int = N_CLASSES) -> np.ndarray:
    m = y >= 0
    return np.bincount(y[m] * n + p[m], minlength=n * n).reshape(n, n)


def scores(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    n = cm.sum()
    po = tp.sum() / max(n, 1)
    pe = (cm.sum(0) * cm.sum(1)).sum() / max(n * n, 1)
    out = {"acc": float(po), "mf1": float(np.nanmean(f1)), "kappa": float((po - pe) / max(1 - pe, 1e-12)), "n": int(n)}
    out.update({f"f1_{s}": float(v) for s, v in zip(STAGES, f1)})
    return out


def per_subject(y: np.ndarray, p: np.ndarray, subj: np.ndarray) -> dict:
    """{subject: confusion matrix}. Pool nights of the same person first."""
    return {s: confusion(y[subj == s], p[subj == s]) for s in np.unique(subj)}


def summarize(cms: dict) -> dict:
    """Pooled scores plus the mean of per-subject macro-F1."""
    pooled = scores(sum(cms.values()))
    pooled["mf1_subject_mean"] = float(np.mean([scores(c)["mf1"] for c in cms.values()]))
    pooled["n_subjects"] = len(cms)
    return pooled


def paired_bootstrap(cms_a: dict, cms_b: dict, metric: str = "mf1", n_boot: int = 10000, seed: int = 0) -> dict:
    """Difference a - b of a pooled metric, resampling subjects with replacement.

    Both dicts must hold the same subjects (same test set, two models or two
    conditions). Returns the point estimate and a 95% percentile interval.
    """
    subs = sorted(cms_a)
    if subs != sorted(cms_b):
        raise ValueError("paired bootstrap needs identical subject sets")
    A = np.stack([cms_a[s] for s in subs]); B = np.stack([cms_b[s] for s in subs])
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(subs), len(subs))
        diffs[i] = scores(A[idx].sum(0))[metric] - scores(B[idx].sum(0))[metric]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(scores(A.sum(0))[metric] - scores(B.sum(0))[metric]), "lo": float(lo), "hi": float(hi),
            "n_subjects": len(subs)}


def normalized_gap(oracle_mf1: float, transfer_mf1: float) -> float:
    """(in-domain oracle - transfer) / oracle, same architecture and same test subjects."""
    return (oracle_mf1 - transfer_mf1) / max(oracle_mf1, 1e-12)
