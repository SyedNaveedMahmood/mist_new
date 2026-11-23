"""In-house REML mixed model (src/mist/mixed.py), used for the pooled H1 effect."""
import numpy as np

from mist.mixed import fit_reml


def design(n_proto=5, n_fold=5, n_seed=3):
    rows = [(a, p, f, s) for p in range(n_proto) for f in range(n_fold) for s in range(n_seed) for a in (0, 1)]
    a, p, f, s = map(np.array, zip(*rows))
    return a, p, f, s


def test_recovers_effect_and_variance_components():
    rng = np.random.default_rng(0)
    a, p, f, s = design()
    est, cover, vp = [], 0, []
    for rep in range(15):
        up, uf, us = rng.normal(0, 0.03, 5), rng.normal(0, 0.01, 5), rng.normal(0, 0.005, 3)
        y = 0.75 + 0.01 * a + up[p] + uf[f] + us[s] + rng.normal(0, 0.01, len(a))
        r = fit_reml(y, np.column_stack([np.ones_like(a), a]), {"protocol": p, "fold": f, "seed": s})
        est.append(r["beta"][1]); vp.append(r["variances"]["protocol"])
        cover += r["ci95"][1, 0] <= 0.01 <= r["ci95"][1, 1]
    assert abs(np.mean(est) - 0.01) < 0.001                     # unbiased for the arm effect
    assert cover >= 12                                         # about 95% coverage (>= 80% of 15)
    assert 0.0004 < np.mean(vp) < 0.002                        # true 0.0009; one fit with 5 levels is noisy, so the mean


def test_equals_ols_when_there_is_no_grouping_variance():
    rng = np.random.default_rng(1)
    a, p, f, s = design()
    y = 0.7 + 0.02 * a + rng.normal(0, 0.01, len(a))
    X = np.column_stack([np.ones_like(a), a]).astype(float)
    r = fit_reml(y, X, {"protocol": p, "fold": f, "seed": s})
    b_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(r["beta"], b_ols, atol=2e-3)
    assert r["variances"]["protocol"] < 1e-4 and r["converged"]
