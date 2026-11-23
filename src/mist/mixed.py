"""Linear mixed model with crossed random intercepts, fitted by REML (EXPERIMENT_PLAN section 7, H1).

    y = X beta + sum_k Z_k u_k + e,   u_k ~ N(0, s2_k I),   e ~ N(0, s2_e I)

Written in-house (numpy / scipy) because statsmodels is not a project dependency and no new download is
allowed. Sizes are small (H1: at most 2 arms x 5 protocols x 5 folds x 3 seeds = 150 rows), so V is formed
densely. Inference on beta uses the GLS covariance (X' V^-1 X)^-1 at the REML estimate, with a normal 95%
interval; with few levels per random factor this interval is approximate and is reported as such.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _design(codes) -> np.ndarray:
    codes = np.asarray(codes)
    levels = {v: i for i, v in enumerate(sorted(set(codes.tolist())))}
    Z = np.zeros((len(codes), len(levels)))
    Z[np.arange(len(codes)), [levels[v] for v in codes.tolist()]] = 1.0
    return Z


def fit_reml(y, X, groups: dict) -> dict:
    """y [n], X [n, p] fixed design, groups {name: level code per row}. Returns beta, se, ci95, variances."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    n, p = X.shape
    G = {k: _design(v) @ _design(v).T for k, v in groups.items()}
    names = list(G)

    from scipy.linalg import cho_factor, cho_solve

    def parts(theta):
        s2 = np.exp(theta)
        V = s2[-1] * np.eye(n) + sum(s2[i] * G[k] for i, k in enumerate(names))
        cf = cho_factor(V, lower=True)
        Vi_X = cho_solve(cf, X)
        XtViX = X.T @ Vi_X
        beta = np.linalg.solve(XtViX, Vi_X.T @ y)
        r = y - X @ beta
        return cf, XtViX, beta, r

    def neg2reml(theta):
        try:
            cf, XtViX, beta, r = parts(theta)
        except np.linalg.LinAlgError:
            return 1e12
        logdetV = 2 * np.log(np.diag(cf[0])).sum()
        sign, logdetX = np.linalg.slogdet(XtViX)
        return logdetV + logdetX + r @ cho_solve(cf, r)

    v0 = np.log(max(np.var(y - X @ np.linalg.lstsq(X, y, rcond=None)[0]), 1e-12))
    bounds = [(v0 - 25, v0 + 10)] * (len(names) + 1)
    best = minimize(neg2reml, np.full(len(names) + 1, v0 - 1), method="L-BFGS-B", bounds=bounds)
    alt = minimize(neg2reml, np.r_[np.full(len(names), v0 - 8), v0], method="L-BFGS-B", bounds=bounds)
    if alt.fun < best.fun:                                   # variance components can sit at the boundary
        best = alt
    cf, XtViX, beta, r = parts(best.x)
    cov = np.linalg.inv(XtViX)
    se = np.sqrt(np.diag(cov))
    s2 = np.exp(best.x)
    return {"beta": beta, "se": se, "ci95": np.stack([beta - 1.96 * se, beta + 1.96 * se], 1),
            "variances": {**{k: float(s2[i]) for i, k in enumerate(names)}, "residual": float(s2[-1])},
            "n": n, "converged": bool(best.success), "neg2reml": float(best.fun)}
