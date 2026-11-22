"""Night-relative event detectors used only to NAME shape slots after training (T8).

They are deliberately simple and label-free: each threshold is relative to the
same night's own distribution, so amplitude calibration does not matter. They are
not validated clinical detectors and their output is never a training target.

  spindle     11-16 Hz RMS envelope above 3x the night median, lasting 0.5-2 s
  slow_wave   0.5-2 Hz trough below -2.5x the night median absolute amplitude,
              trough-to-peak 0.25-1.0 s apart (a half-wave of 0.5-2 Hz)
  alpha       8-12 Hz RMS envelope above 2x the night median, lasting >= 1 s

All take x [n_epochs, T] at fs Hz (one night, one channel) and return a boolean
mask [n_epochs, T] marking samples inside detected events.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def _band(x, fs, lo, hi):
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x.reshape(-1)).reshape(x.shape)


def _rms_env(y, fs, win_s):
    w = max(1, int(win_s * fs))
    k = np.ones(w) / w
    return np.sqrt(np.convolve(y.reshape(-1) ** 2, k, mode="same")).reshape(y.shape)


def _runs(mask_flat, min_len, max_len=None):
    out = np.zeros_like(mask_flat)
    d = np.diff(np.concatenate([[0], mask_flat.astype(np.int8), [0]]))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a >= min_len and (max_len is None or b - a <= max_len):
            out[a:b] = True
    return out


def spindles(x, fs=100, k=3.0):
    env = _rms_env(_band(x, fs, 11, 16), fs, 0.3)
    m = env > k * np.median(env)
    return _runs(m.reshape(-1), int(0.5 * fs), int(2.0 * fs)).reshape(x.shape)


def alpha(x, fs=100, k=2.0):
    env = _rms_env(_band(x, fs, 8, 12), fs, 0.5)
    m = env > k * np.median(env)
    return _runs(m.reshape(-1), int(1.0 * fs)).reshape(x.shape)


def slow_waves(x, fs=100, k=2.5):
    y = _band(x, fs, 0.5, 2.0).reshape(-1)
    thr = -k * np.median(np.abs(y))
    out = np.zeros(y.shape, dtype=bool)
    # troughs: local minima below threshold
    tr = np.flatnonzero((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:]) & (y[1:-1] < thr)) + 1
    for t in tr:
        nxt = t + int(0.25 * fs) + np.argmax(y[t + int(0.25 * fs): t + int(1.0 * fs)]) if t + int(1.0 * fs) < len(y) else None
        if nxt is None:
            continue
        a = max(0, t - int(0.25 * fs))
        out[a:nxt] = True
    return out.reshape(x.shape)


DETECTORS = {"spindle": spindles, "slow_wave": slow_waves, "alpha": alpha}


def token_overlap(mask, token_index):
    """Fraction of each token's samples inside events; token_index rows are (epoch, start_sample, length)."""
    return np.array([mask[e, s:s + n].mean() for e, s, n in token_index])


def name_slots(top_overlap: dict, base: dict, min_lift=2.0, min_overlap=0.3) -> dict:
    """top_overlap[det][slot] and base[det] -> {slot: (name or None, lifts)}."""
    n_slots = len(next(iter(top_overlap.values())))
    out = {}
    for k in range(n_slots):
        lifts = {d: float(top_overlap[d][k] / max(base[d], 1e-9)) for d in top_overlap}
        best = max(lifts, key=lifts.get)
        ok = lifts[best] >= min_lift and top_overlap[best][k] >= min_overlap
        out[k] = {"name": best if ok else None, "lift": lifts,
                  "overlap": {d: float(top_overlap[d][k]) for d in top_overlap}}
    return out
