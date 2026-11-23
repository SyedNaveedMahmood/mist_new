"""Common preprocessing for every external dataset (phase 2): native rate -> 100 Hz, 0.3-35 Hz.

One function, one specification, identical for HMC, DOD-H, ISRUC and CAP (D-017):

1. Anti-alias low-pass at the NATIVE rate, fixed in absolute Hz: zero-phase FIR
   (mne.filter.filter_data, firwin, Hamming), passband edge 45 Hz, transition
   5 Hz, so -6 dB at 47.5 Hz and the stopband (about -50 dB) starts at 50 Hz,
   the Nyquist frequency of the 100 Hz output. Filter length follows MNE's
   'auto' rule (3.3 / transition s), so it is the same duration at every rate.
   It is applied even when the native rate is already 100 Hz, so that every
   recording passes through the same chain.
2. Rational polyphase resampling to 100 Hz (scipy.signal.resample_poly). Its own
   Kaiser filter is linear phase with the delay removed (zero phase); it acts on
   a signal already band-limited to 50 Hz.
3. The Sleep-EDF store's band-pass: 0.3-35 Hz zero-phase FIR
   (mne.filter.filter_data, method fir, phase zero, MNE defaults otherwise).

Why 45 Hz and not lower: content that folds below the 0.3-35 Hz band-pass
stopband (43.75 Hz) must start above 56 Hz before resampling, and a 45 Hz
passband edge leaves the 0.3-35 Hz band untouched by step 1, so steps 1+3 act
like step 3 alone inside the band.

Input must be microvolt; the readers handle units.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from .. import EPOCH_SAMPLES, FS

AA_H_FREQ = 45.0          # Hz, passband edge of the anti-alias low-pass
AA_TRANS = 5.0            # Hz, transition width (stopband from 50 Hz)
BP_L_FREQ = 0.3           # Hz, same band-pass as scripts/prepare_sleepedf.py
BP_H_FREQ = 35.0


def ratio(fs_in: float) -> tuple:
    """(up, down) with up/down == FS/fs_in exactly; raises for rates that are not rational enough."""
    fr = Fraction(FS) / Fraction(fs_in).limit_denominator(10000)
    if abs(float(fr) - FS / float(fs_in)) > 1e-9 or fr.denominator > 10000:
        raise ValueError(f"cannot resample {fs_in} Hz to {FS} Hz with a small rational factor")
    return fr.numerator, fr.denominator


def preprocess(data: np.ndarray, fs_in: float) -> np.ndarray:
    """[C, T] microvolt at fs_in -> [C, T'] microvolt at 100 Hz, float64. Channel by channel to bound memory."""
    import mne
    from scipy.signal import resample_poly

    data = np.atleast_2d(np.asarray(data, dtype=np.float64))
    up, down = ratio(fs_in)
    out = []
    for ch in data:
        lp = mne.filter.filter_data(ch[None], float(fs_in), None, AA_H_FREQ, h_trans_bandwidth=AA_TRANS,
                                    method="fir", phase="zero", fir_window="hamming", fir_design="firwin",
                                    verbose="ERROR")[0]
        rs = lp if (up, down) == (1, 1) else resample_poly(lp, up, down)
        out.append(rs)
    out = np.stack(out)
    return mne.filter.filter_data(out, FS, BP_L_FREQ, BP_H_FREQ, method="fir", phase="zero", verbose="ERROR")


def to_epochs(data100: np.ndarray, start_sample: int = 0) -> np.ndarray:
    """[C, T] at 100 Hz -> [n, C, 3000] float32, complete epochs only, grid starting at start_sample."""
    d = data100[:, start_sample:]
    n = d.shape[1] // EPOCH_SAMPLES
    return d[:, : n * EPOCH_SAMPLES].reshape(d.shape[0], n, EPOCH_SAMPLES).transpose(1, 0, 2).astype(np.float32)


def spec() -> dict:
    """Recorded in every meta.json so a store says how it was made."""
    return {"anti_alias": f"zero-phase FIR low-pass at native rate, pass {AA_H_FREQ} Hz, trans {AA_TRANS} Hz "
                          f"(hamming, firwin, mne)", "resample": "scipy.signal.resample_poly to 100 Hz",
            "bandpass": f"{BP_L_FREQ}-{BP_H_FREQ} Hz zero-phase FIR (mne defaults)"}
