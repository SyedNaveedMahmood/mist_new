"""Read chosen EDF channels in microvolt at their native rate, then the common preprocessing.

Units: MNE scales channels whose EDF dimension is uV/mV/V to volt. A channel
with an empty or unknown dimension comes back as the raw physical value; the
readers treat that number as microvolt (seen on CAP n13-n15, physical range
+-156; the audit checks amplitudes). Channels are read in groups of equal
native rate, because MNE upsamples lower-rate channels to the highest rate in
one read.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from . import resample
from .edf_header import read_header

_TO_UV = {"µV": 1e6, "uV": 1e6, "mV": 1e6, "V": 1e6, "nV": 1e6}


def read_plan(path: Path, plan: list, hdr: dict | None = None) -> tuple:
    """plan = [(store name, ("raw", label) | ("diff", a, b))] -> (data100 [C, T] uV at 100 Hz, info).

    All planned channels must share one native rate after grouping, otherwise a
    ValueError names the rates (no recording seen so far needs mixing).
    """
    import mne

    hdr = hdr or read_header(path)
    labels = sorted({lab for _, how in plan for lab in how[1:]})
    by_fs = defaultdict(list)
    for lab in labels:
        by_fs[hdr["fs"][hdr["label"].index(lab)]].append(lab)
    if len(by_fs) != 1:
        raise ValueError(f"{Path(path).name}: planned channels at several rates {dict(by_fs)}")
    (fs, labs), = by_fs.items()
    raw = mne.io.read_raw_edf(path, include=labs, preload=True, verbose="ERROR")
    if abs(raw.info["sfreq"] - fs) > 1e-6:
        raise ValueError(f"{Path(path).name}: MNE gave {raw.info['sfreq']} Hz, header says {fs}")
    units = dict(raw._orig_units)
    sig = {}
    for lab in labs:
        u = units.get(lab, "n/a")
        sig[lab] = raw.get_data(picks=[lab])[0] * _TO_UV.get(u, 1.0)
    rows = [sig[how[1]] if how[0] == "raw" else sig[how[1]] - sig[how[2]] for _, how in plan]
    data = resample.preprocess(np.stack(rows), fs)
    info = {"fs_orig": float(fs), "units": {lab: units.get(lab, "n/a") for lab in labs},
            "unit_assumed_uv": sorted(lab for lab in labs if units.get(lab, "n/a") not in _TO_UV),
            "seconds_native": raw.n_times / fs}
    return data, info
