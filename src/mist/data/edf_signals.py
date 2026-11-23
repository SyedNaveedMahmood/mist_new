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

    Channels at different native rates are read and preprocessed per rate group, then cut
    to the shortest group (they differ by at most a few samples at 100 Hz). A derived pair
    must have both leads at one rate.
    """
    import mne

    hdr = hdr or read_header(path)
    fs_of = {lab: hdr["fs"][hdr["label"].index(lab)] for _, how in plan for lab in how[1:]}
    by_fs = defaultdict(list)
    for name, how in plan:
        rates = {fs_of[lab] for lab in how[1:]}
        if len(rates) != 1:
            raise ValueError(f"{Path(path).name}: {name} combines leads at rates {rates}")
        by_fs[rates.pop()].append((name, how))
    rows, units, n_times, fs_by = {}, {}, {}, {}
    for fs, group in sorted(by_fs.items()):
        labs = sorted({lab for _, how in group for lab in how[1:]})
        raw = mne.io.read_raw_edf(path, include=labs, preload=True, verbose="ERROR")
        if abs(raw.info["sfreq"] - fs) > 1e-6:
            raise ValueError(f"{Path(path).name}: MNE gave {raw.info['sfreq']} Hz, header says {fs}")
        u = dict(raw._orig_units)
        units.update({lab: u.get(lab, "n/a") for lab in labs})
        sig = {lab: raw.get_data(picks=[lab])[0] * _TO_UV.get(u.get(lab, "n/a"), 1.0) for lab in labs}
        stacked = np.stack([sig[how[1]] if how[0] == "raw" else sig[how[1]] - sig[how[2]] for _, how in group])
        out = resample.preprocess(stacked, fs)
        for (name, _), r in zip(group, out):
            rows[name] = r
            fs_by[name] = float(fs)
        n_times[fs] = raw.n_times / fs
    n = min(len(r) for r in rows.values())
    data = np.stack([rows[name][:n] for name, _ in plan])
    fss = sorted(by_fs)
    info = {"fs_orig": float(fss[0]), "fs_all": [float(f) for f in fss], "fs_by_channel": fs_by,
            "units": units, "unit_assumed_uv": sorted(lab for lab, v in units.items() if v not in _TO_UV),
            "seconds_native": min(n_times.values())}
    return data, info
