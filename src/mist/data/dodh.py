"""Dreem Open Dataset, healthy part (DOD-H) .h5 -> canonical store format. Phase 2 only.

Layout checked on the 25 files (2026-11-24): root attrs `description` (JSON list
of channels with name, domain, fs, unit, path), `duration` (s), `start_time`,
`record_id`; datasets `signals/eeg/<A>_<B>` float32 at 250 Hz, 12 EEG
derivations per file (C3-M2, F4-M1, F3-F4, F3-M2, F4-O2, F3-O1, FP1-F3, FP1-M2,
FP1-O1, FP2-F4, FP2-M1, FP2-O2); `hypnogram` int64, one value per 30 s,
values -1..4, length == duration / 30 == signal length / 7500 in every file.

Units: every EEG channel is labelled "mV", but 30-s standard deviations are
13-88 in these numbers, which is microvolt scale (EEG in mV would be ~0.03).
The numbers are therefore taken as microvolt, and read_record refuses a file
whose median epoch SD is outside [1, 500] (CONDITIONAL: inferred from
magnitudes, contradicting the attribute).

Hypnogram: codes 0..4 taken as W, N1, N2, N3, REM and -1 as unscored
(CONDITIONAL: the files carry no code table; the stage proportions fit this
reading). Whether it is a consensus of the scorers or one scorer: UNKNOWN from
the files (no attribute or dataset says so).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import EPOCH_SAMPLES
from .channels import normalize, pick_rule_channel
from .resample import preprocess, spec, to_epochs


def find_records(folder: Path) -> list:
    return sorted(Path(folder).glob("*.h5"))


def read_record(path: Path) -> tuple:
    """Return (x [n_epochs, C, 3000] microvolt at 100 Hz, y, meta)."""
    import h5py

    with h5py.File(path, "r") as f:
        desc = json.loads(f.attrs["description"])
        eeg = [c for c in desc if c["domain"] == "EEG"]
        fs = {float(c["fs"]) for c in eeg}
        if len(fs) != 1:
            raise ValueError(f"{Path(path).name}: EEG at several rates {fs}")
        fs = fs.pop()
        names = [normalize(c["name"]) for c in eeg]
        if None in names:
            raise ValueError(f"{Path(path).name}: unrecognised EEG name in {[c['name'] for c in eeg]}")
        raw = np.stack([np.asarray(f[c["path"]][:], dtype=np.float64) for c in eeg])
        hyp = np.asarray(f["hypnogram"][:])
        duration = float(f.attrs["duration"])
        units = sorted({c["unit"] for c in eeg})
    n_native = int(round(30 * fs))
    sd = np.median(raw[:, : (raw.shape[1] // n_native) * n_native].reshape(len(eeg), -1, n_native).std(-1))
    if not 1.0 <= sd <= 500.0:
        raise ValueError(f"{Path(path).name}: median epoch SD {sd:.3g}; not microvolt scale")
    if not set(np.unique(hyp)) <= {-1, 0, 1, 2, 3, 4}:
        raise ValueError(f"{Path(path).name}: hypnogram codes {np.unique(hyp)}")
    x = to_epochs(preprocess(raw, fs))
    n_sig = len(x)
    mismatch = {"n_labels": int(len(hyp)), "n_signal_epochs": int(n_sig), "duration_epochs": duration / 30}
    y = np.full(n_sig, -1, dtype=np.int8)
    k = min(n_sig, len(hyp))
    y[:k] = hyp[:k]
    rule, tier = pick_rule_channel(names)
    meta = {"fs_orig": fs, "units_attr": units, "unit_used": "uV (attribute contradicted by magnitude)",
            "median_epoch_sd_native": float(sd), "label_length": mismatch, "channels": names,
            "rule_channel": rule, "rule_tier": tier, "scorer": "UNKNOWN (consensus or single not stated in file)",
            "preprocessing": spec()}
    assert x.shape[2] == EPOCH_SAMPLES
    return x, y, meta
