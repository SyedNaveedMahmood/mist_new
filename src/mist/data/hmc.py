"""HMC sleep staging database v1.1 (PhysioNet, CC BY 4.0) -> canonical store format.

Phase 2 only (T13). Nothing read here may influence a phase-1 choice.

Facts checked on the files (2026-11-22): 151 recordings SNxxx.edf with
SNxxx_sleepscoring.edf/.txt; EEG F4-M1, C4-M1, O2-M1, C3-M2 at 256 Hz; AASM stage
annotations "Sleep stage W/N1/N2/N3/R" on a 30-s grid from the PSG start, plus one
"Lights off" and one "Lights on" event (description suffixed with "@@<channel>").
The EEG prefilter field differs between recordings: 0.2-35 Hz in 83, 0.2-128 Hz
in 68 (all 151 headers, EEG channels only; MNE reports the strictest filter over
the channels loaded, so reading EMG/ECG too gives misleading values).
Only aggregate demographics are published, so there is no per-person age or sex.
Each recording is one person.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .. import EPOCH_SAMPLES, FS
from .sleepedf import labels_from_annotations

STAGE_MAP = {"Sleep stage W": 0, "Sleep stage N1": 1, "Sleep stage N2": 2, "Sleep stage N3": 3, "Sleep stage R": 4}
EVENTS = ("Lights off", "Lights on")
EEG = ("EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EEG C3-M2")
NAME_RE = re.compile(r"^SN(\d{3})\.edf$")


def find_records(folder: Path) -> list:
    folder = Path(folder)
    out = []
    for psg in sorted(folder.glob("SN*.edf")):
        m = NAME_RE.match(psg.name)
        if not m:
            continue
        hyp = folder / f"SN{m.group(1)}_sleepscoring.edf"
        if not hyp.exists():
            raise FileNotFoundError(f"no scoring file for {psg.name}")
        out.append((psg, hyp))
    return out


def parse_annotations(ann, n_epochs: int) -> tuple:
    """-> (y [n_epochs], lights_off_epoch or None, lights_on_epoch or None)."""
    desc = [str(d).split("@@")[0] for d in ann.description]
    y = labels_from_annotations(ann.onset, ann.duration, desc, n_epochs, stage_map=STAGE_MAP, ignore=EVENTS)
    ev = {d: int(on // 30) for on, d in zip(ann.onset, desc) if d in EVENTS}
    return y, ev.get("Lights off"), ev.get("Lights on")


def read_record(psg: Path, hyp: Path, channels=EEG, l_freq: float = 0.3, h_freq: float = 35.0) -> tuple:
    """Return (x [n_epochs, C, 3000] microvolt at 100 Hz, y, meta)."""
    import mne

    raw = mne.io.read_raw_edf(psg, include=list(channels), preload=True, verbose="ERROR")
    meta = {"fs_orig": float(raw.info["sfreq"]), "prefilter_hp": float(raw.info["highpass"]),
            "prefilter_lp": float(raw.info["lowpass"])}
    if int(round(raw.info["sfreq"])) != FS:
        raw.resample(FS, verbose="ERROR")                     # polyphase with anti-alias filter
    data = raw.get_data(picks=list(channels), units="uV")
    data = mne.filter.filter_data(data, FS, l_freq, h_freq, method="fir", phase="zero", verbose="ERROR")
    n = data.shape[1] // EPOCH_SAMPLES
    x = data[:, : n * EPOCH_SAMPLES].reshape(len(channels), n, EPOCH_SAMPLES).transpose(1, 0, 2)
    y, lo, lon = parse_annotations(mne.read_annotations(hyp), n)
    meta.update(lights_off_epoch=lo, lights_on_epoch=lon, channels=[c.replace("EEG ", "") for c in channels])
    return x.astype(np.float32), y, meta
