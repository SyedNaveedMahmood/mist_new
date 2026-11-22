"""Sleep-EDF Expanded (cassette SC and telemetry ST) -> canonical store.

STATUS: logic unit-tested on synthetic annotations only. It has NOT been run on
the real EDF files in this repository's history (PhysioNet was unreachable where
this was written). Task T1 in docs/TASKS.md verifies it against the audited
counts from the old Proto-sleep pipeline before anything is trained.

Decisions carried over from the Proto-sleep data audit (P1.5 / P2-SC):
  * every complete physical 30-s epoch is kept, with unscored epochs as -1;
  * SC persons 06, 23, 36 and 74 are quarantined by default;
  * hypnogram onsets are taken relative to the PSG start. The seven ST
    hypnograms with a previous-day EDF+ Recordingfield were shown there to share
    the PSG's fixed clock, so no date arithmetic is applied.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .. import EPOCH_SAMPLES, FS, UNSCORED

STAGE_MAP = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,   # R&K 3+4 -> N3
    "Sleep stage ?": UNSCORED, "Movement time": UNSCORED,
}
CHANNELS = {"Fpz-Cz": "EEG Fpz-Cz", "Pz-Oz": "EEG Pz-Oz"}
SC_QUARANTINE = (6, 23, 36, 74)
PSG_RE = re.compile(r"^(SC4|ST7)(\d{2})(\d)[A-Z0-9]{2}-PSG\.edf$")


def parse_name(name: str) -> dict:
    m = PSG_RE.match(name)
    if not m:
        raise ValueError(f"not a Sleep-EDF PSG file name: {name}")
    cohort = "SC" if m.group(1) == "SC4" else "ST"
    return {"cohort": cohort, "subject_num": int(m.group(2)), "night": int(m.group(3)),
            "subject": f"{cohort}:{int(m.group(2)):02d}", "record_id": name[:6]}


def find_pairs(folder: Path) -> list:
    """Pair each PSG with its hypnogram by the first six characters of the name."""
    folder = Path(folder)
    hyps = {p.name[:6]: p for p in folder.glob("*-Hypnogram.edf")}
    pairs = []
    for psg in sorted(folder.glob("*-PSG.edf")):
        key = psg.name[:6]
        if key not in hyps:
            raise FileNotFoundError(f"no hypnogram for {psg.name}")
        pairs.append((psg, hyps[key]))
    return pairs


def labels_from_annotations(onsets, durations, descriptions, n_epochs: int, epoch_s: float = 30.0) -> np.ndarray:
    """Fill a 30-s grid. Unknown descriptions raise; partial epochs stay unscored."""
    y = np.full(n_epochs, UNSCORED, dtype=np.int8)
    for on, du, de in zip(onsets, durations, descriptions):
        de = str(de)
        if de not in STAGE_MAP:
            raise ValueError(f"unexpected annotation {de!r}")
        first = int(np.ceil(on / epoch_s - 1e-6))
        last = int(np.floor((on + du) / epoch_s + 1e-6))
        if first < last:
            y[first:min(last, n_epochs)] = STAGE_MAP[de]
    return y


def read_recording(psg: Path, hyp: Path, l_freq: float = 0.3, h_freq: float = 35.0) -> tuple:
    """Return (x [n_epochs, 2, 3000] microvolt, y [n_epochs], channel names)."""
    import mne

    raw = mne.io.read_raw_edf(psg, include=list(CHANNELS.values()), preload=True, verbose="ERROR")
    if int(round(raw.info["sfreq"])) != FS:
        raise ValueError(f"{psg.name}: expected {FS} Hz, got {raw.info['sfreq']}")
    data = raw.get_data(picks=list(CHANNELS.values()), units="uV")
    data = mne.filter.filter_data(data, FS, l_freq, h_freq, method="fir", phase="zero", verbose="ERROR")
    n_epochs = data.shape[1] // EPOCH_SAMPLES
    x = data[:, : n_epochs * EPOCH_SAMPLES].reshape(2, n_epochs, EPOCH_SAMPLES).transpose(1, 0, 2)
    ann = mne.read_annotations(hyp)
    if ann.onset.size and ann.onset[-1] > data.shape[1] / FS + 30:
        raise ValueError(f"{hyp.name}: annotations run past the end of the PSG")
    y = labels_from_annotations(ann.onset, ann.duration, ann.description, n_epochs)
    return x.astype(np.float32), y, list(CHANNELS.keys())


def read_sc_subjects(xls: Path) -> dict:
    """SC-subjects.xls -> {(subject_num, night): {"age": int, "sex": str, "lights_off": str}}.

    Column names are matched loosely; a layout this does not recognise raises.
    Sex assumes the header "sex (F=1)". Both points are unverified against the
    real file: check them in task T1.
    """
    import pandas as pd

    df = pd.read_excel(xls)
    cols = {str(c).strip().lower(): c for c in df.columns}
    pick = lambda key: next((cols[c] for c in cols if c.startswith(key)), None)
    c_sub, c_night, c_age, c_sex, c_lo = pick("subject"), pick("night"), pick("age"), pick("sex"), pick("lightsoff")
    if None in (c_sub, c_night, c_age):
        raise ValueError(f"unrecognised SC-subjects layout: {list(df.columns)}")
    out = {}
    for _, r in df.iterrows():
        out[(int(r[c_sub]), int(r[c_night]))] = {
            "age": int(r[c_age]),
            "sex": None if c_sex is None else ("F" if int(r[c_sex]) == 1 else "M"),
            "lights_off": None if c_lo is None else str(r[c_lo]),
        }
    return out
