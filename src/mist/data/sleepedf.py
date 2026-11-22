"""Sleep-EDF Expanded (cassette SC and telemetry ST) -> canonical store.

STATUS: run on the real Sleep-EDF Expanded 1.0.0 files in T1 (2026-11-22); the
built store reproduces the audited Proto-sleep totals (gate G0): SC 74 people,
146 recordings, 397,832 epochs, 396,378 scored; ST 22 people, 44 recordings.

Decisions carried over from the Proto-sleep data audit (P1.5 / P2-SC):
  * every complete physical 30-s epoch is kept, with unscored epochs as -1;
  * SC persons 06, 23, 36 and 74 are quarantined by default;
  * hypnogram onsets are taken relative to the file start, with no date
    arithmetic. Checked on the real files in T1 (2026-11-22): in the seven ST
    nights ST7021, ST7071, ST7092, ST7131, ST7132, ST7141, ST7142 the
    previous-day EDF+ Recordingfield ("Startdate" one day before the fixed
    header date) is present in BOTH the PSG and the hypnogram, because those
    recordings start just after midnight. In all 197 pairs the PSG and the
    hypnogram have identical fixed start date and time, and mne.read_annotations
    agrees with a raw TAL decode, so onsets relative to the file start are correct.
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


def labels_from_annotations(onsets, durations, descriptions, n_epochs: int, epoch_s: float = 30.0,
                            stage_map: dict | None = None, ignore: tuple = ()) -> np.ndarray:
    """Fill a 30-s grid. Unknown descriptions raise; partial epochs stay unscored.

    stage_map defaults to the Sleep-EDF (R&K) map; descriptions in `ignore` are skipped."""
    stage_map = STAGE_MAP if stage_map is None else stage_map
    y = np.full(n_epochs, UNSCORED, dtype=np.int8)
    for on, du, de in zip(onsets, durations, descriptions):
        de = str(de)
        if de in ignore:
            continue
        if de not in stage_map:
            raise ValueError(f"unexpected annotation {de!r}")
        first = int(np.ceil(on / epoch_s - 1e-6))
        last = int(np.floor((on + du) / epoch_s + 1e-6))
        if first < last:
            y[first:min(last, n_epochs)] = stage_map[de]
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


def read_st_subjects(xls: Path) -> dict:
    """ST-subjects.xls -> {(subject_num, night): {"age", "sex", "lights_off", "condition"}}.

    Layout checked against the real file (T1, 2026-11-22): two header rows, then
    columns Nr | Age | M1/F2 | placebo night nr | lights off | temazepam night nr | lights off.
    Sex is coded M=1, F=2 here, the opposite of SC-subjects.xls (F=1).
    """
    import pandas as pd

    df = pd.read_excel(xls, header=None)
    head = " ".join(str(v) for v in df.iloc[:2].to_numpy().ravel()).lower()
    if not ("placebo" in head and "temazepam" in head and "m1/f2" in head):
        raise ValueError(f"unrecognised ST-subjects layout: {df.iloc[:2].to_numpy().tolist()}")
    out = {}
    for r in df.iloc[2:].itertuples(index=False):
        if pd.isna(r[0]):
            continue
        sex = {1: "M", 2: "F"}[int(r[2])]
        for c_night, c_lo, cond in ((3, 4, "placebo"), (5, 6, "temazepam")):
            out[(int(r[0]), int(r[c_night]))] = {"age": int(r[1]), "sex": sex,
                                                 "lights_off": str(r[c_lo]), "condition": cond}
    return out


def psg_start_hms(psg: Path) -> str:
    """Start time of day from the fixed EDF header (bytes 176-184, "hh.mm.ss")."""
    with Path(psg).open("rb") as f:
        b = f.read(184)
    return b[176:184].decode("ascii").replace(".", ":")


def lights_off_epoch(start_hms: str, lights_off_hms: str) -> int:
    """Epoch index of lights-off, counted from the PSG start; may be slightly negative.

    The sheets give clock times without a date. Lights-off is taken as the first
    occurrence after the start, unless that is more than 20 h away, in which case
    it fell just before the start (ST recordings start at lights-off, and the
    sheet's minute resolution can put it up to a minute earlier).
    """
    sec = lambda s: sum(int(float(v)) * m for v, m in zip(str(s).split(":"), (3600, 60, 1)))
    d = (sec(lights_off_hms) - sec(start_hms)) % 86400
    if d > 20 * 3600:
        d -= 86400
    return int(d // 30)


def window_meta(start_hms: str, lights_off_hms, y) -> dict:
    """Fields stored per night for the label-free `lf` view and the std/lf epoch counts."""
    from .store import lf_slice, std_slice

    out = {"psg_start": start_hms, "lights_off_epoch": None, "n_epochs_std": None, "n_epochs_lf": None,
           "lf_clipped_start": None, "lf_clipped_end": None}
    sl = std_slice(y)
    out["n_epochs_std"] = sl.stop - sl.start
    if lights_off_hms not in (None, "", "nan"):
        lo = lights_off_epoch(start_hms, lights_off_hms)
        sl = lf_slice(lo, len(y))
        out.update(lights_off_epoch=lo, n_epochs_lf=sl.stop - sl.start,
                   lf_clipped_start=sl.start > lo - 60, lf_clipped_end=sl.stop < lo - 60 + 1200)
    return out
