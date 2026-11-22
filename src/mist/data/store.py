"""On-disk format for one recording and loaders.

Canonical layout (written by scripts/prepare_sleepedf.py):

    <root>/<COHORT>/<record_id>/x.npy      float32 [n_epochs, n_channels, 3000], microvolt, 100 Hz
                                y.npy      int8    [n_epochs], 0..4 = W,N1,N2,N3,REM, -1 = unscored
                                meta.json  subject, night, cohort, channels, age, sex, lights_off_epoch

Every complete physical 30-s epoch is kept. Views are masks computed on load:

    full  every epoch
    std   30 min of wake kept before the first and after the last sleep epoch.
          This is the literature convention; it uses stage labels, so it is
          flagged label-conditioned. See docs/EXPERIMENT_PLAN.md, section 4.
    lf    label-free: from 30 min before lights-off (subject spreadsheets) for
          10 h, clipped to the recording. ST recordings start at lights-off and
          last 7.5 to 10.7 h, so for ST the window is mostly the whole recording.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import UNSCORED


@dataclass
class Night:
    record_id: str
    subject: str            # cohort-qualified, e.g. "SC:07"
    night: int
    cohort: str
    channels: list
    x: np.ndarray           # memory-mapped
    y: np.ndarray
    meta: dict

    def channel_index(self, name: str) -> int:
        if name not in self.channels:
            raise KeyError(f"{self.record_id}: channel {name!r} not in {self.channels}")
        return self.channels.index(name)

    def view_slice(self, view: str) -> slice:
        if view == "full":
            return slice(0, len(self.y))
        if view == "std":
            return std_slice(self.y)
        if view == "lf":
            lo = self.meta.get("lights_off_epoch")
            if lo is None:
                raise ValueError(f"{self.record_id}: no lights_off_epoch in meta; run scripts/add_lf_window.py")
            return lf_slice(int(lo), len(self.y))
        raise ValueError(view)


LF_BEFORE_EPOCHS = 60       # 30 min before lights-off
LF_LENGTH_EPOCHS = 1200     # 10 h


def std_slice(y) -> slice:
    """Label-conditioned trim: 30 min of wake either side of the sleep period."""
    y = np.asarray(y)
    sleep = np.flatnonzero(y > 0)
    if sleep.size == 0:
        return slice(0, len(y))
    return slice(max(0, int(sleep[0]) - 60), min(len(y), int(sleep[-1]) + 61))


def lf_slice(lights_off_epoch: int, n_epochs: int) -> slice:
    """Label-free window: 30 min before lights-off, 10 h long, clipped to the recording."""
    start = lights_off_epoch - LF_BEFORE_EPOCHS
    a, b = max(0, start), min(n_epochs, start + LF_LENGTH_EPOCHS)
    if b <= a:
        raise ValueError(f"lf window [{start}, {start + LF_LENGTH_EPOCHS}) misses a recording of {n_epochs} epochs")
    return slice(a, b)


def save_night(root: Path, cohort: str, record_id: str, x: np.ndarray, y: np.ndarray, meta: dict) -> Path:
    d = Path(root) / cohort / record_id
    d.mkdir(parents=True, exist_ok=True)
    assert x.ndim == 3 and x.shape[0] == y.shape[0] and x.shape[2] == 3000, (x.shape, y.shape)
    np.save(d / "x.npy", x.astype(np.float32))
    np.save(d / "y.npy", y.astype(np.int8))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return d


def load_night(d: Path) -> Night:
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    return Night(record_id=d.name, subject=meta["subject"], night=int(meta["night"]), cohort=meta["cohort"],
                 channels=list(meta["channels"]), x=np.load(d / "x.npy", mmap_mode="r"),
                 y=np.load(d / "y.npy"), meta=meta)


DEFAULT_SPLIT = Path(__file__).resolve().parents[3] / "splits" / "sleepedf_v1.json"


def subject_of_record(record_id: str) -> str:
    """'SC4071' -> 'SC:07', 'ST7022' -> 'ST:02' (Sleep-EDF record folder names)."""
    cohort = {"SC4": "SC", "ST7": "ST"}.get(record_id[:3])
    if cohort is None or not record_id[3:5].isdigit():
        raise ValueError(f"not a Sleep-EDF record folder: {record_id}")
    return f"{cohort}:{int(record_id[3:5]):02d}"


def lockbox_subjects(split_file: Path = DEFAULT_SPLIT) -> set:
    from .splits import load
    return set(load(split_file)["lockbox"])


def load_cohort(root: Path, cohort: str, subjects: set | None = None, *, split_file: Path = DEFAULT_SPLIT,
                open_lockbox: bool = False) -> list:
    """Load the nights of one cohort, optionally restricted to `subjects`.

    Lockbox people (read from the frozen split file) are never returned unless
    open_lockbox=True, which only task T12 may pass. Asking for one explicitly
    raises; asking for the whole cohort silently leaves them out. Nights are
    selected by folder name before anything is opened, so lockbox files are not
    even memory-mapped.
    """
    base = Path(root) / cohort
    if not base.is_dir():
        raise FileNotFoundError(f"{base} not found; run scripts/prepare_sleepedf.py first")
    locked = set() if open_lockbox else lockbox_subjects(split_file)
    if subjects is not None and set(subjects) & locked:
        raise PermissionError("lockbox people requested; the lockbox is opened once, by task T12 only")
    dirs = [p for p in sorted(base.iterdir()) if (p / "meta.json").exists()]
    keep = [p for p in dirs if subject_of_record(p.name) not in locked
            and (subjects is None or subject_of_record(p.name) in subjects)]
    nights = [load_night(p) for p in keep]
    assert all(n.subject == subject_of_record(p.name) for n, p in zip(nights, keep)), "folder/subject mismatch"
    return nights


def load_legacy_npz(path: Path, cohort: str = "SC", channel: str = "Fpz-Cz", *,
                    split_file: Path = DEFAULT_SPLIT) -> Night:
    """Read a DeepSleepNet-style NPZ (keys x, y, fs) from the old Proto-sleep folders.

    These files were trimmed with an unknown label-conditioned rule, so they are
    for smoke tests and baseline sanity checks only, never for transfer claims.
    The Sleep-EDF-78 legacy folder holds lockbox people; their files are refused
    before they are opened.
    """
    path = Path(path)
    if f"{cohort}:{int(path.stem[3:5]):02d}" in lockbox_subjects(split_file):
        raise PermissionError(f"{path.name} belongs to a lockbox person; the lockbox is opened by T12 only")
    with np.load(path, allow_pickle=False) as d:
        x = np.asarray(d["x"], dtype=np.float32).squeeze()
        y = np.asarray(d["y"]).reshape(-1).astype(np.int8)
    if x.ndim != 2 or x.shape[1] != 3000:
        raise ValueError(f"{path.name}: expected [n, 3000] after squeeze, got {x.shape}")
    stem = path.stem
    subj, night = int(stem[3:5]), int(stem[5])
    return Night(record_id=stem, subject=f"{cohort}:{subj:02d}", night=night, cohort=cohort,
                 channels=[channel], x=x[:, None, :], y=y,
                 meta={"subject": f"{cohort}:{subj:02d}", "night": night, "cohort": cohort,
                       "channels": [channel], "legacy_trimmed": True})


def stage_counts(y: np.ndarray) -> dict:
    return {int(k): int((y == k).sum()) for k in (UNSCORED, 0, 1, 2, 3, 4)}
