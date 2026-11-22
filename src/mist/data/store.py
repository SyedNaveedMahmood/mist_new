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
            sleep = np.flatnonzero((self.y > 0))
            if sleep.size == 0:
                return slice(0, len(self.y))
            return slice(max(0, sleep[0] - 60), min(len(self.y), sleep[-1] + 61))
        raise ValueError(view)


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


def load_cohort(root: Path, cohort: str, subjects: set | None = None) -> list:
    base = Path(root) / cohort
    if not base.is_dir():
        raise FileNotFoundError(f"{base} not found; run scripts/prepare_sleepedf.py first")
    nights = [load_night(p) for p in sorted(base.iterdir()) if (p / "meta.json").exists()]
    if subjects is not None:
        nights = [n for n in nights if n.subject in subjects]
    return nights


def load_legacy_npz(path: Path, cohort: str = "SC", channel: str = "Fpz-Cz") -> Night:
    """Read a DeepSleepNet-style NPZ (keys x, y, fs) from the old Proto-sleep folders.

    These files were trimmed with an unknown label-conditioned rule, so they are
    for smoke tests and baseline sanity checks only, never for transfer claims.
    """
    path = Path(path)
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
