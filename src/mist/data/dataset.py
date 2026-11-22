"""Sequence dataset: L consecutive epochs of one channel plus the night's statistics."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..models.factorizer import Factorizer, night_reference
from .store import Night


class NightView:
    """One night, one channel, one view, with label-free night statistics per scale."""

    def __init__(self, night: Night, channel: str, view: str, factorizers: list):
        self.night = night
        sl = night.view_slice(view)
        self.ch = night.channel_index(channel)
        self.start, self.stop = sl.start, sl.stop
        self.y = np.asarray(night.y[sl], dtype=np.int64)
        x = torch.from_numpy(np.ascontiguousarray(night.x[sl, self.ch]))
        self.stats = night_reference(x, factorizers)

    def __len__(self):
        return len(self.y)

    def x(self, a: int, b: int) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(self.night.x[self.start + a:self.start + b, self.ch]))


class SequenceDataset(Dataset):
    """train=True: one random window per index. train=False: tiles every night
    without gaps; the last window is aligned to the end and `keep` marks which
    positions are new, so every epoch is scored exactly once."""

    def __init__(self, nights: list, channel: str, scales, view: str = "std", seq_len: int = 20,
                 train: bool = True, fs: int = 100, f_lo: float = 0.5, f_hi: float = 35.0,
                 windows_per_night: int = 64):
        fzs = [Factorizer(fs, w, h, f_lo, f_hi) for w, h in scales]
        self.views = [NightView(n, channel, view, fzs) for n in nights]
        self.views = [v for v in self.views if len(v) >= seq_len]
        self.L, self.train = seq_len, train
        if train:
            self.index = [(i, None) for i, _ in enumerate(self.views) for _ in range(windows_per_night)]
        else:
            self.index = []
            for i, v in enumerate(self.views):
                starts = list(range(0, len(v) - seq_len + 1, seq_len))
                if starts[-1] + seq_len < len(v):
                    starts.append(len(v) - seq_len)
                self.index += [(i, s) for s in starts]

    def __len__(self):
        return len(self.index)

    def class_counts(self) -> np.ndarray:
        y = np.concatenate([v.y for v in self.views])
        return np.bincount(y[y >= 0], minlength=5)

    def __getitem__(self, i):
        vi, s = self.index[i]
        v = self.views[vi]
        if s is None:
            s = int(torch.randint(0, len(v) - self.L + 1, (1,)))
        keep = np.ones(self.L, dtype=bool)
        if not self.train and s % self.L != 0:          # end-aligned last window
            keep[: self.L - (len(v) - (len(v) // self.L) * self.L)] = False
        item = {"x": v.x(s, s + self.L), "y": torch.from_numpy(v.y[s:s + self.L]),
                "keep": torch.from_numpy(keep), "night": vi, "start": s}
        for group, st in v.stats.items():
            for name, t in st.items():
                item[f"{group}.{name}"] = t
        return item


def split_stats(batch: dict, device) -> dict:
    """Collated batch -> {"night": {"envelope", "scale"}, "s0": {"mag_med", "mag_iqr"}, ...}."""
    out = {}
    for key, t in batch.items():
        if "." in key:
            s, name = key.split(".")
            out.setdefault(s, {})[name] = t.to(device)
    return out
