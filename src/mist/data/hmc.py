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

from .channels import pick_rule_channel, plan_channels
from .edf_header import channel_info, parse_prefilter, read_header
from .edf_signals import read_plan
from .resample import spec, to_epochs
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


def read_record(psg: Path, hyp: Path, channels=EEG) -> tuple:
    """Return (x [n_epochs, C, 3000] microvolt at 100 Hz, y, meta).

    Preprocessing is the shared phase-2 chain (resample.preprocess, D-017): native-rate
    anti-alias low-pass, polyphase resampling to 100 Hz, 0.3-35 Hz zero-phase band-pass.
    """
    import mne

    hdr = read_header(psg)
    plan = plan_channels(channels)
    data, info = read_plan(psg, plan, hdr)
    x = to_epochs(data)
    n = len(x)
    y, lo, lon = parse_annotations(mne.read_annotations(hyp), n)
    names = [c for c, _ in plan]
    pre = sorted({channel_info(hdr, c)["prefilter"] for c in channels})
    hp_lp = [parse_prefilter(p) for p in pre]
    rule, tier = pick_rule_channel(names)
    meta = {**info, "prefilter_eeg": pre,
            "prefilter_hp": min((h for h, _ in hp_lp if h is not None), default=None),
            "prefilter_lp": min((lp for _, lp in hp_lp if lp is not None), default=None),
            "lights_off_epoch": lo, "lights_on_epoch": lon, "channels": names,
            "rule_channel": rule, "rule_tier": tier, "preprocessing": spec()}
    return x, y, meta
