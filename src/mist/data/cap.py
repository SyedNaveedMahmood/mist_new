"""CAP Sleep Database 1.0.0 (raw EDF + RemLogic .txt) -> canonical store format. Phase 2 only.

Facts checked on the files (2026-11-24, all 108 recordings in RECORDS):
- Hypnogram .txt: a header block, then a table whose header row starts with
  "Sleep Stage" (with or without a "Position" column). Stage rows have Event
  SLEEP-S0..S4, SLEEP-REM, SLEEP-MT; every stage row has duration 30 s. Other
  rows are CAP A-phases (MCAP-*) and are ignored. Times are clock times
  ("22:09:33" or "4.54.16"); the date is not given, so a time that jumps back
  by more than an hour is taken as the next day.
- R&K -> AASM: S0=W, S1=N1, S2=N2, S3 and S4=N3, REM=REM; MT and anything else
  unscored (-1).
- Alignment: each stage row is placed at (time - EDF start). The epoch grid
  starts at the first stage row's offset modulo 30 s (5 recordings start off
  the EDF's 30-s grid), so the signal before it forms unscored epochs and every
  stage row falls on a grid epoch. Rows outside the signal are dropped and
  counted; rows off the grid raise. More than MAX_ROWS_BEFORE rows before the
  EDF start means the clocks disagree, and the record is refused (nfle27: the
  hypnogram starts 48 min before the EDF and ends 2.7 h before it).
- EDF quirks: sampling rate 100-512 Hz; units uV, mV or empty (empty = physical
  value taken as uV, see edf_signals.py); n13 and n14 have record count -1 in
  the header and brux1's header claims more records than the file holds (MNE
  infers the count from the file size; the audit logs both). Bipolar labels
  vary (F4-C4, F3A2, C4-A1, 'LOC / A2', ...); a few recordings store monopolar
  leads (C3, A1, ...), from which the usual bipolar pairs are formed.
- Channels with an EEG-like label but an EMG-like hardware band (high-pass
  >= 5 Hz, e.g. F7-T3/F8-T4 at 10-100 Hz in some 'ins' recordings) are not
  kept as EEG.
- Diagnosis group = file-name prefix. One recording per person (CONDITIONAL:
  database description; nothing in the files links recordings).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .channels import normalize, pick_rule_channel, plan_channels
from .edf_header import parse_prefilter, read_header, seconds_of_day
from .edf_signals import read_plan
from .resample import spec, to_epochs

STAGE = {"SLEEP-S0": 0, "SLEEP-S1": 1, "SLEEP-S2": 2, "SLEEP-S3": 3, "SLEEP-S4": 3, "SLEEP-REM": 4,
         "SLEEP-MT": -1, "SLEEP-UNSCORED": -1}
GROUP_RE = re.compile(r"^(nfle|narco|brux|plm|rbd|ins|sdb|n)(\d+)$")
DERIVE = ("Fp2-F4", "F4-C4", "C4-P4", "P4-O2", "Fp1-F3", "F3-C3", "C3-P3", "P3-O1", "F8-T4", "T4-T6",
          "F7-T3", "T3-T5", "C4-A1", "C3-A2", "O2-A1", "O1-A2")
MAX_EEG_HP = 5.0
MAX_ROWS_BEFORE = 10      # more stage rows than this before the EDF start: alignment doubtful, record refused


def group_of(record: str) -> str:
    m = GROUP_RE.match(record)
    if not m:
        raise ValueError(f"not a CAP record name: {record}")
    return m.group(1)


def find_records(folder: Path) -> tuple:
    """-> ([(record, edf, txt)], [(record, reason)]) in RECORDS order."""
    folder = Path(folder)
    names = [l.strip().removesuffix(".edf") for l in (folder / "RECORDS").read_text().splitlines() if l.strip()]
    found, skipped = [], []
    for r in names:
        edf, txt = folder / f"{r}.edf", folder / f"{r}.txt"
        if not edf.exists() or not txt.exists():
            skipped.append((r, "edf or txt missing"))
        else:
            found.append((r, edf, txt))
    return found, skipped


def parse_hypnogram(txt: Path) -> tuple:
    """-> (list of (seconds since first-day midnight, stage or -1), notes dict)."""
    lines = Path(txt).read_text(errors="replace").splitlines()
    head = next(i for i, l in enumerate(lines) if l.startswith("Sleep Stage"))
    cols = lines[head].split("\t")
    ie = cols.index("Event")
    it = next(i for i, c in enumerate(cols) if c.startswith("Time"))
    rows, day, prev, notes = [], 0, None, {"truncated_rows": 0, "unknown_events": {}}
    for l in lines[head + 1:]:
        p = l.rstrip("\r").split("\t")
        if not l.strip():
            continue
        if len(p) <= ie:
            notes["truncated_rows"] += 1
            continue
        ev = p[ie].strip()
        if not ev.startswith("SLEEP"):
            continue
        if ev not in STAGE:
            notes["unknown_events"][ev] = notes["unknown_events"].get(ev, 0) + 1
        t = seconds_of_day(p[it])
        if prev is not None and t + day < prev - 3600:
            day += 86400
        rows.append((t + day, STAGE.get(ev, -1)))
        prev = t + day
    return rows, notes


def align(rows: list, start_s: int, n_seconds: float) -> tuple:
    """Place stage rows on a 30-s grid. -> (y, origin_s, notes)."""
    if not rows:
        raise ValueError("no stage rows")
    # only the first row is wrapped to the nearest day; later rows keep their distance from it,
    # so recordings longer than 12 h are not folded back
    off0 = rows[0][0] - start_s
    off0 = off0 - 86400 if off0 > 43200 else (off0 + 86400 if off0 < -43200 else off0)
    off = [off0 + t - rows[0][0] for t, _ in rows]
    origin = off[0] % 30
    n = int((n_seconds - origin) // 30)
    y = np.full(n, -1, dtype=np.int8)
    notes = {"first_row_offset_s": off[0], "grid_origin_s": origin, "rows_before_signal": 0,
             "rows_after_signal": 0, "rows": len(rows)}
    for o, st in zip(off, (s for _, s in rows)):
        k, rem = divmod(o - origin, 30)
        if rem:
            raise ValueError(f"stage row at offset {o} s is off the 30-s grid")
        if k < 0:
            notes["rows_before_signal"] += 1
        elif k >= n:
            notes["rows_after_signal"] += 1
        else:
            y[k] = st
    if notes["rows_before_signal"] > MAX_ROWS_BEFORE:
        raise ValueError(f"{notes['rows_before_signal']} stage rows lie before the EDF start; alignment doubtful")
    return y, origin, notes


def eeg_plan(hdr: dict) -> tuple:
    """-> (plan, excluded labels with reason)."""
    keep, excluded = [], []
    for lab, pre in zip(hdr["label"], hdr["prefilter"]):
        hp, _ = parse_prefilter(pre)
        if hp is not None and hp >= MAX_EEG_HP:
            if normalize(lab) is not None:
                excluded.append((lab, f"hardware high-pass {hp} Hz"))
        else:
            keep.append(lab)
    plan = plan_channels(keep, DERIVE)
    return plan, excluded


def read_record(edf: Path, txt: Path) -> tuple:
    """Return (x [n_epochs, C, 3000] microvolt at 100 Hz, y, meta)."""
    hdr = read_header(edf)
    plan, excluded = eeg_plan(hdr)
    if not plan:
        raise ValueError(f"{Path(edf).name}: no EEG channel recognised")
    data, info = read_plan(edf, plan, hdr)
    rows, hnotes = parse_hypnogram(txt)
    y, origin, anotes = align(rows, seconds_of_day(hdr["start_time"]), info["seconds_native"])
    x = to_epochs(data, start_sample=int(origin * 100))
    if len(x) != len(y):
        y = y[: len(x)] if len(y) > len(x) else np.concatenate([y, np.full(len(x) - len(y), -1, np.int8)])
    names = [c for c, _ in plan]
    rule, tier = pick_rule_channel(names)
    labs = sorted({lab for _, how in plan for lab in how[1:]})
    rule_src = dict(plan)[rule][1] if rule else None
    i = hdr["label"].index(rule_src) if rule_src else None
    meta = {**info, "fs_orig": info["fs_by_channel"][rule] if rule else info["fs_orig"],
            "group": group_of(Path(edf).stem),
            "prefilter_eeg": sorted({hdr["prefilter"][hdr["label"].index(lab)] for lab in labs}),
            "phys_range": f"{hdr['pmin'][i]}..{hdr['pmax'][i]}" if i is not None else None,
            "nrec_header": hdr["nrec_header"], "nrec_file": hdr["nrec_file"],
            "excluded_eeg_like": excluded, "derived": [c for c, how in plan if how[0] == "diff"],
            "hypnogram": {**hnotes, **anotes}, "channels": names, "rule_channel": rule, "rule_tier": tier,
            "preprocessing": spec()}
    return x, y, meta
