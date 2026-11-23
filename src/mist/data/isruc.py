"""ISRUC-Sleep groups I and III -> canonical store format. Phase 2 only.

Facts checked on the files (2026-11-24):
- One folder per subject: <n>/<n>.rec (EDF despite the extension; MNE refuses
  the extension, so the file is passed as an open file object), <n>_1.txt and
  <n>_2.txt (scorer 1 and 2, one code per 30-s epoch: 0=W 1=N1 2=N2 3=N3 5=REM),
  and .xlsx files (not used).
- Group I subjects 4, 6 and 39 have no .rec on disk (download incomplete); they
  are skipped (D-015).
- EEG at 200 Hz, labelled either F3-A2 ... O2-A1 or F3-M2 ... O2-M1; one
  recording has monopolar leads (F3, C3, O1, F4, C4, O2, A1, A2) and its six
  standard derivations are formed by subtraction (DERIVE).
- Scorer 1 is used (D-017). Its label count equals the signal length in 30-s
  epochs for every recording on disk; the length rule below is still applied
  and logged.
- Amplitude: recordings whose header gives a +-25 uV physical range have epoch
  SDs about 8x lower than the +-2490 uV ones. Values are stored as the header
  states; `phys_range` goes into meta and the audit reports the split
  (CONDITIONAL: a header scaling error is suspected, not verified).

Labels are assumed to start at the signal start (CONDITIONAL; nothing in the
files gives an offset).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .channels import pick_rule_channel, plan_channels
from .edf_header import channel_info, read_header
from .resample import spec, to_epochs

CODE = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}
GROUPS = {1: "ISRUC-Sleep-I", 3: "ISRUC-Sleep-III"}
DERIVE = ("F3-A2", "C3-A2", "O1-A2", "F4-A1", "C4-A1", "O2-A1")


def record_id(group: int, n: int) -> str:
    return f"ISRUC{group}-{n:03d}"


def find_records(root: Path) -> tuple:
    """-> ([(group, n, rec_path, scorer1_txt)], [(group, n, reason)])."""
    found, skipped = [], []
    for g, sub in GROUPS.items():
        base = Path(root) / sub
        if not base.is_dir():
            continue
        for d in sorted((p for p in base.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
            n = int(d.name)
            rec, txt = d / f"{n}.rec", d / f"{n}_1.txt"
            if not rec.exists():
                skipped.append((g, n, "no .rec file on disk (not downloaded)"))
            elif not txt.exists():
                skipped.append((g, n, "no scorer-1 file"))
            else:
                found.append((g, n, rec, txt))
    return found, skipped


def read_labels(txt: Path) -> np.ndarray:
    lines = Path(txt).read_text().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if any(not s.strip() for s in lines):
        raise ValueError(f"{Path(txt).name}: blank line inside the label list")
    codes = [int(s.strip()) for s in lines]
    bad = sorted(set(codes) - set(CODE))
    if bad:
        raise ValueError(f"{Path(txt).name}: unexpected codes {bad}")
    return np.array([CODE[c] for c in codes], dtype=np.int8)


def fit_length(labels: np.ndarray, n_epochs: int) -> tuple:
    """Rule: labels start at the signal start; extra labels at the end are dropped,
    missing ones at the end are unscored. -> (y, {"n_labels", "n_signal_epochs", "action"})."""
    y = np.full(n_epochs, -1, dtype=np.int8)
    k = min(n_epochs, len(labels))
    y[:k] = labels[:k]
    action = ("none" if len(labels) == n_epochs else
              f"dropped {len(labels) - n_epochs} trailing labels" if len(labels) > n_epochs else
              f"{n_epochs - len(labels)} trailing epochs unscored")
    return y, {"n_labels": int(len(labels)), "n_signal_epochs": int(n_epochs), "action": action}


def read_record(rec: Path, txt: Path) -> tuple:
    """Return (x [n_epochs, C, 3000] microvolt at 100 Hz, y, meta)."""
    import mne

    from . import resample

    hdr = read_header(rec)
    plan = plan_channels(hdr["label"], DERIVE)
    if not plan:
        raise ValueError(f"{Path(rec).name}: no EEG channel recognised in {hdr['label']}")
    labs = sorted({lab for _, how in plan for lab in how[1:]})
    fs = {hdr["fs"][hdr["label"].index(lab)] for lab in labs}
    if len(fs) != 1:
        raise ValueError(f"{Path(rec).name}: EEG at several rates {fs}")
    fs = fs.pop()
    with open(rec, "rb") as fh:
        raw = mne.io.read_raw_edf(fh, include=labs, preload=True, verbose="ERROR")
        units = dict(raw._orig_units)
        if any(units[lab] not in ("µV", "uV") for lab in labs):
            raise ValueError(f"{Path(rec).name}: EEG units {units}")
        sig = {lab: raw.get_data(picks=[lab])[0] * 1e6 for lab in labs}
    rows = [sig[how[1]] if how[0] == "raw" else sig[how[1]] - sig[how[2]] for _, how in plan]
    x = to_epochs(resample.preprocess(np.stack(rows), fs))
    y, length = fit_length(read_labels(txt), len(x))
    names = [c for c, _ in plan]
    rule, tier = pick_rule_channel(names)
    rule_src = dict(plan)[rule][1] if rule else None
    info = channel_info(hdr, rule_src) if rule_src else {}
    meta = {"fs_orig": float(fs), "units": {lab: units[lab] for lab in labs},
            "prefilter_eeg": sorted({channel_info(hdr, lab)["prefilter"] for lab in labs}),
            "phys_range": f"{info.get('pmin')}..{info.get('pmax')}", "label_length": length,
            "derived": [c for c, how in plan if how[0] == "diff"], "scorer": 1, "channels": names,
            "rule_channel": rule, "rule_tier": tier, "preprocessing": spec()}
    return x, y, meta


def parse_record_id(name: str) -> tuple:
    m = re.fullmatch(r"ISRUC([13])-(\d{3})", name)
    if not m:
        raise ValueError(f"not an ISRUC record folder: {name}")
    return int(m.group(1)), int(m.group(2))
