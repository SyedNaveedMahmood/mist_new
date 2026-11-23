"""Fixed EDF header fields read directly from the file (no signal data).

MNE reports the strictest prefilter over the loaded channels and silently
infers the record count when the header disagrees with the file size, so the
audits read the header themselves.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

FIELDS = (("label", 16), ("transducer", 80), ("dim", 8), ("pmin", 8), ("pmax", 8), ("dmin", 8), ("dmax", 8),
          ("prefilter", 80), ("nsamp", 8), ("reserved", 32))


def read_header(path: Path) -> dict:
    path = Path(path)
    with path.open("rb") as f:
        b = f.read(256)
        ns = int(b[252:256])
        h = f.read(256 * ns)
    out, o = {}, 0
    for name, w in FIELDS:
        block = h[o:o + ns * w]
        out[name] = [block[i * w:(i + 1) * w].decode("latin-1").strip() for i in range(ns)]
        o += ns * w
    nrec, recdur = int(b[236:244]), float(b[244:252])
    bytes_per_rec = 2 * sum(int(n) for n in out["nsamp"])
    nrec_file = (os.path.getsize(path) - 256 * (ns + 1)) // bytes_per_rec if bytes_per_rec else 0
    out.update(start_date=b[168:176].decode("latin-1"), start_time=b[176:184].decode("latin-1"), ns=ns,
               nrec_header=nrec, nrec_file=int(nrec_file), record_s=recdur,
               fs=[int(n) / recdur if recdur else 0.0 for n in out["nsamp"]])
    return out


def channel_info(hdr: dict, label: str) -> dict:
    i = hdr["label"].index(label)
    return {"dim": hdr["dim"][i], "prefilter": hdr["prefilter"][i], "fs": hdr["fs"][i],
            "pmin": hdr["pmin"][i], "pmax": hdr["pmax"][i]}


_HP = (r"HP:\s*([\d.]+)", r"Low Cut:\s*([\d.]+)", r"BP:\s*([\d.]+)\s*HZ")
_LP = (r"LP:\s*([\d.]+)", r"High Cut:\s*([\d.]+)", r"BP:\s*[\d.]+\s*HZ\s*-\s*([\d.]+)")


def parse_prefilter(s: str) -> tuple:
    """'LP:30.00Hz HP:0.30Hz NOTCH:50' / 'High Cut:35 Hz,Low Cut:0.3 Hz' / 'HP:0.2Hz LP:35Hz' -> (hp, lp) or None."""
    def first(pats):
        for p in pats:
            m = re.search(p, s, flags=re.IGNORECASE)
            if m:
                return float(m.group(1))
        return None
    return first(_HP), first(_LP)


def seconds_of_day(hms: str) -> int:
    """'22.10.58' or '22:10:58' or '4.54.16' -> seconds after midnight."""
    p = re.split(r"[.:]", hms.strip())
    if len(p) != 3:
        raise ValueError(f"bad time {hms!r}")
    return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
