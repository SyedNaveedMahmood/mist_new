#!/usr/bin/env python
"""T5 backfill: add the label-free `lf` window fields to an existing store.

Writes psg_start, lights_off_epoch, n_epochs_std, n_epochs_lf and the clipping
flags into every <store>/<cohort>/<record>/meta.json and manifest.json, and a
store-level lf_window_report.json listing nights whose window is clipped. Only
the EDF fixed header (start time) and y.npy are read; signals are not touched.
New builds get the same fields from prepare_sleepedf.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mist.config import load_paths
from mist.data import sleepedf as E


def write_atomic(path: Path, text: str) -> None:
    # training runs may read meta.json concurrently; never expose a half-written file
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    a = ap.parse_args()
    paths = load_paths(a.paths)
    root = Path(paths["store_root"])
    manifest = json.loads((root / "manifest.json").read_text())
    report = {"short_end": [], "short_start": [], "no_lights_off": []}
    for rid, m in manifest.items():
        folder = Path(paths["raw"]["sleep_cassette" if m["cohort"] == "SC" else "sleep_telemetry"])
        d = root / m["cohort"] / rid
        y = np.load(d / "y.npy")
        w = E.window_meta(E.psg_start_hms(folder / m["psg"]), m.get("lights_off"), y)
        meta = json.loads((d / "meta.json").read_text())
        meta.update(w); write_atomic(d / "meta.json", json.dumps(meta, indent=1))
        m.update(w)
        if w["lights_off_epoch"] is None:
            report["no_lights_off"].append(rid)
        else:
            if w["lf_clipped_end"]:
                report["short_end"].append(rid)
            if w["lf_clipped_start"]:
                report["short_start"].append(rid)
    write_atomic(root / "manifest.json", json.dumps(manifest, indent=1))
    for c in ("SC", "ST"):
        r = [v for v in manifest.values() if v["cohort"] == c]
        report[c] = {"nights": len(r), "std_epochs": sum(v["n_epochs_std"] for v in r),
                     "lf_epochs": sum(v["n_epochs_lf"] or 0 for v in r),
                     "lf_short_end": sorted(k for k in report["short_end"] if manifest[k]["cohort"] == c),
                     "lf_short_start": sorted(k for k in report["short_start"] if manifest[k]["cohort"] == c)}
    (root / "lf_window_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({c: {k: (v if not isinstance(v, list) or len(v) < 6 else f"{len(v)} nights")
                          for k, v in report[c].items()} for c in ("SC", "ST")}, indent=1))


if __name__ == "__main__":
    main()
