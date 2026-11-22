#!/usr/bin/env python
"""Raw Sleep-EDF (SC and ST) -> canonical store, then an integrity report.

    python scripts/prepare_sleepedf.py --cohorts SC ST

Reads configs/paths.yaml. Writes <store_root>/<cohort>/<record>/ and
<store_root>/manifest.json. Compare the printed totals with the audited
figures in docs/TASKS.md (T1) before going further.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mist.config import load_paths
from mist.data import sleepedf as E
from mist.data.store import save_night, stage_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["SC", "ST"])
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--keep-quarantined", action="store_true")
    a = ap.parse_args()
    paths = load_paths(a.paths)
    root = Path(paths["store_root"])
    # keyed by cohort: SC and ST reuse subject numbers, so one shared dict would
    # give ST records the demographics of the SC person with the same number
    demo = {"SC": E.read_sc_subjects(Path(paths["raw"]["sc_subjects_xls"])) if "SC" in a.cohorts else {},
            "ST": E.read_st_subjects(Path(paths["raw"]["st_subjects_xls"])) if "ST" in a.cohorts else {}}
    manifest, totals = {}, {"recordings": 0, "epochs": 0, "scored": 0}
    for cohort in a.cohorts:
        folder = Path(paths["raw"]["sleep_cassette" if cohort == "SC" else "sleep_telemetry"])
        for psg, hyp in E.find_pairs(folder):
            info = E.parse_name(psg.name)
            if cohort == "SC" and info["subject_num"] in E.SC_QUARANTINE and not a.keep_quarantined:
                continue
            x, y, chans = E.read_recording(psg, hyp)
            meta = {**info, "channels": chans, "psg": psg.name, "hyp": hyp.name,
                    **demo[cohort].get((info["subject_num"], info["night"]), {"age": None, "sex": None})}
            save_night(root, cohort, info["record_id"], x, y, meta)
            manifest[info["record_id"]] = {**meta, "n_epochs": int(len(y)), "counts": stage_counts(y)}
            totals["recordings"] += 1; totals["epochs"] += int(len(y)); totals["scored"] += int((y >= 0).sum())
            print(info["record_id"], x.shape, stage_counts(y))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    subjects = {m["subject"] for m in manifest.values()}
    print("TOTAL", totals, "subjects", len(subjects))


if __name__ == "__main__":
    main()
