#!/usr/bin/env python
"""Person-level split of one external dataset: train 50% / val 10% / test 40%, stratified, fixed seed.

    python scripts/make_external_splits.py --dataset hmc|dodh|isruc|cap

Reads <store_root>/<DATASET>/manifest.json (written by prepare_external.py) and
writes splits/<dataset>_v1.json once; an existing file is never overwritten.
Strata (D-017): CAP diagnosis group, ISRUC group I vs III, HMC EEG prefilter
field, DOD-H none. Within each stratum people are shuffled with the seed and
cut by rounding (test first, then val); the rest is train. The test part is
the external holdout (mist.data.external.load_external).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from mist.config import load_paths
from mist.data.external import DATASETS

SEED = 20261124
FRACTIONS = {"train": 0.5, "val": 0.1, "test": 0.4}


def stratum(dataset: str, meta: dict) -> str:
    if dataset == "cap":
        return meta["group"]
    if dataset == "isruc":
        return meta["subject"].split(":")[0]
    if dataset == "hmc":
        return " / ".join(meta.get("prefilter_eeg") or ["unknown"])
    return "all"


def make_split(dataset: str, records: dict, seed: int = SEED) -> dict:
    by = defaultdict(set)
    for meta in records.values():
        by[stratum(dataset, meta)].add(meta["subject"])
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    strata = {}
    for s in sorted(by):
        people = sorted(by[s])
        rng.shuffle(people)
        n = len(people)
        n_test = int(round(FRACTIONS["test"] * n))
        n_val = int(round(FRACTIONS["val"] * n))
        if n_test + n_val > n:
            n_val = n - n_test
        cut = {"test": people[:n_test], "val": people[n_test:n_test + n_val], "train": people[n_test + n_val:]}
        for k, v in cut.items():
            parts[k] += v
        strata[s] = {k: len(v) for k, v in cut.items()}
    return {"version": 1, "dataset": dataset, "seed": seed, "unit": "person", "fractions": FRACTIONS,
            "stratified_by": {"cap": "diagnosis group", "isruc": "group (I/III)", "hmc": "EEG prefilter field",
                              "dodh": "none"}[dataset],
            "strata": strata, **{k: sorted(v) for k, v in parts.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"splits/{a.dataset}_v1.json")
    if out.exists():
        sys.exit(f"{out} exists; split files are made once and never overwritten")
    man = Path(a.manifest or Path(load_paths(a.paths)["store_root"]) / DATASETS[a.dataset] / "manifest.json")
    m = json.loads(man.read_text())
    if m.get("dataset") != a.dataset:
        sys.exit(f"{man} is not a {a.dataset} manifest")
    sp = make_split(a.dataset, m["records"])
    out.write_text(json.dumps(sp, indent=1) + "\n")
    print(out, {k: len(sp[k]) for k in ("train", "val", "test")}, sp["strata"])


if __name__ == "__main__":
    main()
