"""Frozen subject-level splits for the Sleep-EDF phase.

Roles (docs/EXPERIMENT_PLAN.md, section 3):

  lockbox  SC persons never used in any earlier Proto-sleep development,
           age-stratified. Opened once, for the final confirmation run.
  dev      every other eligible SC person, partitioned into k folds.
  sets     age bands inside dev, for the age-shift protocol.
  st       all ST persons (development-grade target: already exposed earlier).

`make_splits` is deterministic in its inputs and seed. Run it once, commit the
JSON, and never regenerate it after a model has been trained.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EXPOSED_SC = tuple(range(20))   # historical Sleep-EDF-20 people: used in earlier development


def make_splits(subjects: dict, seed: int = 20261122, n_lockbox: int = 20, k: int = 5,
                quarantine=(6, 23, 36, 74)) -> dict:
    """subjects: {"SC:07": {"age": 30, ...}, "ST:01": {...}}; age may be None for ST."""
    rng = np.random.default_rng(seed)
    num = lambda s: int(s.split(":")[1])
    sc = sorted(s for s in subjects if s.startswith("SC:") and num(s) not in quarantine)
    st = sorted(s for s in subjects if s.startswith("ST:"))
    fresh = [s for s in sc if num(s) not in EXPOSED_SC]
    if any(subjects[s].get("age") is None for s in sc):
        raise ValueError("every SC subject needs an age (SC-subjects.xls)")

    # lockbox: age-stratified draw from the never-exposed people
    order = sorted(fresh, key=lambda s: (subjects[s]["age"], s))
    lockbox = []
    for stratum in np.array_split(np.array(order, dtype=object), 4):
        take = int(round(n_lockbox * len(stratum) / len(order)))
        lockbox += sorted(rng.choice(stratum, size=min(take, len(stratum)), replace=False).tolist())
    lockbox = sorted(lockbox)
    dev = sorted(set(sc) - set(lockbox))

    # folds: balance age by dealing age-sorted people into folds in shuffled blocks
    by_age = sorted(dev, key=lambda s: (subjects[s]["age"], s))
    folds = [[] for _ in range(k)]
    for i in range(0, len(by_age), k):
        block = by_age[i:i + k]
        for f, s in zip(rng.permutation(k)[: len(block)], block):
            folds[int(f)].append(s)
    folds = [sorted(f) for f in folds]

    ages = np.array([subjects[s]["age"] for s in dev])
    lo, hi = np.quantile(ages, [1 / 3, 2 / 3])
    sets = {"young": sorted(s for s in dev if subjects[s]["age"] <= lo),
            "mid": sorted(s for s in dev if lo < subjects[s]["age"] <= hi),
            "old": sorted(s for s in dev if subjects[s]["age"] > hi)}
    out = {"version": 1, "seed": seed, "quarantined_sc": list(quarantine), "lockbox": lockbox, "dev": dev,
           "folds": folds, "sets": sets, "age_cuts": [float(lo), float(hi)], "st": st}
    check(out)
    return out


def check(sp: dict) -> None:
    assert not set(sp["lockbox"]) & set(sp["dev"]), "lockbox leaks into dev"
    flat = [s for f in sp["folds"] for s in f]
    assert sorted(flat) == sorted(sp["dev"]) and len(flat) == len(set(flat)), "folds must partition dev"
    assert not set(sp["st"]) & set(sp["dev"])


def resolve(sp: dict, protocol: str, fold: int = 0) -> dict:
    """Return train / val / test subject lists and the test cohort for a protocol.

    cv         train on k-2 folds, validate on one, test on one (all SC dev)
    young2old  train+val inside the young band, test on the old band
    old2young  the reverse
    sc2st      train/val as in cv, test on every ST person
    lockbox    train/val as in cv, test on the lockbox (final confirmation only)
    """
    k = len(sp["folds"])
    test_f, val_f = sp["folds"][fold % k], sp["folds"][(fold + 1) % k]
    train_f = sorted(set(sp["dev"]) - set(test_f) - set(val_f))
    if protocol == "cv":
        return {"train": train_f, "val": val_f, "test": test_f, "test_cohort": "SC"}
    if protocol in ("young2old", "old2young"):
        src, tgt = ("young", "old") if protocol == "young2old" else ("old", "young")
        pool = sp["sets"][src]
        val = [s for s in pool if s in set(sp["folds"][fold % k])]
        return {"train": sorted(set(pool) - set(val)), "val": val, "test": sp["sets"][tgt], "test_cohort": "SC"}
    if protocol == "sc2st":
        return {"train": sorted(train_f + test_f), "val": val_f, "test": sp["st"], "test_cohort": "ST"}
    if protocol == "lockbox":
        return {"train": sorted(train_f + test_f), "val": val_f, "test": sp["lockbox"], "test_cohort": "SC"}
    raise ValueError(protocol)


def load(path: Path) -> dict:
    sp = json.loads(Path(path).read_text())
    check(sp)
    return sp
