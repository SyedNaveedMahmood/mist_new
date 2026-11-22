#!/usr/bin/env python
"""Paired subject bootstrap between two evaluation cells (same test subjects).

    python scripts/compare.py runs/a/eval_test_cv_Fpz-Cz.json real runs/b/eval_test_cv_Fpz-Cz.json real
"""
import json
import sys

import numpy as np

from mist.metrics import paired_bootstrap


def cell(path, key):
    return {k: np.array(v) for k, v in json.load(open(path))["cells"][key]["subject_cm"].items()}


if __name__ == "__main__":
    pa, ka, pb, kb = sys.argv[1:5]
    for metric in ("mf1", "f1_N1", "kappa"):
        print(metric, paired_bootstrap(cell(pa, ka), cell(pb, kb), metric=metric))
