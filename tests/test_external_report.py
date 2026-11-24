"""scripts/external_report.py: ISRUC sub-domain rows in the owner's order, people summed over fold models, pairs."""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def script():
    spec = importlib.util.spec_from_file_location("external_report", ROOT / "scripts" / "external_report.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def write_eval(path, dataset, subjects, rng, boost):
    meta, ncm = {}, {}
    for i, s in enumerate(subjects):
        rid = s.replace(":", "-")
        meta[rid] = {"subject": s, "condition": None, "channel": "F4-M1",
                     **({"subdomain": "ISRUC-A" if i % 3 == 0 else "ISRUC-M"} if dataset == "isruc" else {})}
        ncm[rid] = (np.eye(5, dtype=int) * (50 + boost) + rng.integers(0, 10, (5, 5))).tolist()
    path.write_text(json.dumps({"night_meta": meta, "cells": {"real": {"night_cm": ncm, "subject_cm": {}}}}))


def test_external_report(tmp_path, monkeypatch):
    er = script()
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    subs = {"isruc": [f"ISRUC1:{i:03d}" for i in range(9)], "hmc": [f"HMC:{i:03d}" for i in range(6)]}
    for arm, prefix, boost in (("raw", "e0", 0), ("cascade2", "e1", 8), ("cascade2_placebo", "e1", 0)):
        for f in range(5):
            r = tmp_path / "runs" / f"{prefix}_{arm}_cv{f}_s0"; r.mkdir(parents=True)
            for ds, ss in subs.items():
                write_eval(r / f"eval_test_ext-{ds}-dev_rule.json", ds, ss, rng, boost)
    (tmp_path / "runs" / "e1_cascade2_cv0_s1").mkdir()            # other seed: not used with --seeds 0
    res = er.build(["raw", "cascade2", "cascade2_placebo"], [0], 300)
    assert list(res) == ["hmc", "isruc"]
    rows = res["isruc"]["arms"]["raw"]["rows"]
    assert list(rows) == ["ISRUC-M", "ISRUC-A", "all"]
    assert rows["ISRUC-A"]["n_subjects"] == 3 and rows["ISRUC-M"]["n_subjects"] == 6 and rows["all"]["n_subjects"] == 9
    assert res["isruc"]["arms"]["raw"]["n_eval_files"] == 5 and res["hmc"]["arms"]["cascade2"]["n_runs"] == 5
    p = [x for x in res["isruc"]["pairs"] if x["pair"] == "cascade2 - cascade2_placebo"]
    assert [x["row"] for x in p] == ["ISRUC-M", "ISRUC-A", "all"] and all(x["mf1"]["lo"] > 0 for x in p)
    er.write(res, [0], tmp_path / "x.md", tmp_path / "x.json")
    md = (tmp_path / "x.md").read_text()
    assert "INTERIM" in md and "comparison with pooled papers only" in md and "holdouts stay closed" in md
    assert er.role("cascade2_absolute", "isruc", "ISRUC-A").startswith("descriptive")
