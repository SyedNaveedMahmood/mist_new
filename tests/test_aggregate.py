import json
from pathlib import Path

import numpy as np

from mist import metrics
from mist.aggregate import collect, paired_diff, scores_batch, summary


def rand_cm(rng, n=200):
    y = rng.integers(0, 5, n); p = np.where(rng.random(n) < 0.7, y, rng.integers(0, 5, n))
    return metrics.confusion(y, p)


def test_scores_batch_matches_metrics_scores():
    rng = np.random.default_rng(0)
    cm = rand_cm(rng)
    a, b = metrics.scores(cm), scores_batch(cm)
    for k in ("mf1", "acc", "kappa", "f1_N1"):
        assert abs(a[k] - float(b[k])) < 1e-12


def test_collect_sums_runs_per_subject_and_filters_nights(tmp_path):
    rng = np.random.default_rng(1)
    c1, c2, c3 = rand_cm(rng), rand_cm(rng), rand_cm(rng)
    for i, (cm_a, cm_b) in enumerate([(c1, c2), (c3, c3)]):
        d = {"night_meta": {"ST7011": {"subject": "ST:01"}, "ST7012": {"subject": "ST:01"}},
             "cells": {"real": {"subject_cm": {"ST:01": (cm_a + cm_b).tolist()},
                                "night_cm": {"ST7011": cm_a.tolist(), "ST7012": cm_b.tolist()}}}}
        (tmp_path / f"e{i}.json").write_text(json.dumps(d))
    paths = sorted(tmp_path.glob("e*.json"))
    assert (collect(paths, "real")["ST:01"] == c1 + c2 + 2 * c3).all()
    assert (collect(paths, "real", nights={"ST7011"})["ST:01"] == c1 + c3).all()


def test_paired_diff_zero_for_identical_and_signed_for_worse():
    rng = np.random.default_rng(2)
    a = {f"s{i}": rand_cm(rng) for i in range(12)}
    r = paired_diff(a, a, n_boot=300)
    assert r["mf1"]["diff"] == 0 and r["mf1"]["lo"] == 0 and r["mf1"]["hi"] == 0
    worse = {k: v + np.full((5, 5), 5) for k, v in a.items()}           # extra confusions everywhere
    r = paired_diff(a, worse, n_boot=300)
    assert r["mf1"]["diff"] > 0 and r["mf1"]["lo"] > 0
    assert summary(a)["n_subjects"] == 12


def test_non_n1_sum_and_restrict():
    from mist.aggregate import restrict
    rng = np.random.default_rng(3)
    cm = rand_cm(rng)
    sc = metrics.scores(cm)
    assert abs(float(scores_batch(cm)["f1_nonN1_sum"]) - (sc["f1_W"] + sc["f1_N2"] + sc["f1_N3"] + sc["f1_REM"])) < 1e-12
    assert list(restrict({"a": 1, "b": 2}, ["b"])) == ["b"]


def test_e1_runs_drop_folds_skipped_by_the_session_queue(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("agg_script", Path(__file__).resolve().parents[1] / "scripts" / "aggregate.py")
    agg = importlib.util.module_from_spec(spec); spec.loader.exec_module(agg)
    monkeypatch.chdir(tmp_path)
    for arm in ("zza", "zzb"):
        for f in range(3):
            (tmp_path / "runs" / f"e1_{arm}_cv{f}_s0").mkdir(parents=True)
    (tmp_path / "runs" / "session_skips.json").write_text(json.dumps({"e1_zza_cv1_s0": "x", "e1_zzb_cv1_s0": "x"}))
    assert [Path(r).name for r in agg.e1_runs("zza", "cv")] == ["e1_zza_cv0_s0", "e1_zza_cv2_s0"]
    assert [Path(r).name for r in agg.e1_runs("zzb", "cv")] == ["e1_zzb_cv0_s0", "e1_zzb_cv2_s0"]


def _agg_script():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("agg_script", Path(__file__).resolve().parents[1] / "scripts" / "aggregate.py")
    agg = importlib.util.module_from_spec(spec); spec.loader.exec_module(agg)
    return agg


def _write_eval(path, subjects, rng, cells=("real",), boost=0):
    cm = lambda: (np.eye(5) * (60 + boost) + rng.integers(0, 12, (5, 5))).tolist()
    d = {"night_meta": {f"{s}_1": {"subject": s, "condition": None} for s in subjects},
         "cells": {c: {"subject_cm": {s: cm() for s in subjects}, "night_cm": {}} for c in cells}}
    path.write_text(json.dumps(d))


def _h1_tree(tmp_path, agg, seeds, rng):
    import shutil
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "splits").mkdir(exist_ok=True)
    shutil.copy(root / "splits" / "sleepedf_v1.json", tmp_path / "splits")
    sp = json.loads((root / "splits" / "sleepedf_v1.json").read_text())
    for arm, boost in (("zc", 6), ("zc_placebo", 0)):
        for s in seeds:
            for f in range(5):
                r = tmp_path / "runs" / f"e1_{arm}_cv{f}_s{s}"; r.mkdir(parents=True, exist_ok=True)
                _write_eval(r / "eval_test_cv_Fpz-Cz.json", sp["folds"][f], rng, boost=boost)
                _write_eval(r / "eval_test_cv_Pz-Oz.json", sp["folds"][f], rng, boost=boost)
                _write_eval(r / "eval_test_sc2st_Fpz-Cz.json", sp["st"], rng, boost=boost)
            for proto, band in (("young2old", "old"), ("old2young", "young")):
                r = tmp_path / "runs" / f"e1_{arm}_{proto}_s{s}"; r.mkdir(parents=True, exist_ok=True)
                _write_eval(r / f"eval_test_{proto}_Fpz-Cz.json", sp["sets"][band], rng, boost=boost)


def test_h1_primary_statistic_interim_complete_and_run_once(tmp_path, monkeypatch):
    import pytest
    agg = _agg_script()
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    _h1_tree(tmp_path, agg, (0, 1), rng)
    out = agg.build_h1("zc", "zc_placebo", n_boot=300)
    assert out["label"] == "INTERIM" and not out["complete"] and out["seeds"] == [0, 1] and out["decision"] is None
    assert out["protocols"] == ["montage", "sc2st", "young2old", "old2young"]          # in-domain is not pooled
    assert out["in_domain_beside"]["n_runs_per_arm"] == 10
    per = out["per_protocol"]
    assert per["sc2st"]["n_subjects"] == 22 and per["montage"]["n_subjects"] == 54 and per["montage"]["n_runs_per_arm"] == 10
    want = np.mean([per[p]["mf1"]["diff"] for p in out["protocols"]])
    assert abs(out["pooled"]["diff"] - want) < 1e-12 and out["pooled"]["lo"] > 0     # boosted arm is better
    assert out["secondary_mixed_model"]["n_rows"] == 2 * (2 * 5 * 2 + 2 * 2)          # shift protocols only
    agg.write_h1(out, tmp_path / "h1.md")
    assert "INTERIM" in (tmp_path / "h1.md").read_text()
    with pytest.raises(SystemExit):
        agg.build_h1("zc", "zc_placebo", confirmatory=True, n_boot=300)                # incomplete: refused
    assert not agg.H1_CONFIRMATORY.exists()
    agg.SEEDS = {0}
    try:
        assert agg.build_h1("zc", "zc_placebo", n_boot=100)["seeds"] == [0]
    finally:
        agg.SEEDS = None
    _h1_tree(tmp_path, agg, (2,), rng)
    (tmp_path / "reports").mkdir()
    out = agg.build_h1("zc", "zc_placebo", confirmatory=True, n_boot=300)
    assert out["complete"] and out["label"] == "CONFIRMATORY" and agg.H1_CONFIRMATORY.exists()
    assert out["decision"]["supported"] == (out["pooled"]["diff"] >= 0.02 and out["pooled"]["lo"] > 0)
    with pytest.raises(SystemExit):
        agg.build_h1("zc", "zc_placebo", confirmatory=True, n_boot=300)                # run once only


def test_h1_pairs_are_balanced(tmp_path, monkeypatch):
    agg = _agg_script()
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(2)
    _h1_tree(tmp_path, agg, (0,), rng)
    import shutil
    shutil.rmtree(tmp_path / "runs" / "e1_zc_placebo_cv3_s0")                       # placebo fold 3 missing
    out = agg.build_h1("zc", "zc_placebo", n_boot=100)
    assert out["per_protocol"]["montage"]["n_runs_per_arm"] == 4 and out["per_protocol"]["montage"]["n_subjects"] == 54 - 11


def test_e3_band_by_stage_costs(tmp_path, monkeypatch):
    agg = _agg_script()
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(1)
    for f in range(2):
        r = tmp_path / "runs" / f"e1_zc_cv{f}_s0"; r.mkdir(parents=True)
        subs = [f"SC:{f}{i}" for i in range(5)]
        for real_fn, e3_fn in agg.E3_FILES.values():
            _write_eval(r / real_fn, subs, rng, boost=10)
            _write_eval(r / e3_fn, subs, rng, cells=("real|destroy_sigma", "real|knock_M"))
    res = agg.build_e3("zc", 200)
    assert set(res) == {"cv", "montage", "sc2st"} and res["cv"]["n_runs"] == 2
    assert res["cv"]["cost"]["real|destroy_sigma"]["mf1"]["diff"] > 0
    assert agg.write_e3("zc", res, tmp_path / "e3.md") == ["real|destroy_sigma"]


def test_e1_pairs_use_only_common_runs(tmp_path, monkeypatch):
    import shutil
    agg = _agg_script()
    monkeypatch.chdir(tmp_path)
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "splits").mkdir()
    shutil.copy(root / "splits" / "sleepedf_v1.json", tmp_path / "splits")
    sp = json.loads((root / "splits" / "sleepedf_v1.json").read_text())
    rng = np.random.default_rng(4)
    for arm, seeds in (("cascade2", (0, 1, 2)), ("cascade2_absolute", (0,))):
        for s in seeds:
            for f in range(5):
                r = tmp_path / "runs" / f"e1_{arm}_cv{f}_s{s}"; r.mkdir(parents=True)
                _write_eval(r / "eval_test_sc2st_Fpz-Cz.json", sp["st"], rng)
    _, pairs = agg.build_e1(100)
    p = next(p for p in pairs if p["pair"] == "cascade2 - cascade2_absolute" and p["protocol"] == "sc2st")
    assert p["n_runs_per_arm"] == 5 and p["n_subjects"] == 22
