import json

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


def test_h1_mixed_model_is_interim_until_complete_and_confirmatory_runs_once(tmp_path, monkeypatch):
    import pytest
    agg = _agg_script()
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    for arm, boost in (("zc", 6), ("zc_placebo", 0)):
        for s in (0, 1):
            for f in range(5):
                r = tmp_path / "runs" / f"e1_{arm}_cv{f}_s{s}"; r.mkdir(parents=True)
                for fn in agg.H1_PROTOCOLS.values():
                    _write_eval(r / fn, [f"SC:{f}{i}" for i in range(4)], rng, boost=boost)
    out = agg.build_h1("zc", "zc_placebo")
    assert out["label"] == "INTERIM" and not out["complete"] and out["seeds"] == [0, 1]
    assert out["n_rows"] == 2 * 3 * 5 * 2 and out["ci95"][0] > 0          # the boosted arm is better
    with pytest.raises(SystemExit):
        agg.build_h1("zc", "zc_placebo", confirmatory=True)                # incomplete: refused


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
