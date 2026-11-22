import json

import numpy as np
import pytest
import torch

from mist import metrics
from mist.data import splits as S
from mist.engine.loop import evaluate, load_checkpoint, make_dataset, train
from mist.data.store import load_cohort
from mist.models.build import build_model, count_parameters

SMALL = {"name": "mist_grm", "seq_len": 6, "n_slots": 8, "d_stream": 16, "d_token": 32, "d_epoch": 32,
         "context_layers": 1, "scales": [[1.0, 0.5], [4.0, 2.0]]}


def stats(b):
    return {"night": {"envelope": torch.ones(b, 1501), "scale": torch.ones(b)},
            "s0": {"mag_med": torch.zeros(b, 35), "mag_iqr": torch.ones(b, 35)},
            "s1": {"mag_med": torch.zeros(b, 139), "mag_iqr": torch.ones(b, 139)}}


@pytest.mark.parametrize("streams", [["G", "R", "M"], ["G", "R"], ["M"], ["X"]])
def test_forward_backward_all_arms(streams):
    m = build_model({**SMALL, "streams": streams, "critic_weight": 0.3 if "M" in streams else 0.0, "allpass_rad": 0.5})
    x = torch.randn(2, 6, 3000)
    logits, aux = m(x, stats(2))
    assert logits.shape == (2, 6, 5)
    (logits.mean() + m.aux_loss(aux)).backward()
    if "M" in streams:
        assert any(k.startswith("critic") for k in aux["losses"]) and aux["assign"][0].shape == (2, 6, 59, 8)
        assert torch.allclose(aux["assign"][0].sum(-1), torch.ones(2, 6, 59), atol=1e-4)


def test_gain_invariance_and_morphology_interventions():
    torch.manual_seed(0)
    m = build_model({**SMALL, "streams": ["R", "M"]}).eval()
    x = torch.randn(2, 6, 3000)
    a, _ = m(x, stats(2)); b, _ = m(11.0 * x, stats(2))
    assert torch.allclose(a, b, atol=1e-3)                    # R and M never see gain
    g = build_model({**SMALL, "streams": ["G"]}).eval()
    assert not torch.allclose(g(x, stats(2))[0], g(11.0 * x, stats(2))[0], atol=1e-3)  # with fixed stats, G does
    k1, _ = m(x, stats(2), knock=("M",))
    k2, _ = m(x, stats(2), knock=("M",), destroy_prob=1.0)
    assert torch.allclose(k1, k2, atol=1e-5)                  # M knocked out: phase cannot matter
    d, _ = m(x, stats(2), destroy_prob=1.0)
    assert not torch.allclose(a, d, atol=1e-3) and not torch.allclose(a, k1, atol=1e-3)


def test_attnsleep_wrapper():
    m = build_model({"name": "attnsleep", "scales": [[1.0, 0.5]], "seq_len": 6})
    logits, _ = m(torch.randn(2, 3, 3000), stats(2))
    assert logits.shape == (2, 3, 5) and count_parameters(m) > 1e5


def test_metrics_and_bootstrap():
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, -1]); p = np.array([0, 0, 1, 0, 2, 2, 3, 3, 4, 4, 0])
    sc = metrics.scores(metrics.confusion(y, p))
    assert sc["n"] == 10 and abs(sc["acc"] - 0.9) < 1e-9 and abs(sc["f1_N1"] - 2 / 3) < 1e-9
    cms = {s: metrics.confusion(y, p) for s in "abc"}
    r = metrics.paired_bootstrap(cms, cms, n_boot=200)
    assert r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0


def test_end_to_end_training_learns_synthetic_stages(store, tmp_path):
    root, subs = store
    sp = {"version": 1, "seed": 0, "quarantined_sc": [], "lockbox": [], "dev": sorted(subs),
          "folds": [sorted(subs)[i::4] for i in range(4)], "sets": {"young": [], "mid": [], "old": []},
          "age_cuts": [0, 0], "st": []}
    spf = tmp_path / "sp.json"; spf.write_text(json.dumps(sp))
    cfg = {"data": {"cohort": "SC", "channel": "Fpz-Cz", "view": "full", "seq_len": 6, "windows_per_night": 8},
           "split": {"file": str(spf), "protocol": "cv", "fold": 0},
           "train": {"epochs": 10, "batch_size": 8, "lr": 3e-3, "seed": 0, "num_workers": 0, "device": "cpu"},
           "model": {**SMALL, "streams": ["G", "R", "M"], "critic_weight": 0.3}}
    out = train(cfg, {"store_root": str(root)}, tmp_path / "run")
    assert out["best_val_mf1"] > 0.45, out                    # chance is 0.2
    model, cfg2 = load_checkpoint(tmp_path / "run", torch.device("cpu"))
    roles = S.resolve(sp, "cv", 0)
    ds = make_dataset(load_cohort(root, "SC", set(roles["test"]), split_file=spf), cfg2, "Pz-Oz", train=False)
    real, _ = evaluate(model, ds, torch.device("cpu"))
    surr, _ = evaluate(model, ds, torch.device("cpu"), condition="global_phase")
    assert real["n"] == surr["n"] == 180 and real["n_subjects"] == 3
