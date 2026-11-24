"""raw_whitened (EXPLORATORY, D-026): the default leaves `raw` unchanged; the new arm is gain-invariant."""
import glob
from pathlib import Path

import pytest
import torch

from mist.config import load_config
from mist.models.build import build_model, count_parameters
from mist.models.factorizer import night_reference

ROOT = Path(__file__).resolve().parents[1]
SMALL = {"name": "mist_grm", "seq_len": 6, "d_stream": 16, "d_token": 32, "d_epoch": 32, "context_layers": 1,
         "scales": [[1.0, 0.5], [4.0, 2.0]], "streams": ["X"]}


def night(seed=0, n=40):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(3000) / 100
    x = torch.randn(n, 3000, generator=g) * 20 + 30 * torch.sin(2 * torch.pi * 10 * t) * torch.rand(n, 1, generator=g)
    return x


def batch_stats(x, model):
    ref = night_reference(x, [b.fz for b in model.branches])
    return {k: {kk: vv[None] for kk, vv in v.items()} for k, v in ref.items()}


def test_default_raw_is_unchanged_and_whitened_differs():
    torch.manual_seed(0)
    a = build_model(SMALL).eval()
    torch.manual_seed(0)
    b = build_model({**SMALL, "raw_input": "scaled"}).eval()
    torch.manual_seed(0)
    w = build_model({**SMALL, "raw_input": "whitened"}).eval()
    assert a.state_dict().keys() == w.state_dict().keys() and count_parameters(a) == count_parameters(w)
    x = night()[:6][None]
    st = batch_stats(night()[:6], a)
    with torch.no_grad():
        assert torch.equal(a(x, st)[0], b(x, st)[0])
        w.load_state_dict(a.state_dict())                       # same parameters, other input
        assert not torch.allclose(a(x, st)[0], w(x, st)[0], atol=1e-3)


def test_raw_whitened_is_gain_invariant_and_raw_scaled_too():
    torch.manual_seed(1)
    for mode in ("whitened", "scaled"):
        m = build_model({**SMALL, "raw_input": mode}).eval()
        x = night(2)
        with torch.no_grad():
            o1 = m(x[:6][None], batch_stats(x, m))[0]
            o2 = m(7.3 * x[:6][None], batch_stats(7.3 * x, m))[0]
        assert torch.allclose(o1, o2, atol=1e-4), mode


def test_whitened_only_where_it_means_something():
    for bad in ({"raw_input": "other"}, {"raw_input": "whitened", "night_relative": False},
                {"raw_input": "whitened", "streams": ["G", "R"]}):
        with pytest.raises(ValueError):
            build_model({**SMALL, **bad})


def test_arm_configs_and_existing_raw_checkpoint():
    raw, rw = (load_config([ROOT / "configs/base.yaml", ROOT / f"configs/arms/{a}.yaml"]) for a in ("raw", "raw_whitened"))
    assert rw["model"]["raw_input"] == "whitened" and "raw_input" not in raw["model"]
    assert {k: v for k, v in rw["model"].items() if k != "raw_input"} == raw["model"]
    assert {k: v for k, v in rw.items() if k != "model"} == {k: v for k, v in raw.items() if k != "model"}   # same recipe
    cks = sorted(glob.glob(str(ROOT / "runs/e0_raw_cv0_s0/best.pt")))
    if not cks:
        pytest.skip("no local raw checkpoint (private, not in git)")
    ck = torch.load(cks[0], map_location="cpu", weights_only=False)
    m = build_model(ck["cfg"]["model"])
    m.load_state_dict(ck["model"])                              # strict: the switch adds no parameter
    assert m.cfg.raw_input == "scaled"
