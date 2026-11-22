"""T8 slot knockouts and T9 stage-targeted critic."""
import torch

from mist.models.build import build_model
from test_model import SMALL, stats


def test_knocking_all_slots_equals_knock_m_without_residual():
    torch.manual_seed(0)
    m = build_model({**SMALL, "streams": ["G", "R", "M"], "morph_residual": 0}).eval()
    x = torch.randn(2, 6, 3000)
    all_slots, _ = m(x, stats(2), knock_slots=range(SMALL["n_slots"]))
    knock_m, _ = m(x, stats(2), knock=("M",))
    assert torch.allclose(all_slots, knock_m, atol=1e-6)


def test_knocking_all_slots_leaves_only_the_residual():
    torch.manual_seed(0)
    m = build_model({**SMALL, "streams": ["G", "R", "M"], "morph_residual": 8}).eval()
    x = torch.randn(2, 6, 3000)
    _, aux = m(x, stats(2), knock_slots=range(SMALL["n_slots"]))
    assert all(float(a.detach().abs().sum()) == 0 for a in aux["assign"])
    assert not torch.allclose(m(x, stats(2), knock_slots=range(8))[0], m(x, stats(2), knock=("M",))[0], atol=1e-4)


def test_partial_knock_renormalises_and_zeroes_chosen_slots():
    torch.manual_seed(0)
    m = build_model({**SMALL, "streams": ["M"]}).eval()
    _, aux = m(torch.randn(2, 6, 3000), stats(2), knock_slots=(0, 3))
    for a in aux["assign"]:
        assert float(a[..., [0, 3]].detach().abs().sum()) == 0
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)


def test_stage_targeted_critic_uses_only_chosen_stages():
    torch.manual_seed(0)
    cfg = {**SMALL, "streams": ["M"], "critic_weight": 1.0, "critic_stages": [0, 1, 4]}
    m = build_model(cfg).train()
    x = torch.randn(2, 6, 3000)
    # every epoch is N2 or N3: no eligible token, critic loss is exactly 0 but still differentiable
    _, aux = m(x, stats(2), y=torch.tensor([[2, 3, 2, 3, 2, 3]] * 2))
    assert all(float(v.detach()) == 0 for k, v in aux["losses"].items() if k.startswith("critic"))
    _, aux = m(x, stats(2), y=torch.tensor([[0, 1, 4, 2, 2, 2]] * 2))
    assert all(float(v.detach()) > 0 for k, v in aux["losses"].items() if k.startswith("critic"))
    mask = m.branches[0]._critic_mask(torch.tensor([[0, 2, 4, 3, 1, 2]]), 1, 6, 3)
    assert mask.view(6, 3)[:, 0].tolist() == [True, False, True, False, True, False]


def test_critic_stages_requires_labels():
    import pytest
    m = build_model({**SMALL, "streams": ["M"], "critic_weight": 1.0, "critic_stages": [1]}).train()
    with pytest.raises(ValueError):
        m(torch.randn(1, 6, 3000), stats(1))
