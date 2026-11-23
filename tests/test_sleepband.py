"""SleepBand re-implementation (T14, D-023): filterbank properties and the model interface."""
import math

import torch

from mist.models.build import build_model, count_parameters
from mist.models.sleepband import GaborBank


def stats(b, scale=1.0):
    return {"night": {"scale": torch.full((b,), scale)}}


def test_gabor_filter_nearest_to_a_sine_responds_most():
    bank = GaborBank(20)
    t = torch.arange(3000) / 100.0
    for f in (1.0, 6.0, 12.0, 25.0):
        r = bank(torch.sin(2 * math.pi * f * t)[None])[0, :, 500:2500].mean(-1)
        assert abs(math.log(bank.eta[r.argmax()].item() / f)) < 0.35, f


def test_constant_q_width_and_frequency_range():
    bank = GaborBank(20)
    eta = bank.eta.detach()
    assert 0.5 <= eta.min() and eta.max() <= 35.0 and torch.all(eta[1:] > eta[:-1])
    k = bank.kernels().detach()
    width = lambda row: (row.abs() > row.abs().max() / 2).sum().item()
    assert width(k[0]) > width(k[-1])                           # low frequencies: longer filters


def test_forward_backward_scale_invariance_and_size():
    torch.manual_seed(0)
    m = build_model({"name": "sleepband", "seq_len": 6})
    x = torch.randn(2, 6, 3000) * 20
    logits, aux = m(x, stats(2, 20.0))
    assert logits.shape == (2, 6, 5) and m.aux_loss(aux) == 0.0
    logits.mean().backward()
    assert m.bank.eta_raw.grad is not None and m.bank.eta_raw.grad.abs().sum() > 0
    m.eval()
    with torch.no_grad():
        a, _ = m(x, stats(2, 20.0)); b, _ = m(x * 7, stats(2, 140.0))
    assert torch.allclose(a, b, atol=1e-4)                      # input scaled by the night's robust SD
    assert 1e6 < count_parameters(m) < 5e6
