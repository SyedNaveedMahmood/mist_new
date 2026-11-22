import torch
from conftest import eeg_like

from mist.data.surrogates import flatten_tilt, local_phase


def x():
    return eeg_like(4, 0)


def band_power(sig, lo, hi):
    X = torch.fft.rfft(sig, dim=-1).abs() ** 2
    f = torch.fft.rfftfreq(sig.shape[-1], 0.01)
    return X[..., (f >= lo) & (f < hi)].sum(-1)


def test_local_phase_keeps_band_power_roughly():
    a = x(); b = local_phase(a, generator=torch.Generator().manual_seed(3))
    assert b.shape == a.shape and torch.isfinite(b).all()
    for lo, hi in [(0.5, 4), (4, 8), (8, 13), (13, 30)]:
        ratio = band_power(b, lo, hi) / band_power(a, lo, hi)
        assert ((ratio > 0.3) & (ratio < 3.0)).all(), (lo, hi, ratio)
    assert (a - b).abs().mean() > 0.1 * a.abs().mean()


def test_flatten_tilt_keeps_phase_and_power_removes_slope():
    a = x(); b = flatten_tilt(a)
    f = torch.fft.rfftfreq(3000, 0.01); band = (f >= 1) & (f <= 35)
    pa, pb = torch.fft.rfft(a)[..., band], torch.fft.rfft(b)[..., band]
    assert torch.allclose(torch.angle(pa), torch.angle(pb), atol=1e-3)
    assert torch.allclose((a ** 2).mean(-1), (b ** 2).mean(-1), rtol=1e-3)
    def slope(P):
        lf = torch.log(f[band]); lf = lf - lf.mean()
        return (torch.log(P.abs() + 1e-8) * lf).sum(-1) / (lf ** 2).sum()
    assert slope(pa).abs().mean() > 0.3 and slope(pb).abs().max() < 1e-3
