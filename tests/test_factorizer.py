import torch
from conftest import eeg_like

from mist.data.surrogates import global_phase
from mist.models.factorizer import (Factorizer, Factors, gain_rhythm, night_codes, night_envelope,
                                     night_reference, whiten)


def sig(n=8, seed=0):
    return eeg_like(n, seed)


def rel(a, b):
    return float((a - b).norm() / a.norm())


def test_token_count_and_band():
    fz = Factorizer(100, 1.0, 0.5)
    assert fz.n_tokens(3000) == 59 and fz.n_freq == 35
    fz4 = Factorizer(100, 4.0, 2.0)
    assert fz4.n_tokens(3000) == 14 and fz4.n_freq == 139


def test_lossless_reconstruction_from_g_r_phi():
    fz = Factorizer(100, 1.0, 0.5)
    x = sig(40)
    ref = night_reference(x, [fz])
    xw = whiten(x, ref["night"]["envelope"])
    f = fz.factorize(xw)
    med, iqr = ref["s0"]["mag_med"][None], ref["s0"]["mag_iqr"][None]
    g, r = gain_rhythm(f.log_mag[None], med, iqr)
    log_mag = (r + g.unsqueeze(-1)) * (iqr + 0.1) + med            # invert the coding
    rebuilt = fz.reconstruct(Factors(log_mag[0], f.phasor))
    target = fz.synth(fz.spectra(xw))
    assert rel(target, rebuilt) < 1e-3
    assert torch.allclose(r.mean(-1), torch.zeros_like(g), atol=1e-4)   # R carries no level


def test_all_codes_are_gain_invariant():
    fz = Factorizer(100, 1.0, 0.5)
    x = sig(40)
    a, b = night_codes(x, fz), night_codes(7.5 * x, fz)
    for u, v in zip(a, b):
        assert rel(u, v) < 1e-3


def test_gain_and_rhythm_are_blind_to_phase():
    fz = Factorizer(100, 1.0, 0.5)
    f = fz.factorize(sig())
    scrambled = fz.synth(torch.exp(f.log_mag) * fz.random_phasor(f.phasor))
    mag2 = torch.fft.rfft(scrambled, dim=-1)[..., fz.band_idx].abs()
    assert torch.allclose(mag2, torch.exp(f.log_mag), atol=1e-3, rtol=1e-3)   # same |X| -> same G and R


def test_phase_only_morphology_is_blind_to_magnitude():
    fz = Factorizer(100, 1.0, 0.5)
    f = fz.factorize(sig())
    tilt = (fz.freqs / 10.0) ** -0.8                           # any positive magnitude response
    f2 = Factors(f.log_mag + torch.log(tilt), f.phasor)
    assert torch.allclose(fz.morph_token(f), fz.morph_token(f2), atol=1e-5)
    # the loose ablation is not blind, so it is a real contrast
    assert rel(fz.morph_token(f, "unit_raw"), fz.morph_token(f2, "unit_raw")) > 0.05


def test_night_whitening_cancels_recording_level_filters():
    """Signal-level check: filter a whole night, then code it. Raw tokens move a lot; G, R, M barely."""
    fz = Factorizer(100, 1.0, 0.5)
    x = sig(120)
    X, f = torch.fft.rfft(x), torch.fft.rfftfreq(3000, 0.01)
    base = night_codes(x, fz)
    bump = 1 + 0.8 * torch.exp(-((f - 12) / 6) ** 2) - 0.5 * torch.exp(-((f - 3) / 2) ** 2)
    for response, raw_floor in ((f.clamp(min=0.3) ** -0.5, 0.3), (bump, 0.15)):
        tilted = torch.fft.irfft(X * response, n=3000) * 4.0
        moved = night_codes(tilted, fz)
        assert rel(fz.tokens(x / x.std()), fz.tokens(tilted / tilted.std())) > raw_floor
        assert max(rel(u, v) for u, v in zip(base, moved)) < 0.03
    tilted = torch.fft.irfft(X * f.clamp(min=0.3) ** -0.5, n=3000) * 4.0
    moved = night_codes(tilted, fz)
    assert max(rel(u, v) for u, v in zip(base, moved)) < 0.03


def test_envelope_reproduces_a_power_law_inside_the_band():
    x = sig(60)
    env = night_envelope(x)
    f = torch.fft.rfftfreq(3000, 0.01)
    assert (env[f > 36.5] == 1).all() and (env[f < 0.29] == 1).all()
    band = (f >= 2) & (f <= 20)
    lf = torch.log(f[band]); lf = lf - lf.mean()
    slope = (torch.log(env[band]) * lf).sum() / (lf ** 2).sum()
    assert abs(float(slope) + 1.0) < 0.1                       # eeg_like has a 1/f amplitude spectrum


def test_global_surrogate_keeps_spectrum_changes_waveform():
    x = sig()
    s = global_phase(x, torch.Generator().manual_seed(1))
    assert torch.allclose(torch.fft.rfft(x).abs(), torch.fft.rfft(s).abs(), atol=1e-2, rtol=1e-3)
    assert rel(x, s) > 0.5


def test_night_reference_shapes():
    ref = night_reference(sig(30), [Factorizer(100, 1.0, 0.5), Factorizer(100, 4.0, 2.0)])
    assert ref["night"]["envelope"].shape == (1501,) and ref["night"]["scale"] > 0
    assert ref["s0"]["mag_med"].shape == (35,) and ref["s1"]["mag_iqr"].shape == (139,)
    assert (ref["s1"]["mag_iqr"] > 0).all()
