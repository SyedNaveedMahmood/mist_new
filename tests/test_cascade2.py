"""Revision 3, own-rate cascade (docs/ARCHITECTURE.md section 9; owner's tests 1 to 8, task T6c)."""
import math

import torch
from conftest import eeg_like
from test_cascade import SMALL, rel, signal_codes
from test_model import stats

from mist.models.build import build_model, count_parameters
from mist.models.cascade import BAND_NAMES
from mist.models.factorizer import Factors

OWN = {**SMALL, "band_rate": "own", "band_scales": [4.0], "band_width": 8, "critic_weight": 0.3, "allpass_rad": 0.5}
CODES = ("g", "r", "gates", "m", "e", "a")


def model(**kw):
    torch.manual_seed(0)
    return build_model({**OWN, **kw}).eval()


def factors(br, seed=0):
    x = eeg_like(4, seed)
    return br.fz.factorize(x[None] / 20.0), {"mag_med": torch.zeros(1, br.fz.n_freq),
                                            "mag_iqr": torch.ones(1, br.fz.n_freq)}


def same(a, b):
    if isinstance(a, list):
        return all(torch.equal(u, v) for u, v in zip(a, b))
    return torch.equal(a, b)


def close(a, b, tol):
    if isinstance(a, list):
        return all(rel(u, v) < tol for u, v in zip(a, b))
    return rel(a, b) < tol


def test_channels_and_lengths_follow_the_revision():
    m = model()
    s1, s4 = m.branches
    assert s1.channels == ("broad",) and s1.ch_len == [72]
    assert s4.channels == BAND_NAMES + ("broad",) and s4.ch_len == [32, 64, 96, 128, 288, 288]
    assert s1.morph_band is None and s4.morph_band is not None          # band encoder only where bands are


@torch.no_grad()
def test_1_own_rate_waveform_is_the_full_rate_waveform_on_the_coarse_grid():
    for br in model().branches:
        f, _ = factors(br)
        own = br.waveforms(f.phasor, normalise=False)
        win, F = br.fz.win, f.phasor.shape[-1]
        k = br.fz.band_idx.double()
        for i, c in enumerate(br.channels):
            pos = getattr(br, f"ch_pos{i}")
            sel = torch.zeros(F, dtype=torch.bool); sel[pos] = True
            spec = torch.where(sel, f.phasor, torch.zeros_like(f.phasor)).to(torch.complex128)
            n = br.ch_len[i]
            t = torch.arange(n, dtype=torch.float64) * win / n                  # coarse grid, in samples of the token
            ph = torch.exp(2j * math.pi * k[:, None] * t[None] / win)           # [F, n]
            cont = 2 * (spec @ ph).real / win                                   # band-limited waveform at the grid points
            w = own[i].double()
            assert torch.allclose(w * n / win, cont, atol=1e-5), c              # irfft's 1/n vs 1/win
            full = br.fz.synth(spec)                                            # full-rate waveform, win samples
            rms_full = full.pow(2).mean(-1).sqrt()
            rms_own = (w * n / win).pow(2).mean(-1).sqrt()
            assert torch.allclose(rms_own, rms_full, rtol=1e-5), c              # Parseval: same RMS
            unit = br.waveforms(f.phasor)[i].double()
            assert torch.allclose(unit, cont / cont.pow(2).mean(-1, keepdim=True).sqrt(), atol=1e-4), c


@torch.no_grad()
def test_2_every_code_is_gain_invariant():
    for br in model().branches:
        x = eeg_like(30)
        a, b = signal_codes(br, x), signal_codes(br, 7.5 * x)
        for key in CODES:
            assert close(a[key], b[key], 1e-3), key


@torch.no_grad()
def test_3_gain_rhythm_and_gates_depend_on_magnitude_only():
    for br in model().branches:
        f, st = factors(br)
        a = br.codes(f, st)
        b = br.codes(Factors(f.log_mag, br.fz.random_phasor(f.phasor, torch.Generator().manual_seed(3))), st)
        for key in ("g", "r", "gates"):
            assert torch.equal(a[key], b[key]), key
        assert not close(a["m"], b["m"], 0.5)


@torch.no_grad()
def test_4_shape_codes_depend_on_phase_only():
    for br in model().branches:
        f, st = factors(br)
        a = br.codes(f, st)
        tilt = torch.log((br.fz.freqs / 10.0) ** -0.8)
        b = br.codes(Factors(f.log_mag + tilt + 2.0, f.phasor), st)
        for key in ("m", "e", "a"):
            assert same(a[key], b[key]), key
        assert not torch.allclose(a["r"], b["r"])


@torch.no_grad()
def test_5_rfft_of_each_own_rate_waveform_returns_its_phasor_bins():
    for br in model().branches:
        f, _ = factors(br)
        own = br.waveforms(f.phasor, normalise=False)
        for i, c in enumerate(br.channels):
            pos = getattr(br, f"ch_pos{i}")
            Z = torch.fft.rfft(own[i].double(), dim=-1)
            k = br.fz.band_idx[pos]
            assert torch.allclose(Z[..., k], f.phasor[..., pos].to(Z.dtype), atol=1e-5), c
            rest = torch.ones(Z.shape[-1], dtype=torch.bool); rest[k] = False
            assert Z[..., rest].abs().max() < 1e-5, c                           # nothing else, no aliasing


def test_6_destroy_prob_one_same_parameters_no_shape_information():
    a, b = model(), model(destroy_prob=1.0, critic_weight=0.0)
    assert count_parameters(a) == count_parameters(b)
    with torch.no_grad():
        for br in b.branches:
            f, st = factors(br)
            other = Factors(f.log_mag, br.fz.random_phasor(f.phasor, torch.Generator().manual_seed(9)))
            h1, _ = br.from_factors(f, st, destroy_prob=1.0, gen=torch.Generator().manual_seed(5))
            h2, _ = br.from_factors(other, st, destroy_prob=1.0, gen=torch.Generator().manual_seed(5))
            assert torch.equal(h1, h2)
            assert not torch.allclose(h1, br.from_factors(f, st)[0])


@torch.no_grad()
def test_7_destroying_a_band_changes_only_that_band_and_broadband():
    s1, s4 = model().branches
    for band in BAND_NAMES:
        f, st = factors(s4)
        a = s4.codes(f, st)
        d = s4.codes(f, st, gen=torch.Generator().manual_seed(2), destroy_bands=(band,))
        for j, c in enumerate(s4.channels):
            assert torch.equal(a["m"][j], d["m"][j]) == (c not in (band, "broad")), (band, c)
        f, st = factors(s1)                                                     # 1 s: broadband only, still hit
        a = s1.codes(f, st)
        d = s1.codes(f, st, gen=torch.Generator().manual_seed(2), destroy_bands=(band,))
        assert not torch.equal(a["m"][0], d["m"][0]), band


def test_8_defaults_reproduce_the_section_8_cascade():
    ref = torch.load("tests/fixtures/cascade_v1_reference.pt")
    m = build_model(ref["cfg"])
    m.load_state_dict(ref["state"], strict=True)                                # old checkpoints load
    m.eval()
    calls = {"real": {}, "knock_R": {"knock": ("R",)},
             "destroy_sigma": {"destroy_bands": ("sigma",), "gen": torch.Generator().manual_seed(4)},
             "destroy_all": {"destroy_prob": 1.0, "gen": torch.Generator().manual_seed(5)}}
    with torch.no_grad():
        for k, kw in calls.items():
            # allclose, not equal: other torch builds differ by ~3e-7 (owner, 2026-11-23); bit-identical on the workstation
            assert torch.allclose(m(ref["x"], stats(2), **kw)[0], ref["out"][k], atol=1e-5), k


def test_forward_backward_switches_and_separate_band_encoder():
    m = model()
    m.train()
    x = torch.randn(2, 6, 3000)
    logits, aux = m(x, stats(2), y=torch.randint(0, 5, (2, 6)))
    assert "critic_s0" in aux["losses"] and "critic_s1" in aux["losses"]
    assert aux["assign"][0].shape == (2, 6, 59, 1 * 4) and aux["assign"][1].shape == (2, 6, 14, 6 * 4)
    (logits.mean() + m.aux_loss(aux)).backward()
    band = m.branches[1].morph_band
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in band.parameters())
    assert band.conv[0][0].out_channels == 8 and m.branches[1].morph.conv[0][0].out_channels == 32
    m.eval()
    with torch.no_grad():
        base, _ = m(x, stats(2))
        for kw in ({"knock": ("R",)}, {"destroy_bands": ("delta",)}, {"destroy_bands": ("broad",)}):
            assert not torch.allclose(base, m(x, stats(2), **kw)[0]), kw


def test_checkpointing_switch_does_not_change_values_or_gradients():
    torch.manual_seed(1)
    br = model().branches[1]
    m = [torch.randn(2, 3, n) for n in br.ch_len]
    br.CKPT_CHUNK = 5
    e1 = br.encode(m); g1 = torch.autograd.grad(e1.square().sum(), list(br.parameters()), allow_unused=True)
    br.cfg.encoder_checkpoint = False
    try:
        e2 = br.encode(m); g2 = torch.autograd.grad(e2.square().sum(), list(br.parameters()), allow_unused=True)
    finally:
        br.cfg.encoder_checkpoint = True
    assert torch.allclose(e1, e2, atol=1e-5)
    assert all((a is None and b is None) or torch.allclose(a, b, atol=1e-4, rtol=1e-4) for a, b in zip(g1, g2))
