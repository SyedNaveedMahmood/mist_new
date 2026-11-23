"""Properties of the cascade topology (docs/ARCHITECTURE.md, section 8.5) and T6b acceptance checks."""
import pytest
import torch
from conftest import eeg_like

from mist.models.build import build_model, count_parameters
from mist.models.cascade import BAND_NAMES, band_masks
from mist.models.factorizer import Factorizer, Factors, night_reference, whiten

SMALL = {"name": "mist_grm", "seq_len": 6, "d_stream": 16, "d_token": 32, "d_epoch": 32, "band_slots": 4,
         "context_layers": 1, "scales": [[1.0, 0.5], [4.0, 2.0]], "topology": "cascade",
         "streams": ["G", "R", "M"]}


def rel(a, b):
    return float((a - b).norm() / a.norm())


def branch(k=0, **kw):
    torch.manual_seed(0)
    return build_model({**SMALL, **kw}).eval().branches[k]


def signal_codes(br, x):
    """Codes of a whole synthetic night x [n, 3000] through night whitening, as the model sees it."""
    ref = night_reference(x, [br.fz])
    f = br.fz.factorize(whiten(x, ref["night"]["envelope"])[None])
    return br.codes(f, {k: v[None] for k, v in ref["s0"].items()})


def factors(seed=0, k=0):
    br = branch(k)
    x = eeg_like(4, seed)
    return br, br.fz.factorize(x[None] / 20.0), {"mag_med": torch.zeros(1, br.fz.n_freq),
                                                 "mag_iqr": torch.ones(1, br.fz.n_freq)}


def test_bands_cover_the_token_band_without_overlap():
    for win in (1.0, 4.0):
        fz = Factorizer(100, win, win / 2)
        m = band_masks(fz.freqs)
        assert (m.sum(0) == 1).all()                         # every in-band bin in exactly one band
        assert (m.sum(1) >= 3).all()                         # narrow bands hold 3-4 bins at 1 s


@torch.no_grad()
def test_g_r_and_band_waveforms_are_gain_invariant():
    br = branch()
    x = eeg_like(30)
    a, b = signal_codes(br, x), signal_codes(br, 7.5 * x)
    for k in ("g", "r", "gates", "m"):
        assert rel(a[k], b[k]) < 1e-3, k


@torch.no_grad()
def test_rhythm_code_and_gates_are_functions_of_magnitude_only():
    br, f, st = factors()
    a = br.codes(f, st)
    b = br.codes(Factors(f.log_mag, br.fz.random_phasor(f.phasor, torch.Generator().manual_seed(3))), st)
    for k in ("g", "r", "gates"):
        assert torch.equal(a[k], b[k]), k
    assert rel(a["m"], b["m"]) > 0.5                          # the contrast is real


@torch.no_grad()
def test_band_waveforms_codes_and_slots_are_functions_of_phase_only():
    br, f, st = factors()
    a = br.codes(f, st)
    tilt = torch.log((br.fz.freqs / 10.0) ** -0.8)            # any positive magnitude response
    b = br.codes(Factors(f.log_mag + tilt + 2.0, f.phasor), st)
    for k in ("m", "e", "a"):
        assert torch.equal(a[k], b[k]), k
    assert not torch.allclose(a["r"], b["r"])


@torch.no_grad()
def test_unnormalised_band_waveforms_sum_to_the_v1_morphology_token():
    br, f, _ = factors()
    bands = br.waveforms(f.phasor, normalise=False)[..., :len(BAND_NAMES), :]
    v1 = br.fz.synth(f.phasor)                                # morph_token before unit-RMS scaling
    assert torch.allclose(bands.sum(-2), v1, atol=1e-5)
    assert torch.allclose(br.waveforms(f.phasor, normalise=False)[..., -1, :], v1, atol=1e-6)   # broadband channel


@torch.no_grad()
def test_destroying_morphology_leaves_gain_and_rhythm_untouched():
    br, f, st = factors()
    a = br.codes(f, st)
    d = br.codes(f, st, destroy_prob=1.0, gen=torch.Generator().manual_seed(1))
    for k in ("g", "r", "gates"):
        assert torch.equal(a[k], d[k]), k
    assert rel(a["m"], d["m"]) > 0.5


@torch.no_grad()
def test_smooth_zero_phase_filter_barely_moves_g_r_and_band_waveforms():
    """Approximate, as in ARCHITECTURE section 3 (same threshold as tests/test_factorizer.py)."""
    br = branch()
    x = eeg_like(120)
    X, fr = torch.fft.rfft(x), torch.fft.rfftfreq(3000, 0.01)
    base = signal_codes(br, x)
    bump = 1 + 0.8 * torch.exp(-((fr - 12) / 6) ** 2) - 0.5 * torch.exp(-((fr - 3) / 2) ** 2)
    for response in (fr.clamp(min=0.3) ** -0.5, bump):
        moved = signal_codes(br, torch.fft.irfft(X * response, n=3000) * 4.0)
        for k in ("g", "R", "m"):
            assert rel(base[k], moved[k]) < 0.03, (k, rel(base[k], moved[k]))


@torch.no_grad()
def test_placebo_output_identical_when_only_the_phase_differs():
    """T6b: with destroy_prob 1.0, a token and the same token with its phase replaced give the same output."""
    for k in (0, 1):
        br, f, st = factors(k=k)
        other = Factors(f.log_mag, br.fz.random_phasor(f.phasor, torch.Generator().manual_seed(9)))
        h1, _ = br.from_factors(f, st, destroy_prob=1.0, gen=torch.Generator().manual_seed(5))
        h2, _ = br.from_factors(other, st, destroy_prob=1.0, gen=torch.Generator().manual_seed(5))
        assert torch.equal(h1, h2)
        h3, _ = br.from_factors(f, st)
        assert not torch.allclose(h1, h3)


def test_cascade_and_placebo_have_equal_parameter_counts():
    a = build_model({**SMALL, "critic_weight": 0.3, "allpass_rad": 0.5})
    b = build_model({**SMALL, "destroy_prob": 1.0, "critic_weight": 0.0})
    assert count_parameters(a) == count_parameters(b)


@pytest.mark.parametrize("channels", ["all", "bands", "broad"])
def test_forward_backward_and_switches(channels):
    torch.manual_seed(0)
    m = build_model({**SMALL, "cascade_channels": channels, "critic_weight": 0.3, "allpass_rad": 0.5})
    from test_model import stats
    x = torch.randn(2, 6, 3000)
    logits, aux = m(x, stats(2), y=torch.randint(0, 5, (2, 6)))
    assert logits.shape == (2, 6, 5) and "critic_s0" in aux["losses"] and "diversity" in aux["losses"]
    (logits.mean() + m.aux_loss(aux)).backward()
    n_ch = {"all": 6, "bands": 5, "broad": 1}[channels]
    assert aux["assign"][0].shape == (2, 6, 59, n_ch * 4)
    m.eval()
    with torch.no_grad():
        base, _ = m(x, stats(2))
        assert not torch.allclose(base, m(x, stats(2), knock=("R",))[0])
        assert not torch.allclose(base, m(x, stats(2), destroy_bands=("sigma",))[0])
        if channels != "bands":
            assert not torch.allclose(base, m(x, stats(2), destroy_bands=("broad",))[0])
        else:
            with pytest.raises(ValueError):
                m(x, stats(2), destroy_bands=("broad",))
        k1, _ = m(x, stats(2), knock=("M",))
        k2, _ = m(x, stats(2), knock=("M",), destroy_prob=1.0)
        assert torch.allclose(k1, k2, atol=1e-6)              # M knocked out: phase cannot matter


@torch.no_grad()
def test_destroying_a_band_only_changes_that_band_and_broadband():
    br, f, st = factors()
    a = br.codes(f, st)
    d = br.codes(f, st, gen=torch.Generator().manual_seed(2), destroy_bands=("sigma",))
    i = BAND_NAMES.index("sigma")
    for j in range(len(BAND_NAMES)):
        same = torch.equal(a["m"][..., j, :], d["m"][..., j, :])
        assert same == (j != i), BAND_NAMES[j]
    assert not torch.equal(a["m"][..., -1, :], d["m"][..., -1, :])


def test_parallel_topology_rejects_band_switches():
    from test_model import SMALL as PSMALL, stats
    m = build_model({**PSMALL, "streams": ["G", "R", "M"]}).eval()
    with pytest.raises(ValueError):
        m(torch.randn(1, 6, 3000), stats(1), destroy_bands=("sigma",))


def test_cascade_trains_end_to_end_and_checkpoint_keeps_gate_means(store, tmp_path):
    import json
    from mist.data import splits as S
    from mist.data.store import load_cohort
    from mist.engine.loop import evaluate, load_checkpoint, make_dataset, train
    root, subs = store
    sp = {"version": 1, "seed": 0, "quarantined_sc": [], "lockbox": [], "dev": sorted(subs),
          "folds": [sorted(subs)[i::4] for i in range(4)], "sets": {"young": [], "mid": [], "old": []},
          "age_cuts": [0, 0], "st": []}
    spf = tmp_path / "sp.json"; spf.write_text(json.dumps(sp))
    cfg = {"data": {"cohort": "SC", "channel": "Fpz-Cz", "view": "full", "seq_len": 6, "windows_per_night": 8},
           "split": {"file": str(spf), "protocol": "cv", "fold": 0},
           "train": {"epochs": 3, "batch_size": 8, "lr": 3e-3, "seed": 0, "num_workers": 0, "device": "cpu"},
           "model": {**SMALL, "critic_weight": 0.3, "allpass_rad": 0.5}}
    out = train(cfg, {"store_root": str(root)}, tmp_path / "run")
    assert out["best_val_mf1"] > 0.2, out
    model, cfg2 = load_checkpoint(tmp_path / "run", torch.device("cpu"))
    gm = model.branches[0].gate_mean
    assert not torch.allclose(gm, torch.full_like(gm, 0.6931), atol=1e-4)   # learned, and restored from disk
    ds = make_dataset(load_cohort(root, "SC", set(S.resolve(sp, "cv", 0)["test"]), split_file=spf), cfg2,
                      "Fpz-Cz", train=False)
    real, _ = evaluate(model, ds, torch.device("cpu"))
    band, _ = evaluate(model, ds, torch.device("cpu"), destroy_bands=("sigma",))
    assert real["n"] == band["n"] > 0


def test_checkpointed_encoder_matches_plain_encoder_values_and_gradients():
    br = branch()
    br.CKPT_CHUNK = 7                                         # force several chunks on a small input
    m = torch.randn(2, 3, 6, br.fz.win)
    e1 = br.encode(m); g1 = torch.autograd.grad(e1.square().sum(), list(br.morph.parameters()))
    br.CKPT_CHUNK = 10 ** 9                                   # no checkpointing
    e2 = br.encode(m); g2 = torch.autograd.grad(e2.square().sum(), list(br.morph.parameters()))
    assert torch.allclose(e1, e2, atol=1e-5)
    assert all(torch.allclose(a, b, atol=1e-4, rtol=1e-4) for a, b in zip(g1, g2))
