"""Shared phase-2 preprocessing (resample.preprocess): amplitude, alias rejection, zero phase."""
import numpy as np
import pytest

pytest.importorskip("mne")

from mist.data import resample  # noqa: E402


def sine(fs, f, seconds=120, amp=50.0, phase=0.3):
    t = np.arange(int(seconds * fs)) / fs
    return amp * np.sin(2 * np.pi * f * t + phase)


@pytest.mark.parametrize("fs", [100, 128, 200, 250, 256, 512])
def test_10hz_sine_keeps_amplitude_and_phase(fs):
    out = resample.preprocess(sine(fs, 10.0)[None], fs)[0]
    ref = sine(100, 10.0)
    assert out.shape == ref.shape
    mid = slice(2000, len(ref) - 2000)                       # away from the edges
    assert np.std(out[mid]) == pytest.approx(np.std(ref[mid]), rel=0.01)
    assert np.max(np.abs(out[mid] - ref[mid])) < 0.02 * 50   # same phase: sample-by-sample agreement
    xc = [np.dot(out[mid], np.roll(ref, k)[mid]) for k in range(-3, 4)]
    assert int(np.argmax(xc)) - 3 == 0


@pytest.mark.parametrize("fs", [200, 256])
def test_60hz_component_is_removed(fs):
    x = sine(fs, 10.0) + sine(fs, 60.0, amp=100.0, phase=1.0)
    out = resample.preprocess(x[None], fs)[0]
    resid = out - sine(100, 10.0)
    assert np.std(resid[2000:-2000]) < 0.005 * 100 / np.sqrt(2)   # 60 Hz (and its 40 Hz alias) down > 46 dB


def test_band_edges_and_epoching():
    fs = 256
    slow = resample.preprocess(sine(fs, 0.05, amp=100)[None], fs)[0]
    assert np.std(slow[3000:-3000]) < 5.0                    # below the 0.3 Hz high-pass
    x = resample.preprocess(np.random.default_rng(0).standard_normal((2, fs * 95)), fs)
    ep = resample.to_epochs(x, start_sample=500)
    assert ep.shape == (3, 2, 3000) and ep.dtype == np.float32
    assert np.array_equal(ep[1, 0], x[0, 3500:6500].astype(np.float32))


def test_ratio():
    assert resample.ratio(256) == (25, 64) and resample.ratio(250) == (2, 5) and resample.ratio(100) == (1, 1)
    assert resample.ratio(512.0) == (25, 128)
