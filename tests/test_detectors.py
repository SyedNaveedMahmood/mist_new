import numpy as np

from mist.detectors import alpha, slow_waves, spindles, token_overlap


def background(n_ep=40, fs=100, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n_ep * 30 * fs)
    w = np.cumsum(w) * 0.05 + rng.standard_normal(w.size) * 5          # crude 1/f-ish background, microvolt
    return w.reshape(n_ep, 30 * fs)


def test_spindle_detected_where_inserted_and_scale_free():
    x = background()
    t = np.arange(100) / 100
    burst = 40 * np.sin(2 * np.pi * 13 * t) * np.hanning(100)            # 1 s, 13 Hz
    x[10, 1000:1100] += burst
    m = spindles(x)
    assert m[10, 1020:1080].mean() > 0.8 and m[20].sum() == 0
    assert (spindles(7.0 * x) == m).all()                                  # night-relative threshold


def test_alpha_and_slow_wave():
    x = background(seed=1)
    t = np.arange(300) / 100
    x[5, 0:300] += 30 * np.sin(2 * np.pi * 10 * t)
    assert alpha(x)[5, 50:250].mean() > 0.8
    tt = np.arange(100) / 100
    x[30, 1500:1600] += -150 * np.sin(np.pi * tt)                         # 1 s negative half-wave
    sw = slow_waves(x)
    assert sw[30, 1540:1560].all()


def test_token_overlap():
    m = np.zeros((2, 3000), bool); m[1, 100:200] = True
    assert token_overlap(m, [(1, 100, 100), (1, 150, 100), (0, 0, 100)]).tolist() == [1.0, 0.5, 0.0]


def test_name_slots_requires_lift_and_overlap():
    from mist.detectors import name_slots
    top = {"spindle": np.array([0.6, 0.1, 0.25]), "alpha": np.array([0.1, 0.1, 0.05])}
    base = {"spindle": 0.1, "alpha": 0.1}
    n = name_slots(top, base)
    assert n[0]["name"] == "spindle"          # lift 6, overlap 0.6
    assert n[1]["name"] is None               # lift 1
    assert n[2]["name"] is None               # lift 2.5 but overlap 0.25 < 0.3
