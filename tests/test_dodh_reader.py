"""DOD-H reader on a synthetic .h5 in the layout of the real files (250 Hz, 'mV' attribute, uV numbers)."""
import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("mne")

from mist.data import dodh  # noqa: E402

NAMES = ("C3-M2", "F4-M1", "F3-M2", "FP1-F3")


def write(path, n_epochs=8, scale=20.0, hyp=None, f0=10.0):
    fs = 250
    t = np.arange(n_epochs * 30 * fs) / fs
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        desc = []
        for nm in NAMES:
            p = f"signals/eeg/{nm.replace('-', '_')}"
            sig = scale * (np.sin(2 * np.pi * f0 * t) + 0.1 * rng.standard_normal(t.size))
            f.create_dataset(p, data=sig.astype(np.float32))
            desc.append({"default": True, "domain": "EEG", "fs": fs, "name": nm, "path": p, "unit": "mV"})
        f.create_dataset("signals/emg/EMG", data=np.zeros(t.size, np.float32))
        desc.append({"domain": "EMG", "fs": fs, "name": "EMG", "path": "signals/emg/EMG", "unit": "mV"})
        f.attrs["description"] = json.dumps(desc)
        f.attrs["duration"] = n_epochs * 30
        f.create_dataset("hypnogram", data=np.array(hyp if hyp is not None else [0, 1, 2, 3, 4, -1, 2, 2]))


def test_read_record(tmp_path):
    p = tmp_path / "0d79f4b1-e74f-5e87-8e42-f9dd7112ada5.h5"
    write(p)
    (q,) = dodh.find_records(tmp_path)
    x, y, meta = dodh.read_record(q)
    assert x.shape == (8, 4, 3000) and x.dtype == np.float32
    assert y.tolist() == [0, 1, 2, 3, 4, -1, 2, 2]
    assert meta["channels"] == ["C3-M2", "F4-M1", "F3-M2", "Fp1-F3"] and meta["fs_orig"] == 250.0
    assert meta["rule_channel"] == "F4-M1" and meta["rule_tier"] == "frontal-mastoid"
    assert meta["units_attr"] == ["mV"]
    mid = x[2:6, 1].ravel()                                  # 10 Hz sine of amplitude 20 survives in uV
    assert np.std(mid) == pytest.approx(20 / np.sqrt(2), rel=0.05)
    assert meta["scorer"].startswith("UNKNOWN")


def test_millivolt_scale_numbers_are_refused(tmp_path):
    p = tmp_path / "a.h5"
    write(p, scale=0.02)
    with pytest.raises(ValueError, match="microvolt"):
        dodh.read_record(p)


def test_bad_hypnogram_code_raises(tmp_path):
    p = tmp_path / "b.h5"
    write(p, hyp=[0, 1, 2, 3, 4, 5, 2, 2])
    with pytest.raises(ValueError, match="hypnogram"):
        dodh.read_record(p)
