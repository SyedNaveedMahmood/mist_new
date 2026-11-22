"""HMC reader on a synthetic EDF pair (256 Hz, AASM labels, lights events)."""
import numpy as np
import pytest

mne = pytest.importorskip("mne")
pytest.importorskip("edfio")

from mist.data import hmc  # noqa: E402


def test_hmc_read_record_resamples_and_labels(tmp_path):
    rng = np.random.default_rng(0)
    n = 256 * 60 * 5 + 100
    names = list(hmc.EEG) + ["EMG chin"]
    raw = mne.io.RawArray(rng.standard_normal((5, n)) * 20e-6, mne.create_info(names, 256.0, ["eeg"] * 4 + ["emg"]),
                          verbose="ERROR")
    mne.export.export_raw(tmp_path / "SN001.edf", raw, verbose="ERROR")
    hyp = mne.io.RawArray(np.zeros((1, 256 * 300)), mne.create_info(["x"], 256.0, ["misc"]), verbose="ERROR")
    hyp.set_annotations(mne.Annotations(
        onset=[0, 30, 33.4, 60, 90, 120, 180, 250], duration=[30, 30, 0, 30, 30, 60, 60, 0],
        description=["Sleep stage W", "Sleep stage W", "Lights off@@EEG F4-A1", "Sleep stage N1", "Sleep stage N2",
                     "Sleep stage N3", "Sleep stage R", "Lights on@@EEG Fpz-Cz"]))
    mne.export.export_raw(tmp_path / "SN001_sleepscoring.edf", hyp, verbose="ERROR")
    (psg, h), = hmc.find_records(tmp_path)
    x, y, meta = hmc.read_record(psg, h)
    assert x.shape == (10, 4, 3000) and meta["fs_orig"] == 256.0
    assert y.tolist() == [0, 0, 1, 2, 3, 3, 4, 4, -1, -1]
    assert meta["lights_off_epoch"] == 1 and meta["lights_on_epoch"] == 8
    assert 5 < x.std() < 40


def test_hmc_unknown_annotation_raises():
    from types import SimpleNamespace
    ann = SimpleNamespace(onset=np.array([0.0]), duration=np.array([30.0]), description=np.array(["Arousal"]))
    with pytest.raises(ValueError):
        hmc.parse_annotations(ann, 2)
