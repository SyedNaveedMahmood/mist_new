"""End-to-end check of the Sleep-EDF reader on a synthetic EDF pair.

This exercises file pairing, channel picking, microvolt units, the 30-s grid and
the stage mapping. It cannot stand in for task T1, which runs the reader on the
real recordings and compares counts with the earlier audit.
"""
import numpy as np
import pytest

mne = pytest.importorskip("mne")
pytest.importorskip("edfio")

from mist.data import sleepedf as E  # noqa: E402


def test_read_recording_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    n = 100 * 60 * 10 + 1234                                   # 10 min plus a partial epoch
    data = rng.standard_normal((3, n)) * 20e-6                 # volts
    info = mne.create_info(["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"], 100.0, ["eeg", "eeg", "eog"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    mne.export.export_raw(tmp_path / "SC4991E0-PSG.edf", raw, verbose="ERROR")
    hyp = mne.io.RawArray(np.zeros((1, 100 * 60 * 10)), mne.create_info(["x"], 100.0, ["misc"]), verbose="ERROR")
    hyp.set_annotations(mne.Annotations(
        onset=[0, 120, 150, 300, 420, 540], duration=[120, 30, 150, 120, 120, 30],
        description=["Sleep stage W", "Sleep stage 1", "Sleep stage 2", "Sleep stage 4", "Sleep stage R",
                     "Sleep stage ?"]))
    mne.export.export_raw(tmp_path / "SC4991EC-Hypnogram.edf", hyp, verbose="ERROR")

    (psg, hy), = E.find_pairs(tmp_path)
    assert E.parse_name(psg.name)["subject"] == "SC:99"
    x, y, chans = E.read_recording(psg, hy)
    assert x.shape == (20, 2, 3000) and chans == ["Fpz-Cz", "Pz-Oz"]
    assert y.tolist() == [0] * 4 + [1] + [2] * 5 + [3] * 4 + [4] * 4 + [-1] + [-1]
    assert 5 < x.std() < 40                                    # microvolt scale after 0.3-35 Hz filtering
