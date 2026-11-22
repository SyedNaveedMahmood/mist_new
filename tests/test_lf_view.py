"""Label-free `lf` view (T5): 30 min before lights-off, 10 h, clipped to the recording."""
import numpy as np
import pytest

from mist.data import sleepedf as E
from mist.data.store import Night, lf_slice, std_slice


def test_lights_off_epoch_crosses_midnight_and_tolerates_early_lights_off():
    assert E.lights_off_epoch("16:13:00", "00:38:00") == (8 * 3600 + 25 * 60) // 30
    assert E.lights_off_epoch("23:00:30", "23:00:00") == -1        # sheet minute resolution
    assert E.lights_off_epoch("21:48:00", "23:42:00") == 228


def test_lf_slice_full_and_clipped():
    assert lf_slice(500, 3000) == slice(440, 1640)                 # 1200 epochs = 10 h
    assert lf_slice(0, 1000) == slice(0, 1000)                     # ST: starts at lights-off, ends early
    assert lf_slice(-1, 900) == slice(0, 900)
    with pytest.raises(ValueError):
        lf_slice(5000, 1000)


def test_window_meta_counts_and_flags_do_not_use_labels_for_lf():
    y = np.full(3000, 0, np.int8); y[1000:1900] = 2
    a = E.window_meta("16:00:00", "00:00:00", y)
    y2 = y.copy(); y2[:] = 0                                       # different labels, same lf window
    b = E.window_meta("16:00:00", "00:00:00", y2)
    assert a["lights_off_epoch"] == b["lights_off_epoch"] == 960
    assert a["n_epochs_lf"] == b["n_epochs_lf"] == 1200 and not a["lf_clipped_end"]
    assert a["n_epochs_std"] == 900 + 120
    short = E.window_meta("16:00:00", "00:00:00", y[:1500])
    assert short["n_epochs_lf"] == 1500 - 900 and short["lf_clipped_end"] and not short["lf_clipped_start"]
    assert E.window_meta("16:00:00", None, y)["n_epochs_lf"] is None


def test_night_view_slice_lf_needs_meta():
    y = np.zeros(2000, np.int8)
    n = Night("SC4001", "SC:00", 1, "SC", ["Fpz-Cz"], np.zeros((2000, 1, 3000), np.float32), y,
              {"lights_off_epoch": 100})
    assert n.view_slice("lf") == slice(40, 1240)
    assert n.view_slice("std") == std_slice(y)
    with pytest.raises(ValueError):
        Night("SC4001", "SC:00", 1, "SC", ["Fpz-Cz"], n.x, y, {}).view_slice("lf")
