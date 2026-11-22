import numpy as np
import pytest

from mist.data import splits as S
from mist.data.dataset import SequenceDataset
from mist.data.sleepedf import labels_from_annotations, parse_name
from mist.data.store import load_cohort
from conftest import write_split


def test_parse_name():
    assert parse_name("SC4071E0-PSG.edf") == {"cohort": "SC", "subject_num": 7, "night": 1, "subject": "SC:07",
                                              "record_id": "SC4071"}
    assert parse_name("ST7022J0-PSG.edf")["subject"] == "ST:02"
    with pytest.raises(ValueError):
        parse_name("SC4071EC-Hypnogram.edf")


def test_labels_from_annotations_merges_n3_and_marks_unscored():
    on = [0, 60, 90, 150, 180, 240]
    du = [60, 30, 60, 30, 60, 30]
    de = ["Sleep stage W", "Sleep stage 1", "Sleep stage 3", "Sleep stage 4", "Sleep stage R", "Movement time"]
    y = labels_from_annotations(on, du, de, 10)
    assert y.tolist() == [0, 0, 1, 3, 3, 3, 4, 4, -1, -1]
    with pytest.raises(ValueError):
        labels_from_annotations([0], [30], ["Lights off"], 2)


def test_store_views_and_eval_tiling(store):
    root, subs = store
    nights = load_cohort(root, "SC", split_file=write_split(root, subs, lockbox=[]))
    assert len(nights) == 12 and nights[0].x.shape == (60, 2, 3000)
    ds = SequenceDataset(nights[:2], "Pz-Oz", [(1.0, 0.5), (4.0, 2.0)], view="full", seq_len=25, train=False)
    seen = {}
    for i in range(len(ds)):
        it = ds[i]
        for j in np.flatnonzero(it["keep"].numpy()):
            key = (it["night"], it["start"] + int(j))
            assert key not in seen
            seen[key] = 1
    assert len(seen) == 120                      # every epoch of both nights exactly once
    assert it["night.envelope"].shape == (1501,) and it["s1.mag_med"].shape == (139,)


def test_std_view_trims_wake():
    from mist.data.store import Night
    y = np.array([0] * 200 + [2] * 50 + [0] * 200, dtype=np.int8)
    n = Night("r", "SC:01", 1, "SC", ["a"], np.zeros((450, 1, 1)), y, {})
    sl = n.view_slice("std")
    assert (sl.start, sl.stop) == (140, 310)


def test_splits_are_disjoint_and_deterministic(store):
    _, subs = store
    subs = dict(subs)
    subs.update({f"SC:{i:02d}": {"age": 25 + i % 10} for i in range(20)})      # exposed people
    subs.update({f"SC:{i:02d}": {"age": 60 + i % 30} for i in range(32, 80)})  # fresh people
    subs.update({f"ST:{i:02d}": {"age": None} for i in range(1, 23)})
    a, b = S.make_splits(subs, seed=1), S.make_splits(subs, seed=1)
    assert a == b
    assert len(a["lockbox"]) == 20
    assert all(int(s.split(":")[1]) >= 20 for s in a["lockbox"])               # never an exposed person
    assert not {"SC:06", "SC:23", "SC:36", "SC:74"} & set(a["dev"] + a["lockbox"])
    for proto in ("cv", "young2old", "old2young", "sc2st", "lockbox"):
        r = S.resolve(a, proto, fold=2)
        assert not set(r["train"]) & set(r["val"]) and not set(r["train"]) & set(r["test"])
        assert not set(r["val"]) & set(r["test"])
        if proto != "lockbox":
            assert not set(r["train"] + r["val"] + r["test"]) & set(a["lockbox"])
