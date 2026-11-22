"""The lockbox (CLAUDE.md rule 1) cannot be reached through store.load_cohort by default."""
import json
from pathlib import Path

import pytest

from conftest import write_split
from mist.data import splits as S
from mist.data.store import DEFAULT_SPLIT, load_cohort, subject_of_record


def test_default_excludes_lockbox_and_never_opens_its_files(store):
    root, subs = store
    spf = write_split(root, subs, lockbox=["SC:21", "SC:22"])
    (root / "SC" / "SC4211" / "x.npy").write_bytes(b"not an array")   # would fail if memory-mapped
    got = {n.subject for n in load_cohort(root, "SC", split_file=spf)}
    assert got == set(subs) - {"SC:21", "SC:22"}


def test_explicit_lockbox_request_raises(store):
    root, subs = store
    spf = write_split(root, subs, lockbox=["SC:21"])
    with pytest.raises(PermissionError):
        load_cohort(root, "SC", {"SC:20", "SC:21"}, split_file=spf)


def test_open_lockbox_flag_returns_them(store):
    root, subs = store
    spf = write_split(root, subs, lockbox=["SC:21"])
    assert [n.subject for n in load_cohort(root, "SC", {"SC:21"}, split_file=spf, open_lockbox=True)] == ["SC:21"]


def test_default_split_is_the_frozen_file():
    assert DEFAULT_SPLIT == Path(__file__).resolve().parents[1] / "splits" / "sleepedf_v1.json"
    if DEFAULT_SPLIT.exists():
        sp = S.load(DEFAULT_SPLIT)
        assert len(sp["lockbox"]) == 20 and not set(sp["lockbox"]) & set(sp["dev"])


def test_record_folder_names():
    assert subject_of_record("SC4071") == "SC:07" and subject_of_record("ST7022") == "ST:02"
    with pytest.raises(ValueError):
        subject_of_record("XX1234")


def test_legacy_npz_refuses_lockbox_files(store, tmp_path):
    import numpy as np
    from mist.data.store import load_legacy_npz
    root, subs = store
    spf = write_split(root, subs, lockbox=["SC:21"])
    for name in ("SC4211E0.npz", "SC4201E0.npz"):
        np.savez(tmp_path / name, x=np.zeros((3, 3000), np.float32), y=np.zeros(3, np.int8), fs=100)
    with pytest.raises(PermissionError):
        load_legacy_npz(tmp_path / "SC4211E0.npz", split_file=spf)
    assert load_legacy_npz(tmp_path / "SC4201E0.npz", split_file=spf).subject == "SC:20"
