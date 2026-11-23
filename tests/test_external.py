"""External stores: subject ids, the holdout guard, the split script and evaluate.py --dataset."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import synth_night
from mist.data.external import load_external, subject_of_folder
from mist.data.store import save_night

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ext_store(tmp_path):
    """Ten synthetic HMC-like recordings; odd ones have C4-M1 as their rule channel."""
    records = {}
    for k in range(1, 11):
        x, y = synth_night(n_epochs=30, seed=k)
        rid = f"SN{k:03d}"
        meta = {"subject": f"HMC:{k:03d}", "night": 1, "cohort": "HMC", "channels": ["F4-M1", "C4-M1"],
                "rule_channel": "C4-M1" if k % 2 else "F4-M1", "rule_tier": "x",
                "prefilter_eeg": ["HP:0.2Hz LP:35.0Hz" if k <= 6 else "HP:0.2Hz LP:128.0Hz"]}
        save_night(tmp_path, "HMC", rid, x, y, meta)
        records[rid] = meta
    (tmp_path / "HMC" / "manifest.json").write_text(json.dumps({"dataset": "hmc", "records": records}))
    split = tmp_path / "hmc_v1.json"
    mk = script("make_external_splits")
    split.write_text(json.dumps(mk.make_split("hmc", records)))
    return tmp_path, split, records


def test_subject_ids():
    assert subject_of_folder("hmc", "SN001") == "HMC:001"
    assert subject_of_folder("isruc", "ISRUC1-017") == "ISRUC1:017"
    assert subject_of_folder("isruc", "ISRUC3-002") == "ISRUC3:002"
    assert subject_of_folder("cap", "nfle12") == "CAP:nfle12"
    assert subject_of_folder("dodh", "095d6e40-5f19-55b6-a0ec-6e0ad3793da0").startswith("DODH:095d6e40")
    with pytest.raises(ValueError):
        subject_of_folder("hmc", "SC4011")


def test_split_is_by_person_stratified_and_complete(ext_store):
    _, split, records = ext_store
    sp = json.loads(split.read_text())
    parts = [set(sp[k]) for k in ("train", "val", "test")]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert set.union(*parts) == {m["subject"] for m in records.values()}
    assert sum(v["test"] for v in sp["strata"].values()) == len(sp["test"]) == 4      # 40% of 6 and of 4
    assert len(sp["strata"]) == 2


def test_holdout_guard(ext_store):
    root, split, _ = ext_store
    sp = json.loads(split.read_text())
    with pytest.raises(PermissionError):
        load_external(root, "hmc", "test", split)
    for s in sp["test"]:                                      # would fail if a test folder were memory-mapped
        (root / "HMC" / f"SN{s.split(':')[1]}" / "x.npy").write_bytes(b"not an array")
    dev = load_external(root, "hmc", "dev", split)
    assert {n.subject for n in dev} == set(sp["train"]) | set(sp["val"])
    assert {n.subject for n in load_external(root, "hmc", "val", split)} == set(sp["val"])
    with pytest.raises(ValueError):
        load_external(root, "hmc", "test", split, open_holdout=True)   # opened only with the flag (corrupt here)


def test_split_script_refuses_to_overwrite(ext_store, tmp_path, monkeypatch):
    root, _, _ = ext_store
    out = tmp_path / "new_split.json"
    mk = script("make_external_splits")
    argv = ["x", "--dataset", "hmc", "--manifest", str(root / "HMC" / "manifest.json"), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    mk.main()
    first = out.read_text()
    with pytest.raises(SystemExit):
        mk.main()
    assert out.read_text() == first


def test_evaluate_external_dev_with_rule_channel(ext_store, tmp_path, monkeypatch):
    import torch

    from mist.models.build import build_model
    root, split, records = ext_store
    small = {"name": "mist_grm", "seq_len": 6, "n_slots": 8, "d_stream": 16, "d_token": 32, "d_epoch": 32,
             "context_layers": 1, "scales": [[1.0, 0.5], [4.0, 2.0]], "streams": ["G", "R"]}
    cfg = {"data": {"channel": "Fpz-Cz", "view": "full", "seq_len": 6}, "model": small,
           "split": {"file": "splits/sleepedf_v1.json", "protocol": "cv", "fold": 0}}
    run = tmp_path / "run"
    run.mkdir()
    torch.save({"model": build_model(small).state_dict(), "cfg": cfg, "epoch": 0}, run / "best.pt")
    paths = tmp_path / "paths.yaml"
    paths.write_text(f"store_root: {root}\n")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ev = script("evaluate")
    monkeypatch.setattr(sys, "argv", ["x", "--run", str(run), "--dataset", "hmc", "--part", "dev", "--paths", str(paths),
                                      "--split-file", str(split), "--cells", "core"])
    ev.main()
    out = json.loads((run / "eval_test_ext-hmc-dev_rule_full.json").read_text())
    sp = json.loads(split.read_text())
    assert out["protocol"] == "ext-hmc-dev" and out["channel"] == "rule"
    assert {m["subject"] for m in out["night_meta"].values()} == set(sp["train"]) | set(sp["val"])
    for rid, m in out["night_meta"].items():
        assert m["channel"] == records[rid]["rule_channel"]
    assert np.isfinite(out["cells"]["real"]["summary"]["mf1"])
    monkeypatch.setattr(sys, "argv", ["x", "--run", str(run), "--dataset", "hmc", "--part", "test", "--paths", str(paths)])
    with pytest.raises(SystemExit):                           # argparse rejects the holdout part
        ev.main()


def test_isruc_v2_split_stratifies_by_group_and_amplitude_and_is_the_default():
    from mist.data.external import default_split_file
    mk = script("make_external_splits")
    recs = {}
    for g, amp, n in (("ISRUC1", "-2490..2490", 20), ("ISRUC1", "-25..25", 10), ("ISRUC3", "-25..25", 10)):
        for i in range(n):
            s = f"{g}:{len(recs):03d}"
            recs[s] = {"subject": s, "phys_range": amp}
    v1, v2 = mk.make_split("isruc", recs), mk.make_split("isruc", recs, version=2)
    assert set(v1["strata"]) == {"ISRUC1", "ISRUC3"} and v1["version"] == 1
    assert set(v2["strata"]) == {"ISRUC1 / normal", "ISRUC1 / low", "ISRUC3 / low"} and v2["version"] == 2
    assert v2["strata"]["ISRUC1 / low"] == {"test": 4, "val": 1, "train": 5}
    assert default_split_file("isruc").name == "isruc_v2.json" and default_split_file("hmc").name == "hmc_v1.json"


def test_isruc_subdomains_named_by_reference_label():
    from mist.data.external import ISRUC_ROWS, isruc_subdomain
    assert isruc_subdomain({"phys_range": "-25..25"}) == "ISRUC-A"
    assert isruc_subdomain({"phys_range": "-2490..2490"}) == isruc_subdomain({"phys_range": "-5013..5013"}) == "ISRUC-M"
    assert ISRUC_ROWS == ("ISRUC-M", "ISRUC-A", "all")
