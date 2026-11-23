"""ISRUC reader on a synthetic subject folder (<n>.rec is an EDF, scorer files <n>_1.txt / <n>_2.txt)."""
import numpy as np
import pytest

mne = pytest.importorskip("mne")
pytest.importorskip("edfio")

from mist.data import isruc  # noqa: E402


def write_rec(path, labels, n_epochs=6, fs=200, amp_uv=40.0, amps=None, noise=0.05):
    t = np.arange(n_epochs * 30 * fs) / fs
    rng = np.random.default_rng(1)
    amps = amps or {}
    data = np.stack([amps.get(lab, amp_uv) * 1e-6 * (np.sin(2 * np.pi * 10 * t) + noise * rng.standard_normal(t.size))
                     for lab in labels])
    raw = mne.io.RawArray(data, mne.create_info(list(labels), float(fs), "eeg"), verbose="ERROR")
    edf = path.with_suffix(".edf")
    mne.export.export_raw(edf, raw, verbose="ERROR")
    edf.rename(path)                                         # the real files carry .rec


def make_subject(root, group, n, labels, codes, rec=True, **kw):
    d = root / isruc.GROUPS[group] / str(n)
    d.mkdir(parents=True)
    if rec:
        write_rec(d / f"{n}.rec", labels, **kw)
    (d / f"{n}_1.txt").write_text("\n".join(map(str, codes)) + "\n\n")
    (d / f"{n}_2.txt").write_text("0\n" * len(codes))
    return d


def test_read_record_and_find(tmp_path):
    make_subject(tmp_path, 1, 7, ["LOC-A2", "F3-A2", "C3-A2", "F4-A1", "C4-A1", "X1"], [0, 1, 2, 3, 5, 5])
    make_subject(tmp_path, 1, 4, ["F4-A1"], [0] * 6, rec=False)
    make_subject(tmp_path, 3, 2, ["E1-M2", "F3-M2", "C4-M1"], [0, 0, 2, 2, 2, 2, 2, 2])     # 2 extra labels
    found, skipped = isruc.find_records(tmp_path)
    assert [(g, n) for g, n, _, _ in found] == [(1, 7), (3, 2)]
    assert skipped == [(1, 4, "no .rec file on disk (not downloaded)")]
    g, n, rec, txt = found[0]
    x, y, meta = isruc.read_record(rec, txt)
    assert x.shape == (6, 4, 3000) and meta["fs_orig"] == 200.0
    assert meta["channels"] == ["F3-A2", "C3-A2", "F4-A1", "C4-A1"]
    assert y.tolist() == [0, 1, 2, 3, 4, 4]                  # ISRUC 5 = REM -> 4
    assert meta["rule_channel"] == "F4-A1" and meta["rule_tier"] == "frontal-mastoid" and meta["scorer"] == 1
    assert set(meta["units"].values()) == {"µV"}
    assert np.std(x[1:5, 2]) == pytest.approx(40 / np.sqrt(2), rel=0.05)   # microvolt after resampling
    x3, y3, m3 = isruc.read_record(found[1][2], found[1][3])
    assert len(y3) == 6 and m3["label_length"]["action"] == "dropped 2 trailing labels"
    assert m3["rule_channel"] == "F3-M2" and m3["rule_tier"] == "frontal-mastoid"
    assert isruc.record_id(3, 2) == "ISRUC3-002" and isruc.parse_record_id("ISRUC3-002") == (3, 2)


def test_monopolar_leads_are_derived(tmp_path):
    d = make_subject(tmp_path, 1, 40, ["ROC", "A1", "C4", "F4", "A2", "F3"], [0] * 6, noise=0.0,
                     amps={"A1": 10.0, "A2": 0.0, "C4": 40.0, "F4": 30.0, "F3": 40.0})
    x, y, meta = isruc.read_record(d / "40.rec", d / "40_1.txt")
    assert meta["channels"] == ["F3-A2", "F4-A1", "C4-A1"] and meta["derived"] == meta["channels"]
    sd = x[1:5].std(axis=(0, 2))                             # lead minus reference, in phase: 40-0, 30-10, 40-10
    assert sd == pytest.approx(np.array([40, 20, 30]) / np.sqrt(2), rel=0.05)


def test_length_rule_and_codes(tmp_path):
    y, info = isruc.fit_length(np.array([0, 1, 2], np.int8), 5)
    assert y.tolist() == [0, 1, 2, -1, -1] and info["action"] == "2 trailing epochs unscored"
    p = tmp_path / "1_1.txt"
    p.write_text("0\n4\n")
    with pytest.raises(ValueError):
        isruc.read_labels(p)
