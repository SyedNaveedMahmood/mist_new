"""CAP reader on a synthetic EDF + RemLogic-style hypnogram (R&K stages, clock times, grid offset)."""
from datetime import datetime, timezone

import numpy as np
import pytest

mne = pytest.importorskip("mne")
pytest.importorskip("edfio")

from mist.data import cap  # noqa: E402

HEAD = ("RemLogic Event Export\nPatient:\tX\n\nEvents Included:\nSLEEP-S0\n\n"
        "Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation\n")


def write_edf(path, labels, fs=128, seconds=320, start=(22, 10, 0), amp_uv=30.0):
    t = np.arange(seconds * fs) / fs
    data = np.stack([amp_uv * 1e-6 * np.sin(2 * np.pi * 10 * t) for _ in labels])
    raw = mne.io.RawArray(data, mne.create_info(list(labels), float(fs), "eeg"), verbose="ERROR")
    raw.set_meas_date(datetime(2009, 1, 1, *start, tzinfo=timezone.utc))
    mne.export.export_raw(path, raw, verbose="ERROR")


def row(stage, hms, event, dur=30):
    return f"{stage}\tUnknown Position\t{hms}\t{event}\t{dur}\tROC-LOC\n"


def test_read_record_aligns_and_maps_stages(tmp_path):
    write_edf(tmp_path / "rbd3.edf", ["ROC-LOC", "Fp2-F4", "F4-C4", "C4-A1", "EMG1-EMG2"])
    rows = [row("W", "22:10:20", "SLEEP-S0"), row("S1", "22:10:50", "SLEEP-S1"), row("S2", "22:11:20", "SLEEP-S2"),
            row("S2", "22:11:25", "MCAP-A1", 4), row("S3", "22:11:50", "SLEEP-S3"), row("S4", "22:12:20", "SLEEP-S4"),
            row("R", "22:12:50", "SLEEP-REM"), row("MT", "22:13:20", "SLEEP-MT"),
            row("S2", "22.14.20", "SLEEP-S2"),           # one epoch missing before it; dotted time format
            row("S2", "22:15:20", "SLEEP-S2")]           # past the end of the signal
    (tmp_path / "rbd3.txt").write_text(HEAD + "".join(rows))
    (tmp_path / "RECORDS").write_text("rbd3.edf\nrbd9.edf\n")
    found, skipped = cap.find_records(tmp_path)
    assert [r for r, _, _ in found] == ["rbd3"] and skipped == [("rbd9", "edf or txt missing")]
    _, edf, txt = found[0]
    x, y, meta = cap.read_record(edf, txt)
    assert meta["hypnogram"]["grid_origin_s"] == 20 and meta["hypnogram"]["rows_after_signal"] == 1
    assert x.shape == (10, 3, 3000) and meta["fs_orig"] == 128.0
    assert y.tolist() == [0, 1, 2, 3, 3, 4, -1, -1, 2, -1]
    assert meta["channels"] == ["Fp2-F4", "F4-C4", "C4-A1"] and meta["group"] == "rbd"
    assert meta["rule_channel"] == "F4-C4" and meta["rule_tier"] == "frontal-central"
    assert np.std(x[2:8, 1]) == pytest.approx(30 / np.sqrt(2), rel=0.05)     # uV preserved through 128 -> 100 Hz


def test_midnight_and_refusals(tmp_path):
    p = tmp_path / "n1.txt"
    p.write_text(HEAD + row("W", "23:59:30", "SLEEP-S0") + row("S1", "0:00:00", "SLEEP-S1") + "S2\tUnknown\t00:0")
    rows, notes = cap.parse_hypnogram(p)
    assert [t for t, _ in rows] == [86370, 86400] and notes["truncated_rows"] == 1
    y, origin, _ = cap.align(rows, start_s=86340, n_seconds=120)
    assert origin == 0 and y.tolist() == [-1, 0, 1, -1]
    with pytest.raises(ValueError, match="before the EDF start"):
        cap.align([(36000 + 30 * k, 2) for k in range(20)], start_s=36000 + 30 * 15, n_seconds=600)
    with pytest.raises(ValueError, match="grid"):
        cap.align([(100, 0), (145, 0)], start_s=0, n_seconds=600)


def test_emg_band_channels_and_groups():
    hdr = {"label": ["F4-C4", "F8-T4", "C4", "A1", "ROC-LOC"],
           "prefilter": ["LP:30.00Hz HP:0.30Hz", "LP:100.00Hz HP:10.00Hz", "", "", ""]}
    plan, excluded = cap.eeg_plan(hdr)
    assert [c for c, _ in plan] == ["F4-C4", "C4-A1"] and excluded == [("F8-T4", "hardware high-pass 10.0 Hz")]
    assert [cap.group_of(r) for r in ("n3", "nfle12", "narco1", "brux2", "sdb4")] == ["n", "nfle", "narco", "brux", "sdb"]
