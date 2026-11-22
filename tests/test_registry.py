import pytest
import yaml

from mist.registry import append_run

BASE = "# header comment\nruns: []\n\ndatasets:\n  x: {licence: L, status: VERIFIED}\n"


def entry(i, status="VERIFIED"):
    return {"id": i, "date": "2026-11-22", "commit": "abc", "status": status, "note": "n", "evals": ["a.json"]}


def test_append_keeps_order_comments_and_datasets(tmp_path):
    p = tmp_path / "r.yaml"; p.write_text(BASE)
    append_run(p, entry("a")); append_run(p, entry("b"))
    d = yaml.safe_load(p.read_text())
    assert [r["id"] for r in d["runs"]] == ["a", "b"] and d["datasets"]["x"]["licence"] == "L"
    assert p.read_text().startswith("# header comment")


def test_rejects_duplicates_and_bad_status(tmp_path):
    p = tmp_path / "r.yaml"; p.write_text(BASE)
    append_run(p, entry("a"))
    with pytest.raises(ValueError):
        append_run(p, entry("a"))
    with pytest.raises(ValueError):
        append_run(p, entry("c", status="GREAT"))
