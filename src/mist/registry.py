"""Append-only writer for experiments/registry.yaml.

Entries are appended as text under the `runs:` key so earlier entries (and the
comments at the top of the file) are never rewritten. An id may appear once.
"""
from __future__ import annotations

from pathlib import Path

import yaml

STATUSES = {"VERIFIED", "CONDITIONAL", "BLOCKED", "UNKNOWN", "NOT TESTED"}
REQUIRED = ("id", "date", "commit", "status", "note")


def append_run(path: Path, entry: dict) -> None:
    path = Path(path)
    missing = [k for k in REQUIRED if k not in entry]
    if missing:
        raise ValueError(f"registry entry lacks {missing}")
    if entry["status"] not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}")
    text = path.read_text()
    data = yaml.safe_load(text) or {}
    runs = data.get("runs") or []
    if any(r.get("id") == entry["id"] for r in runs):
        raise ValueError(f"run id {entry['id']!r} already in the registry")
    if "runs: []" in text:                      # first entry: open the block list in place
        text = text.replace("runs: []", "runs:", 1)
    block = yaml.safe_dump([entry], sort_keys=False, default_flow_style=None, width=200)
    block = "".join("  " + line if line.strip() else line for line in block.splitlines(True))
    head, sep, tail = text.partition("\ndatasets:")
    path.write_text(head.rstrip("\n") + "\n" + block + (sep + tail if sep else ""))
    after = yaml.safe_load(path.read_text())
    assert [r["id"] for r in after["runs"]][-1] == entry["id"] and after.get("datasets") == data.get("datasets")
