#!/usr/bin/env python
"""Freeze the subject-level splits from the store manifest. Run once, commit the JSON."""
import argparse
import json
from pathlib import Path

from mist.config import load_paths
from mist.data.splits import make_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="splits/sleepedf_v1.json")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--seed", type=int, default=20261122)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists. Splits are frozen; bump the version name to make a new one.")
    manifest = json.loads((Path(load_paths(a.paths)["store_root"]) / "manifest.json").read_text())
    subjects = {}
    for m in manifest.values():
        subjects.setdefault(m["subject"], {"age": m.get("age"), "sex": m.get("sex")})
    sp = make_splits(subjects, seed=a.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sp, indent=1))
    print({k: (len(v) if isinstance(v, list) else v) for k, v in sp.items() if k != "sets"},
          {k: len(v) for k, v in sp["sets"].items()})


if __name__ == "__main__":
    main()
