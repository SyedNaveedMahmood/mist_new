#!/usr/bin/env python
"""Train one run.

    python scripts/train.py --arm grm --name grm_cv0_s0 split.fold=0 train.seed=0
"""
import argparse
from pathlib import Path

from mist.config import load_config, load_paths
from mist.engine.loop import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="file name in configs/arms without .yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(["configs/base.yaml", f"configs/arms/{a.arm}.yaml"], a.overrides)
    cfg["arm"] = a.arm
    paths = load_paths(a.paths)
    print(train(cfg, paths, Path(paths.get("runs_root", "runs")) / a.name))


if __name__ == "__main__":
    main()
