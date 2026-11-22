#!/usr/bin/env python
"""Evaluate a trained run on its protocol's test subjects.

Reports every (condition, intervention) cell needed by E0 and E3:
  conditions     real, global_phase, local_phase, flatten_tilt   (input surrogates)
  interventions  none, knock G / R / M, destroy M (random phase inside the model);
                 cascade only: destroy_<band> for delta..beta and broad (ARCHITECTURE 8.4)

    python scripts/evaluate.py --run runs/grm_cv0_s0 --channel Pz-Oz
"""
import argparse
import json
from pathlib import Path

import torch

from mist.config import load_paths
from mist.data import splits as S
from mist.data.store import load_cohort
from mist.engine.loop import evaluate, load_checkpoint, make_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--channel", default=None, help="test channel; default = training channel")
    ap.add_argument("--protocol", default=None, help="override the test protocol, e.g. sc2st")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--tag", default="test")
    ap.add_argument("--view", default=None, help="evaluation window; default = training view (std). lf = label-free")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(a.run, device)
    protocol = a.protocol or cfg["split"]["protocol"]
    if protocol == "lockbox" and not cfg["split"].get("i_am_running_the_final_confirmation"):
        raise SystemExit("the lockbox is opened once, by task T12 only")
    roles = S.resolve(S.load(cfg["split"]["file"]), protocol, cfg["split"].get("fold", 0))
    nights = load_cohort(load_paths(a.paths)["store_root"], roles["test_cohort"], set(roles["test"]),
                         split_file=cfg["split"]["file"],
                         open_lockbox=protocol == "lockbox" and bool(cfg["split"].get("i_am_running_the_final_confirmation")))
    channel = a.channel or cfg["data"]["channel"]
    if a.view:
        cfg["data"]["view"] = a.view          # window only; the night reference is recomputed inside it
    ds = make_dataset(nights, cfg, channel, train=False)
    streams = cfg["model"].get("streams", []) if cfg["model"].get("name", "mist_grm") == "mist_grm" else []
    cells = [("real", (), None, ()), ("global_phase", (), None, ()), ("local_phase", (), None, ()),
             ("flatten_tilt", (), None, ())]
    cells += [("real", (s,), None, ()) for s in streams if s in "GRM" and len(streams) > 1]
    if "M" in streams:
        cells.append(("real", (), 1.0, ()))
    if cfg["model"].get("topology") == "cascade":
        from mist.models.cascade import BAND_NAMES, CHANNEL_SETS
        has_broad = "broad" in CHANNEL_SETS[cfg["model"].get("cascade_channels", "all")]
        cells += [("real", (), None, (b,)) for b in BAND_NAMES + (("broad",) if has_broad else ())]
    out = {"protocol": protocol, "channel": channel, "n_nights": len(nights), "view": cfg["data"].get("view", "std"),
           "night_meta": {n.record_id: {"subject": n.subject, "condition": n.meta.get("condition")} for n in nights},
           "cells": {}}
    for cond, knock, destroy, bands in cells:
        key = (cond + ("" if not knock else f"|knock_{''.join(knock)}") + ("" if destroy is None else "|destroy_M")
               + "".join(f"|destroy_{b}" for b in bands))
        summ, cms, ncms = evaluate(model, ds, device, return_night_cms=True, condition=cond, knock=knock,
                                   destroy_prob=destroy, destroy_bands=bands)
        out["cells"][key] = {"summary": summ, "subject_cm": {k: v.tolist() for k, v in cms.items()},
                             "night_cm": {k: v.tolist() for k, v in ncms.items()}}
        print(key, {k: round(v, 4) for k, v in summ.items() if k in ("mf1", "acc", "kappa", "f1_N1")})
    view = cfg["data"].get("view", "std")
    name = f"eval_{a.tag}_{protocol}_{channel}" + ("" if view == "std" else f"_{view}") + ".json"
    (Path(a.run) / name).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
