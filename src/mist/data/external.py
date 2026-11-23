"""External datasets in the canonical store: subject ids and the guarded holdout loader (phase 2).

    <store_root>/<DATASET>/<record_id>/{x.npy, y.npy, meta.json}
    <store_root>/<DATASET>/manifest.json          (participant rows; stays outside git)

Split files splits/<dataset>_v1.json hold train / val / test lists of
dataset-qualified subject ids (made once by scripts/make_external_splits.py).
The test part is the external holdout: load_external refuses it unless
open_holdout=True, and folders of test people are selected out by folder name
before anything is opened, so they are not even memory-mapped.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .store import load_night

DATASETS = {"hmc": "HMC", "dodh": "DODH", "isruc": "ISRUC", "cap": "CAP"}
PARTS = ("dev", "train", "val", "test")
SPLIT_DIR = Path(__file__).resolve().parents[3] / "splits"


# Split version in use per dataset. isruc v2 (owner, 2026-11-24, Q9): stratified by group and by amplitude
# subgroup; splits/isruc_v1.json stays on disk, unused.
SPLIT_VERSION = {"isruc": 2}


def default_split_file(dataset: str) -> Path:
    return SPLIT_DIR / f"{dataset}_v{SPLIT_VERSION.get(dataset, 1)}.json"


def isruc_amplitude_subgroup(meta: dict) -> str:
    """'low' for the 26 recordings whose EEG header range is +/-25 uV (D-018, Q9), else 'normal'."""
    return "low" if meta.get("phys_range") == "-25..25" else "normal"


def subject_of_folder(dataset: str, name: str) -> str:
    """Record folder name -> dataset-qualified subject id. One recording per person in all four datasets."""
    if dataset == "hmc":
        m = re.fullmatch(r"SN(\d{3})", name)
        if m:
            return f"HMC:{m.group(1)}"
    elif dataset == "isruc":
        m = re.fullmatch(r"ISRUC([13])-(\d{3})", name)
        if m:
            return f"ISRUC{m.group(1)}:{m.group(2)}"
    elif dataset == "dodh":
        if re.fullmatch(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", name):
            return f"DODH:{name}"
    elif dataset == "cap":
        if re.fullmatch(r"(nfle|narco|brux|plm|rbd|ins|sdb|n)\d+", name):
            return f"CAP:{name}"
    raise ValueError(f"{name!r} is not a {dataset} record folder")


def load_split(split_file: Path) -> dict:
    sp = json.loads(Path(split_file).read_text())
    parts = [set(sp[k]) for k in ("train", "val", "test")]
    if parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]:
        raise ValueError(f"{split_file}: a subject is in more than one part")
    return sp


def subjects_of_part(sp: dict, part: str) -> set:
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")
    return set(sp["train"]) | set(sp["val"]) if part == "dev" else set(sp[part])


def load_external(root: Path, dataset: str, part: str, split_file: Path | None = None, *,
                  open_holdout: bool = False) -> list:
    """Nights of one external dataset for part dev (= train + val), train, val or test."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; one of {sorted(DATASETS)}")
    if part == "test" and not open_holdout:
        raise PermissionError("the external test part is a holdout; it is not loaded without open_holdout=True")
    sp = load_split(split_file or default_split_file(dataset))
    if sp.get("dataset") not in (None, dataset):
        raise ValueError(f"split file is for {sp.get('dataset')}, not {dataset}")
    want = subjects_of_part(sp, part)
    held = set() if open_holdout else set(sp["test"])
    base = Path(root) / DATASETS[dataset]
    if not base.is_dir():
        raise FileNotFoundError(f"{base} not found; run scripts/prepare_external.py --dataset {dataset}")
    dirs = [p for p in sorted(base.iterdir()) if p.is_dir()]
    keep = [p for p in dirs if subject_of_folder(dataset, p.name) in want - held]
    nights = [load_night(p) for p in keep]
    for n, p in zip(nights, keep):
        if n.subject != subject_of_folder(dataset, p.name):
            raise AssertionError(f"{p}: meta subject {n.subject} does not match its folder")
    return nights
