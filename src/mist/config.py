"""YAML configs: base file, optional arm files, then key=value overrides."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(files: list, overrides: list | None = None) -> dict:
    cfg = {}
    for f in files:
        cfg = deep_merge(cfg, yaml.safe_load(Path(f).read_text()) or {})
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def load_paths(path: str = "configs/paths.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: copy configs/paths.example.yaml and edit it")
    return yaml.safe_load(p.read_text())
