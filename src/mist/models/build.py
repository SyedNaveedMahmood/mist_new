"""Model factory. Every model maps (x [B, L, T], stats) -> (logits [B, L, 5], aux)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .mist_grm import MistConfig, MistGRM


class EpochwiseBaseline(nn.Module):
    """Wraps a single-epoch network (e.g. AttnSleep). The input is divided by the
    night's robust standard deviation, i.e. ordinary per-recording scaling."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x, stats, **_):
        b, l, t = x.shape
        logits = self.net((x / stats["night"]["scale"].view(b, 1, 1)).reshape(b * l, 1, t))
        return logits.view(b, l, -1), {"losses": {}, "assign": []}

    def aux_loss(self, aux):
        return 0.0


def build_model(cfg: dict) -> nn.Module:
    cfg = dict(cfg)
    name = cfg.pop("name", "mist_grm")
    if name == "mist_grm":
        cfg["streams"] = tuple(cfg.get("streams", ("G", "R", "M")))
        cfg["scales"] = tuple(tuple(s) for s in cfg.get("scales", ((1.0, 0.5), (4.0, 2.0))))
        if cfg.get("band_scales") is not None:
            cfg["band_scales"] = tuple(float(v) for v in cfg["band_scales"])
        if cfg.get("critic_stages") is not None:
            cfg["critic_stages"] = tuple(int(v) for v in cfg["critic_stages"])
        return MistGRM(MistConfig(**cfg))
    if name == "attnsleep":
        from .attnsleep import AttnSleepBaseline, init_attnsleep_weights
        net = AttnSleepBaseline()
        net.apply(init_attnsleep_weights)
        return EpochwiseBaseline(net)
    if name == "sleepband":                 # T14 phase-2 baseline (D-023); not used in phase 1
        from .sleepband import SleepBand
        return SleepBand(**cfg)
    raise ValueError(f"unknown model {name!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
