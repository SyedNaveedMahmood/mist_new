"""Writes tests/fixtures/cascade_v1_reference.pt from the cascade code at commit f96f7e0 (before revision 3).

Small synthetic model and input only (no EEG). tests/test_cascade2.py checks that the default
configuration still produces these outputs after the cascade2 changes.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from test_cascade import SMALL          # noqa: E402
from test_model import stats            # noqa: E402

from mist.models.build import build_model

torch.manual_seed(0)
cfg = {**SMALL, "critic_weight": 0.3, "allpass_rad": 0.5}
m = build_model(cfg)
m.train()
x = torch.randn(2, 6, 3000, generator=torch.Generator().manual_seed(1))
y = torch.randint(0, 5, (2, 6), generator=torch.Generator().manual_seed(2))
for _ in range(3):                                   # move gate_mean off its initial value
    m(x, stats(2), y=y, gen=torch.Generator().manual_seed(3))
m.eval()
with torch.no_grad():
    out = {k: m(x, stats(2), **kw)[0] for k, kw in {
        "real": {}, "knock_R": {"knock": ("R",)}, "destroy_sigma": {"destroy_bands": ("sigma",), "gen": torch.Generator().manual_seed(4)},
        "destroy_all": {"destroy_prob": 1.0, "gen": torch.Generator().manual_seed(5)}}.items()}
torch.save({"cfg": cfg, "state": m.state_dict(), "x": x, "out": out}, "tests/fixtures/cascade_v1_reference.pt")
print({k: v.shape for k, v in out.items()})
