"""Literature replication of AttnSleep on the legacy Sleep-EDF-20 NPZ folder.

REPLICATION CHECK ONLY. It reproduces the published protocol (Eldele et al.,
TNSRE 2021, doi:10.1109/TNSRE.2021.3076234; repo emadeldeen24/AttnSleep) to test
whether the ported baseline code can reach the published figures. Its numbers are
never mixed with project results; the legacy NPZ files are label-trimmed and
subjects 00-19 only.

Published recipe (paper Sec. III-C and repo config.json): per-epoch input, raw
signal without normalisation, 100 epochs, batch 128, Adam (amsgrad, betas 0.9 /
0.999, eps 1e-8, weight decay 1e-3), lr 1e-3 then 1e-4 from epoch 10, weighted
cross-entropy with the class-aware weights below. 20-fold leave-one-subject-out,
metrics on predictions pooled over folds.
"""
from __future__ import annotations

import math

import numpy as np

# mu per class [W, N1, N2, N3, REM] / K, repo utils/util.py ("for Sleep-EDF-20 only")
MU = (1.5, 2.0, 1.5, 1.0, 1.5)


def calc_class_weight(counts) -> list:
    """Class-aware weights, paper Eq. 12, ported from repo utils/util.py calc_class_weight."""
    counts = np.asarray(counts, dtype=float)
    total, k = counts.sum(), len(counts)
    out = []
    for c, m in zip(counts, MU):
        mu = m / k
        score = math.log(mu * total / c)
        out.append(round(max(score, 1.0) * mu, 2))
    return out


def lr_at(epoch: int) -> float:
    """Repo trainer.py: 1e-3, reduced to 1e-4 from epoch 10 (0-based) on."""
    return 1e-3 if epoch < 10 else 1e-4
