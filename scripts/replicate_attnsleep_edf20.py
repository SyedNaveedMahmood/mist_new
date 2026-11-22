#!/usr/bin/env python
"""REPLICATION CHECK ONLY: AttnSleep, legacy Sleep-EDF-20 NPZ, published protocol.

    python scripts/replicate_attnsleep_edf20.py [--folds 0-19] [--epochs 100]

20-fold leave-one-subject-out over SC persons 00-19 (both nights of a person in
the same fold). Writes runs/replication_edf20/fold_XX.npz (test labels and
predictions per fold: final epoch, and the epoch with the best test accuracy as
the published repo does) and summary.json with pooled metrics for both.
Never mix these numbers with project results (see src/mist/replication.py).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from mist.config import load_paths
from mist.data.store import load_legacy_npz
from mist.models.attnsleep import AttnSleepBaseline, init_attnsleep_weights
from mist.replication import calc_class_weight, lr_at



def load_subjects(folder: Path) -> dict:
    subs = {}
    for f in sorted(folder.glob("SC4*.npz")):
        n = load_legacy_npz(f)                       # refuses lockbox files (none among 00-19)
        keep = n.y >= 0
        subs.setdefault(int(n.subject.split(":")[1]), []).append((n.x[keep, 0], n.y[keep].astype(np.int64)))
    assert sorted(subs) == list(range(20)), sorted(subs)
    return {s: (np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v])) for s, v in subs.items()}


@torch.no_grad()
def predict(model, x, device, bs=512):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        out.append(model(torch.from_numpy(x[i:i + bs]).float().unsqueeze(1).to(device)).argmax(-1).cpu().numpy())
    return np.concatenate(out)


def metrics(y, p) -> dict:
    return {"acc": float((y == p).mean()), "mf1": float(f1_score(y, p, average="macro", labels=range(5))),
            "kappa": float(cohen_kappa_score(y, p)),
            "f1": [float(v) for v in f1_score(y, p, average=None, labels=range(5))],
            "cm": confusion_matrix(y, p, labels=range(5)).tolist(), "n": int(len(y))}


def run_fold(subs, test_s, epochs, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    xtr = np.concatenate([subs[s][0] for s in subs if s != test_s]).astype(np.float32)
    ytr = np.concatenate([subs[s][1] for s in subs if s != test_s])
    xte, yte = subs[test_s][0].astype(np.float32), subs[test_s][1]
    w = calc_class_weight(np.bincount(ytr, minlength=5))       # training subjects only (decision D-002)
    model = AttnSleepBaseline().to(device); model.apply(init_attnsleep_weights)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3,
                           amsgrad=True)
    xt, yt = torch.from_numpy(xtr), torch.from_numpy(ytr)
    wt = torch.tensor(w, dtype=torch.float32, device=device)
    best_acc, best_pred, t0 = -1.0, None, time.time()
    for ep in range(epochs):
        for g in opt.param_groups:
            g["lr"] = lr_at(ep)
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            logits = model(xt[idx].unsqueeze(1).to(device))
            loss = F.cross_entropy(logits, yt[idx].to(device), weight=wt)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        p = predict(model, xte, device)
        acc = float((p == yte).mean())
        if acc > best_acc:                                      # published repo's selection (on test)
            best_acc, best_pred = acc, p
    return yte, p, best_pred, w, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0-19")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)           # repo default
    ap.add_argument("--out", default="runs/replication_edf20")
    a = ap.parse_args()
    lo, hi = map(int, a.folds.split("-"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subs = load_subjects(Path(load_paths()["legacy"]["edf20_fpzcz_npz"]))
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    for s in range(lo, hi + 1):
        f = OUT / f"fold_{s:02d}.npz"
        if f.exists():
            continue
        y, p_final, p_best, w, sec = run_fold(subs, s, a.epochs, device, a.seed)
        np.savez(f, y=y, p_final=p_final, p_best_test=p_best, weights=np.array(w), sec=sec)
        print(f"fold {s:02d} final acc {np.mean(p_final == y):.3f} best-test acc {np.mean(p_best == y):.3f} "
              f"{sec:.0f}s", flush=True)
    folds = sorted(OUT.glob("fold_*.npz"))
    if len(folds) == 20:
        d = [np.load(f) for f in folds]
        y = np.concatenate([z["y"] for z in d])
        summary = {"label": "REPLICATION CHECK ONLY - legacy Sleep-EDF-20 NPZ, not a project result",
                   "protocol": "20-fold LOSO, per-epoch AttnSleep, published recipe, seed %d" % a.seed,
                   "final_epoch": metrics(y, np.concatenate([z["p_final"] for z in d])),
                   "best_test_epoch_OPTIMISTIC": metrics(y, np.concatenate([z["p_best_test"] for z in d])),
                   "published_table_v": {"acc": 0.844, "mf1": 0.781, "kappa": 0.79,
                                         "f1": [0.897, 0.426, 0.888, 0.902, 0.790],
                                         "source": "Eldele et al., IEEE TNSRE 29 (2021) 809-818, Table V"},
                   "train_sec_per_fold": float(np.mean([float(z["sec"]) for z in d]))}
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps({k: {m: v[m] for m in ("acc", "mf1", "kappa")} for k, v in summary.items()
                          if isinstance(v, dict) and "acc" in v}, indent=1))


if __name__ == "__main__":
    main()
