"""Training and evaluation loops shared by every arm and baseline."""
from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .. import metrics
from ..data import splits as S
from ..data.dataset import SequenceDataset, split_stats
from ..data.store import load_cohort
from ..data.surrogates import CONDITIONS
from ..models.build import build_model, count_parameters


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def make_dataset(nights, cfg, channel, train):
    d, m = cfg["data"], cfg["model"]
    scales = m.get("scales", [[1.0, 0.5], [4.0, 2.0]])
    return SequenceDataset(nights, channel, scales, view=d.get("view", "std"), seq_len=d.get("seq_len", 20),
                           train=train, f_lo=m.get("f_lo", 0.5), f_hi=m.get("f_hi", 35.0),
                           windows_per_night=d.get("windows_per_night", 64))


@torch.no_grad()
def predict(model, ds, device, condition="real", knock=(), destroy_prob=None, batch_size=32, seed=0):
    """Return true labels, predictions and subject ids for every scored epoch of `ds`."""
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ys, ps, ss = [], [], []
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
        x = CONDITIONS[condition](batch["x"].float(), gen).to(device)
        logits, _ = model(x, split_stats(batch, device), knock=knock, destroy_prob=destroy_prob)
        keep = batch["keep"].numpy()
        ys.append(batch["y"].numpy()[keep])
        ps.append(logits.argmax(-1).cpu().numpy()[keep])
        subj = np.array([ds.views[int(i)].night.subject for i in batch["night"]])
        ss.append(np.repeat(subj[:, None], keep.shape[1], 1)[keep])
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ss)


def evaluate(model, ds, device, **kw) -> tuple:
    y, p, s = predict(model, ds, device, **kw)
    cms = metrics.per_subject(y, p, s)
    return metrics.summarize(cms), cms


def train(cfg: dict, paths: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t, d = cfg["train"], cfg["data"]
    seed_all(t.get("seed", 0))
    device = torch.device(t.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    sp = S.load(cfg["split"]["file"])
    roles = S.resolve(sp, cfg["split"]["protocol"], cfg["split"].get("fold", 0))
    if cfg["split"]["protocol"] == "lockbox" and not cfg["split"].get("i_am_running_the_final_confirmation"):
        raise RuntimeError("the lockbox is opened once, by task T12 only")
    nights = load_cohort(paths["store_root"], d.get("cohort", "SC"), set(roles["train"]) | set(roles["val"]),
                         split_file=cfg["split"]["file"])
    tr = make_dataset([n for n in nights if n.subject in set(roles["train"])], cfg, d["channel"], True)
    va = make_dataset([n for n in nights if n.subject in set(roles["val"])], cfg, d["channel"], False)

    model = build_model(cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t.get("lr", 1e-3), weight_decay=t.get("weight_decay", 1e-2))
    epochs, bs = t.get("epochs", 40), t.get("batch_size", 32)
    loader = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True, num_workers=t.get("num_workers", 0))
    total = epochs * max(len(loader), 1)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i, total) / total)))
    weight = None
    if t.get("class_weight") == "sqrt_inv":
        c = tr.class_counts().astype(float)
        w = 1.0 / np.sqrt(np.maximum(c, 1))
        weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    n1_weight = t.get("n1_weight")
    if n1_weight:
        weight = torch.ones(5, device=device) if weight is None else weight
        weight[1] = weight[1] * float(n1_weight)

    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    log = (out_dir / "log.jsonl").open("a")
    # no early stopping: the cosine schedule runs to zero and the best
    # source-validation checkpoint is kept (stopping early left the LR undecayed)
    if t.get("patience") is not None:
        raise ValueError("train.patience is no longer supported: every run trains the full schedule")
    best, best_epoch = -1.0, -1
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.time()
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for batch in loader:
            x, y = batch["x"].float().to(device), batch["y"].to(device)
            logits, aux = model(x, split_stats(batch, device))
            loss = F.cross_entropy(logits.reshape(-1, 5), y.reshape(-1), weight=weight, ignore_index=-1,
                                   label_smoothing=t.get("label_smoothing", 0.0))
            loss = loss + model.aux_loss(aux)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("grad_clip", 1.0))
            opt.step(); sched.step()
            tot += float(loss.detach()); n += 1
        val, _ = evaluate(model, va, device, batch_size=bs)
        rec = {"epoch": ep, "loss": tot / max(n, 1), "val_mf1": val["mf1"], "val_acc": val["acc"],
               "sec": round(time.time() - t0, 1)}
        if cuda:
            rec["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated(device) / 2**20)
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(rec)
        if val["mf1"] > best:
            best, best_epoch = val["mf1"], ep
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep}, out_dir / "best.pt")
    summary = {"best_val_mf1": best, "best_epoch": best_epoch, "params": count_parameters(model),
               "epochs_run": ep + 1, "wall_sec": round(time.time() - t_start, 1),
               "gpu_peak_mb": round(torch.cuda.max_memory_allocated(device) / 2**20) if cuda else None,
               "commit": git_commit(), "roles": {k: v for k, v in roles.items()}}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def load_checkpoint(run_dir: Path, device):
    ck = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=False)
    model = build_model(ck["cfg"]["model"]).to(device)
    model.load_state_dict(ck["model"])
    return model, ck["cfg"]
