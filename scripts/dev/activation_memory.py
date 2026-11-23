"""CPU estimate of activation memory saved for backward, per arm (no GPU). Bytes of distinct tensors saved
by autograd during one training forward at the given batch, as a proxy for GPU peak memory."""
import sys

import torch
import yaml

sys.path.insert(0, "tests")
from mist.models.build import build_model  # noqa: E402


def stats(b):
    return {"night": {"envelope": torch.ones(b, 1501), "scale": torch.ones(b)},
            "s0": {"mag_med": torch.zeros(b, 35), "mag_iqr": torch.ones(b, 35)},
            "s1": {"mag_med": torch.zeros(b, 139), "mag_iqr": torch.ones(b, 139)}}


def measure(arm, batch, extra=None):
    base = yaml.safe_load(open("configs/base.yaml"))["model"]
    cfg = {**base, **yaml.safe_load(open(f"configs/arms/{arm}.yaml"))["model"], **(extra or {})}
    torch.manual_seed(0)
    m = build_model(cfg).train()
    seen, total = set(), [0]

    def pack(t):
        key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        if key not in seen:
            seen.add(key); total[0] += key[1]
        return t

    x = torch.randn(batch, cfg.get("seq_len", 20), 3000)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        logits, aux = m(x, stats(batch), y=torch.randint(0, 5, logits_shape(batch, cfg)))
        loss = logits.mean() + m.aux_loss(aux)
    return total[0] / 2 ** 30, sum(p.numel() for p in m.parameters())


def logits_shape(b, cfg):
    return (b, cfg.get("seq_len", 20))


if __name__ == "__main__":
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    for arm, extra in [("grm", None), ("cascade", None), ("cascade", {"encoder_checkpoint": False}),
                       ("cascade2", None), ("cascade2", {"encoder_checkpoint": False})]:
        gb, n = measure(arm, batch, extra)
        print(f"{arm:10s} {str(extra):32s} params {n:8d}  saved {gb:6.2f} GB at batch {batch} -> x{32 // batch} = {gb * 32 / batch:6.2f} GB")
