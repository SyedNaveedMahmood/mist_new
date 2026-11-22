import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def eeg_like(n=8, seed=0, T=3000):
    """1/f noise limited to 0.3-45 Hz, like a preprocessed EEG epoch (torch tensor)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    f = torch.fft.rfftfreq(T, 0.01)
    amp = torch.where((f >= 0.3) & (f <= 45), 1.0 / f.clamp(min=0.3), torch.zeros_like(f))
    X = amp * torch.polar(torch.ones(n, f.numel()), torch.rand(n, f.numel(), generator=g) * 6.2832)
    x = torch.fft.irfft(X, n=T)
    return 20.0 * x / x.std()


def synth_night(n_epochs=60, seed=0, n_ch=2):
    """Pink-ish noise with stage-dependent bursts; labels cycle through stages."""
    rng = np.random.default_rng(seed)
    t = np.arange(3000) / 100.0
    y = (np.arange(n_epochs) // 6) % 5
    x = np.zeros((n_epochs, n_ch, 3000), dtype=np.float32)
    for i in range(n_epochs):
        for c in range(n_ch):
            w = np.cumsum(rng.standard_normal(3000)) * 0.5 + rng.standard_normal(3000) * 3
            w -= w.mean()
            f0 = [10.0, 6.0, 13.0, 1.0, 5.0][y[i]]
            amp = [8, 4, 6, 30, 4][y[i]]
            w += amp * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 6.28)) * (np.sin(2 * np.pi * 0.2 * t) > 0)
            x[i, c] = w
    return x, y.astype(np.int8)


@pytest.fixture
def store(tmp_path):
    from mist.data.store import save_night
    subs = {}
    for k in range(12):
        x, y = synth_night(seed=k)
        s = f"SC:{20 + k:02d}"
        save_night(tmp_path, "SC", f"SC4{20 + k:02d}1", x, y,
                   {"subject": s, "night": 1, "cohort": "SC", "channels": ["Fpz-Cz", "Pz-Oz"], "age": 50 + 3 * k})
        subs[s] = {"age": 50 + 3 * k}
    return tmp_path, subs
