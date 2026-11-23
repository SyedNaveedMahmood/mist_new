"""Q9 diagnostic (owner, 2026-11-24): is the ISRUC low-amplitude subgroup a pure scale? CPU, label-free, aggregates only.

(a) robust amplitude (median over 30-s epochs of the epoch SD, physical units from the header) of EOG, ECG and chin
    EMG read raw from the .rec files; (b) median log spectrum of the stored rule channel after per-night scaling
    (divide by the night's median epoch SD); (c) header fields are summarised by scripts/dev/isruc_header_summary.py.
Writes reports/isruc_amplitude_check.json.
"""
import json
from pathlib import Path

import numpy as np

from mist.data.edf_header import read_header

ROOT = Path("/home/FA006/Desktop/Dimension/dataset/isruc")
STORE = Path("/home/FA006/Desktop/Dimension/dataset/mist_store/ISRUC")
KIND = {"EOG_Channel": "eog", "EOG2_Channel": "eog", "EKG_Channel": "ecg", "Chin_Channel": "chin",
        "EEG_Channel": "eeg", "EEG2_Channel": "eeg"}


def read_raw(path, hdr, idx):
    ns, nsamp = hdr["ns"], [int(n) for n in hdr["nsamp"]]
    rec = sum(nsamp)
    d = np.fromfile(path, dtype="<i2", offset=256 * (ns + 1))
    d = d[: (len(d) // rec) * rec].reshape(-1, rec)
    o = sum(nsamp[:idx])
    dig = d[:, o:o + nsamp[idx]].reshape(-1).astype(np.float64)
    pmin, pmax, dmin, dmax = (float(hdr[k][idx]) for k in ("pmin", "pmax", "dmin", "dmax"))
    return (dig - dmin) * (pmax - pmin) / (dmax - dmin) + pmin, hdr["fs"][idx]


def robust_sd(x, fs):
    n = int(30 * fs)
    e = x[: (len(x) // n) * n].reshape(-1, n)
    return float(np.median(e.std(1)))


out = {"low": {}, "normal": {}}
spec = {"low": [], "normal": []}
f = np.fft.rfftfreq(3000, 0.01)
band = (f >= 0.5) & (f <= 35)
for mf in sorted(STORE.glob("*/meta.json")):
    m = json.loads(mf.read_text())
    grp = "low" if m["phys_range"] == "-25..25" else "normal"
    folder = "ISRUC-Sleep-I" if m["subject"].startswith("ISRUC1") else "ISRUC-Sleep-III"
    p = ROOT / folder / str(int(m["subject"].split(":")[1])) / m["source"][0]
    hdr = read_header(p)
    amp = {}
    for i, t in enumerate(hdr["transducer"]):
        k = KIND.get(t.strip())
        if k:
            x, fs = read_raw(p, hdr, i)
            amp.setdefault(k, []).append(robust_sd(x, fs))
    for k, v in amp.items():
        out[grp].setdefault(k, []).append(float(np.median(v)))
    x = np.load(mf.parent / "x.npy", mmap_mode="r")[:, m["channels"].index(m["rule_channel"])]
    x = np.asarray(x, dtype=np.float64)
    x = x / np.median(x.std(1))
    P = np.abs(np.fft.rfft(x * np.hanning(3000), axis=1)) ** 2
    spec[grp].append(np.log10(np.median(P, 0)[band]))
res = {"n": {g: len(spec[g]) for g in spec}, "amplitude_median_uv": {}, "ratio_normal_over_low": {}}
for k in ("eeg", "eog", "ecg", "chin"):
    lo, no = np.median(out["low"].get(k, [np.nan])), np.median(out["normal"].get(k, [np.nan]))
    res["amplitude_median_uv"][k] = {"low": float(lo), "normal": float(no),
                                     "low_iqr": [float(np.percentile(out["low"][k], q)) for q in (25, 75)],
                                     "normal_iqr": [float(np.percentile(out["normal"][k], q)) for q in (25, 75)]}
    res["ratio_normal_over_low"][k] = float(no / lo)
S = {g: np.stack(v) for g, v in spec.items()}
med = {g: np.median(S[g], 0) for g in S}
diff = med["low"] - med["normal"]                                   # log10 power, after per-night scaling
spread = np.median(np.abs(S["normal"] - med["normal"]), 0)           # within-group MAD of the normal group
edges = [(0.5, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 25), (25, 35)]
fb = f[band]
res["spectrum_shape"] = {
    "metric": "median log10 power (rule channel, per-night scaled) low minus normal, per band; within-normal MAD",
    "bands": {f"{a}-{b} Hz": {"low_minus_normal": float(diff[(fb >= a) & (fb < b)].mean()),
                               "normal_within_MAD": float(spread[(fb >= a) & (fb < b)].mean())} for a, b in edges}}
Path("reports/isruc_amplitude_check.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
