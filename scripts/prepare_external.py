#!/usr/bin/env python
"""Raw external dataset -> canonical store + aggregate audit (phase 2, T13).

    CUDA_VISIBLE_DEVICES= python scripts/prepare_external.py --dataset hmc|dodh|isruc|cap [--workers 4]

Writes <store_root>/<DATASET>/<record_id>/{x.npy, y.npy, meta.json},
<store_root>/<DATASET>/manifest.json (participant rows, outside git) and
reports/external_audit_<dataset>.md (aggregates only). Source folders are read
only. Records already in the store are not rewritten unless --overwrite.
Stops before writing when free disk would fall under --min-free-gb.
"""
import argparse
import json
import shutil
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from mist.config import load_paths
from mist.data.external import DATASETS, subject_of_folder
from mist.data.store import save_night, stage_counts

DEFAULT_SRC = {
    "hmc": "/home/FA006/Desktop/Dimension/dataset/hmc/hmc-sleep-staging/1.1/recordings",
    "dodh": "/home/FA006/Desktop/Dimension/dataset/dreem_dod/dodh",
    "isruc": "/home/FA006/Desktop/Dimension/dataset/isruc",
    "cap": "/home/FA006/Desktop/Dimension/dataset/cap/physionet.org/files/capslpdb/1.0.0",
}
STAGES = ("W", "N1", "N2", "N3", "REM")


def discover(dataset: str, src: Path) -> tuple:
    """-> ([(record_id, reader args)], [(record_id or None, reason)])."""
    if dataset == "hmc":
        from mist.data import hmc
        return [(p.stem, (str(p), str(h))) for p, h in hmc.find_records(src)], []
    if dataset == "dodh":
        from mist.data import dodh
        return [(p.stem, (str(p),)) for p in dodh.find_records(src)], []
    if dataset == "isruc":
        from mist.data import isruc
        found, skipped = isruc.find_records(src)
        return ([(isruc.record_id(g, n), (str(r), str(t))) for g, n, r, t in found],
                [(f"group {'I' if g == 1 else 'III'} subject {n}", why) for g, n, why in skipped])
    if dataset == "cap":
        from mist.data import cap
        found, skipped = cap.find_records(src)
        return [(r, (str(e), str(t))) for r, e, t in found], [(r, why) for r, why in skipped]
    raise ValueError(dataset)


def read(dataset: str, args: tuple):
    from mist.data import cap, dodh, hmc, isruc
    return {"hmc": hmc, "dodh": dodh, "isruc": isruc, "cap": cap}[dataset].read_record(*map(Path, args))


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def work(job):
    dataset, rid, args, root, overwrite, min_free = job
    d = Path(root) / DATASETS[dataset] / rid
    if (d / "meta.json").exists() and not overwrite:
        meta = json.loads((d / "meta.json").read_text())
        return rid, "existing", meta, None
    try:
        x, y, meta = read(dataset, args)
    except Exception as e:                          # recorded as a skip with its reason
        return rid, "error", None, f"{type(e).__name__}: {e}".splitlines()[0][:200]
    if meta.get("rule_channel") is None:
        return rid, "error", None, "no channel matches the channel rule"
    need = x.nbytes / 1e9
    if free_gb(root) - need < min_free:
        return rid, "disk", None, f"free disk would fall below {min_free} GB"
    rc = meta["channels"].index(meta["rule_channel"])
    meta.update(subject=subject_of_folder(dataset, rid), night=1, cohort=DATASETS[dataset],
                source=[Path(a).name for a in args], n_epochs=int(len(y)), counts=stage_counts(y),
                rule_median_epoch_sd_uv=float(np.median(x[:, rc].std(-1))))
    save_night(root, DATASETS[dataset], rid, x, y, meta)
    return rid, "written", meta, None


def pct(v, qs=(5, 25, 50, 75, 95)):
    return ", ".join(f"p{q} {np.percentile(v, q):.1f}" for q in qs) if len(v) else "n/a"


def table(counter: Counter, head=("value", "recordings")) -> list:
    out = [f"| {head[0]} | {head[1]} |", "|---|---|"]
    out += [f"| {k if k not in ('', None) else '(empty)'} | {v} |" for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))]
    return out


def audit(dataset: str, n_found: int, metas: dict, skipped: list, src: Path) -> str:
    M = list(metas.values())
    ep = sum(m["n_epochs"] for m in M)
    counts = Counter()
    for m in M:
        for k, v in m["counts"].items():
            counts[int(k)] += v
    scored = sum(v for k, v in counts.items() if k >= 0)
    L = [f"# External audit: {DATASETS[dataset]}", "",
         "Written by scripts/prepare_external.py. Aggregates only (no per-participant rows).",
         "Development-grade phase-2 data description; nothing here may change a phase-1 choice.",
         "Labels: VERIFIED = counted from the files by this script; CONDITIONAL = depends on the stated assumption.", "",
         f"Source folder: `{src}` (read only). Preprocessing: {json.dumps(M[0]['preprocessing']) if M else 'n/a'}.", "",
         "## Recordings (VERIFIED)", "",
         f"- found: {n_found}",
         f"- read and stored: {len(M)}",
         f"- skipped: {len(skipped)}"]
    reasons = Counter(why for _, why in skipped)
    L += [f"  - {v} x {why}" for why, v in reasons.most_common()]
    named = [str(r) for r, why in skipped if dataset == "isruc" and "not downloaded" in why]
    if named:
        L += [f"  - ISRUC skips by subject number (owner-listed): {', '.join(named)}"]
    L += ["", "## Epochs and stages (VERIFIED)", "",
          f"- epochs: {ep}; scored: {scored}; unscored (-1): {counts[-1]}", "",
          "| stage | epochs | share of scored |", "|---|---|---|"]
    L += [f"| {s} | {counts[i]} | {counts[i] / max(scored, 1):.3f} |" for i, s in enumerate(STAGES)]
    L += ["", "## Sampling rates and hardware prefilters (VERIFIED, EEG channels as stated in the headers)", ""]
    L += table(Counter(f"{m.get('fs_orig')} Hz" for m in M), ("native rate", "recordings"))
    if any("prefilter_eeg" in m for m in M):
        L += [""] + table(Counter(" / ".join(m.get("prefilter_eeg", [])) or "(not stated)" for m in M),
                          ("EEG prefilter field(s)", "recordings"))
    else:
        L += ["", "Prefilter: not stated in the files (UNKNOWN)."]
    L += ["", "## Channels", "", "Channel rule (D-017): first of Fz-Cz, F4-C4, F3-C3 | F4-M1, F3-M2, F4-M2, F3-M1 | C4-M1, C3-M2 "
          "(A1/A2 treated as M1/M2). VERIFIED counts:", ""]
    L += table(Counter(f"{m['rule_tier']}: {m['rule_channel']}" for m in M), ("rule outcome", "recordings"))
    L += ["", "Channels kept per recording:", ""]
    L += table(Counter(len(m["channels"]) for m in M), ("EEG channels", "recordings"))
    L += ["", "Channel names kept (recordings containing each):", ""]
    L += table(Counter(c for m in M for c in m["channels"]), ("channel", "recordings"))
    der = Counter(len(m.get("derived", [])) > 0 for m in M)
    L += ["", f"- recordings with channels derived from monopolar leads: {der[True]}"]
    if dataset == "cap":
        exc = Counter(lab for m in M for lab, _ in m.get("excluded_eeg_like", []))
        L += [f"- EEG-like labels dropped for an EMG-like hardware band (high-pass >= 5 Hz): "
              f"{dict(exc) if exc else 'none'} (recording counts)"]
    L += ["", "## Units and amplitude", ""]
    if dataset == "dodh":
        L += ["- Unit attribute in the files: " + ", ".join(sorted({u for m in M for u in m['units_attr']})) +
              " (VERIFIED). Median native 30-s SD per file: " + pct([m["median_epoch_sd_native"] for m in M]) +
              ". These magnitudes are microvolt scale, so the numbers are used as microvolt (CONDITIONAL)."]
    else:
        L += table(Counter(u for m in M for u in sorted(set(m.get("units", {}).values()))), ("EEG unit field", "recordings with it"))
        ass = sum(1 for m in M if m.get("unit_assumed_uv"))
        L += ["", f"- recordings with an empty/unknown unit field, physical value taken as microvolt: {ass} (CONDITIONAL)"]
    sd = [m["rule_median_epoch_sd_uv"] for m in M]
    L += [f"- median epoch SD of the rule channel after preprocessing (uV), across recordings: {pct(sd)} (VERIFIED)",
          f"- recordings with that median below 5 uV: {sum(v < 5 for v in sd)}; above 150 uV: {sum(v > 150 for v in sd)}"]
    if any("phys_range" in m for m in M):
        by = {}
        for m in M:
            by.setdefault(m.get("phys_range"), []).append(m["rule_median_epoch_sd_uv"])
        L += ["", "| rule-channel physical range in header | recordings | median epoch SD (uV), median over recordings |", "|---|---|---|"]
        L += [f"| {k} | {len(v)} | {np.median(v):.2f} |" for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))]
    L += ["", "## Label / signal length (VERIFIED)", ""]
    if dataset in ("isruc", "dodh"):
        acts = Counter(m["label_length"].get("action", "none" if m["label_length"]["n_labels"] == m["label_length"]["n_signal_epochs"] else "mismatch") for m in M)
        L += table(acts, ("label-length rule outcome", "recordings"))
    if dataset == "isruc":
        L += ["", "Rule: labels start at the signal start; extra trailing labels dropped, missing trailing epochs unscored (CONDITIONAL on the start alignment). Scorer 1 used."]
    if dataset == "cap":
        H = [m["hypnogram"] for m in M]
        L += [f"- stage rows before the EDF start, dropped: {sum(h['rows_before_signal'] for h in H)} rows in {sum(h['rows_before_signal'] > 0 for h in H)} recordings",
              f"- stage rows after the end of the signal, dropped: {sum(h['rows_after_signal'] for h in H)} rows in {sum(h['rows_after_signal'] > 0 for h in H)} recordings",
              f"- recordings whose grid starts off the EDF's 30-s grid (origin != 0 s): {sum(h['grid_origin_s'] != 0 for h in H)}",
              f"- truncated table rows ignored: {sum(h['truncated_rows'] for h in H)} in {sum(h['truncated_rows'] > 0 for h in H)} recordings",
              f"- MT epochs (unscored): counted in unscored above",
              f"- EDF record count in header differs from file size: {sum(m['nrec_header'] != m['nrec_file'] for m in M)} recordings (count taken from file size)"]
        L += ["", "Diagnosis groups:", ""] + table(Counter(m["group"] for m in M), ("group", "recordings"))
    if dataset == "hmc":
        L += ["Annotations are on the 30-s grid from the PSG start (hmc.py); unannotated epochs stay unscored."]
    if dataset == "isruc":
        L += ["", "Groups:", ""] + table(Counter(m["subject"].split(":")[0] for m in M), ("group", "recordings"))
    if dataset == "dodh":
        L += ["", "Hypnogram source (consensus or single scorer): UNKNOWN from the files."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--src", default=None)
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=52.0)
    ap.add_argument("--audit-dir", default="reports")
    a = ap.parse_args()
    paths = load_paths(a.paths)
    root = Path(paths["store_root"])
    src = Path(a.src or paths.get("raw", {}).get(a.dataset) or DEFAULT_SRC[a.dataset])
    print(f"free disk {free_gb(root):.1f} GB")
    if free_gb(root) < a.min_free_gb:
        sys.exit("not enough free disk")
    jobs, skipped = discover(a.dataset, src)
    n_found = len(jobs) + len(skipped)
    metas = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for rid, status, meta, err in ex.map(work, [(a.dataset, r, args, root, a.overwrite, a.min_free_gb) for r, args in jobs]):
            if status in ("written", "existing"):
                metas[rid] = meta
            else:
                skipped.append((rid, err))
            print(rid, status, err or "", flush=True)
    (root / DATASETS[a.dataset]).mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": a.dataset, "records": metas, "skipped": skipped}
    (root / DATASETS[a.dataset] / "manifest.json").write_text(json.dumps(manifest, indent=1))
    Path(a.audit_dir).mkdir(exist_ok=True)
    out = Path(a.audit_dir) / f"external_audit_{a.dataset}.md"
    out.write_text(audit(a.dataset, n_found, metas, skipped, src))
    print(f"{len(metas)} stored, {len(skipped)} skipped; audit -> {out}; free disk {free_gb(root):.1f} GB")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
