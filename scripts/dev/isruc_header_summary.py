"""Aggregated EDF header fields of ISRUC recordings, by amplitude subgroup (header range of the EEG). No signals."""
import collections, glob, json
from pathlib import Path
from mist.data.edf_header import read_header
ROOT = Path("/home/FA006/Desktop/Dimension/dataset/isruc")
STORE = Path("/home/FA006/Desktop/Dimension/dataset/mist_store/ISRUC")
agg = collections.defaultdict(collections.Counter)
for mf in sorted(STORE.glob("*/meta.json")):
    m = json.loads(mf.read_text())
    grp = "low" if m["phys_range"] == "-25..25" else "normal"
    folder = "ISRUC-Sleep-I" if m["subject"].startswith("ISRUC1") else "ISRUC-Sleep-III"
    n = int(m["subject"].split(":")[1])
    h = read_header(ROOT / folder / str(n) / m["source"][0])
    for i, lab in enumerate(h["label"]):
        agg[grp][(lab, h["transducer"][i][:30], h["dim"][i], h["pmin"][i], h["pmax"][i], h["dmin"][i], h["dmax"][i],
                  h["prefilter"][i][:45], round(h["fs"][i]))] += 1
for grp, c in agg.items():
    print(f"== {grp}")
    for k, v in sorted(c.items()):
        print(v, k)
