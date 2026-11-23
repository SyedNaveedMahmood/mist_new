"""EEG channel names across external datasets, and the fixed channel rule (D-017).

Store channel names keep the source's electrodes (A1/A2 stay A1/A2, M1/M2 stay
M1/M2) in a canonical spelling "<active>-<reference>", e.g. "F4-A1", "Fp2-F4".
For the rule, A1 == M1 and A2 == M2: the files do not say whether an earlobe or
mastoid electrode was used, so the two cannot be told apart (CONDITIONAL).

The rule is fixed before any external evaluation and never uses a score:
first available of
  1 frontal-central   Fz-Cz, F4-C4, F3-C3
  2 frontal-mastoid   F4-M1, F3-M2 (contralateral), then F4-M2, F3-M1 (ipsilateral)
  3 central-mastoid   C4-M1, C3-M2 (contralateral only)
Within a tier the midline comes first (closest to Sleep-EDF's Fpz-Cz), then the
right hemisphere, as in the AASM recommended derivations (F4-M1, C4-M1; left
side as backup).
"""
from __future__ import annotations

import re

ELECTRODES = ("Fp1", "Fp2", "Fpz", "F1", "F2", "F3", "F4", "F7", "F8", "Fz", "C3", "C4", "Cz", "P3", "P4", "Pz",
              "O1", "O2", "Oz", "T3", "T4", "T5", "T6", "A1", "A2", "M1", "M2")
REFERENCES = ("A1", "A2", "M1", "M2")
_CANON = {e.upper(): e for e in ELECTRODES}

RULE = (("frontal-central", ("Fz-Cz", "F4-C4", "F3-C3")),
        ("frontal-mastoid", ("F4-M1", "F3-M2", "F4-M2", "F3-M1")),
        ("central-mastoid", ("C4-M1", "C3-M2")))


def electrode(tok: str):
    return _CANON.get(tok.strip().upper())


def normalize(label: str):
    """Source label -> canonical bipolar name, "<electrode>" for a monopolar lead, or None if not EEG.

    Handles 'EEG F4-M1', 'F3_M2', 'FP1-F3', 'F3A2', 'C4-A1'. EOG/EMG names (LOC, ROC, E1, ...) give None.
    """
    s = re.sub(r"^EEG\s+", "", label.strip(), flags=re.IGNORECASE).replace("_", "-").replace(" ", "")
    s = s.replace("/", "-")
    parts = s.split("-")
    if len(parts) == 2:
        a, b = electrode(parts[0]), electrode(parts[1])
        if a and b and a not in REFERENCES:
            return f"{a}-{b}"
        return None
    if len(parts) == 1:
        if electrode(s):
            return electrode(s)
        m = re.fullmatch(r"([A-Za-z]{1,2}[0-9zZ])([AM][12])", s)       # 'F3A2'
        if m and electrode(m.group(1)) and electrode(m.group(2)) and electrode(m.group(1)) not in REFERENCES:
            return f"{electrode(m.group(1))}-{electrode(m.group(2))}"
    return None


def rule_key(name: str) -> str:
    return name.replace("A1", "M1").replace("A2", "M2")


def pick_rule_channel(channels) -> tuple:
    """-> (store channel name, tier name) or (None, None). Order fixed in RULE."""
    by_key = {rule_key(c): c for c in channels}
    for tier, names in RULE:
        for n in names:
            if n in by_key:
                return by_key[n], tier
    return None, None


def plan_channels(labels, derive: tuple = ()) -> list:
    """Source labels -> [(store name, how)], how = ("raw", label) or ("diff", label_a, label_b).

    Bipolar EEG labels are kept as recorded. Monopolar leads (all against one
    common reference) are combined into the bipolar pairs listed in `derive`
    when both leads exist and the pair is not already recorded.
    """
    out, seen, mono = [], set(), {}
    for lab in labels:
        n = normalize(lab)
        if n is None:
            continue
        if "-" in n:
            if n not in seen:
                out.append((n, ("raw", lab)))
                seen.add(n)
        else:
            mono.setdefault(n, lab)
    for pair in derive:
        a, b = pair.split("-")
        if pair not in seen and a in mono and b in mono:
            out.append((pair, ("diff", mono[a], mono[b])))
            seen.add(pair)
    return out
