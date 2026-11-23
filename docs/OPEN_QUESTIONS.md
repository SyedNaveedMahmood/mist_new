# Open questions for the owner

Items reserved for the owner (delegation of 2026-11-22). Claude Code has moved on to work that does not depend on them.

| # | Date | Question | Why it is reserved | Blocking |
| --- | --- | --- | --- | --- |
| Q1 | 2026-11-22 | CinC 2018 training partition (~135 GB) and DOD-O (36 GB zip) need about 150 GB more free disk. | Disk / owner decision (deferred by owner) | Phase 2 breadth only |
| Q2 | 2026-11-22 | SHHS, MASS, MESA, MrOS need data-use applications (NSRR DUA; MASS via CEAMS with ethics approval). | Applications and third-party contact | Phase 2 extensions, T16 |
| Q3 | 2026-11-22 | `src/mist/models/attnsleep.py` came from the earlier project; upstream AttnSleep licence must be checked before the repo is made public. | Making the repo public / licence | Publication only |
| Q4 | 2026-11-22 | T12/E6 lockbox evaluation. | Lockbox | E6 only |
| Q5 | 2026-11-22 | Supervisor: does R -> M mean morphology within each rhythm band (adopted in ARCHITECTURE section 8), or rhythm as a conditioning signal for one broadband morphology stream? | Supervisor's intent | T6b default is the per-band reading; `cascade_broad` covers the other |
| Q6 | 2026-11-22 | Supervisor: should rhythm have no direct path to the composer at all? A skip connection was left out because the arrow was crossed out. | Supervisor's intent | Not blocking; affects gate G2 if the cascade loses in-domain accuracy |
| Q7 | 2026-11-22 | Supervisor: the slide carried a figure of a 1-D EEG trace with sharp transients. Instruction or illustration? | Supervisor's intent | Not blocking |
| Q8 | 2026-11-23 | ISRUC group I: subjects 4, 6 and 39 have no .rec file (MEGA anonymous quota). Owner fetches the three files by hand into the ISRUC folder; until then group I is used without them. | Owner instruction 2026-11-23 | No; phase 2 runs on 97 group I subjects |
| Q9 | 2026-11-24 | ISRUC amplitude: 26 recordings (A1/A2-labelled, header range +/-25 uV) are about 11x lower than the other 81. **Decided by the owner 2026-11-24 (D-019):** keep as stored; three-row reporting. The pure-scale diagnostic says **not a pure scale** (2-8 Hz shape differs after scaling, also within group I; ECG factor 33 vs about 11 for EEG/EOG/EMG), so by the owner's rule ISRUC is not evaluated. Open: how to proceed with ISRUC (evaluate the normal-range 81 only, keep the three rows with a caveat, or leave ISRUC out). | Owner's rule stops ISRUC evaluation | ISRUC external evaluation only; HMC, DOD-H, CAP unaffected |
