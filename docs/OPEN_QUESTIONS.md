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
| Q9 | 2026-11-24 | ISRUC amplitude subgroup. **Closed by the owner 2026-11-24 (D-019, D-020):** kept as stored; evaluated as sub-domains ISRUC-M (81) and ISRUC-A (26) plus "all"; `*_absolute` arms on ISRUC-M only; no group III or healthy-versus-patient claim. | closed | no |
| Q10 | 2026-11-24 | ISRUC as a training source: ISRUC-M only, or ISRUC-M and ISRUC-A? (D-020) | Owner's instruction | Any ISRUC-trained model (phase 2, T15) |
| Q11 | 2026-11-24 | SleepBand (Lu, Hu, Chen; arXiv:2607.04851, 2026) has public code (github.com/lzcn/sleep-band) but no licence file or licence statement, so it is all rights reserved by default. T14 therefore re-implements it from the paper text only; the repository code was not cloned or read. Asking the authors for a licence (to compare against the official code) means contacting a third party. | Contacting anyone is reserved | No; the re-implementation proceeds |
