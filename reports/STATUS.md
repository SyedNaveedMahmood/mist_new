# MIST-GRM status

**Updated:** 2026-11-22 (Claude Code, owner offline under the delegation in docs/DECISIONS.md)

## Summary
- **Done:** T0 setup (tests pass), T1 store (G0 passed, from the built manifest), T2 splits frozen, T4 real-data invariance, T5 `lf` view, lockbox hardening, early-stopping removal.
- **Running:** recipe step 1b (class weighting null vs sqrt_inv, attnsleep + raw, 5 folds, seed 0, validation only); downloads HMC, ISRUC I+III, DOD-H.
- **Failed / open:** G0b under the pre-fix recipe (attnsleep fold 0: 0.6999 < 0.70; kept in the registry as pre-fix). G0b is re-decided once, on 5 folds, under the frozen recipe.
- **Next:** freeze recipe -> G0b (5-fold attnsleep test, once) -> `raw`/`grm` timing -> Sleep-EDF-20 replication (check only) -> T6 / E0.

## Gates
| Gate | Status | Evidence |
| --- | --- | --- |
| G0 data | PASSED (VERIFIED) | store manifest: SC 74 people / 146 rec / 397,832 epochs / 396,378 scored; ST 22 / 44 |
| G0b baseline | pre-fix FAILED (0.6999, fold 0); redefined, pending | registry `t3_attnsleep_cv0_s0` |
| G1 reliance | NOT TESTED | |
| G2, G3 | NOT TESTED | |

## Downloads (free disk is reported after each)
| Dataset | State |
| --- | --- |
| HMC v1.1 (15.7 GB) | downloading |
| CAP | complete on disk, checksum-verified; nothing to download |
| ISRUC I + III (16.5 GB) | downloading (allow-list; 5 non-standard files skipped) |
| DOD-H (21.9 GB zip) | downloading; md5 check, extract, verify, delete zip |
| DOD-O, CinC 2018 | deferred by owner (disk) |
| DREAMS, SHHS, MASS, MESA, MrOS | not downloaded (licence / applications) |

## Where to look
- docs/DECISION_LOG.md: every decision taken while the owner is offline.
- docs/OPEN_QUESTIONS.md: items reserved for the owner.
- experiments/registry.yaml: every run and dataset.
