# MIST-GRM status

**Updated:** 2026-11-23 (Claude Code, owner offline under the delegation in docs/DECISIONS.md)

## Summary
- **Done:**
  - T0 setup; T1 store (G0 passed); T2 splits frozen; T4 real-data invariance; T5 `lf` view; lockbox hardening.
  - Recipe fix: no early stopping. **Recipe frozen**: `class_weight: sqrt_inv`, chosen on validation only (+0.64 points over null).
  - **G0b passed** (attnsleep 5-fold pooled test macro-F1 0.7277).
  - T6b cascade topology implemented, tested on synthetic signals only (timing still pending).
  - T9 stage-targeted critic; T8 code; E0/E1 runners; HMC reader (phase 2, code only).
  - HMC and DOD-H downloaded and verified.
  - Sleep-EDF-20 AttnSleep replication (check only, own registry section): the published MF1 78.1 is reproduced, 78.5, under the published repo's test-epoch selection; 75.0 at the final epoch without test selection.
- **Running:**
  - E0 (T6): raw and attnsleep, seeds 1 and 2, then all evaluations.
  - ISRUC group I download (resumed 2026-11-23).
- **Session break 2026-11-22/23:** the class-weight sweep and the ISRUC retry stopped when the previous session ended. 19 of 20 sweep runs had finished. The interrupted run was retrained from scratch (partial run kept in runs/_interrupted/, D-008), and ISRUC was restarted.
- **Plan change found on the branch:** D7 = cascade (owner relaying the supervisor, commit 175911e). The cascade is the main model; `grm` and `gr` become the topology ablation. E0 is unchanged.
- **Next:**
  1. Finish E0, then `aggregate.py e0` and decide G1 (reports/e0.md).
  2. Time `grm` and `cascade` alone on the GPU; record the budget in EXPERIMENT_PLAN section 6; close T6b.
  3. E1 with the section 11 arms.

## Gates
| Gate | Status | Evidence |
| --- | --- | --- |
| G0 data | PASSED (VERIFIED) | store manifest: SC 74 people / 146 rec / 397,832 epochs / 396,378 scored; ST 22 / 44 |
| G0b baseline | **PASSED (VERIFIED)**: 0.7277 >= 0.70, 54 dev people, std, Fpz-Cz, seed 0, frozen recipe. The pre-fix single-fold 0.6999 is kept as history | registry `g0b_attnsleep_5fold_s0`; `t3_attnsleep_cv0_s0` + correction |
| G1 reliance | NOT TESTED (E0 running) | |
| G2, G3 | NOT TESTED | |

## Frozen recipe (validation macro-F1, mean over 5 folds, seed 0)
| Model | null | sqrt_inv |
| --- | --- | --- |
| attnsleep | 0.7222 | 0.7298 |
| raw | 0.7523 | 0.7576 |
| mean | 0.7373 | **0.7437** |

## Downloads (free disk: 105 GB on 2026-11-23)
| Dataset | State |
| --- | --- |
| HMC v1.1 (15.7 GB) | complete, all 456 files match SHA256SUMS, 151 recordings |
| CAP | complete on disk, checksum-verified; nothing to download |
| DOD-H (22.8 GB) | complete: md5 matched, 25 .h5 files open with signals + hypnogram and match the zip sizes, zip deleted. Contents not yet read by project code |
| ISRUC I + III (16.5 GB) | group III complete (10 subjects); group I resuming from 55 of 500 files (MEGA quota, D-007) |
| DOD-O, CinC 2018 | deferred by owner (disk) |
| DREAMS, SHHS, MASS, MESA, MrOS | not downloaded (licence / applications) |

## Where to look
- docs/DECISION_LOG.md: every decision taken while the owner is offline (D-001 to D-010).
- docs/OPEN_QUESTIONS.md: items reserved for the owner or the supervisor (Q1 to Q7).
- experiments/registry.yaml: every run and dataset.
