# MIST-GRM status

**Updated:** 2026-11-23 (Claude Code, owner offline under the delegation in docs/DECISIONS.md)

## Summary
- **Done:**
  - T0 setup; T1 store (G0 passed); T2 splits frozen; T4 real-data invariance; T5 `lf` view; lockbox hardening.
  - Recipe fix: no early stopping. **Recipe frozen**: `class_weight: sqrt_inv`, chosen on validation only (+0.64 points over null).
  - **G0b passed** (attnsleep 5-fold pooled test macro-F1 0.7277).
  - T6b cascade topology implemented and timed (closed). Property tests are on synthetic signals only. Cost per run: cascade 2.44 h vs grm 20 min (7.2x), so the pre-decided seed cuts for non-decisive arms apply (D-014).
  - T9 stage-targeted critic; T8 code; E0/E1 runners; HMC reader (phase 2, code only).
  - HMC and DOD-H downloaded and verified.
  - Sleep-EDF-20 AttnSleep replication (check only, own registry section): the published MF1 78.1 is reproduced, 78.5, under the published repo's test-epoch selection; 75.0 at the final epoch without test selection.
- **E0 (T6) done, G1 PASSED** (reports/e0.md). Morphology reliance (cost of the corrected `local_phase` surrogate, macro-F1): attnsleep 16.5 in-domain / 4.9 montage / 6.1 sc2st; raw 11.8 / 1.8 / -0.6 (sc2st interval spans 0). A surrogate bug (local_phase halved band power) was found by an audit after the first pass, fixed with a test, and all E0 evaluations rerun (D-011, registry `correction_e0_local_phase`).
- **Running:**
  - **E1** (T7), serial queue scripts/e1_queue.sh: 297 trainings, about 13.5 GPU-days plus evaluation (EXPERIMENT_PLAN section 6). Started 2026-11-23 with seed 0 of cascade, cascade_placebo, cascade_absolute.
  - ISRUC: 547 of 550 allow-listed files complete; 3 group I .rec files (subjects 4, 6, 39) still incomplete after two passes, retry scheduled 3 h later (MEGA quota, D-007).
- **Session break 2026-11-22/23:** the class-weight sweep and the ISRUC retry stopped when the previous session ended. 19 of 20 sweep runs had finished. The interrupted run was retrained from scratch (partial run kept in runs/_interrupted/, D-008), and ISRUC was restarted.
- **Plan change found on the branch:** D7 = cascade (owner relaying the supervisor, commit 175911e). The cascade is the main model; `grm` and `gr` become the topology ablation. E0 is unchanged.
- **Next:**
  1. When seed 0 of the decisive cascade arms is done: interim G2 check (cascade vs raw, recorded as interim; the gate is decided on all seeds).
  2. Finish E1, write reports/e1.md, decide G2 and G3.

## Gates
| Gate | Status | Evidence |
| --- | --- | --- |
| G0 data | PASSED (VERIFIED) | store manifest: SC 74 people / 146 rec / 397,832 epochs / 396,378 scored; ST 22 / 44 |
| G0b baseline | **PASSED (VERIFIED)**: 0.7277 >= 0.70, 54 dev people, std, Fpz-Cz, seed 0, frozen recipe. The pre-fix single-fold 0.6999 is kept as history | registry `g0b_attnsleep_5fold_s0`; `t3_attnsleep_cv0_s0` + correction |
| G1 reliance | **PASSED (VERIFIED, rule D-005)**: every stage, both models, smallest raw N2 +5.8 [+4.2, +7.6]. Effects are upper bounds on reliance (probe-fidelity caveats in reports/e0.md) | registry `g1_decision`, `e0`, `correction_e0_local_phase` |
| G2, G3 | NOT TESTED | |

## Frozen recipe (validation macro-F1, mean over 5 folds, seed 0)
| Model | null | sqrt_inv |
| --- | --- | --- |
| attnsleep | 0.7222 | 0.7298 |
| raw | 0.7523 | 0.7576 |
| mean | 0.7373 | **0.7437** |

## Downloads (free disk: 93 GB on 2026-11-23)
| Dataset | State |
| --- | --- |
| HMC v1.1 (15.7 GB) | complete, all 456 files match SHA256SUMS, 151 recordings |
| CAP | complete on disk, checksum-verified; nothing to download |
| DOD-H (22.8 GB) | complete: md5 matched, 25 .h5 files open with signals + hypnogram and match the zip sizes, zip deleted. Contents not yet read by project code |
| ISRUC I + III (16.5 GB) | group III complete (10 subjects); group I 497 of 500 files complete, 3 .rec files being retried (MEGA quota, D-007) |
| DOD-O, CinC 2018 | deferred by owner (disk) |
| DREAMS, SHHS, MASS, MESA, MrOS | not downloaded (licence / applications) |

## Where to look
- docs/DECISION_LOG.md: every decision taken while the owner is offline (D-001 to D-014).
- docs/OPEN_QUESTIONS.md: items reserved for the owner or the supervisor (Q1 to Q7).
- experiments/registry.yaml: every run and dataset.
