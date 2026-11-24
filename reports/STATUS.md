# MIST-GRM status

**Updated:** 2026-11-24 (Claude Code, owner offline under the overnight delegation of 2026-11-24 01:10)

## Summary
- **GPU session 1 done** (2026-11-24 00:19 to 11:46, deadline 12:04; reports/session_1.md). 33 runs trained, 102 evaluations; no failures, retries or skips. One queue restart at 01:24 with the same end time (D-021). Since 11:46 the GPU is idle until the owner writes SESSION START.
  - P1 and P2 complete: cascade2 and cascade2_placebo, 5 folds x 3 seeds, each with in-domain, Pz-Oz and sc2st evaluations (std), plus the lf-view evaluations run on CPU afterwards.
  - P3 started: cascade2_absolute folds 0-2, seed 0. Folds 3 and 4 roll over.
  - Measured: cascade2 23.5 min per run (10.7 GB peak), cascade2_placebo 16.5 min (7.4 GB), cascade2_absolute 23.3 min.
- **Validation gate: PASSED** (validation only). cascade2 76.26 vs full-rate cascade 76.34 mean best val macro-F1, seed 0, -0.08 points (one-sided tolerance 1.0).
- **H1, INTERIM** (seeds 0-2, montage and sc2st only; the age block is not run): cascade2 - cascade2_placebo -1.9 macro-F1 points [-2.4, -1.4] by the corrected primary statistic (EXPERIMENT_PLAN section 7, D-024). In-domain +0.1 [-0.4, +0.6]. The loss is mostly Wake and N1. Changes no plan (owner B6). The confirmatory analysis waits for P5.
- **G2, INTERIM:** cascade2 - raw in-domain +0.6 [-0.3, +1.5], within 1.5 points.
- **B8 leakage check** on the large shift gains of the cascade arms over raw: none found (subjects, checkpoint selection, night statistics, lf window). See the notes at the top of reports/session_1.md.
- **Plan correction D-024 (owner):** the H1 primary statistic is the joint subject bootstrap over the four shift protocols. The mixed model is secondary. No H1 number existed before the change.
- **CPU now:** external dev evaluation (D2): raw and attnsleep seed 0 on HMC, DOD-H, CAP and ISRUC, then cascade2 and cascade2_placebo seed 0 (INTERIM), into reports/external_dev.md. The missing e1_cascade_cv4_s0 sc2st evaluation runs last.
- **Revision 4 (owner, 2026-11-24; DECISIONS D9; made after the interim H1 result was seen):** research questions re-aimed at transfer (what carries over: gain, rhythm or shape; does a model restricted to it beat standard models on unseen datasets). H1 to H6 unchanged; H3 is the central claim; the confirmatory H1 runs in session 3 and is reported whatever it shows. EXPERIMENT_PLAN section 13.
- **Recorded:** on seed 0 the own-rate cascade2 is below the full-rate cascade under shift (Pz-Oz -1.8 [-3.3, -0.6], sc2st -1.1 [-1.5, -0.7]); the in-domain revision-3 gate could not detect this. EXPLORATORY: full-rate cascade - cascade2_placebo is about zero (in-domain +0.2, Pz-Oz -0.1, sc2st -0.4; intervals span 0).
- **New EXPLORATORY control `raw_whitened`** (raw on night-whitened tokens): code and tests done; trains in session 2.
- **Next GPU session (2), on SESSION START:** `nohup scripts/session_queue.sh 2 "<end>" &`. Order S2-1 E3 cells on cascade2 (evaluation only), S2-2 raw_whitened, S2-3 cascade2_absolute folds 3-4, S2-4 P9 (grm, gr, grm_placebo), S2-5 cascade2 critic 0, S2-6 age block as far as it fits. Dry run: S2-1 to S2-5 fit; S2-6 rolls to session 3. Full-rate cascade destroy-M runs on CPU.

## Gates
| Gate | Status | Evidence |
| --- | --- | --- |
| G0 data | PASSED (VERIFIED) | store manifest: SC 74 people / 146 rec / 397,832 epochs / 396,378 scored; ST 22 / 44 |
| G0b baseline | **PASSED (VERIFIED)**: 0.7277 >= 0.70, 54 dev people, std, Fpz-Cz, seed 0, frozen recipe | registry `g0b_attnsleep_5fold_s0` |
| G1 reliance | **PASSED (VERIFIED, rule D-005)** | registry `g1_decision`, `e0`, `correction_e0_local_phase` |
| Validation gate (revision 3) | **PASSED (VERIFIED, validation only, seed 0)**: -0.08 points | runs/gates/val_gate_cascade2_s0.json; registry `session1_gate` |
| G2 | INTERIM: +0.6 [-0.3, +1.5] in-domain (seeds 0-2); decided on all seeds | reports/e1_interim_s012.md |
| G3 | NOT TESTED (parallel arms, P9) | |

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
- docs/DECISION_LOG.md: every decision taken while the owner is offline (D-001 to D-024).
- docs/OPEN_QUESTIONS.md: items reserved for the owner or the supervisor (Q1 to Q11).
- experiments/registry.yaml: every run and dataset.
