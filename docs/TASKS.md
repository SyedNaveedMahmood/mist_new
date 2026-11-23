# Task list for Claude Code

Work top to bottom. Each task has an acceptance test. Do not start a task whose gate is open. Record every run in `experiments/registry.yaml`.

## Phase 1a: data and baseline (gates G0, G0b)

- [x] **T0 Setup.** `pip install -e ".[data,dev]"`, copy `configs/paths.example.yaml` to `configs/paths.yaml`, fix paths, run `pytest`. *Accept:* 25 tests pass on the workstation.
- [x] **T1 Build the store.** `python scripts/prepare_sleepedf.py --cohorts SC ST`. Check the two unverified parsers first: `read_sc_subjects` column names and sex coding against the real `SC-subjects.xls`; `mne.read_annotations` on the seven ST hypnograms (ST7021JM, ST7071JA, ST7092JE, ST7131JR, ST7132JR, ST7141JE, ST7142JE). *Accept (G0):* SC 74 people, 146 recordings, 397,832 epochs, 396,378 scored; ST 22 people, 44 recordings; no annotation outside the stage map. If totals differ, explain every difference before continuing (the old pipeline is on Proto-sleep branch `research/mist-transfer-v3-edf-shhs`, `src/mist_transfer/`).
- [x] **T2 Freeze splits.** Only after decision D1 is confirmed by the owner. `python scripts/make_splits.py`, commit `splits/sleepedf_v1.json`. *Accept:* 20 lockbox, 54 dev, 5 folds of 10 or 11, three age sets, 22 ST; file committed; never regenerated.
- [ ] **T3 Baseline sanity and timing.** Train `attnsleep` and `raw` on fold 0, seed 0. Record wall time per epoch and GPU memory. *Accept (G0b):* `attnsleep` test macro-F1 >= 0.70 on the `std` view. Write the timing into EXPERIMENT_PLAN section 6 and trim the budget if needed.
- [x] **T4 Real-data invariance.** `python scripts/check_invariance.py --n-nights 10`. *Accept:* table saved under `runs/e5a/`; gain change 0; tilt change of G, R, M below 0.05; report the all-pass figures as they are.
- [x] **T5 Label-free window.** Add a `lf` view: from 30 minutes before lights-off for 10 hours, from the subject spreadsheets. *Accept:* unit test; `lf` and `std` epoch counts per night written to the manifest.

## Phase 1b: experiments

- [x] **T6 E0 audit.** 30 runs, then `scripts/evaluate.py` for in-domain, `--channel Pz-Oz`, `--protocol sc2st`. Write `scripts/aggregate.py` to build the condition x stage tables with subject-bootstrap intervals. *Accept:* tables in `reports/e0.md`; gate G1 decided and recorded.
- [x] **T6b Cascade topology (supervisor revision, 2026-11-22).** Implement docs/ARCHITECTURE.md section 8 as `topology: cascade` beside the existing parallel code path; do not remove or change the parallel path. Add the arms in section 8.4, `destroy_bands`, and the six property tests in section 8.5. Time one `cascade` run. *Accept:* all existing tests still pass unchanged; the six new tests pass; with `destroy_prob: 1.0` the model output is identical for a token and for the same token with its phase replaced; parameter counts of `cascade` and `cascade_placebo` are equal; timing recorded. E1 does not start for the cascade arms until this is done. Runs already made with parallel arms are kept.
- [ ] **T7 E1 factor arms.** As EXPERIMENT_PLAN section 5, with the changes in section 11. *Accept:* `reports/e1.md` with every decisive pair from section 1; gates G2 and G3 decided.
- [ ] **T8 Slot knockouts.** Add `knock_slots` to `ScaleBranch.forward` (zero the chosen assignments and renormalise), a slot gallery (top-activating tokens as waveforms), and after-the-fact naming by agreement with relative-threshold spindle, slow-wave and alpha detectors. *Accept:* unit test that knocking all slots equals `knock_M` up to the residual; gallery for one checkpoint.
- [x] **T9 Stage-targeted critic.** Config `critic_stages: [0, 1, 4]` restricts the critic to tokens from those stages (training labels only). *Accept:* unit test; used by E4 arm N-target.
- [ ] **T10 E2, E3, E4, E5.** *Accept:* `reports/e2.md` to `e5.md`; the mixed model for H1; Holm-corrected table of H1 to H6.
- [ ] **T11 Freeze.** Write the final hyperparameters and analysis script hash into the registry. *Accept:* owner signs off in `docs/DECISIONS.md`.
- [ ] **T12 E6 lockbox.** One evaluation with `split.i_am_running_the_final_confirmation=true`. *Accept:* result reported unchanged, whatever it is.

## Phase 2: external (start downloads during phase 1)

- [ ] **T13 HMC reader and audit**, then ISRUC, DOD, CinC 2018 (training part), CAP rebuild from EDF.
- [ ] **T14 Baselines.** SleepBand re-implementation (check for released code first), magnitude/phase STFT model in the spirit of PhASER, TinySleepNet, WaveSleepNet.
- [ ] **T15 Single-source and leave-one-dataset-out benchmark.**
- [ ] **T16 Morphology-competence probes** on MODA / MASS-SS2 or DREAMS once access and rights are settled.
- [ ] **T17 Masked-token pretraining** on (R, shape-slot) targets, only if phase 1 shows M is useful.

## Standing instructions

- Run `pytest` before every commit. Add a test with every new module.
- One change per commit; message says what and why.
- Never open the lockbox, never tune on ST after its role is declared, never regenerate the split file.
- If a result looks too good, look for leakage first: subject overlap, night reference computed on the wrong window, checkpoint chosen on test.
- When blocked by a decision that is the owner's, stop and ask; do not pick silently.
