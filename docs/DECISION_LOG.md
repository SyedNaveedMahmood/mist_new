# Decision log (append-only)

Decisions taken by Claude Code while the owner is offline (delegation of 2026-11-22, see docs/DECISIONS.md). Rule: pick the more conservative option (less favourable to our method, more reversible, closer to the written plan). Never edit an entry; add a new one that points to it.

## Owner's pre-decided answers (copied 2026-11-22, for reference)

- G0b fails: no tuning. If the Sleep-EDF-20 replication is within 2 macro-F1 points of the published figure, attnsleep is CONDITIONAL ("code reproduces the literature; our split is harder") and work continues. Otherwise bug fixes only (judged on validation), then continue with `raw` as main control, attnsleep CONDITIONAL.
- G1 fails: EXPERIMENT_PLAN section 8 fallback, but still run E1 core arms (raw, gr, grm_placebo, grm, grm_absolute); skip E2.
- G2 fails: try grm_loose, then morph_residual 32, on validation only; report the trade-off.
- G3 fails: shrink M or widen gr until parameter counts and validation scores match; rerun affected arms.
- Budget too large: 2 seeds for r, m, grm_loose, grm_nocritic; then 1 seed for E2. Never cut raw, gr, grm_placebo, grm.
- Result-changing bug: fix with a test, rerun affected runs, registry correction entry.
- Too-good result: check leakage first (subject overlap, night-reference window, checkpoint selection).
- T11 freeze may be done by Claude Code after E0-E5; then phase 2 (T13-T15) with data on disk; phase 2 never changes phase 1.

## Entries

### 2026-11-22 - D-001: Sleep-EDF-20 replication, which predictions to report
- **Came up:** The published AttnSleep repo saves, for each fold, the predictions from the epoch with the highest *test-fold* accuracy (its "validation" loader is the test subject), i.e. checkpoint selection on test. The paper does not state this.
- **Options:** (a) report only final-epoch (epoch 100) predictions; (b) copy the repo's best-test-epoch selection; (c) report both, (a) as primary.
- **Chosen:** (c). Primary number = final epoch, no selection (honest, less favourable). The best-test-epoch figure is reported beside it only to show how much the published protocol gains from test selection, and labelled as optimistic.
- **Why:** Conservative and comparable at the same time; the replication is a check on the code, not a project result.
- **Reversible:** yes (both are computed from the same runs).

### 2026-11-22 - D-002: Sleep-EDF-20 replication, class weights
- **Came up:** The repo computes class weights from all labels, test fold included.
- **Chosen:** Same formula and constants (mu = 1.5, 2, 1.5, 1, 1.5 over K), counts from the training subjects of each fold only.
- **Why:** Avoids test information; the difference in weights is tiny.
- **Reversible:** yes.

### 2026-11-22 - D-003: where the short `lf` nights are listed
- **Came up:** The owner asked for the lf counts and the two short SC nights "in the manifest". The manifest is a dict of records; a top-level list would break code that iterates its values.
- **Chosen:** per-record fields `n_epochs_std`, `n_epochs_lf`, `lf_clipped_start`, `lf_clipped_end` in manifest.json and meta.json, plus `<store>/lf_window_report.json` listing clipped nights (SC: SC4492, SC4592 end early; ST: 42 of 44 end before the 10 h window, 41 start at lights-off so the 30 min lead-in is missing).
- **Reversible:** yes (backfill script is idempotent).

### 2026-11-22 - D-004: how seeds and folds are combined in E0 tables
- **Came up:** EXPERIMENT_PLAN section 7 asks for the bootstrap "on seed-averaged predictions"; evaluation files hold argmax confusion matrices, not probabilities.
- **Options:** (a) re-run evaluation saving probabilities and average them; (b) sum each subject's confusion matrices over seeds (and over folds for sc2st, where every ST person is tested by all 5 fold models) and resample subjects.
- **Chosen:** (b). Each seed counts equally, the resampling unit stays the person, and no extra evaluation pass is needed. It does not reduce prediction noise the way probability averaging would, so intervals are, if anything, slightly wider (conservative).
- **Reversible:** yes (probabilities can be added to evaluate.py later).

### 2026-11-22 - D-005: exact G1 rule
- **Came up:** EXPERIMENT_PLAN gives G1 as "local_phase costs less than 1 macro-F1 point for every stage and every model" and H0 as "MR >= 1 point for at least one stage, interval excluding 0", without saying which protocol decides.
- **Chosen:** G1 passes iff, **in-domain** (cv, Fpz-Cz, std), at least one of raw / attnsleep has at least one stage whose F1 drops by >= 1 point under `local_phase` with the 95% paired-bootstrap interval excluding 0. Montage, sc2st and lf results are reported in reports/e0.md but do not decide the gate.
- **Why:** Combines both written criteria (size and interval); deciding on one pre-named protocol avoids passing by searching across protocols (conservative).
- **Reversible:** the tables for every protocol are kept.

### 2026-11-22 - D-006: E0 scope
- **Chosen:** E0 = raw and attnsleep under `cv`, evaluated in-domain, Pz-Oz (montage) and sc2st (all ST nights, then placebo and temazepam nights separately), plus the lf window for in-domain and sc2st as sensitivity. The age-shift protocols need separate training and are covered in E1, not E0 (the owner's T6 list names only in-domain, Pz-Oz and sc2st).
- **Seed 0** of E0 reuses the recipe-sweep runs whose class-weight setting is selected, because their configuration is identical to the frozen recipe; seeds 1 and 2 are trained new.

### 2026-11-22 - D-007: ISRUC download stalled on MEGA's anonymous transfer quota
- **Came up:** After about 3 GB (all of group III, 11 subjects of group I) MEGA stopped serving (0 bytes/s even for a 2.4 KB file).
- **Options:** (a) wait for the quota to reset and resume; (b) switch to the Google Drive mirror listed on the ISRUC site; (c) create a MEGA account.
- **Chosen:** (a). The fetch is resumable (complete files are skipped by size); retry scheduled 3 h later with a 30 min timeout per file. (c) is reserved for the owner (account creation). (b) stays as a fallback if MEGA keeps refusing.
- **Reversible:** yes.

### 2026-11-23 - D-008: class-weight sweep finished after a session break; recipe frozen
- **Came up:** The session ended while the last sweep run (cw_raw_sqrt_inv_cv4_s0) was at epoch 35 of 40. Training has no resume.
- **Chosen:** moved the partial run to runs/_interrupted/ (kept, not deleted) and retrained it from scratch with the same seed. The 20 runs were trained at different commits; the only training-path changes between them are a behaviour-preserving refactor of the std trim and passing `y` to the model (read only by the GRM critic), so the runs are comparable.
- **Result (validation only):** sqrt_inv 0.7437 vs null 0.7373 mean over attnsleep and raw, +0.64 points, so by the owner's rule class_weight = sqrt_inv. The margin over the 0.5-point threshold is small (0.14 points); recorded so the choice is not read as a strong effect.
- **Reversible:** no, by owner instruction the recipe is frozen for phase 1.

### 2026-11-23 - D-009: G0b evaluation shared with E0 seed 0
- **Chosen:** G0b was evaluated once with scripts/evaluate.py defaults on runs/cw_attnsleep_sqrt_inv_cv*_s0. E0 seed 0 links to the same run folders, so E0 reads the same eval files instead of recomputing them. No score was looked at before the recipe was frozen (commit 630371d precedes the evaluation).

### 2026-11-23 - D-010: cascade implementation details (T6b)
- **Came up:** A docs-only commit from another session (175911e, 2026-11-22) recorded D7 = cascade (owner relaying the supervisor) and added task T6b. ARCHITECTURE 8.3 leaves several details open.
- **Chosen:** the simplest reading of each, listed in ARCHITECTURE section 8.8. The one with consequences for results: `destroy_bands: [b]` randomises band b's phase in every channel (band and broadband), so it removes that band's shape from the whole model; the alternative (band channel only) would let the broadband channel leak the band's shape and understate reliance. `broad` removes only the broadband channel (cross-band alignment).
- **Not changed:** the parallel path (ScaleBranch), its tests and configs. Existing runs stay valid.
- **Order of work:** E0 (raw, attnsleep) runs unchanged on the GPU while T6b was implemented; the cascade and grm timing runs follow E0 so timings are not distorted by a shared GPU.
- **Reversible:** yes; no cascade result exists yet.

### 2026-11-23 - D-011: local_phase surrogate bug found after E0; fixed and E0 re-evaluated
- **Came up:** the first E0 tables showed local_phase costs up to +45 points (attnsleep W). Following the owner's rule ("a result that looks too good: assume a fault first"), a model-free fidelity audit (scripts/audit_surrogates.py, fold-0 training nights) showed the surrogate lost about half the power in every band.
- **Chosen:** treated as a result-changing bug under the pre-decided answer: analytic fix (no fitted constant), regression test, rerun of all affected evaluations (150 files), correction entry in the registry, pre-fix report and files kept. The G1 rule (D-005) and the probe itself were **not** changed: replacing local_phase by another surrogate after seeing test scores would be a test-informed analysis choice (hard limit 3).
- **Result:** G1 PASSED both before and after the fix; effect sizes changed (for example raw N3 +36 -> +10 points). The probe-fidelity table and its reading are in reports/e0.md.
- **Reversible:** the fix is a correctness fix; not reversible in spirit.

### 2026-11-23 - D-012: G1 status label
- **Chosen:** G1 = PASSED (VERIFIED for the pre-registered rule). The interpretation caveats (sub-2 s timing, off-distribution inputs) are stated in reports/e0.md and the registry, not used to re-decide the gate. E1 runs the full plan, which is also the more complete option.

### 2026-11-23 - D-013: cascade does not fit the GPU at the frozen batch size; activation checkpointing
- **Came up:** the first cascade timing run ran out of memory (16 GB GPU) in the first step: the shared WaveEncoder runs on 6 channel waveforms per token (about 226k waveforms per batch at the 1 s scale). `grm` peaks at 9.8 GB.
- **Options:** (a) smaller batch; (b) mixed precision; (c) gradient accumulation; (d) activation checkpointing of the encoder in chunks.
- **Chosen:** (d). It leaves the arithmetic unchanged (encoder has no dropout, only per-sample GroupNorm; a test checks values and gradients match), so the frozen recipe is untouched. (a) and (b) change the recipe; (c) changes how the loss is averaged over unequal scored-epoch counts. Cost: extra compute, measured in the timing run.
- **Reversible:** yes.

### 2026-11-23 - D-014: measured cost of the cascade; E1 budget and order
- **Measured (alone on the GPU, frozen recipe, fold 0 seed 0):** `grm` 30.5 s/epoch, 1218 s/run, 9.8 GB peak; `cascade` (with D-013 checkpointing) 219.6 s/epoch, 8782 s/run (2.44 h), 7.1 GB peak. The cascade is 7.2x slower: its shared encoder runs on 6 channel waveforms per token, plus the checkpoint recompute. Best validation macro-F1 at fold 0: grm 0.780 (epoch 30), cascade 0.779 (epoch 15); validation only, no test score was read.
- **Came up:** E1 as written (section 6 plus section 11) is about 360 trainings; with 117+ cascade trainings at 2.44 h it is well over two weeks. EXPERIMENT_PLAN section 6 says to cut seeds for non-decisive arms when a run exceeds 30 minutes.
- **Chosen (the pre-decided cuts, nothing decisive cut):**
  - Full 5 folds x 3 seeds: `cascade`, `cascade_placebo`, `cascade_absolute`, `gr`, `grm_placebo`, `grm`, `grm_absolute` (raw and attnsleep come from E0).
  - 2 seeds: `grm_nocritic`, `cascade_nocritic`. 1 seed: `r`, `m`, `grm_loose` (section 11 order), `cascade_bands`, `cascade_broad` (non-decisive ablations, treated like `r`/`m`).
  - Age shift and Pz-Oz `cv`, 3 seeds: `raw`, `gr`, `grm_placebo`, `grm`, `grm_absolute`, `cascade`, `cascade_placebo`.
  - E2 (dose) will use 1 seed, as planned.
- **Plan gap found:** section 11 states H3 as `cascade` vs a cascade without the night reference but its training count (75) omits `cascade_absolute`. Added at full seeds (+15 trainings), because H3 is in the Holm family.
- **Order:** scripts/e1_queue.sh, serial on the one GPU (a second concurrent job would exceed 16 GB with a cascade run and gives little throughput at 98% utilisation). Seed 0 of the decisive cascade arms first, then the decisive parallel arms, then further seeds, then ablations, then the age and Pz-Oz blocks.
- **Estimate:** 297 trainings, about 324 GPU hours of training (about 13.5 days) plus evaluation. Recorded in EXPERIMENT_PLAN section 6.
- **Not chosen:** mixed precision or a smaller batch for the cascade only (would change the frozen recipe for one arm and confound the cascade vs raw comparison).
- **Reversible:** yes; more seeds can be added later without changing anything already run.

### 2026-11-23 - D-015: owner's revision of 23 Nov (GPU sessions, cascade2); E1 queue stopped
- **Owner instruction (message of 2026-11-23, recorded in docs/DECISIONS.md D8):** the GPU is available in two 12-hour sessions a week; E1 is re-planned around that, with an own-rate cascade (`cascade2`, ARCHITECTURE section 9). **Blind change:** no E1 test score had been read or aggregated when the plan changed. The E1 evaluation files that exist (cascade seed 0) were written by the queue and have not been opened, printed or aggregated.
- **Stopped 2026-11-23 23:33:** scripts/e1_queue.sh (waiting process), scripts/run_e1.sh and the running job. The running job was the sc2st evaluation of `e1_cascade_cv4_s0`, not a training: all five seed-0 `cascade` runs finished training (40 epochs each), so there was no partial training run to move to runs/_interrupted. The interrupted evaluation wrote no file. `e1_cascade_cv4_s0` therefore lacks its sc2st and lf evaluations; `e1_grm_cv0_s0` has none yet. Both are listed as pending evaluations.
- **Kept unchanged:** `e1_cascade_cv{0..4}_s0` are the "full-rate cascade" reference for the cascade2 validation gate; `e1_grm_cv0_s0` is reused in P9.
- **ISRUC:** the retry was cancelled. Group I goes ahead without subjects 4, 6 and 39: their .rec files never downloaded (MEGA quota); their scoring files are on disk. 547 of 550 allow-listed files are complete. The owner fetches the three files by hand (OPEN_QUESTIONS Q8).
- **Superseded:** D-014's queue and the EXPERIMENT_PLAN section 6 E1 table (replaced by section 12).

### 2026-11-23 - D-016: implementing revision 3 (T6c) and the session queue: choices not fixed by the owner
All taken before any E1 test score was read; none uses a test score.
- **Defaults reproduce the old cascade:** checked bit for bit against a fixture written by the pre-revision code (tests/fixtures/cascade_v1_reference.pt, synthetic model and input, no EEG), and `runs/e1_cascade_cv0_s0/best.pt` loads unchanged. The per-channel index buffers are non-persistent, so old and new checkpoints stay compatible.
- **Checkpointing dropped for cascade2** (`encoder_checkpoint: false`). This rests on a CPU count of the activations saved for backward at batch 32: cascade2 10.5 GB; grm 9.6 GB (measured GPU peak 9.8 GB); full-rate cascade 55 GB, consistent with its out-of-memory failure. No GPU was used. If a training still runs out of memory, the queue retries once with checkpointing, which gives the same arithmetic.
- **Band encoder:** `band_width: null` keeps one shared encoder (old behaviour); an integer adds the second encoder. The gate fallback `band_width: 32` therefore keeps two encoders and widens only the band encoder.
- **Validation gate read two-sided** ("within 1.0 point": |diff| <= 1.0). If `cascade2` were more than 1 point *above* the reference, the rule would still send it to the fallback; the session report would flag that case for the owner.
- **In-session evaluations write the real-input cell only** (`evaluate.py --cells core`). E1, G2 and H1 read only that cell. The other cells are E3 (P6) and go to `eval_e3_*.json`. This keeps session minutes for training, and the owner's 30-minute target includes three evaluations.
- **Queue semantics:**
  - A pair starts only if both runs fit, so an early end leaves balanced folds.
  - The walk stops at the first unit that does not fit rather than skipping ahead to smaller items, which keeps the priority order.
  - Estimates use the largest measured time per arm and block.
  - The prior for an unmeasured `cascade2` is the full-rate cascade time (conservative).
- **E4 weights** `train.n1_weight` in {1.5, 2.0, 3.0}. They multiply the frozen `sqrt_inv` class weight for N1 and were chosen without data as a spread around a doubling. The match to `cascade2` uses validation N1 F1 at the best-validation epoch, which training now logs (`val_f1` per epoch, `best_val_f1` in train_summary.json). This is logging only, not a recipe change; runs trained earlier lack the field.
- **E3 cells (P6):** the four input conditions, knock G/R/M, destroy M and the six band cells, on in-domain, Pz-Oz and sc2st. The age-shift checkpoints (P5) are not in P6, which names the P1 to P3 checkpoints.
- **Run names** keep the E1 pattern (`e1_<arm>_cv<f>_s<s>`, `e1_<arm>_<protocol>_s<s>`). E2 and E4 variants are `e2_cascade2_{dp0.5,cw0,cw1}_*`, `e4_cascade2_placebo_n1w{1.5,2.0,3.0}_*`, `e4ctx0_*` and `e4_cascade2_ntarget_*`.

### 2026-11-23 - Correction to D-016 (owner, before session 1)
- **Validation gate is one-sided.** `cascade2` fails only if its mean best-validation macro-F1 is more than 1.0 point below the full-rate reference; being above passes. scripts/gate_cascade2.py changed (`diff >= -0.010`); the queue's use is unchanged (exit 0 pass, 1 fail -> band_width 32 branch, 2 incomplete -> stop), and the w32 recheck uses the same one-sided rule. Replaces the two-sided reading in D-016. Decided before any cascade2 run exists.
- **tests/test_cascade2.py::test_8** compares with `torch.allclose(atol=1e-5)` instead of `torch.equal`: on another machine (torch 2.15, CPU) the outputs differ by about 3e-7. This is a portability fix to a test written the same day, not a weakening of the invariance tests. On the workstation the outputs are still bit-identical (checked 2026-11-23).

### 2026-11-24 - D-017: phase-2 external pipeline: preprocessing, channel rule, labels, splits (fixed before any external evaluation)
Owner's instruction of 2026-11-23 ("Part 5"). Written before any model has been evaluated on HMC, DOD-H, ISRUC or CAP; no external score exists. Development-grade; nothing here changes a phase-1 choice.
- **Preprocessing (one function, `src/mist/data/resample.py`, all four datasets, HMC included):** (1) zero-phase FIR low-pass at the native rate, passband edge 45 Hz, transition 5 Hz (-6 dB at 47.5 Hz, about -50 dB from 50 Hz; Hamming, firwin, MNE 'auto' length), applied even at a native 100 Hz; (2) `scipy.signal.resample_poly` to 100 Hz; (3) the Sleep-EDF store's 0.3-35 Hz zero-phase FIR band-pass. 45 Hz keeps the 0.3-35 Hz band untouched by step 1 and puts the anti-alias stopband at the output Nyquist. HMC previously resampled with MNE's default; it now uses this path (its test passes unchanged). VERIFIED on synthetic signals (tests/test_resample.py: 10 Hz amplitude within 1% and no lag at 100-512 Hz input; 60 Hz at 200/256 Hz removed by > 46 dB).
- **Channel rule (per recording, never by score):** first available of (1) frontal-central Fz-Cz, F4-C4, F3-C3; (2) frontal-mastoid F4-M1, F3-M2, then ipsilateral F4-M2, F3-M1; (3) central-mastoid C4-M1, C3-M2 (contralateral only). Contralateral is preferred because it is the AASM recommended form (F4-M1, backup F3-M2) and gives the larger inter-electrode distance; within a tier the midline comes first (nearest Sleep-EDF's Fpz-Cz), then the right side (AASM primary). A1/A2 are treated as M1/M2: the files do not say earlobe or mastoid (CONDITIONAL). The chosen channel is `rule_channel` in each meta.json; recordings with no rule channel are skipped. Every identifiable EEG channel is still stored.
- **Labels:** W/N1/N2/N3/REM = 0..4, unscored -1. ISRUC: scorer 1 (`<n>_1.txt`; 5 = REM). CAP: R&K S3 and S4 -> N3, MT and non-stages -> -1. DOD-H: h5 `hypnogram` codes taken as 0..4 = W..REM and -1 unscored (CONDITIONAL: no code table in the files); whether it is a consensus or one scorer is UNKNOWN from the files. HMC: unchanged (AASM annotations).
- **Splits:** by person (one recording per person in all four datasets as stored), 50/10/40 train/val/test, seed 20261124, stratified by CAP diagnosis group, ISRUC group (I vs III), HMC EEG prefilter field, DOD-H none; `scripts/make_external_splits.py`, never overwritten. Subject ids follow splits/sleepedf_v1.json, which already lists dataset-qualified ids ("SC:07"): "HMC:001", "ISRUC1:017", "ISRUC3:002", "DODH:<file uuid>", "CAP:<record>".
- **Holdout:** the 40% test part of each dataset is an external holdout. `load_external(..., part="test")` raises unless `open_holdout=True`; test folders are not opened otherwise. `scripts/evaluate.py --dataset <d> --part dev|train|val` cannot request it.
- **Reversible:** yes, until the first external evaluation; after that a change needs a new entry and new file names (`*_v2`).

### 2026-11-24 - D-018: phase-2 stores built; data findings and judgement calls for the lead
Stores built on CPU with scripts/prepare_external.py; splits made with scripts/make_external_splits.py. No model has been evaluated on any external dataset. Aggregates are in reports/external_audit_<dataset>.md. Development-grade; no phase-1 choice depends on any of this.
- **Counts (VERIFIED):** HMC 151/151 recordings, 137248 epochs; DOD-H 25/25, 24665; ISRUC 107 of 110 subject folders (group I subjects 4, 6, 39 have no .rec), 96316; CAP 107/108, 119425 (12156 unscored). Splits train/val/test: HMC 76/15/60, DOD-H 13/2/10, ISRUC 53/11/43, CAP 53/10/44.
- **Channel rule outcome (VERIFIED):** HMC F4-M1 151; DOD-H F4-M1 25; ISRUC F4-M1 81, F4-A1 25, C4-A1 1; CAP F4-C4 100, F4-A1 3, C3-A2 3, C4-A1 1.
- **CAP bug found and fixed before any split or evaluation:** the first alignment code wrapped every hypnogram row to within 12 h of the EDF start, so rows of recordings longer than 12 h were folded back (8 recordings refused, up to 10 rows silently lost in others). Now only the first row is wrapped; a test covers a 14-h recording; the whole CAP store was rebuilt with --overwrite.
- **CAP refusal rule:** a recording with more than 10 stage rows before the EDF start is refused (clocks disagree). Only nfle27 meets it (hypnogram starts 48 min before the EDF, ends 2.7 h before it). One other recording loses one leading row.
- **CAP channel handling:** nfle20 has EEG at 128 and 256 Hz; channels are now preprocessed per rate group (each through the same chain) and cut to a common length. EEG-labelled channels with a hardware high-pass >= 5 Hz (F7-T3/F8-T4 in 22 recordings, T3-T5/T4-T6 in 1) are EMG-like and not kept. Two recordings with monopolar leads get the usual bipolar pairs by subtraction.
- **Units:** HMC, ISRUC uV in all headers. CAP: uV 102, mV 3 (scaled), empty 3 (n13-n15; physical value taken as uV; CONDITIONAL, their rule-channel epoch SD of about 18-25 uV fits uV). DOD-H: attribute "mV", numbers uV-scale, used as uV (CONDITIONAL). CAP saturation check (scratch, not in the audit; 102 of 107 rule channels): at most 0.11% of samples at the observed extreme, so no meaningful clipping despite +-62.5 uV header ranges on 86 rule channels.
- **ISRUC amplitude anomaly (for the lead/owner; not fixed):** the 26 recordings whose rule channel has a +-25 uV header range (the A1/A2-labelled ones) have a median epoch SD of 1.5 uV against about 17 uV for the +-2490/+-5013 uV ones. A header scaling error is suspected but not verified, and the factor (about 11) is not a clean power of ten. Values are stored exactly as the headers state; no rescaling or exclusion was applied. Gain-normalised models are less affected than `raw`. Whether to rescale, exclude or report separately is the owner's call (proposed open question).
- **HMC:** two recordings have a rule-channel median epoch SD above 150 uV (artefact-heavy or poor contact; not verified). Kept.
- **Labels:** ISRUC scorer 1 label counts equal the signal length for all 107 (scorer 2 differs by 7 epochs in one recording; not used). DOD-H hypnogram source (consensus or single scorer): UNKNOWN. One recording per person: HMC (dataset description, hmc.py), CAP, DOD-H and ISRUC groups I/III are CONDITIONAL on the database descriptions; nothing in the files links recordings.
- **Practical note:** ISRUC channel names keep the source electrodes (F4-A1 vs F4-M1), so `evaluate.py --channel F4-M1` on ISRUC fails for the A-labelled recordings; the default `rule` channel works for all.

### 2026-11-24 - D-019: ISRUC amplitude (owner's Q9 decision) and correction to D-018
- **Owner's decision, before any external evaluation:** keep the 26 low-amplitude ISRUC recordings as stored, with no rescaling and no exclusion.
- **Reporting rule:** every ISRUC table shows three rows: all, normal-range (81) and low-amplitude (26).
  - For comparisons between gain-normalised arms, "all" is primary.
  - For any comparison involving a `*_absolute` arm (H3), the normal-range row is primary and the low-amplitude row is descriptive only, not evidence for H3.
- **Correction to D-018** (append-only; the D-018 text stays as written): D-018 says gain-normalised models are less affected than `raw`. That is wrong.
  - `raw` is gain-normalised: `xraw = x / per-night robust std` (mist_grm.py).
  - The attnsleep baseline divides its input by the same per-night robust std (build.py, EpochwiseBaseline). This is per-recording scaling without mean removal, not a z-score.
  - The arms exposed to the anomaly are the `*_absolute` ones (`x / 50`).
  - Checked 2026-11-24 on a synthetic night, with the input and its night statistics scaled by 1/11. Maximum relative change of the logits: `raw` 8e-7, `cascade2` 3e-7, `cascade2_absolute` 7.5e-2, `grm_absolute` 3.0e-1 (VERIFIED, synthetic, untrained weights).
- **Pure-scale diagnostic (owner's item 4): not a pure scale** (reports/external_audit_isruc.md, last section).
  - After per-night scaling, the low subgroup has more 2-8 Hz power than the normal subgroup, beyond the normal between-person spread, and this holds within group I.
  - The non-EEG scale factors differ: about 11 for EEG, EOG and EMG, 33 for ECG.
  - The headers show a different acquisition set-up: A1/A2 labels and per-channel-type ranges.
  - **By the owner's rule, ISRUC is not evaluated.** HMC, DOD-H and CAP proceed after session 1.
- **Split:** splits/isruc_v2.json, stratified by group and amplitude subgroup, same seed (20261124) and proportions: 54 / 11 / 42 people. Strata: group I normal 41/8/32, group I low 8/2/6, group III low 5/1/4. It is the default from now on (`external.SPLIT_VERSION`); splits/isruc_v1.json stays in place, unused. Made before any external evaluation.
- **DOD-H is a target only,** never a training source (owner: 2 validation people are too few).

### 2026-11-24 - D-020: ISRUC sub-domains (owner closes Q9, before any external evaluation)
1. **Two sub-domains, named by reference label.** ISRUC is evaluated as ISRUC-M (81 recordings, M1/M2 labels, normal range) and ISRUC-A (26, A1/A2 labels, +/-25 uV header range).
   - Every ISRUC table shows ISRUC-M, ISRUC-A and "all", in that order.
   - Claims are made per sub-domain. The "all" row is shown only for comparison with papers that pool ISRUC; no claim rests on it alone.
   - Code: `external.isruc_subdomain`, `external.ISRUC_ROWS`.
2. **`*_absolute` arms are evaluated on ISRUC-M only.** On ISRUC-A the calibration is unknown, so their scores there are descriptive and not evidence for or against H3. This replaces the D-019 reporting rule for these arms.
3. **Group III is entirely inside ISRUC-A.** Any statement about group III says it is confounded with the recording set-up. No healthy-versus-patient claim is made from ISRUC.
4. **Wording in reports:** the ISRUC-A difference is "a different recording set-up or a different population; not separable from the files" (CONDITIONAL).
5. **Open (OPEN_QUESTIONS Q10):** whether ISRUC as a training source uses ISRUC-M only or both. To be decided before any ISRUC-trained model is run.
6. **attnsleep docstring corrected:** its input is divided by the night's robust standard deviation without mean removal; it is not z-scored.

The external dev evaluation of the E0 runs goes ahead on all four datasets after session 1 ends. D-019's "ISRUC is not evaluated" is lifted by this decision.

### 2026-11-24 - D-021: overnight rules built into the session queue; queue restarted with the same end time
- **Came up:** the owner's overnight message (2026-11-24 01:10) sets rules the running queue did not implement:
  - B3: retry any failed run once; a second failure skips that fold for both arms.
  - B5: a GPU or driver fault waits 5 minutes and retries once; if it persists, GPU work stops for the night.
  - B1: after a second gate failure, P9 is followed by the `raw` runs of P5.
- **Done (B4: fix on CPU, add a test, run pytest, commit, restart with the SAME end time):**
  - scripts/session_queue.py now retries any failed training once: out of memory with encoder checkpointing (as before), a GPU or driver fault after a 5-minute wait, anything else as is.
  - A second failure records every run of the unit in runs/session_skips.json and the walk continues; a second GPU fault stops GPU work, and no pending evaluations run.
  - A failed evaluation is retried once.
  - scripts/aggregate.py drops skipped runs, so a skipped fold is missing for every arm.
  - experiments/priority.txt: the second-failure branch now ends with the six `raw` age-shift runs. The main order is unchanged.
  - Tests: 5 new (dry-run failure simulation); full suite 122 passed.
- **Restart:** the old queue was stopped without touching the running training or evaluation process. The new queue started only after that process had ended (otherwise it would treat the folder as partial), with the same end time, 2026-11-24 12:19 (deadline 12:04). Completed units are skipped automatically.
- **Reversible:** yes.

### 2026-11-24 - D-022: H1 mixed model and E3 aggregation (code only, no results)
- **H1 mixed model:** EXPERIMENT_PLAN section 7 specifies "macro-F1 ~ arm with random intercepts for protocol, fold and seed".
  - Implemented in-house as a REML fit with crossed random intercepts (src/mist/mixed.py, numpy/scipy). `statsmodels` is not a dependency, and the overnight limits forbid new downloads.
  - Unit: run-level macro-F1 on the real cell, one row per (arm, protocol, fold, seed), kept only where both arms exist. Protocols: in-domain, Pz-Oz, sc2st, and both age directions.
  - Interval: normal 95% from the GLS covariance, approximate with so few levels per factor; stated in the output.
  - Tests: the effect is unbiased with about 95% coverage on simulated data, and the fit equals OLS when there is no grouping variance.
  - `aggregate.py h1` labels output INTERIM, with the seeds covered. `--confirmatory` refuses unless 3 seeds x 5 folds of both arms and the age block are complete, and refuses if reports/h1_confirmatory.json exists (owner rule B7: run once).
- **E3 band x stage:** `aggregate.py e3 --arm cascade2` gives the cost of each intervention cell (real minus intervened, paired subject bootstrap) per protocol, from the P6 files `eval_e3_*.json`.
- Nothing was run on real results.

### 2026-11-24 - D-023: T14 SleepBand baseline is an independent re-implementation from the paper
- **Checked first (per T14):** SleepBand (Lu, Hu, Chen; arXiv:2607.04851, v1, 6 Jul 2026). The abstract says "Code is available at https://github.com/lzcn/sleep-band"; the conclusion says "Code will be released". The repository exists but states no licence (VERIFIED from the repository page, 2026-11-24).
- **Chosen:** re-implement the architecture from the paper text only, without cloning or reading the repository code. Copying unlicensed code is not allowed, and the overnight limits forbid new downloads. Licence clarification would need contact with the authors (reserved; OPEN_QUESTIONS Q11).
- **Scope tonight: code and tests only, no training.** In scope: the model (learnable constant-Q Gabor filterbank, magnitude responses, spectral integration with per-band temporal convolution, mixing and gates, epoch encoder, one-layer sequence attention).
- **Not in scope:** the paper's training scheme (mean-teacher consistency on bandwise Fourier perturbations, 50 epochs, Adam 1e-4 weight decay) differs from the frozen phase-1 recipe. Whether a phase-2 baseline uses its own published recipe or ours is a phase-2 protocol choice, made before any baseline is trained; it is not decided here.
- Details the paper leaves open are listed in the module docstring (src/mist/models/sleepband.py). Parameters with those choices: 3,268,130 (the paper gives no absolute count to check against; UNKNOWN). Tests: tests/test_sleepband.py (filter selectivity, constant-Q width, scale invariance, gradients).

### 2026-11-24 - D-024: plan correction, H1 primary statistic (owner)
- **Fault (owner):** EXPERIMENT_PLAN section 7 named a run-level mixed model as the pooled H1 analysis. That contradicts its own first rule (the unit of analysis is the test subject): runs of different seeds and folds re-test the same people, so run-level rows are not independent. D-022 implemented the plan as written.
- **Corrected section 7:**
  - Primary H1 statistic. Protocols: the four shifts only (montage, sc2st, young2old, old2young); in-domain is reported beside them, not pooled.
  - Per protocol: `cascade2` - `cascade2_placebo` by the paired subject bootstrap, with subject confusion matrices summed over folds and seeds, on balanced (fold, seed) pairs.
  - Pooled: the equal-weight mean of the four per-protocol macro-F1 differences.
  - Interval: one joint bootstrap, 10,000 resamples, over people in two strata (the 54 SC development people, the 22 ST people). Every resample recomputes all four effects, so a person tested in several protocols moves together. 95% percentile interval; the two-sided bootstrap p feeds Holm.
  - Decision rule unchanged: pooled gain >= 2 points and the interval excludes 0.
  - The mixed model stays as a secondary analysis, shift protocols only, described as "variability over seeds and folds on the same people". No claim rests on it.
  - The run-once lock (reports/h1_confirmatory.json) and the completeness check (3 seeds x 5 folds of both arms plus P5) moved to the primary statistic.
  - Interim tables use the same statistics, labelled INTERIM with the seeds covered.
- **Code:**
  - `mist.aggregate.joint_shift_bootstrap`.
  - `scripts/aggregate.py h1 [--seeds]` writes reports/h1_interim[_sS].md/.json.
  - `scripts/session_report.py` now shows the H1 primary statistic per seed set instead of the mixed model.
  - EXPERIMENT_PLAN section 12.2 had pooled over "the protocols actually run" including in-domain; corrected to match.
- **Tests:**
  - tests/test_h1_bootstrap.py, simulated people with a lasting ability and a lasting response to the arm, re-tested by 3 seeds. The known population effect is recovered (estimate within 1 point); coverage of the pooled 95% interval was 95.8% over 120 replicates (test runs 300 replicates and requires 92-98%).
  - tests/test_aggregate.py covers the interim, complete and run-once paths and balanced pairing.
- **Was any H1 number computed or read before this change? No.**
  - No H1, E1 or G2 statistic has been computed on real runs. `aggregate.py e1`/`h1` and `session_report.py` were run only on synthetic data (tests; one synthetic dry run in the scratchpad, 2026-11-23).
  - The session queue log records only ok/failed and minutes. One cascade2 test-evaluation file (e1_cascade2_cv0_s0, sc2st) was opened today only to list its subject identifiers and cell names, to write this code. No score in it was read.
  - The only cascade2 numbers seen are validation scores (timing run, the gate). Those are allowed.
- The session queue was not affected and was not restarted.
- **Reversible:** yes (plan text and code).

### 2026-11-24 - D-025: E1 pairs balanced on common runs; session-1 report fixes (bug fixes, B4)
- **Found while reading the first draft of reports/session_1.md:** `aggregate.py e1` paired arms whose run sets differed (e.g. 15 cascade2 runs against 3 cascade2_absolute runs on sc2st, where the subject sets match because every fold model tests all ST people). Those rows mixed seeds and folds unevenly.
- **Fix:** every pair now uses only the (fold, seed) runs both arms have; the table shows runs per arm. Test: tests/test_aggregate.py::test_e1_pairs_use_only_common_runs. The H1 and G2 rows were already balanced (15 vs 15) and did not change; the H1 primary statistic (D-024) pairs this way by construction.
- **Also fixed:**
  - The report missed the run that finished across the 01:24 queue restart (e1_cascade2_cv1_s0).
  - It now prints the gate per fold, and lists only failures, restarts, checks and roll-overs instead of every successful evaluation.
  - A CPU job chain waited on `pgrep -f 'view lf'`, which matched its own command line, so the external evaluations started about 4 minutes late, by hand.
- **B8 leakage check** on the cascade arms' large sc2st gains over raw: none found. Details are in reports/session_1_notes.md.
- **Reversible:** yes.

### 2026-11-24 - D-026: owner's revision 4 (reframing D9, raw_whitened, session 2 order); what the revision-3 gate missed
- **Reframing (DECISIONS D9, owner's sign-off):** research questions re-aimed at transfer. **This reframing was made after the interim H1 result of session 1 was seen** (cascade2 - cascade2_placebo -1.9 [-2.4, -1.4] under shift, seeds 0-2, montage and sc2st only, INTERIM).
  - The original questions are kept as history (EXPERIMENT_PLAN section 1, README).
  - H1 to H6 are unchanged. The confirmatory H1 keeps its definition and is reported whatever it shows.
  - H3 becomes the central claim to test.
  - Rule 3 unchanged. Docs: README, CLAUDE.md, EXPERIMENT_PLAN section 13.
- **Recorded as instructed:** on seed 0 the own-rate cascade2 is below the full-rate cascade under shift: Pz-Oz -1.8 [-3.3, -0.6], sc2st -1.1 [-1.5, -0.7] (reports/e1_interim_s0.md, balanced pairs).
  - The revision-3 gate compared in-domain validation scores only and could not detect this.
  - "Morphology hurts transfer" is not to be claimed from cascade2 alone (ARCHITECTURE section 9.5).
  - **EXPLORATORY** (reports/session_1.md, under the H1 tables): full-rate cascade - cascade2_placebo, seed 0, on the folds both have. Results: in-domain +0.2 [-0.3, +0.9], Pz-Oz -0.1 [-0.8, +0.6], sc2st -0.4 [-1.0, +0.3] (4 folds until the cascade fold-4 sc2st evaluation, running on CPU, lands).
- **New control arm `raw_whitened` (EXPLORATORY everywhere; added after interim results):**
  - One switch, `model.raw_input`: `scaled` (default, the `raw` arm) or `whitened` (tokens from the night-whitened signal). It is accepted only for the X stream of the parallel topology with night_relative.
  - The default leaves `raw` exactly as it was:
    - The switch adds no parameter, so existing checkpoints load strictly.
    - e0_raw_cv0_s0 re-evaluated on CPU with the old code and with the new code gave identical night confusion matrices (VERIFIED).
    - The stored GPU evaluation differs by 1 of 28,989 epochs, which is CPU-vs-GPU arithmetic and not the switch.
  - Tests (tests/test_raw_whitened.py): default unchanged; the whitened arm differs from raw with the same weights; both arms gain-invariant under recomputed night statistics; invalid combinations refused; same recipe and parameter count (568,423).
- **Session 2 order (experiments/priority.txt from `scripts/make_priority.py --session 2`; the session-1 list kept as experiments/priority_session1.txt and reproduced identically by `--session 1`):** S2-1 to S2-6 as in EXPERIMENT_PLAN section 13.4. Decisions inside the owner's order:
  - `--cells e3` now gives exactly the owner's cells: knock G, R, M; destroy M; the five band cells and broad. The three E0 input surrogates are dropped. No E3 file existed, so nothing is lost.
  - New `--cells destroyM` gives the destroy-M cell only.
  - **Full-rate cascade destroy-M goes to CPU.** Its earlier GPU evaluations took about 9 min per run for the three protocols (file times of e1_cascade_cv3_s0), so 5 runs come to about 45 min. That is not clearly under the owner's 45-minute limit.
  - **Age block:** per (seed, direction), raw first, then the balanced pair, so every stopping point leaves balanced comparisons.
- **Queue estimates fixed** (scripts/session_queue.py; no change to the fit rule):
  - E3 and destroy-M evaluations are estimated as cells x the 90th percentile of the arm's measured core evaluation, until measured. Before this fix they took the 600 s prior each, 7.5 h for S2-1.
  - Unmeasured arms' core evaluations are estimated at 3x the slowest measured arm's 90th percentile, instead of 300 s.
  - CPU dry run of session 2 at measured speeds: S2-1 to S2-5 fit (S2-1 about 1.3 h); S2-6 does not start (raw young2old misses by a few minutes).
- **Reversible:** yes. The reframing is the owner's.
- **Addendum (same day, after the e1_cascade_cv4_s0 sc2st evaluation finished on CPU):**
  - The owner's sc2st figure for cascade2 - cascade (-1.1 [-1.5, -0.7]) was on 4 folds. On all 5 folds it is -1.0 [-1.3, -0.6]; Pz-Oz is unchanged (5 folds already).
  - EXPLORATORY full-rate cascade - cascade2_placebo, sc2st, on 5 folds: -0.6 [-1.2, +0.0]; in-domain and Pz-Oz as above.
- **External dev evaluation (D2) done:**
  - 81 CPU evaluations, no failures; reports/external_dev.md.
  - Person counts match the splits (HMC 91, DOD-H 15, CAP 63, ISRUC v2 65 = 49 ISRUC-M + 16 ISRUC-A). No holdout person was evaluated (checked against the split files).
- **Addition (owner, 2026-11-24, after session 2 started; CPU only):** full-rate cascade seed-0 fold models on the external development parts (real cell, D-017 to D-020 rules), pair `cascade - cascade2_placebo` per dataset, EXPLORATORY.
  - **5% rule:** a guard waits for 8 epochs of the first S2-2 training (`e1_raw_whitened_cv0_s0`), takes the mean epoch time of epochs 3-8 as baseline, starts the 20 evaluations (CPU, nice 19, 2 parallel, 4 threads each), and compares the mean of the next 6 epochs. Over 5% slower: it stops them and they run after the session. Log: runs/logs/cascade_ext_guard.log.
  - **Session-2 report line:** `scripts/external_report.py` writes one line (reports/external_dev_shape_line.txt). Rule fixed before any cascade external number was read: per dataset row (ISRUC-M and ISRUC-A separately, not "all"), macro-F1 difference is negative if the 95% interval lies below 0, positive if above, else neutral. `scripts/session_report.py 2` adds it after sections a) to f).
- **Refinement to the report-line rule (owner, 2026-11-24, 15:58), made before any full-rate cascade external number existed** (checked at the time: 0 eval_test_ext files in runs/e1_cascade_cv*_s0; the guard had not yet started the evaluations). Per dataset row (ISRUC-M and ISRUC-A separate), from the 95% interval of the macro-F1 difference cascade - cascade2_placebo:
  - negative: interval entirely below 0;
  - positive: interval entirely above 0;
  - neutral: interval includes 0 AND lies entirely within -2 to +2 points;
  - inconclusive: interval includes 0 but reaches beyond -2 or +2.
  - The interval and the number of people are always printed beside the label. Replaces the three-way rule above. Code: shape_label in scripts/external_report.py; test_shape_label.
