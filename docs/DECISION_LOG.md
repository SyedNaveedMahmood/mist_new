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
