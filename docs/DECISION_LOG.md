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
