# Experiment plan

Phase 1 runs entirely on Sleep-EDF Expanded. Phase 2 repeats the decisive comparisons on independent datasets. Nothing in phase 2 may be used to tune phase 1 choices.

## 1. Questions and hypotheses

*Research questions as first written (history). Since 2026-11-24 the questions are those of section 13 (DECISIONS D9), adopted after the interim H1 result was seen; the hypotheses below are unchanged.*

**RQ1.** Can a model that understands EEG morphology transfer better?
**RQ2.** Which morphology matters, and does better morphology for N1 raise every stage?

Operational definitions, fixed before any run:

- **Morphology** = what the phase spectrum of a short token adds to its magnitude spectrum. In the model it is the M stream. A model "understands morphology" to the degree that its predictions change when M is destroyed.
- **Morphology reliance (MR)** = macro-F1 with real input minus macro-F1 when morphology is destroyed (`destroy_M` inside MIST-GRM; `local_phase` surrogate for any model). Reported per stage.
- **Transfer** in phase 1 = three shifts available inside Sleep-EDF (section 3). They are within-database shifts and must be called that.
- **Normalised gap** = (oracle - transfer) / oracle, same architecture, same test subjects.

| ID | Hypothesis | Decisive comparison | Supported when | Refuted when |
| --- | --- | --- | --- | --- |
| H0 (gate) | Trained stagers use morphology at all | MR of `raw` and `attnsleep`, per stage | MR >= 1 macro-F1 point for at least one stage, interval excluding 0 | MR < 1 for every stage |
| H1 | Access to morphology improves transfer | `grm` vs `gr` under each shift | Pooled gain >= 2 macro-F1 points, interval excluding 0, and `grm_placebo` does not show it | Gain < 2, or the placebo matches it |
| H2 | More morphology, more transfer (dose) | macro-F1 under shift against `destroy_prob` and `critic_weight` | Monotone trend over the dose levels | Flat or reversed |
| H3 | The factorisation itself helps transfer | `gr` vs `raw`; `grm` vs `grm_absolute` | Each >= 1 point under shift, in-domain loss < 1 point | No gain under shift |
| H4 | Factors differ in transfer value by stage | Stream-by-stage reliance matrix, shift minus in-domain | At least one cell differs by >= 2 F1 points, same sign in two of three shifts | No cell does |
| H5 | Morphology raises N1 and the other four stages | `grm` vs `gr`: N1 F1 and summed F1 of W, N2, N3, REM | Both differences have intervals above 0 | Non-N1 sum is flat or negative |
| H6 | The lift is morphology, not attention to N1 | `grm` vs `gr` + N1 loss weight matched on N1 F1 | Non-N1 gain of `grm` exceeds the re-weighted arm | The re-weighted arm matches it |

Priors, stated so they can be checked later (speculation): H0 passes weakly, driven by N2 and N3. H3 is the most likely to hold. H1 is close to a coin flip, and the all-pass sensitivity of M (docs/ARCHITECTURE.md, section 3) could make it negative on the cohort shift. H5 is the least likely: N1 gains usually cost Wake or N2 precision.

## 2. Data

Sleep-EDF Expanded 1.0.0: cassette (SC, home recordings, healthy, ages 25 to 101) and telemetry (ST, hospital, mild difficulty falling asleep, temazepam and placebo nights). Channels Fpz-Cz and Pz-Oz at 100 Hz. Scoring is Rechtschaffen and Kales; stages 3 and 4 merge to N3; movement time and unscored are -1.

Carried over from the earlier audit (docs/LESSONS_FROM_PROTO_SLEEP.md):

- Raw EDF only. The old trimmed NPZ folders are for smoke tests.
- SC persons 06, 23, 36, 74 are quarantined. Eligible SC: 74 people, 146 recordings.
- Every complete physical epoch is stored; views are masks.

Preprocessing: zero-phase FIR band-pass 0.3 to 35 Hz, no resampling (already 100 Hz), microvolt units, 30 s epochs. No amplitude normalisation on disk; models normalise against the night reference.

## 3. Splits and shift protocols

Frozen once by `scripts/make_splits.py` (seed 20261122) into `splits/sleepedf_v1.json`.

| Role | Who | Use |
| --- | --- | --- |
| Lockbox | 20 SC people drawn, age-stratified, from the 55 never used in earlier development | One final confirmation run (E6) |
| Dev | The other 54 eligible SC people (19 historical + 35 newly released) | 5 age-balanced folds; all development |
| ST | All 22 ST people | Cohort-shift target, development grade (already exposed in earlier work) |

This releases 35 previously reserved people. It is decision D1 in docs/DECISIONS.md (confirmed by the owner 2026-11-22; splits frozen in T2).

**Lockbox age range.** The frozen lockbox covers ages 50 to 101 only, because everyone under 50 in SC (persons 00 to 19, ages 25 to 34) was used in earlier development and is not eligible. E6 therefore confirms results on older adults only; say so wherever E6 is reported.

| Protocol | Train / validate | Test | Shift | Oracle for the gap |
| --- | --- | --- | --- | --- |
| `cv` | 3 folds / 1 fold, Fpz-Cz | 1 fold, Fpz-Cz | none (in-domain) | itself |
| montage | as `cv`, Fpz-Cz | same test fold, **Pz-Oz** | derivation | `cv` model trained on Pz-Oz |
| montage, reverse | as `cv`, Pz-Oz | same test fold, Fpz-Cz | derivation | `cv` on Fpz-Cz |
| `young2old` | youngest age third | oldest age third | age | `cv` restricted to the old third |
| `old2young` | oldest third | youngest third | age | `cv` restricted to the young third |
| `sc2st` | SC dev | all ST, Fpz-Cz | cohort, hardware, setting, drug | none (ST is too small to train on); report raw scores |

Montage needs no extra training: evaluate the `cv` checkpoints with `--channel Pz-Oz`. `sc2st` likewise: `--protocol sc2st`.

Caveats to print next to every result: all three shifts are inside one database; SC and ST person linkage is unknown; ST mixes drug and placebo nights (report them separately).

## 4. Evaluation window

- **Primary: `std` view.** Thirty minutes of wake kept either side of sleep. This is the convention in the Sleep-EDF literature. It uses stage labels to choose the window, so it is label-conditioned; the earlier project barred it for zero-shot claims.
- **Sensitivity: label-free window.** From 30 minutes before lights-off (SC-subjects.xls, ST-subjects.xls) for 10 hours. To be implemented in T5; every headline comparison is re-run in this window and both are reported.
- The night reference is computed inside whichever window is evaluated.

Why not `full`: SC recordings are about 20 hours and mostly daytime wake, which no other sleep dataset contains.

## 5. Experiments

Standard run: 5 folds x 3 seeds, 40 epochs maximum, early stopping on source-validation macro-F1, AdamW 1e-3, batch 32 sequences of 20 epochs. Checkpoint selection never sees test subjects or the target derivation.

### E0. Gate: do stagers use morphology? (H0)

Train `raw` and `attnsleep` under `cv`. Evaluate with `real`, `global_phase`, `local_phase`, `flatten_tilt`, in-domain and under each shift.

Output: MR per stage and model; a 4-condition x 5-stage table for each protocol.

Gate G1: if `local_phase` costs less than 1 macro-F1 point for every stage and every model, morphology is not being used. Then follow the fallback in section 8 instead of E1 as written.

### E1. Factor arms (H1, H3)

Arms: `raw`, `r`, `gr`, `grm_placebo`, `grm_nocritic`, `grm`, `m`, `grm_loose`, `grm_absolute`, plus `attnsleep`. All under `cv`; evaluate every checkpoint in-domain, on Pz-Oz and on ST. Train the five arms `raw`, `gr`, `grm_placebo`, `grm`, `grm_absolute` additionally under `young2old`, `old2young` and Pz-Oz `cv`.

Output: one table per protocol (macro-F1, kappa, per-stage F1, normalised gap), and paired subject-bootstrap differences for the decisive pairs in section 1.

### E2. Dose-response (H2)

One seed, 5 folds:

- `critic_weight` in {0, 0.1, 0.3, 1.0}
- `destroy_prob` in {0, 0.25, 0.5, 0.75, 1.0}, same value at train and test

Record, for each setting: critic accuracy on held-out tokens (a label-free measure of how much beyond-spectrum structure M has captured), in-domain macro-F1, macro-F1 under each shift. Plot shift macro-F1 against dose.

### E3. Which factor matters where (H4, RQ2a)

On the `grm` checkpoints from E1, evaluate `knock_G`, `knock_R`, `knock_M` and `destroy_M`, in-domain and under each shift. Build the stream x stage matrix of F1 change. Report shift minus in-domain: a stream whose removal hurts more under shift carries transferable information; one whose removal helps is dataset-specific.

Then, after T8 is implemented: the same for shape-slot families, with slots named after the fact by their agreement with relative-threshold detectors (and with expert events once a rights-cleared set is available).

### E4. N1 (H5, H6, RQ2b)

| Arm | Config |
| --- | --- |
| N0 | `gr` |
| N-morph | `grm` |
| N-placebo | `grm_placebo` |
| N-weight | `gr` with `train.n1_weight` tuned on validation so N1 F1 matches N-morph within 0.5 points |
| N-target (after T9) | `grm` with the critic applied only to tokens from Wake, N1 and REM epochs |

Readouts: per-stage F1 differences against N0; the summed non-N1 difference with a subject-bootstrap interval; confusion counts for Wake/N1, N1/N2, N1/REM; all repeated with `context_layers: 0`, because spill-over may travel through sequence context.

Across every run of E1, E2 and E4, regress the non-N1 F1 sum on N1 F1 with fold and protocol as random effects. A positive slope within arms is descriptive only; the causal reading rests on N-morph against N-weight.

### E5. Mechanism checks

- E5a `scripts/check_invariance.py` on real nights: change in G, R, M under gain, tilt and all-pass.
- E5b Per-stream domain probes: a linear classifier on each stream's token embeddings predicts derivation, age third or cohort, conditional on stage. Expectation under the design: M is the most identifiable for cohort (phase response), G and R the least.
- E5c Stage-conditional usage of each shape slot in source and target (Jensen-Shannon divergence).
- E5d Critic accuracy by stage: where does beyond-spectrum structure live?

### E6. Lockbox confirmation

After every choice above is frozen and written into `experiments/registry.yaml`: train `gr`, `grm_placebo` and `grm` on all dev folds (3 seeds), evaluate once on the lockbox (Fpz-Cz and Pz-Oz). Report whatever comes out.

## 6. Run budget (phase 1)

| Block | Trainings |
| --- | ---: |
| E0 | 2 arms x 5 folds x 3 seeds = 30 |
| E1 `cv` Fpz-Cz | 8 further arms x 15 = 120 |
| E1 age shift | 5 arms x 2 directions x 3 seeds (fold 0 as validation) = 30 |
| E1 Pz-Oz `cv` | 5 arms x 15 = 75 |
| E2 | 7 new settings x 5 folds = 35 |
| E4 | N-weight tuning and runs, context-free repeats: about 60 |
| E6 | 9 |
| Total | about 360 |

Time per run is unknown until T3 measures it on the workstation. If a run exceeds about 30 minutes, cut seeds to 2 for the non-decisive arms (`r`, `m`, `grm_loose`, `grm_nocritic`) before cutting folds.

**Superseded by section 12 (owner, 2026-11-23).** Kept for the record. **Measured 2026-11-23 (DECISION_LOG D-014).** Per run, alone on the RTX 4080 SUPER: `attnsleep` about 5 min, `raw` about 26 min, `grm` 20 min (30.5 s/epoch), `cascade` 2.44 h (219.6 s/epoch). The cut above therefore applies. E1 as run (scripts/e1_queue.sh):

| Block | Trainings | GPU hours (approx.) |
| --- | ---: | ---: |
| `cv` decisive, 3 seeds: cascade, cascade_placebo, cascade_absolute | 45 | 110 |
| `cv` decisive, 3 seeds: gr, grm_placebo, grm, grm_absolute | 60 | 21 |
| `cv` 2 seeds: grm_nocritic, cascade_nocritic | 20 | 28 |
| `cv` 1 seed: r, m, grm_loose, cascade_bands, cascade_broad | 25 | 27 |
| age shift, 3 seeds: raw, gr, grm_placebo, grm, grm_absolute, cascade, cascade_placebo | 42 | 40 |
| Pz-Oz `cv`, 3 seeds: same 7 arms | 105 | 99 |
| E1 total | 297 | about 324 (13.5 days) plus evaluation |

`raw` and `attnsleep` under `cv` are the E0 runs. E2 then runs on `cascade` with one seed (35 trainings, about 85 h).

## 7. Statistics

- Unit of analysis: the test subject. Nights of one person are pooled first.
- Pairwise claims: paired subject bootstrap, 10,000 resamples, 95% percentile interval (`scripts/compare.py`), on seed-averaged predictions.
- Pooled shift effect (H1), primary statistic (plan correction 2026-11-24, DECISION_LOG D-024; replaces the run-level mixed model, which contradicted the first rule above):
  - Protocols: the four shifts only, Fpz-Cz to Pz-Oz (montage), `sc2st`, `young2old`, `old2young`. In-domain is reported beside them, not pooled.
  - Per protocol: main arm minus its placebo (`cascade2` - `cascade2_placebo`) by the paired subject bootstrap above, each person's confusion matrices summed over folds and seeds. Only (fold, seed) runs present for both arms enter.
  - Pooled: the equal-weight mean of the four per-protocol macro-F1 differences. Interval: one joint bootstrap, 10,000 resamples, over people in two strata (the 54 SC development people; the 22 ST people). Each resample recomputes all four protocol effects with the same draw, so a person who is a test subject in several protocols (montage and an age shift) moves together. 95% percentile interval; the two-sided bootstrap p-value enters the Holm correction.
  - Decision rule unchanged: supported when the pooled gain is at least 2 macro-F1 points and the interval excludes 0.
  - Run once only (`scripts/aggregate.py h1 --confirmatory`, which writes `reports/h1_confirmatory.json` and refuses a second run), and only when 3 seeds x 5 folds of both arms and the age-shift block (P5) are complete. Before that every H1 number is INTERIM, with the seeds covered, and uses the same per-protocol and pooled statistics.
  - Tested on simulated people re-tested by several seeds: the known effect is recovered and coverage is near 95% (`tests/test_h1_bootstrap.py`).
- Secondary H1 analysis: the linear mixed model, run-level macro-F1 ~ arm with random intercepts for protocol, fold and seed, shift protocols only (`src/mist/mixed.py`). It describes variability over seeds and folds on the same people; its rows are not independent people, and no claim rests on it.
- Six confirmatory hypotheses; Holm correction across them. Everything else is labelled exploratory.
- Smallest effect of interest: 2 macro-F1 points for H1, 1 point for H3.
- With 54 dev people and about 11 test people per fold, intervals will be wide. A null result here is weak evidence; say so.

## 8. Gates and fallbacks

| Gate | Check | If it fails |
| --- | --- | --- |
| G0 data | Store totals equal the earlier audit: 74 SC people, 146 recordings, 397,832 epochs of which 396,378 scored; 44 ST recordings, 22 people | Stop. Fix the reader. |
| G0b baseline | `attnsleep` under `cv`, `std` view, reaches at least 0.70 macro-F1 | Stop. A pipeline fault is more likely than a weak model (the earlier adapter scored 0.31). |
| G1 reliance | E0 | Reframe: "what transfers is spectrum and calibration". Keep H3 as the main claim, M becomes the negative control, drop E2. |
| G2 cost | `grm` within 1.5 macro-F1 points of `raw` in-domain | Try `grm_loose`, then wider `morph_residual`. Report the trade-off. |
| G3 placebo | `grm_placebo` equals `gr` within noise | If the placebo beats `gr`, extra capacity is helping; shrink M or enlarge `gr` until matched. |

## 9. Phase 2: external transfer

Adopt the single-source protocol of SleepBand (train on one of Sleep-EDF, HMC, ISRUC, SHHS1, CinC 2018; test on the other four), single EEG channel. Arms: `raw`, `gr`, `grm_placebo`, `grm`, `attnsleep`, a SleepBand re-implementation, a magnitude/phase STFT model in the spirit of PhASER. Add leave-one-dataset-out. Morphology competence probes on expert spindle and K-complex annotations become possible here (MODA with MASS; DREAMS if rights are cleared). Access steps are in docs/DATASETS_AND_ACCESS.md.

## 10. Reporting rules

- Every number in a write-up traces to a `runs/<name>/eval_*.json` and an entry in `experiments/registry.yaml`.
- Labels: VERIFIED, CONDITIONAL, BLOCKED, UNKNOWN, NOT TESTED.
- Negative and null results are recorded in the registry with the same detail as positive ones.
- No "transfer" wording for anything inside Sleep-EDF without the qualifier "within-database".

## 11. Revision 2 (2026-11-22): what the cascade changes in this plan

The supervisor's cascade topology (docs/ARCHITECTURE.md, section 8) becomes the main model. No model with a morphology stream had been evaluated on any test set when this was written, so the changes below are made before the fact.

| Item | Before | Now |
| --- | --- | --- |
| H1 decisive pair | `grm` vs `gr`, with `grm_placebo` as control | `cascade` vs `cascade_placebo` (same network, phase real vs scrambled). `grm` vs `gr` is reported as the parallel-topology replicate |
| H2 dose | on `grm` | on `cascade` |
| H3 | `gr` vs `raw`; `grm` vs `grm_absolute` | `cascade_placebo` vs `raw`; `cascade` vs a cascade without the night reference |
| H4 / RQ2a | stream x stage matrix, then slot families named after the fact | band x stage matrix from `destroy_bands` (delta, theta, alpha, sigma, beta, broadband), in-domain and under each shift. Slot galleries become descriptive only |
| H5, H6 | `grm` vs `gr`, `gr` + N1 weight | `cascade` vs `cascade_placebo`, `cascade_placebo` + N1 weight |
| Gate G2 | `grm` within 1.5 points of `raw` | `cascade` within 1.5 points of `raw`; if it fails, compare with `grm` to see whether the routing is the cause |
| Gate G3 | placebo vs `gr` | not needed for the cascade (identical parameters by construction); kept for the parallel arms |
| New, exploratory | | `cascade` vs `grm` under each shift: does routing rhythm through morphology matter? Outside the Holm family |

E0 is unchanged: it uses `raw` and `attnsleep` only. The recipe stays frozen.

Extra trainings: `cascade`, `cascade_placebo`, `cascade_nocritic`, `cascade_bands`, `cascade_broad` under `cv` (5 x 15 = 75), plus `cascade` and `cascade_placebo` under the age protocols and Pz-Oz (2 x 21 = 42). To stay inside the budget, drop `r`, `m` and `grm_loose` to one seed first.

## 12. Revision 3 (2026-11-23): GPU sessions and the own-rate cascade

Owner's decision D8. It replaces the E1 table in section 6 and the D-014 queue. **No E1 test score had been read when this was fixed**; every change below is blind to E1 results.

### 12.1 Session rules

- The GPU is used only inside a session. A session starts when the owner writes SESSION START and lasts 12 hours unless the owner gives another end time. Every GPU process has ended 15 minutes before the end.
- `scripts/session_queue.sh <N> "<end>"` walks `experiments/priority.txt` (written by `scripts/make_priority.py`) in order:
  - **Fit rule.** A training starts only if the time left before the deadline (end minus 15 min) is at least 1.3 x its estimated duration, training plus its evaluations. Estimates are the largest measured time for that arm, from `runs/session_times.json`, else a conservative prior (the full-rate cascade time for `cascade2` until the timing run measures it).
  - **Hard stop.** Each process also runs under a timeout that expires at the deadline. A training stopped that way moves to `runs/_interrupted`.
  - **Pairs.** Pairs are interleaved by fold (`cascade2` fold f, then `cascade2_placebo` fold f) and start only if both fit.
  - **Roll-over.** The walk stops at the first unit that does not fit, so the order is kept. Pending evaluations then run while they fit, and the queue stops.
  - **Out of memory.** A training that fails with an out-of-memory error is retried once with encoder checkpointing (same arithmetic).
- Inside a session each cv run writes three evaluations, real-input cell only (`--cells core`): in-domain, Pz-Oz and sc2st, std view. The lf-view evaluations run on CPU outside sessions (`CUDA_VISIBLE_DEVICES=`). The other cells (surrogates, knock-outs, `destroy_bands`) are E3 (P6), written to `eval_e3_*.json`.
- All coding, tests and dry runs happen on CPU before a session (`scripts/session_queue.py --dry-run`). If `cascade2` is not ready when a session starts, the GPU runs P9 instead.
- After each session: `scripts/session_report.py <N>` writes `reports/session_N.md` (runs done, minutes per run, results tables with intervals, what comes next); STATUS updated; pushed.

### 12.2 Gates fixed before the session

- **Time:** the first `cascade2` run (fold 0, seed 0) is the timing run. Target: 30 minutes or less including its three evaluations. If slower, no redesign: report the number and let the queue fit fewer runs.
- **Accuracy, validation only:** after `cascade2` seed 0 has 5 folds, its mean best validation macro-F1 must be within 1.0 point of the full-rate cascade on the same folds. The reference is `e1_cascade_cv{0..4}_s0`, mean 0.7634. The gate is one-sided: it fails only if `cascade2` is more than 1.0 point below the reference; being above passes (owner's correction, 2026-11-23).
  - If it fails, set `band_width` to 32 (`cascade2w32`), rerun those 5 folds and check again.
  - If that also fails, stop the `cascade2` queue and report; the GPU then runs P9.
  - No test score is read for this decision (`scripts/gate_cascade2.py` reads only `train_summary.json`).
- G2 (section 11 rule): `cascade2` within 1.5 points of `raw`, in-domain. H1 = `cascade2` - `cascade2_placebo`, pooled over the four shift protocols by the statistic in section 7 (in-domain reported beside, not pooled; corrected 2026-11-24, D-024, before any H1 number was computed on real runs). The original "pooled over the protocols actually run" was fixed before any E1 test score was read. After P1, G2 and H1 are reported on seed 0 as INTERIM (`aggregate.py e1 --seeds 0`, `aggregate.py h1 --seeds 0`); H1 becomes confirmatory on 3 seeds (P3).

### 12.3 Priority list (walked in order; what does not fit rolls over)

| Session | Item | Content | Trainings |
| --- | --- | --- | ---: |
| 1 | P1 | `cascade2` and `cascade2_placebo`, cv, seed 0. Then the validation gate, and G2 and H1 on seed 0 (INTERIM) | 10 |
| 1 | P2 | the same two arms, seed 1 | 10 |
| 2 | P3 | the same two arms, seed 2; H1 confirmatory on 3 seeds | 10 |
| 2 | P4 | `cascade2_absolute`, cv, seed 0 (H3) | 5 |
| 2 | P5 | age shift, both directions: `raw`, `cascade2`, `cascade2_placebo`, 3 seeds. Then reports/e1.md | 18 |
| 3 | P6 | E3 on the P1 to P3 checkpoints, evaluation only: knock G, R, M and each band, in-domain and under each shift; band x stage table | 0 (90 evaluations) |
| 3 | P7 | E2 trimmed: `destroy_prob` 0.5, `critic_weight` 0 and 1.0, cv, seed 0 (0 / 1.0 and 0.3 exist as `cascade2` and `cascade2_placebo`) | 15 |
| 4 | P8 | E4: `cascade2_placebo` with `train.n1_weight` 1.5, 2.0 and 3.0, cv, seed 0. Take the weight whose validation N1 F1 (at the best-validation epoch, now logged) matches `cascade2` within 0.5 points; read test scores for that weight only | 15 |
| 4 | P9 | parallel model: `grm`, `gr`, `grm_placebo`, cv, seed 0 (`grm` fold 0 exists) | 14 |
| 5 | P10 | E4 with `context_layers: 0`: `cascade2`, `cascade2_placebo`, cv, seed 0 | 10 |
| 5 | P11 | `cascade2_absolute` seeds 1 and 2; N-target arm (`critic_stages: [0, 1, 4]`) if time remains | 10 + 5 |
| 6 | | freeze (T11). The lockbox stays the owner's to trigger | |

Session numbers say where an item is expected; the queue simply continues down the list, so early finishes pull work forward and late ones roll it over.

Deferred until the owner says so: the Pz-Oz-trained block, further seeds of the parallel arms, `grm_absolute`, `r`, `m`, `grm_loose`, `grm_nocritic`, `cascade2_bands`, `cascade2_broad`, the other E2 levels.

The five full-rate `cascade` seed-0 runs are kept as the reference for the validation gate and reported descriptively (`cascade2` - `cascade`). The pending evaluations from before the revision (`e1_cascade_cv4_s0` sc2st; `e1_grm_cv0_s0` in-domain, Pz-Oz and sc2st) run in the pending phase.

## 13. Revision 4 (2026-11-24): research questions re-aimed at transfer; session 2 order (DECISIONS D9)

**Made after the interim H1 result of session 1 was seen** (cascade2 - cascade2_placebo -1.9 [-2.4, -1.4] macro-F1 under shift, seeds 0-2, montage and sc2st only; INTERIM). The original questions stay in section 1 as history.

### 13.1 Questions
- **RQ1.** What in single-channel sleep EEG carries over to new recordings: gain, rhythm or waveform shape?
- **RQ2.** Does a model restricted to the part that carries over beat standard models on unseen datasets?
- H1 to H6 stay as pre-specified (section 1, statistics in section 7). H1 (shape improves transfer) is one tested factor, not the paper's premise. **H3 (the factorisation without shape helps transfer) is the central claim to test.**
- The confirmatory H1 keeps its definition (section 7) and is reported whatever it shows. It needs the complete age block and runs in session 3.
- Rule 3 is unchanged: no development-target rule exists, so no architecture, band setting or hyperparameter is chosen from any shift, test or external score. No arms beyond `raw_whitened`; no design variants.

### 13.2 What session 1 showed that the gate could not
On seed 0 the own-rate `cascade2` is below the full-rate `cascade` under shift: Pz-Oz -1.8 [-3.3, -0.6], sc2st -1.1 [-1.5, -0.7] macro-F1 (in-domain -0.4 [-0.8, +0.1]). The revision-3 validation gate compared in-domain validation scores only and could not detect this. "Morphology hurts transfer" is not to be claimed from `cascade2` alone. EXPLORATORY check: full-rate `cascade` - `cascade2_placebo` on the folds both have (reports/session_1.md).

### 13.3 New control arm `raw_whitened` (EXPLORATORY; added after interim results)
The `raw` model (streams [X], same parameters, same recipe) with its tokens taken from the night-whitened signal (`model.raw_input: whitened`) instead of the scale-only signal. The default (`raw_input: scaled`) leaves `raw` exactly as it was: existing checkpoints load strictly and give identical outputs (tests/test_raw_whitened.py; re-evaluation of e0_raw_cv0_s0 on CPU identical before and after the change). The arm is gain-invariant (tested). It separates how much of the transfer gain of the factorised arms is night whitening alone (`raw_whitened` - `raw`) from what the gain/rhythm split adds on top (`cascade2_placebo` - `raw_whitened`).

### 13.4 Session 2 order (experiments/priority.txt, `scripts/make_priority.py --session 2`)
| Step | Content |
| --- | --- |
| S2-1 | P6, evaluation only: E3 cells (knock G, R, M; destroy M; six band cells) on `cascade2` seeds 0-2; in-domain, Pz-Oz, sc2st (45 evaluations). Full-rate `cascade` seed 0: destroy-M cell only; estimated at about 45 GPU minutes, not clearly under the owner's 45-minute limit, so it runs on CPU |
| S2-2 | `raw_whitened`, cv, seed 0 (5 runs), three standard evaluations |
| S2-3 | `cascade2_absolute` folds 3-4, seed 0 |
| S2-4 | P9: `grm` folds 1-4, `gr` and `grm_placebo` 5 folds, seed 0 |
| S2-5 | From P7: `cascade2` with `critic_weight` 0, cv, seed 0 |
| S2-6 | Age block (P5) as far as it fits; per (seed, direction) `raw` first, then the balanced `cascade2` / `cascade2_placebo` pair |

All overnight rules hold (retry once, balanced skips, GPU only inside the session, hard stop 15 minutes before the end). CPU dry run at the measured speeds: S2-1 to S2-5 fit; S2-6 does not start. After the session, on CPU: external dev evaluation of `raw_whitened` seed 0, added to reports/external_dev.md.

### 13.5 Session 2 report, in this order
a) `raw_whitened` - `raw` and `cascade2_placebo` - `raw_whitened`, per protocol.
b) Destroy-M cell against real, per protocol, for `cascade2` (does scrambling phase at test time bring it up to `cascade2_placebo` under shift?) and for the full-rate `cascade`.
c) Band-by-stage table, shift minus in-domain, with Wake and N1 called out.
d) `grm` - `grm_placebo` and `grm` - `gr`, per protocol (parallel replicate of H1).
e) `cascade2` with `critic_weight` 0 - `cascade2_placebo`.
f) `cascade2` - `cascade2_absolute` on all 5 folds.
Everything labelled INTERIM or EXPLORATORY, with the seeds and folds covered.

