# Experiment plan

Phase 1 runs entirely on Sleep-EDF Expanded. Phase 2 repeats the decisive comparisons on independent datasets. Nothing in phase 2 may be used to tune phase 1 choices.

## 1. Questions and hypotheses

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

**Measured 2026-11-23 (DECISION_LOG D-014).** Per run, alone on the RTX 4080 SUPER: `attnsleep` about 5 min, `raw` about 26 min, `grm` 20 min (30.5 s/epoch), `cascade` 2.44 h (219.6 s/epoch). The cut above therefore applies. E1 as run (scripts/e1_queue.sh):

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
- Pooled shift effect (H1): linear mixed model, macro-F1 ~ arm with random intercepts for protocol, fold and seed.
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
