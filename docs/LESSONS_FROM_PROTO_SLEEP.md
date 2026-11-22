# What the earlier work (Proto-sleep) established, and what changes because of it

Source: `RESEARCH_STATUS_AND_HISTORY.md` on the `main` branch of `SyedNaveedMahmood/Proto-sleep` (last reviewed there 2026-10-31) and the `research/mist-transfer-v3-edf-shhs` branch. Numbers below are quoted from that file, not re-derived.

## Findings to keep

| Finding | Consequence here |
| --- | --- |
| Sleep-EDF-20 and Sleep-EDF-78 share 20 people and 39 identical recordings | Never called transfer. Splits are by person across the whole SC cohort |
| The old NPZ files were trimmed by a label-conditioned rule whose script is unknown | Rebuild from raw EDF; NPZ only for smoke tests |
| SC persons 06, 23, 36, 74 have unresolved demographic or annotation conflicts | Quarantined by default (`sleepedf.SC_QUARANTINE`) |
| Audited raw SC subset: 74 people, 146 recordings, 397,832 complete 30-s epochs, 396,378 scored | Gate G0: the new reader must reproduce these totals |
| Seven ST hypnograms fail strict EDF+ parsing because of a previous-day Recordingfield; all 44 ST pairs were validated | The new reader takes onsets relative to PSG start and must still be checked on those seven files |
| ST was used in several exploratory analyses | ST is a development-grade target; never presented as untouched |
| 55 SC extension people were never used | 20 stay locked; the release of the other 35 is decision D1 |
| HMC evaluation was blocked by storage (about 15.7 GB needed, 7 GB free), not by access | Free the space; HMC is the first external target |

## Approaches that did not work

| Approach | Outcome there | What this project does instead |
| --- | --- | --- |
| Prototype layer on a latent CNN feature map (ProtoMAE, MorphMAE, WaveSleepNet-style) | Some representation measures looked promising; no downstream or cross-dataset gain shown | Prototypes sit on a power-blind input, so they can only encode shape; their value is tested by knockout |
| Morphology-preservation invariance loss (L2 alignment plus KL consistency of prototype assignments under bounded views) | Representation drift fell, but the pre-specified +0.02 robustness gain was missed (about +0.006) | No consistency loss. Invariance to gain and magnitude response is arithmetic, exact or near-exact, and checked by unit tests |
| Internal explanation supports as event localisers | Worse than an equal-duration sigma-power baseline on ST (0.171 vs 0.217 interval precision) | No claim that slots are clinical events until expert-annotated data are available |
| AttnSleep "repository adapter" as baseline | 0.31 macro-F1 on four validation people, far below the published model | Gate G0b: the baseline must reach a plausible score before any comparison counts |
| Development on 15 training and 4 validation people | Every estimate was too noisy to steer by | 54 dev people in 5 folds |
| Many parallel branches with their own gates | Evidence could not be combined | One branch, one registry, one frozen split file |

## Process rules inherited

- Status labels: VERIFIED, CONDITIONAL, BLOCKED, UNKNOWN, NOT TESTED, always with the scope.
- No target labels or target statistics for fitting, model selection or design. One deliberate departure: the night reference uses the test recording's own unlabelled signal (decision D3); the `grm_absolute` arm reports the strict alternative.
- A passed software gate is not a scientific result.
- Participant-level signals, per-epoch outputs and checkpoints stay local.
