# Novelty check

Date: 2026-11-22. Method: about a dozen targeted web searches (titles, abstracts and result snippets; three full-page summaries from earlier in the project). **This is a screening, not a systematic review.** Several pages could not be opened in this session, so details marked "snippet" were read from search results only.

## Verdict

No single ingredient of MIST-GRM is new. The combination and the question it is built to answer were not found in the search. "Absolutely novel" cannot be promised from a screening; what can be said is narrower:

- **Not found:** a sleep-staging model whose representation is an exact gain / magnitude / phase factorisation with a power-blind morphology stream and night-relative coding, and no study that manipulates access to waveform morphology and measures cross-dataset staging transfer.
- **Found, and close:** separate magnitude and phase encoders for time-series domain generalisation, with sleep staging among the test tasks (PhASER). A reviewer will raise it. The paper must position against it explicitly and include a baseline in its spirit.
- **Found, and opposed:** a 2026 single-source domain-generalisation paper arguing that narrow-band spectral magnitude is what transfers in sleep staging, and that phase should be perturbed, not used (SleepBand).

Whether the work reaches a top-quartile journal depends on the results and on the breadth of the external evaluation, not on the architecture. A null result on RQ1 is publishable only if the instrument is convincing, which is why the by-construction properties and the gates matter.

## The literature disagrees, which is the opening

| Claim | Source | Status |
| --- | --- | --- |
| Randomising Fourier phase makes a sleep-staging network rely less on waveform and more on the power spectrum; used as augmentation | Rommel et al., 2022 | read (snippet + earlier summary) |
| Learning the coupling of phase and amplitude improves generalisation across subjects and recording sessions, sleep staging included | Lemkhenter & Favaro, 2020 (PhaseSwap) | abstract |
| Sleep staging rests on a few narrow-band rhythms that are stable across datasets; band magnitudes are used, phase is jittered | Lu, Hu & Chen, 2026 (SleepBand) | full-page summary |
| Phase-focused features generalise best across subjects in motor-imagery BCI | KnIFE, arXiv:2405.11163 | snippet |
| In vision, amplitude carries style and phase carries content | e.g. arXiv:2202.04287 and the FACT / FDA line | snippet |
| Sleep-wake classifiers lean heavily on the aperiodic 1/f envelope | Bindra & Panwar, 2026 | abstract summary |

Nobody in this list has a model in which the two kinds of information are separated exactly, so none of them can say which one carries transfer. That is the gap.

## Closest prior work, claim by claim

| Our element | Closest prior work | What they do | What differs here |
| --- | --- | --- | --- |
| Separate encoders for magnitude and phase | PhASER (Mohapatra, Wang & Zhu, TMLR 2025; arXiv:2402.05960) | STFT magnitude and phase as two modalities, Hilbert-transform phase augmentation, phase-residual broadcasting; evaluated on activity, gesture and sleep-stage data | No gain factor and no night-relative coding were found in the abstract and snippets; it is a general time-series method with no prototypes and no intervention on morphology. Whether its phase input is strictly power-blind was NOT checked against the paper (an earlier version of this row said it was not; that was unverified). An input ablation in the paper reports magnitude-only above phase-only (0.81 vs 0.62, dataset not identified in the snippet) |
| Rhythm decomposition before feature learning | Zhou et al., 2026 (ERRL + SCRL) | Five zero-phase band-pass signals (delta to gamma) with channel attention and per-rhythm extractors, fused with a raw-EEG branch trained by supervised contrastive learning; Sleep-EDF-20 and MASS-SS3, within-dataset | Each band signal keeps amplitude and phase together and the raw branch sees everything. The cascade (ARCHITECTURE section 8) also decomposes by rhythm first, but separates level from shape inside each band and tests transfer. Read in full from the uploaded paper |
| Real-versus-surrogate objective | PhaseSwap (Lemkhenter & Favaro, GCPR 2020) | Self-supervised task: detect whether amplitude and phase come from different signals | Whole-segment binary pretext; ours is token-level, inside a factorised model, with a dose control |
| Spectrum-structured encoder for transfer | SleepBand (arXiv:2607.04851) | Learnable constant-Q Gabor bank, band magnitudes, spectral-consistency regulariser; single-source DG over Sleep-EDF, HMC, ISRUC, SHHS1, CinC 2018 with EEG+EOG | Uses magnitude only. It is the natural strong competitor and its five-dataset protocol is the benchmark to adopt in phase 2 |
| Waveform prototypes | WaveSleepNet (arXiv:2404.15342); SLEEPER (arXiv:1910.06100) | Prototypes in a CNN latent space; prototypes from expert rules | Their prototypes can encode band power. Ours cannot, by construction |
| Waveform-shaped kernels | Interpretable Gabor-kernel layer (arXiv:2207.04585) | Kernels shaped like spindles, slow oscillations, K-complexes | Fixed parametric shapes; no transfer question |
| Raw and spectral branches, N1 focus | DualViewSeqSleepNet (Frontiers in Medicine, 2026) | Raw-waveform plus spectral encoders, auxiliary loss on Wake-N1 and N1-N2; Sleep-EDF and SHHS | Branches are not disjoint in information: the raw branch still sees the spectrum. Relevant to RQ2 |
| Phase-randomised surrogates as a measurement | Spectral audit (arXiv:2606.08583); Ellis et al., 2022 | Fourier interventions to audit EEG models; waveform versus spectral feature importance | Audits of existing models. We add local surrogates, per-stage reliance under transfer, and a model built to be audited |
| Joint staging and event detection | arXiv:2501.09519 and earlier multi-task work | Shared network for stages and micro-events | Needs event labels; not about transfer |
| Per-recording normalisation | Common practice (per-night z-score; YASA-style per-night feature scaling) | Scale by recording statistics | We whiten by the night's smooth spectral envelope and code magnitudes against per-frequency night medians |

## Claims the paper can defend today

1. **Instrument.** A lossless gain / rhythm / morphology token factorisation in which the rhythm and gain streams are provably blind to waveform shape and the morphology stream is provably blind to power. (Proved by unit tests; closest: PhASER.)
2. **By-construction acquisition invariance.** Night whitening plus the factorisation removes gain exactly and smooth zero-phase responses to within about 1% on synthetic signals, leaving phase response as the single exposed nuisance, confined to one stream. (Synthetic evidence only so far.)
3. **Power-blind shape prototypes.** A codebook that can only encode morphology. (Closest: WaveSleepNet.)
4. **Causal protocol.** A morphology dial (off, placebo, available, trained, partial dose) with a factor-by-stage reliance matrix, in-domain versus under montage, age and cohort shift. (Closest: Rommel et al. as augmentation; the spectral audit as measurement.)

## Claims to avoid

- "First to use phase for sleep staging" or "first to separate magnitude and phase": false (PhaseSwap, PhASER).
- "Morphology-aware" as a headline without the operational definition: reviewers will read it as K-complex and spindle detection, which this design does not supervise.
- Any state-of-the-art claim before the external benchmark.

## Residual risks to novelty

- Work after mid-2026 moves fast in this area (SleepBand appeared in July 2026). Repeat the search before submission.
- Not searched well: Chinese-language venues, theses, and speech or audio literature on phase-only representations applied to biosignals.
- Bispectral (higher-order) features are the classical way to capture phase coupling in EEG. A reviewer may ask why M is not a bispectrum. Answer to prepare: the phase-only token is a complete phase representation and the CNN with pooling can learn shift-invariant functions of it; an explicit biphase input is a reasonable ablation.

## Search log

Queries run (WebSearch): PhaseSwap title; sleep staging domain generalisation Fourier amplitude phase; staging with auxiliary spindle / K-complex detection and cross-dataset generalisation; concept bottleneck sleep staging graphoelements; self-supervised EEG real versus phase-randomised surrogate; EEG separate amplitude and phase streams, phase-only representation; amplitude-invariant waveform shape tokens, rank normalisation within night; do staging networks rely on waveform morphology or spectral power; phase-only or spectral whitening EEG waveform prototypes; PhASER venue and datasets. Earlier in the project: phase-randomised surrogates in EEG deep learning; foundation-model probes for spindles and K-complexes.
