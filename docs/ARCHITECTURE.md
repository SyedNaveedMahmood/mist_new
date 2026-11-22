# MIST-GRM architecture

Status: implemented in `src/mist/models/`, unit-tested on synthetic signals, **not yet trained on real EEG**.

## 1. Idea in one paragraph

A recording system changes an EEG in three ways: it scales it (gain), it colours its spectrum (magnitude response, plus age- and montage-dependent 1/f tilt), and it shifts phases (group delay). A sleep stage is defined by rhythms (which frequencies carry power, and when) and by waveform shapes (K-complexes, vertex waves, sawtooth waves, slow-wave asymmetry). MIST-GRM splits every short waveform token into three factors that are separated **by arithmetic, not by a loss**:

| Factor | Symbol | Definition | Cannot see |
| --- | --- | --- | --- |
| Gain | G | mean over frequency of the night-relative log magnitude | waveform shape |
| Rhythm | R | night-relative log magnitude spectrum minus G | waveform shape, overall level |
| Morphology | M | the phase spectrum, rendered as a time-domain token with a flat magnitude spectrum | power at any frequency, level |

Because the split is exact, the research questions become switch settings: remove M and the model is provably blind to waveform shape; keep only M and it is provably blind to power.

## 2. Modules

```
x [B, L=20 epochs, 3000 samples, microvolt]        night reference (label-free): envelope, scale, per-scale median/IQR
        |                                                         |
M0  night whitening  ---------------------------------------------+
        |
M1  tokeniser        fast: 1 s window, 0.5 s hop -> 59 tokens/epoch, 35 bins (1-35 Hz)
        |            slow: 4 s window, 2 s hop   -> 14 tokens/epoch, 139 bins (0.5-35 Hz)
M2  GRM factoriser   G scalar | R [F] | M phase-only token [win]
        |
M3  stream encoders  GainEmbed | RhythmEncoder (MLP over bins) | WaveEncoder (1-D CNN) -> ShapeCodebook (32 slots)
        |                                                                   \-> surrogate critic (training only)
M4  epoch composer   concat streams -> token transformer (1 layer) -> attention pool, per scale -> fuse
        |
M5  context encoder  transformer over the 20 epochs (2 layers)
        |
M6  stage head       5 classes per epoch
```

| Module | File | Parameters | Role | Switch that ablates it |
| --- | --- | --- | --- | --- |
| M0 Night whitening | `factorizer.py: night_envelope, whiten` | none | Divide each epoch's spectrum by the smooth envelope (1/3-octave, log-frequency) of the night's median amplitude spectrum | `night_relative: false` |
| M1 Tokeniser | `factorizer.py: Factorizer.tokens` | none | Hann tokens on the waveform, two scales, never crossing an epoch boundary | `scales` |
| M2 Factoriser | `factorizer.py: factorize, gain_rhythm, morph_token` | none | Lossless G / R / M split | `streams`, `morph_input` |
| M3 Encoders | `mist_grm.py: GainEmbed, RhythmEncoder, WaveEncoder, ShapeCodebook` | most | One encoder per factor; M is quantised softly onto shape prototypes | `n_slots`, `morph_residual` |
| M4 Composer | `mist_grm.py: ScaleBranch` | some | Tokens to one vector per epoch; stream dropout during training makes inference-time knockouts valid | `stream_dropout`, `knock` |
| M5 Context | `mist_grm.py: MistGRM.context` | some | Identical in every arm, so stage-transition priors are not a confound | `context_layers: 0` |
| M6 Heads | `mist_grm.py: head, critic` | few | Stage logits; real-versus-surrogate critic on M embeddings | `critic_weight` |

The full model has about 0.62 M parameters; the spectrum-only arm 0.50 M; the raw-token control 0.57 M; AttnSleep 0.52 M (counts printed by the smoke run on 2026-11-22).

## 3. The factorisation, exactly

Night reference, computed once per recording and channel from the unlabelled signal:

- `envelope(f)`: median over epochs of |FFT(epoch)|, smoothed in log frequency over 1/3 octave with 1/f weights, inside 0.3 to 36 Hz.
- `scale`: robust standard deviation (used only by raw-input models).
- `mag_med(f)`, `mag_iqr(f)` per token scale: median and IQR over all tokens of log|X(f)| after whitening.

Per token, with X(f) the in-band spectrum of the Hann-windowed whitened token:

```
z(f)   = (log|X(f)| - mag_med(f)) / (mag_iqr(f) + 0.1)
G      = mean_f z(f)
R(f)   = z(f) - G
phi(f) = arg X(f)
M      = irfft( exp(i*phi(f)) on in-band bins, 0 elsewhere ), scaled to unit RMS
```

`(G, R, phi)` with the night reference reconstructs the token exactly (`test_lossless_reconstruction_from_g_r_phi`).

### What holds by construction

| Property | Exact? | Test |
| --- | --- | --- |
| G, R, M unchanged by any positive gain | exact | `test_all_codes_are_gain_invariant` |
| G and R are functions of |X| only | exact | `test_gain_and_rhythm_are_blind_to_phase` |
| phase-only M is a function of arg X only | exact | `test_phase_only_morphology_is_blind_to_magnitude` |
| G, R, M unchanged by a recording-level smooth zero-phase filter | approximate | `test_night_whitening_cancels_recording_level_filters` |

Measured on synthetic 1/f noise (`scripts/sanity_phase_only.py`, 200 epochs, 1 s tokens); 0 means unchanged:

| Nuisance applied to the whole night | G | R | M | z-scored raw tokens |
| --- | ---: | ---: | ---: | ---: |
| gain x3 | 0.000 | 0.000 | 0.000 | 0.000 |
| zero-phase tilt f^-0.5 | 0.004 | 0.007 | 0.004 | 0.330 |
| zero-phase smooth response | 0.005 | 0.008 | 0.005 | 0.172 |
| all-pass, quadratic phase (1.5 rad at 35 Hz) | 0.058 | 0.090 | 0.659 | 0.102 |

Measured on real Sleep-EDF nights (`scripts/check_invariance.py --n-nights 10`, task T4, 2026-11-22; first 10 dev nights, SC 00-05, `std` view, 1 s tokens; mean over nights, max in brackets; tables in `runs/e5a/`, registry `e5a_invariance_10n`):

| Nuisance applied to the whole night | Fpz-Cz G | R | M | Pz-Oz G | R | M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gain x3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| zero-phase tilt f^-0.5 | 0.003 (0.003) | 0.007 (0.007) | 0.006 (0.006) | 0.002 (0.003) | 0.007 (0.008) | 0.006 (0.006) |
| all-pass, quadratic phase (1.5 rad at 35 Hz) | 0.035 (0.038) | 0.081 (0.083) | 0.659 (0.660) | 0.030 (0.042) | 0.080 (0.085) | 0.659 (0.660) |

A dispersive (all-pass) filter applied to the signal moves energy between neighbouring tokens, so G and R change slightly (about 0.03 and 0.08 on real nights, 0.06 and 0.09 on synthetic) even though each token's code ignores its own phase.

Reading: magnitude-type nuisances are removed almost completely. **Phase-type nuisances are not, and they hit M harder than they hit raw tokens.** That is the main technical risk of the design (section 6).

### What does not hold

- A non-zero-phase response (analogue high-pass filters, anti-alias filters) changes phi. Only M is exposed. `allpass_rad` adds a random smooth phase response per sequence during training so the M encoder can learn to ignore it.
- The night reference is label-free but it does use the test recording's own signal. That is ordinary per-recording normalisation, yet it is not strictly inductive. The `grm_absolute` arm quantifies what it buys.
- Whitening by a night median removes between-night differences in average spectrum. If a night has unusually much N3, its envelope shifts. Median (not mean) limits this; it does not remove it.

## 4. The morphology stream in detail

- **Input.** `phase_only` (default) or `unit_raw` (ablation: unit-RMS whitened waveform, spectrum included). With `phase_only`, every transient becomes a phase-coherent event in a flat-spectrum token. A synthetic check shows it is learnable: a small encoder detects a biphasic transient from phase-only tokens at 0.77 accuracy against 0.48 when no transient exists (chance 0.5; `scripts/sanity_phase_only.py`, part B). That is a plumbing check on synthetic data, not evidence about EEG.
- **Encoder.** Four 1-D conv blocks, then max and mean pooling over time, so the code does not depend on where in the token the event sits.
- **Shape codebook.** 32 cosine prototypes per scale; each token gets a soft assignment. With phase-only input a prototype cannot encode band power or amplitude, which is the property the old prototype work could not guarantee. A capped residual (8 dimensions) carries what the prototypes miss; its width is a knob for leakage tests.
- **Surrogate critic (training only).** For a random quarter of tokens the phase is replaced by random phase (same magnitude spectrum) and a linear head on the M embedding must tell real from surrogate. Its weight is the first setting of the **morphology dial**.

### The morphology dial

| Setting | Config | Meaning |
| --- | --- | --- |
| Off | `streams: [G, R]` | No access to phase |
| Placebo | `destroy_prob: 1.0` | Same parameters as the full model; M only ever sees random phase |
| Available | `critic_weight: 0` | M is there, trained by the stage loss only |
| Trained | `critic_weight: 0.1 / 0.3 / 1.0` | M is also pushed to encode beyond-spectrum structure |
| Partial dose | `destroy_prob: 0.25 / 0.5 / 0.75` | Fraction of tokens whose morphology is destroyed, at train and test |

At test time, `destroy_prob=1.0` or `knock=("M",)` measures how much a trained model relies on morphology.

## 5. Arms (configs/arms)

| Arm | Streams | Purpose |
| --- | --- | --- |
| `raw` | X | Unfactorised control with the same tokeniser, composer and context |
| `r` | R | Rhythm alone |
| `gr` | G, R | Best model that cannot see waveform shape |
| `grm_placebo` | G, R, M (random phase) | Capacity control for `grm` |
| `grm_nocritic` | G, R, M | Morphology available, dial at zero |
| `grm` | G, R, M + critic | Full model |
| `m` | M | Morphology alone: no power, no gain |
| `grm_loose` | G, R, M (`unit_raw`) | Does spectrum-blindness of M matter? |
| `grm_absolute` | G, R, M, no night reference | Does night-relative coding matter? |
| `attnsleep` | baseline | External reference point |

## 6. Known risks in the design

1. **Phase distortion.** Datasets recorded through different analogue chains have different phase responses. M may therefore transfer worse, not better. E5 measures this directly, and `allpass_rad` is the only mitigation implemented.
2. **Phase-only tokens may be a poor learning substrate on real EEG.** Flat magnitude gives equal weight to low-power bins whose phase is noise. If `grm` is not better than `grm_placebo` in-domain, try `grm_loose` before concluding morphology is useless.
3. **Spindles are mostly rhythm under this split.** A narrow-band burst is visible in R; only its envelope shape inside a token is in M. So "morphology" here means transients and asymmetries, plus within-token burst structure. The paper must say this plainly.
4. **Night-relative coding and short recordings.** The night reference needs enough data to be stable. Minimum length is untested.
5. **Two scales only.** Events longer than 4 s are left to the context encoder.

## 7. Not built yet (see docs/TASKS.md)

Slot-family knockouts and naming; stage-targeted critic; masked-token pretraining; training on more than one derivation at once; the magnitude/phase STFT baseline that mirrors the closest prior work; SleepBand, WaveSleepNet and TinySleepNet baselines.
