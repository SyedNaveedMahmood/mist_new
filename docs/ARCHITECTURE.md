# MIST-GRM architecture

Status: sections 1 to 7 describe the **parallel** topology (v1), implemented in `src/mist/models/` and unit-tested. **Revision 2 (2026-11-22, supervisor): the main model is now the cascade in section 8**, where rhythm feeds morphology and only gain and morphology reach the composer. The cascade is specified but NOT IMPLEMENTED and NOT TESTED; the parallel model stays as an ablation arm.

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

These are the parallel-topology arms that exist today. The cascade arms are listed in section 8.4.

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

## 8. Revision 2 (2026-11-22): cascade topology, rhythm before morphology

Status: **IMPLEMENTED 2026-11-23** (`src/mist/models/cascade.py`, `topology: cascade`; task T6b). The section 8.5 properties are VERIFIED on synthetic 1/f signals (`tests/test_cascade.py`); NOT TESTED on real EEG; no cascade model has been trained on real data yet. Requested by the supervisor on the architecture slide and relayed by the owner. Implementation choices the specification left open are in section 8.8.

### 8.1 What changed on the diagram

Three marks on the slide: the arrow factoriser -> M is crossed out, an arrow R -> M is added, and the arrow R -> composer is crossed out. G -> composer is untouched.

```
v1 parallel (sections 1-7)            v2 cascade (this section)

factorise -> G -> compose             factorise -> G ---------> compose
factorise -> R -> compose             factorise -> R -> M ----> compose
factorise -> M -> compose
```

### 8.2 How the drawing is read (an assumption; confirm with the supervisor)

Read literally with the v1 definitions, the cascade cannot work: R is built from the magnitude spectrum only, so it holds no phase, and morphology cannot be computed from it. The reading adopted here keeps every v1 quantity and changes the routing:

- **R becomes the rhythm-decomposition stage.** It still produces the phase-blind rhythm code. It also splits the token's bins into rhythm bands and hands each band's phase on to M.
- **M becomes "the shape of each rhythm".** It encodes a power-blind waveform per band and is modulated by the rhythm code, so a shape is read in the context of which rhythm is active.
- **Only G and the rhythm-gated M output reach the composer.** Rhythm has no path of its own.

The alternative reading, "rhythm is only a conditioning signal for a broadband M", is the special case with one band. It is kept as the `cascade_broad` ablation.

### 8.3 Specification

Unchanged from section 3: night whitening, tokenising, `z(f)`, `G`, `R(f)`, `phi(f)`.

Bands (Hz, lower edge inclusive): delta 0.5-4, theta 4-8, alpha 8-12, sigma 12-16, beta 16-35. Sigma is separate because spindles define N2; there is no gamma band because the signal is limited to 35 Hz. With 1 s tokens the narrow bands hold 3 to 4 bins each; with 4 s tokens 14 to 16.

```
r        = RhythmEncoder(R)                      rhythm code, phase-blind (as in v1)
m_b      = irfft( exp(i*phi(f)) for f in band b ), unit RMS      b = delta..beta
m_0      = irfft( exp(i*phi(f)) for all in-band f ), unit RMS    broadband channel = the v1 M input
e_b      = WaveEncoder(m_b) + band_embedding_b   shared encoder, b = 0..5
a_b      = ShapeCodebook_b(e_b)                  soft assignment over 16 shape slots per band
g_b      = softplus( Linear_b(r) )               one positive gate vector per band, from rhythm
u        = sum_b  g_b * ( P_b a_b + W_b e_b )    P_b: slot embeddings; W_b: capped residual (8 dims)
token    = proj( [ GainEmbed(G), u ] )           then composer, context and head as in v1
```

Notes:

- The band waveforms sum (before normalisation) to the v1 morphology token, so nothing v1 could see is removed. They are organised by rhythm.
- `m_0` keeps phase alignment across bands, which is what makes a transient sharp. Without it a K-complex edge or a sawtooth wave would be split across bands.
- Rhythm enters only multiplicatively, through `g_b`. There is no additive bypass, in line with the crossed-out arrow.
- The critic acts on each `e_b`, so its accuracy per band says where beyond-spectrum structure lives.

### 8.4 Switches and arms

| Switch | Config (planned) | Effect |
| --- | --- | --- |
| Morphology off | `destroy_prob: 1.0` | Every band gets random phase. `G` and `r` are untouched, so the model is exactly a gain-plus-rhythm model with the same parameters |
| Morphology of one rhythm off | `destroy_bands: [sigma]` at test | Removes the shape of that band only |
| Rhythm off | `knock: [R]` (gates set to their training mean) | Shape without rhythm context |
| Gain off | `knock: [G]` | As in v1 |
| Dial | `critic_weight`, `destroy_prob` | As in v1 |

| Arm | Topology | Purpose |
| --- | --- | --- |
| `cascade` | cascade, 5 bands + broadband, critic on | **Main model** |
| `cascade_placebo` | cascade, `destroy_prob: 1.0` | Gain plus rhythm through the same path; replaces `gr` and `grm_placebo` as the reference for H1 |
| `cascade_nocritic` | cascade, `critic_weight: 0` | Dial at zero |
| `cascade_bands` | cascade without the broadband channel | Does cross-band phase alignment matter? |
| `cascade_broad` | cascade with the broadband channel only | Rhythm as conditioning only (the alternative reading) |
| `grm`, `gr` | parallel (v1) | Topology ablation: does routing rhythm through morphology change anything? |

### 8.5 Properties to prove before any cascade result counts

Each needs a unit test in `tests/` next to the existing invariance tests (CLAUDE.md rule 6).

| Property | Expected |
| --- | --- |
| `G`, `r`, every `m_b` unchanged by a positive gain | exact |
| `r` and every gate `g_b` are functions of |X| only | exact |
| every `m_b`, `e_b`, `a_b` is a function of arg X only | exact |
| the unnormalised band waveforms sum to the v1 morphology token | exact |
| with `destroy_prob: 1.0`, `G` and `r` equal their values on the real token | exact |
| smooth zero-phase filter leaves `G`, `r`, `m_b` nearly unchanged | approximate, as in section 3 |

What no longer holds: the vector that reaches the composer from M is not power-blind. Power-blindness holds for M's inputs and its shape slots, not for its gated output. "Remove R" and "remove M" are no longer independent switches.

### 8.6 What the change buys and what it costs

Buys:

- RQ2 gets families by construction. "Which morphology matters" becomes a band-by-stage table (shape of delta, theta, alpha, sigma, beta, broadband), with no after-the-fact naming of slots.
- The reference model for RQ1 is the same network with scrambled phase, so parameter counts match exactly and gate G3 is not needed for the cascade.
- The design sits closer to Zhou et al. (2026), who also decompose into rhythms first. Their band signals keep amplitude and phase together; here the two are separated inside each band. That makes their model a natural comparator.

Costs and risks:

- All rhythm information must pass through the M stage. If that stage is a bottleneck, in-domain accuracy falls. Gate G2 is checked for `cascade` against `raw` and against `grm`.
- With 1 s tokens a narrow band has 3 to 4 phases, so its shape code carries little. Most per-band shape information will come from the 4 s tokens. (inference)
- Phase distortion still reaches every band. Inside a narrow band a smooth phase response is close to a constant offset plus a delay, which may be easier to ignore. (speculation)
- More parts: six encoder passes per token instead of one. The M stage will cost several times what it does in `grm`; the factor is unknown until T6b times a run. (speculation)

### 8.7 Questions for the supervisor

1. Does R -> M mean "morphology within each rhythm band" (adopted), or "rhythm as a conditioning signal for one broadband morphology stream" (`cascade_broad`)?
2. Should rhythm really have no direct path to the composer? A skip connection is the usual safeguard; it is left out because the arrow was crossed out.
3. The slide also carried a figure of a 1-D EEG trace with sharp transients. Was that meant as an instruction (for example, morphology should be read from the raw trace), or as an illustration?

### 8.8 Implementation choices (2026-11-23, T6b; docs/DECISION_LOG.md D-010)

The specification in 8.3 leaves these open. Each choice is the simplest reading; none was tuned.

| Item | Choice |
| --- | --- |
| Width of `u`, `g_b`, `P_b` | `d_stream` (64), so `u` has the width of one v1 stream |
| `W_b` "capped residual (8 dims)" | rank-8 map per channel: `e_b` (64) -> 8 -> 64 (`morph_residual`) |
| Band edges on the bin grid | lower edge inclusive, upper exclusive, except beta which keeps 35 Hz. Every in-band bin falls in exactly one band. 1 s tokens: delta 3 bins, theta/alpha/sigma 4, beta 20 |
| `destroy_bands: [b]` | randomises the phase of band b's bins **in every channel**, the band channel and the broadband channel, so the band's shape is removed from the whole model. `broad` randomises the broadband channel's whole input and leaves the band channels intact (measures reliance on cross-band alignment) |
| `knock: [R]` | gates replaced by a running mean of the training gates (EMA, momentum 0.01, stored in the checkpoint) |
| `knock: [M]` | zeroes `u`; since rhythm reaches the composer only through `u`, this also removes rhythm (the switches are not independent, as 8.5 says) |
| Critic | one linear critic shared by all channels, applied to `e_b` (which includes the band embedding); real channel waveform vs the same channel built from random phase; mean BCE over channels |
| Stream dropout | applied to the gain embedding and to `u`, as v1 does per stream |
| Slot indices for `knock_slots` and galleries | flattened as channel x 16 + slot; renormalised within each channel |

Measured on synthetic 1/f nights (tests/test_cascade.py, smooth zero-phase tilt and bump responses): relative change of G <= 0.005, R <= 0.008, band waveforms <= 0.006 overall and <= 0.012 for the worst channel (delta). Parameter counts with base.yaml: `cascade` = `cascade_placebo` = 699,337; `cascade_bands` 684,745; `cascade_broad` 626,377; `grm` 624,601.


## 9. Revision 3 (2026-11-23): own-rate cascade (`cascade2`)

Owner's decision D8 (docs/DECISIONS.md), made for the GPU budget of two 12-hour sessions a week. The idea of section 8 is unchanged: G goes directly to the composer; R makes positive per-channel gates; M is per-band phase-only shape plus broadband; only G and the gated M reach the composer. Three changes to the shape encoder.

### 9.1 Changes

| | Section 8 cascade | `cascade2` |
| --- | --- | --- |
| (a) Synthesis | every channel at the token length (100 or 400 samples) | each channel at its own length `n = 2 (k_top + 1)`, rounded up to a multiple of 8, minimum 16, `k_top` = the channel's highest FFT bin; `irfft` over bins `0..n/2` with `n` points, then unit RMS |
| (b) Band channels | at both token scales | only at the 4 s scale; the 1 s scale keeps the broadband channel only, still gated by rhythm |
| (c) Encoders | one WaveEncoder (width 32) for all six channels | WaveEncoder width 32 for broadband; a second WaveEncoder, width 16, shared by the five band channels |

Lengths: 1 s token, broad 72. 4 s token, delta 32, theta 64, alpha 96, sigma 128, beta 288, broad 288 (full rate: 6 x 400).

`destroy_bands` still acts on the broadband channel's bins at both scales, so at 1 s it randomises that band's bins in the broadband channel.

### 9.2 What (a) does and does not change

The own-rate waveform is the same band-limited waveform sampled on a coarser grid: at the grid points `t_j = j * win / n` it equals the full-rate waveform (up to the common factor `win / n`, which unit RMS removes), and its RMS equals the full-rate RMS (Parseval). Because `n / 2 > k_top`, the Nyquist bin is empty and `rfft` of the own-rate waveform returns exactly the channel's phasor bins, so no shape information is lost (tests 1 and 5). What changes is what the encoder sees per sample: a delta waveform is 32 samples long, so the encoder's kernels (9, 9, 7, 5 samples with three stride-2 steps) span a larger share of the token for slow bands than for fast ones. That is a design change, not a numerical identity, and it is why the validation gate in EXPERIMENT_PLAN section 12 exists.

### 9.3 Configuration

`MistConfig` gains `band_rate` (`full` | `own`), `band_scales` (token lengths in s that carry band channels; `null` = every scale), `band_width` (`null` = one encoder for all channels; an integer = the band encoder's width) and `encoder_checkpoint` (D-013 activation checkpointing; on by default). The defaults reproduce the section 8 model exactly: existing checkpoints load with `strict=True` and give bit-identical outputs (test 8 compares against `tests/fixtures/cascade_v1_reference.pt`, written by the pre-revision code). The new per-channel index buffers are not saved in checkpoints.

Arms: `cascade2`, `cascade2_placebo` (destroy_prob 1.0, critic 0), `cascade2_absolute` (night_relative false); `cascade2w32*` (band_width 32) exist only for the validation-gate fallback.

`cascade2` runs without checkpointing (`encoder_checkpoint: false`). The decision rests on a CPU count of the activations autograd saves in one training step at batch 32 (`scripts/dev/activation_memory.py`). That count gives 10.5 GB for `cascade2`, against 9.6 GB for `grm`, whose measured GPU peak is 9.8 GB. Without checkpointing the full-rate cascade needs 55 GB. If a run still runs out of memory, the session queue retries it once with checkpointing, which gives the same arithmetic.

### 9.4 Properties (tests/test_cascade2.py)

1. Own-rate waveform = full-rate waveform at the coarse grid points; equal RMS.
2. Every code (g, r, gates, m, e, a) is invariant to a positive gain.
3. G, r and the gates depend on |X| only.
4. m, e and a depend on arg X only.
5. `rfft` of each own-rate waveform returns that channel's phasor bins and nothing else.
6. With destroy_prob 1: same parameters as `cascade2`, and the output is identical for a token and for the same token with its phase replaced.
7. `destroy_bands: [b]` changes only band b's channel and the broadband channel (4 s), and the broadband channel at 1 s.
8. The defaults reproduce the section 8 cascade outputs bit for bit.

Plus: forward and backward with the critic, the band encoder receives gradients and has width 16, and the checkpointing switch changes neither values nor gradients. Tests 1 and 5 were checked to fail on two deliberate synthesis faults (a one-sample shift, a dropped bin). All on synthetic signals; nothing here is evidence on real EEG.

Parameter counts with base.yaml: `cascade2` = `cascade2_placebo` = 678,457 (`cascade` 699,337).
