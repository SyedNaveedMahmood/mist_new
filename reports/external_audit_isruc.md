# External audit: ISRUC

Written by scripts/prepare_external.py. Aggregates only (no per-participant rows).
Development-grade phase-2 data description; nothing here may change a phase-1 choice.
Labels: VERIFIED = counted from the files by this script; CONDITIONAL = depends on the stated assumption.

Source folder: `/home/FA006/Desktop/Dimension/dataset/isruc` (read only). Preprocessing: {"anti_alias": "zero-phase FIR low-pass at native rate, pass 45.0 Hz, trans 5.0 Hz (hamming, firwin, mne)", "resample": "scipy.signal.resample_poly to 100 Hz", "bandpass": "0.3-35.0 Hz zero-phase FIR (mne defaults)"}.

## Recordings (VERIFIED)

- found: 110
- read and stored: 107
- skipped: 3
  - 3 x no .rec file on disk (not downloaded)
  - ISRUC skips by subject number (owner-listed): group I subject 4, group I subject 6, group I subject 39

## Epochs and stages (VERIFIED)

- epochs: 96316; scored: 96316; unscored (-1): 0

| stage | epochs | share of scored |
|---|---|---|
| W | 21709 | 0.225 |
| N1 | 12540 | 0.130 |
| N2 | 30271 | 0.314 |
| N3 | 19072 | 0.198 |
| REM | 12724 | 0.132 |

## Sampling rates and hardware prefilters (VERIFIED, EEG channels as stated in the headers)

| native rate | recordings |
|---|---|
| 200.0 Hz | 107 |

| EEG prefilter field(s) | recordings |
|---|---|
| High Cut:35 Hz,Low Cut:0.3 Hz,Notch:50 Hz | 103 |
| High Cut:35 Hz,Low Cut:0.3 Hz,Notch:50 Hz / High Cut:35 Hz,Low Cut:1 Hz,Notch:50 Hz | 3 |
| High Cut:35 Hz,Low Cut:1 Hz,Notch:50 Hz | 1 |

## Channels

Channel rule (D-017): first of Fz-Cz, F4-C4, F3-C3 | F4-M1, F3-M2, F4-M2, F3-M1 | C4-M1, C3-M2 (A1/A2 treated as M1/M2). VERIFIED counts:

| rule outcome | recordings |
|---|---|
| frontal-mastoid: F4-M1 | 81 |
| frontal-mastoid: F4-A1 | 25 |
| central-mastoid: C4-A1 | 1 |

Channels kept per recording:

| EEG channels | recordings |
|---|---|
| 6 | 106 |
| 4 | 1 |

Channel names kept (recordings containing each):

| channel | recordings |
|---|---|
| C3-M2 | 81 |
| C4-M1 | 81 |
| F3-M2 | 81 |
| F4-M1 | 81 |
| O1-M2 | 81 |
| O2-M1 | 81 |
| C3-A2 | 26 |
| C4-A1 | 26 |
| O1-A2 | 26 |
| O2-A1 | 26 |
| F3-A2 | 25 |
| F4-A1 | 25 |

- recordings with channels derived from monopolar leads: 1

## Units and amplitude

| EEG unit field | recordings with it |
|---|---|
| µV | 107 |

- recordings with an empty/unknown unit field, physical value taken as microvolt: 0 (CONDITIONAL)
- median epoch SD of the rule channel after preprocessing (uV), across recordings: p5 1.3, p25 9.6, p50 15.6, p75 19.1, p95 24.9 (VERIFIED)
- recordings with that median below 5 uV: 26; above 150 uV: 0

| rule-channel physical range in header | recordings | median epoch SD (uV), median over recordings |
|---|---|---|
| -2490..2490 | 49 | 16.84 |
| -5013..5013 | 32 | 17.78 |
| -25..25 | 26 | 1.51 |

## Label / signal length (VERIFIED)

| label-length rule outcome | recordings |
|---|---|
| none | 107 |

Rule: labels start at the signal start; extra trailing labels dropped, missing trailing epochs unscored (CONDITIONAL on the start alignment). Scorer 1 used.

Groups:

| group | recordings |
|---|---|
| ISRUC1 | 97 |
| ISRUC3 | 10 |

## Amplitude subgroups: is the low-amplitude subgroup a pure scale? (owner's Q9 diagnostic, 2026-11-24)

CPU only, no labels, no model; aggregates only. Scripts: `scripts/dev/isruc_header_summary.py`, `scripts/dev/isruc_amplitude_check.py`; numbers in `reports/isruc_amplitude_check.json`. Subgroups: **low** = the 26 recordings whose EEG header range is +/-25 uV (16 in group I, all 10 of group III); **normal** = the other 81 (all group I; header range +/-2490 uV in 49, +/-5013 uV in 32).

**(c) Header fields (VERIFIED).** The two subgroups come from different acquisition set-ups:
- **Electrode labels:** low uses A1/A2; normal uses M1/M2. One low recording stores monopolar leads.
- **EEG and EOG range:** +/-25 uV in low (one O1-A2 at +/-15); a common +/-2490 or +/-5013 uV in normal.
- **Other channels:**
  - In low, each channel type has its own range: ECG +/-84 to 97, chin +/-26 to 101, leg and snore channels different again.
  - In normal, every non-EEG channel of a file shares one range (+/-9404 or +/-5013).
- **Same in both subgroups:** digital range -32768..32767 and units "uV".
- **EEG prefilter:** "High Cut:35 Hz, Low Cut:0.3 Hz, Notch:50 Hz", except one low recording with a 1 Hz low cut.

**(a) Amplitude of the other channels (VERIFIED, physical units as stated in the headers).** Median over recordings of the median 30-s epoch SD:

| channel | low (uV) | normal (uV) | normal / low |
|---|---:|---:|---:|
| EEG (all EEG channels) | 1.40 | 15.2 | 10.9 |
| EOG | 1.35 | 17.0 | 12.5 |
| chin EMG | 0.36 | 3.48 | 9.7 |
| ECG | 7.1 | 235 | 33.3 |

EEG, EOG and chin EMG are lower by a similar factor of 10 to 12; ECG is lower by 33.

**(b) Shape of the median spectrum after per-night scaling (VERIFIED).** Rule channel from the store, each night divided by its median epoch SD; median log10 power per band. The comparison is the low-subgroup median minus the normal-subgroup median, against the normal subgroup's between-person spread (median absolute deviation, MAD). Group III is entirely low-amplitude and healthy, so the comparison is repeated within group I only:

| band | low I - normal I | low III - normal I | normal I MAD |
|---|---:|---:|---:|
| 0.5-2 Hz | +0.03 | +0.05 | 0.07 |
| 2-4 Hz | **+0.12** | **+0.12** | 0.08 |
| 4-8 Hz | **+0.12** | +0.05 | 0.10 |
| 8-12 Hz | +0.08 | +0.06 | 0.14 |
| 12-16 Hz | +0.09 | +0.03 | 0.13 |
| 16-25 Hz | +0.03 | -0.08 | 0.18 |
| 25-35 Hz | -0.08 | **-0.28** | 0.18 |

**Verdict: not a pure scale (CONDITIONAL on these three checks).**
- After per-night scaling, the low subgroup has relatively more 2-8 Hz power than the normal subgroup, beyond the normal between-person spread, and the difference holds within group I.
- The scale factor differs by channel type: about 11 for EEG, EOG and EMG, 33 for ECG.
- Most likely a different amplifier or recording set-up, with its own gain and possibly its own frequency response. A population difference cannot be excluded: the subgroups may differ in diagnosis or recording period.
- **Following the owner's rule, ISRUC is not evaluated.** HMC, DOD-H and CAP are not affected.
