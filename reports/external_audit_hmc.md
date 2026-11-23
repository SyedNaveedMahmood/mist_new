# External audit: HMC

Written by scripts/prepare_external.py. Aggregates only (no per-participant rows).
Development-grade phase-2 data description; nothing here may change a phase-1 choice.
Labels: VERIFIED = counted from the files by this script; CONDITIONAL = depends on the stated assumption.

Source folder: `/home/FA006/Desktop/Dimension/dataset/hmc/hmc-sleep-staging/1.1/recordings` (read only). Preprocessing: {"anti_alias": "zero-phase FIR low-pass at native rate, pass 45.0 Hz, trans 5.0 Hz (hamming, firwin, mne)", "resample": "scipy.signal.resample_poly to 100 Hz", "bandpass": "0.3-35.0 Hz zero-phase FIR (mne defaults)"}.

## Recordings (VERIFIED)

- found: 151
- read and stored: 151
- skipped: 0

## Epochs and stages (VERIFIED)

- epochs: 137248; scored: 137243; unscored (-1): 5

| stage | epochs | share of scored |
|---|---|---|
| W | 23686 | 0.173 |
| N1 | 15548 | 0.113 |
| N2 | 50083 | 0.365 |
| N3 | 26671 | 0.194 |
| REM | 21255 | 0.155 |

## Sampling rates and hardware prefilters (VERIFIED, EEG channels as stated in the headers)

| native rate | recordings |
|---|---|
| 256.0 Hz | 151 |

| EEG prefilter field(s) | recordings |
|---|---|
| HP:0.2Hz LP:35.0Hz | 83 |
| HP:0.2Hz LP:128.0Hz | 68 |

## Channels

Channel rule (D-017): first of Fz-Cz, F4-C4, F3-C3 | F4-M1, F3-M2, F4-M2, F3-M1 | C4-M1, C3-M2 (A1/A2 treated as M1/M2). VERIFIED counts:

| rule outcome | recordings |
|---|---|
| frontal-mastoid: F4-M1 | 151 |

Channels kept per recording:

| EEG channels | recordings |
|---|---|
| 4 | 151 |

Channel names kept (recordings containing each):

| channel | recordings |
|---|---|
| C3-M2 | 151 |
| C4-M1 | 151 |
| F4-M1 | 151 |
| O2-M1 | 151 |

- recordings with channels derived from monopolar leads: 0

## Units and amplitude

| EEG unit field | recordings with it |
|---|---|
| µV | 151 |

- recordings with an empty/unknown unit field, physical value taken as microvolt: 0 (CONDITIONAL)
- median epoch SD of the rule channel after preprocessing (uV), across recordings: p5 13.1, p25 16.1, p50 18.9, p75 22.7, p95 35.4 (VERIFIED)
- recordings with that median below 5 uV: 0; above 150 uV: 2

## Label / signal length (VERIFIED)

Annotations are on the 30-s grid from the PSG start (hmc.py); unannotated epochs stay unscored.
