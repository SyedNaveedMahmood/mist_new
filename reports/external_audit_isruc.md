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
