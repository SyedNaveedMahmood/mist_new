# External audit: DODH

Written by scripts/prepare_external.py. Aggregates only (no per-participant rows).
Development-grade phase-2 data description; nothing here may change a phase-1 choice.
Labels: VERIFIED = counted from the files by this script; CONDITIONAL = depends on the stated assumption.

Source folder: `/home/FA006/Desktop/Dimension/dataset/dreem_dod/dodh` (read only). Preprocessing: {"anti_alias": "zero-phase FIR low-pass at native rate, pass 45.0 Hz, trans 5.0 Hz (hamming, firwin, mne)", "resample": "scipy.signal.resample_poly to 100 Hz", "bandpass": "0.3-35.0 Hz zero-phase FIR (mne defaults)"}.

## Recordings (VERIFIED)

- found: 25
- read and stored: 25
- skipped: 0

## Epochs and stages (VERIFIED)

- epochs: 24665; scored: 24662; unscored (-1): 3

| stage | epochs | share of scored |
|---|---|---|
| W | 3037 | 0.123 |
| N1 | 1505 | 0.061 |
| N2 | 11879 | 0.482 |
| N3 | 3514 | 0.142 |
| REM | 4727 | 0.192 |

## Sampling rates and hardware prefilters (VERIFIED, EEG channels as stated in the headers)

| native rate | recordings |
|---|---|
| 250.0 Hz | 25 |

Prefilter: not stated in the files (UNKNOWN).

## Channels

Channel rule (D-017): first of Fz-Cz, F4-C4, F3-C3 | F4-M1, F3-M2, F4-M2, F3-M1 | C4-M1, C3-M2 (A1/A2 treated as M1/M2). VERIFIED counts:

| rule outcome | recordings |
|---|---|
| frontal-mastoid: F4-M1 | 25 |

Channels kept per recording:

| EEG channels | recordings |
|---|---|
| 12 | 25 |

Channel names kept (recordings containing each):

| channel | recordings |
|---|---|
| C3-M2 | 25 |
| F3-F4 | 25 |
| F3-M2 | 25 |
| F3-O1 | 25 |
| F4-M1 | 25 |
| F4-O2 | 25 |
| Fp1-F3 | 25 |
| Fp1-M2 | 25 |
| Fp1-O1 | 25 |
| Fp2-F4 | 25 |
| Fp2-M1 | 25 |
| Fp2-O2 | 25 |

- recordings with channels derived from monopolar leads: 0

## Units and amplitude

- Unit attribute in the files: mV (VERIFIED). Median native 30-s SD per file: p5 10.7, p25 11.6, p50 14.3, p75 17.0, p95 19.7. These magnitudes are microvolt scale, so the numbers are used as microvolt (CONDITIONAL).
- median epoch SD of the rule channel after preprocessing (uV), across recordings: p5 12.1, p25 14.0, p50 15.0, p75 23.0, p95 26.0 (VERIFIED)
- recordings with that median below 5 uV: 0; above 150 uV: 0

## Label / signal length (VERIFIED)

| label-length rule outcome | recordings |
|---|---|
| none | 25 |

Hypnogram source (consensus or single scorer): UNKNOWN from the files.
