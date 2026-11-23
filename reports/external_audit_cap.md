# External audit: CAP

Written by scripts/prepare_external.py. Aggregates only (no per-participant rows).
Development-grade phase-2 data description; nothing here may change a phase-1 choice.
Labels: VERIFIED = counted from the files by this script; CONDITIONAL = depends on the stated assumption.

Source folder: `/home/FA006/Desktop/Dimension/dataset/cap/physionet.org/files/capslpdb/1.0.0` (read only). Preprocessing: {"anti_alias": "zero-phase FIR low-pass at native rate, pass 45.0 Hz, trans 5.0 Hz (hamming, firwin, mne)", "resample": "scipy.signal.resample_poly to 100 Hz", "bandpass": "0.3-35.0 Hz zero-phase FIR (mne defaults)"}.

## Recordings (VERIFIED)

- found: 108
- read and stored: 107
- skipped: 1
  - 1 x ValueError: 97 stage rows lie before the EDF start; alignment doubtful

## Epochs and stages (VERIFIED)

- epochs: 119425; scored: 107269; unscored (-1): 12156

| stage | epochs | share of scored |
|---|---|---|
| W | 19435 | 0.181 |
| N1 | 4576 | 0.043 |
| N2 | 38967 | 0.363 |
| N3 | 25499 | 0.238 |
| REM | 18792 | 0.175 |

## Sampling rates and hardware prefilters (VERIFIED, EEG channels as stated in the headers)

| native rate | recordings |
|---|---|
| 512.0 Hz | 80 |
| 128.0 Hz | 12 |
| 256.0 Hz | 8 |
| 100.0 Hz | 4 |
| 200.0 Hz | 3 |

| EEG prefilter field(s) | recordings |
|---|---|
| LP:30.00Hz HP:0.30Hz NOTCH:50 | 85 |
| (not stated) | 11 |
| LP:30.00Hz HP:0.30Hz NOTCH:1 | 6 |
| LP:30.00Hz HP:0.50Hz NOTCH:50 | 2 |
| BP: 0.1HZ -100HZ | 1 |
| LP:100.00Hz HP:0.30Hz NOTCH:50 / LP:30.00Hz HP:0.30Hz NOTCH:50 | 1 |
| LP:30.00Hz HP:0.10Hz NOTCH:50 / LP:30.00Hz HP:0.30Hz NOTCH:50 | 1 |

## Channels

Channel rule (D-017): first of Fz-Cz, F4-C4, F3-C3 | F4-M1, F3-M2, F4-M2, F3-M1 | C4-M1, C3-M2 (A1/A2 treated as M1/M2). VERIFIED counts:

| rule outcome | recordings |
|---|---|
| frontal-central: F4-C4 | 100 |
| central-mastoid: C3-A2 | 3 |
| frontal-mastoid: F4-A1 | 3 |
| central-mastoid: C4-A1 | 1 |

Channels kept per recording:

| EEG channels | recordings |
|---|---|
| 13 | 42 |
| 5 | 29 |
| 9 | 24 |
| 2 | 3 |
| 6 | 3 |
| 12 | 2 |
| 16 | 2 |
| 11 | 1 |
| 4 | 1 |

Channel names kept (recordings containing each):

| channel | recordings |
|---|---|
| C4-A1 | 103 |
| C4-P4 | 100 |
| F4-C4 | 100 |
| P4-O2 | 100 |
| Fp2-F4 | 97 |
| C3-P3 | 71 |
| F3-C3 | 71 |
| P3-O1 | 71 |
| Fp1-F3 | 70 |
| F7-T3 | 47 |
| F8-T4 | 47 |
| T3-T5 | 45 |
| T4-T6 | 45 |
| C3-A2 | 10 |
| O2-A1 | 9 |
| O1-A2 | 6 |
| F2-F4 | 3 |
| F3-A2 | 3 |
| F4-A1 | 3 |
| F1-F3 | 1 |

- recordings with channels derived from monopolar leads: 2
- EEG-like labels dropped for an EMG-like hardware band (high-pass >= 5 Hz): {'F8-T4': 22, 'F7-T3': 22, 'T4-T6': 1, 'T3-T5': 1} (recording counts)

## Units and amplitude

| EEG unit field | recordings with it |
|---|---|
| µV | 102 |
| mV | 3 |
| n/a | 3 |

- recordings with an empty/unknown unit field, physical value taken as microvolt: 3 (CONDITIONAL)
- median epoch SD of the rule channel after preprocessing (uV), across recordings: p5 3.9, p25 4.9, p50 6.1, p75 8.1, p95 13.5 (VERIFIED)
- recordings with that median below 5 uV: 28; above 150 uV: 0

| rule-channel physical range in header | recordings | median epoch SD (uV), median over recordings |
|---|---|---|
| -62.50..62.50 | 86 | 5.86 |
| -71.43..71.43 | 3 | 6.25 |
| -142.86..142.86 | 3 | 6.27 |
| -156.250..156.250 | 2 | 23.54 |
| -250.00..250.00 | 2 | 10.29 |
| -50..50 | 1 | 13.78 |
| -156.097..156.097 | 1 | 17.92 |
| -728.756..728.756 | 1 | 4.20 |
| -671.398..671.398 | 1 | 7.20 |
| -556.793..555.9782 | 1 | 12.88 |
| -572.408..574.6498 | 1 | 21.03 |
| -7.79854..7.798546 | 1 | 5.08 |
| -305.423..303.0460 | 1 | 11.91 |
| 62.50..-62.50 | 1 | 5.41 |
| -1.00..1.00 | 1 | 5.86 |
| -66.67..66.67 | 1 | 14.77 |

## Label / signal length (VERIFIED)

- stage rows before the EDF start, dropped: 1 rows in 1 recordings
- stage rows after the end of the signal, dropped: 635 rows in 21 recordings
- recordings whose grid starts off the EDF's 30-s grid (origin != 0 s): 5
- truncated table rows ignored: 0 in 0 recordings
- MT epochs (unscored): counted in unscored above
- EDF record count in header differs from file size: 3 recordings (count taken from file size)

Diagnosis groups:

| group | recordings |
|---|---|
| nfle | 39 |
| rbd | 22 |
| n | 16 |
| plm | 10 |
| ins | 9 |
| narco | 5 |
| sdb | 4 |
| brux | 2 |
