# Datasets and access plan

Checked on 2026-11-22. "Verified" means read from the dataset's own page or registry entry in search results this session. "From memory" means not checked; confirm before relying on it.

## Phase 1: already on disk

| Dataset | Path recorded in the old repository | Use | Notes |
| --- | --- | --- | --- |
| Sleep-EDF cassette, raw EDF | `/home/FA006/Desktop/Sleep_dl/SLeep_edf_fulldataset/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette` | All of phase 1 | Folder name has a capital L in `SLeep`. Licence: Open Data Commons Attribution 1.0 (verified) |
| Sleep-EDF telemetry, raw EDF | `.../1.0.0/sleep-telemetry` | Cohort-shift target | Seven hypnograms carry a previous-day EDF+ Recordingfield; the old audit validated all 44 pairs |
| `SC-subjects.xls`, `ST-subjects.xls` | `.../1.0.0/` | Age, sex, lights-off | Column layout unverified in code (T1) |
| Legacy trimmed NPZ (EDF-20, EDF-78, Pz-Oz) | `/home/FA006/Desktop/Dimension/dataset/Preprocessed ...` | Smoke tests only | Trimming rule unknown and label-conditioned |

Paths have not been checked from outside that machine. The canonical store needs roughly 12 GB.

## Phase 2: what to get, in what order

| Priority | Dataset | Why | Access | Size and channels | Action |
| --- | --- | --- | --- | --- | --- |
| 1 | **HMC** sleep staging database v1.1 | First independent target. Clinical cohort, AASM scoring, different hardware | Open on PhysioNet; CC BY 4.0 per the AWS open-data registry (verified). DOI 10.13026/t79q-fr32 | 151 whole-night recordings (verified). About 15.7 GB (old project note). EEG derivations F4-M1, C4-M1, O2-M1, C3-M2 at 256 Hz (from memory) | Download now. The earlier attempt was blocked only by 7 GB of free storage |
| 1 | **CAP Sleep Database** | Already on disk (`cap`, `cap_npz_out*` folders). 108 recordings, 16 healthy and 92 with disorders; R&K staging; expert **A-phase** micro-structure annotations (verified) | Open on PhysioNet; ODC Attribution 1.0 (verified) | At least 3 EEG channels per record; montage varies by record (verified) | Inspect what the `cap_npz_out*` folders contain and which script made them. Treat as untrusted until rebuilt from EDF |
| 2 | **PhysioNet/CinC Challenge 2018** | Large open clinical set; in the SleepBand benchmark | Open on PhysioNet (verified). Stage labels for the 994 training subjects; 989 test subjects are unlabelled (training/test counts verified; label availability from memory) | 1,985 subjects in total; tens of GB (from memory) | Plan storage first; download the training partition only |
| 2 | **ISRUC-Sleep** | In the SleepBand benchmark; two scorers on subgroup 1 | Open download from the ISRUC site (from memory) | About 100 subjects in subgroup 1, C3-A2 and others, 200 Hz (from memory) | Download; verify licence text on the site |
| 2 | **DOD-H / DOD-O** | Five scorers per night: the only way to put a ceiling on N1 for H5 | Open, via the Dreem open-data tools (from memory; check that the download endpoint still works) | 25 healthy, 55 with apnoea (verified earlier in the project) | Download |
| 3 | **SHHS visit 1** | Large community cohort; in every benchmark | NSRR data-use agreement. The configured folder `SHHS_NSRR` does not exist; a lowercase `shhs` folder does, contents unknown | More than 5,000 nights, C4-A1 and C3-A2, 125 Hz (from memory) | Check what `shhs` holds and under which agreement. If nothing usable, apply to NSRR now: approval takes days to weeks |
| 3 | **MASS** (+ **MODA** spindle consensus) | Expert spindle and K-complex events for morphology-competence probes; central channels | MASS needs an application with ethics approval. MODA annotations are CC0 on OSF (verified) but are useless without MASS signals | 200 nights at 256 Hz; MODA covers 180 subjects, 5,342 spindles (verified earlier) | Start the application now: it is the slowest item |
| 3 | **DREAMS** spindle and K-complex excerpts | Small expert-event sets, central channels | On Zenodo (record 2650142). The earlier project judged the usage permission unclear; a Hugging Face mirror labels it CC BY-NC-SA 4.0 (secondary source) | 8 spindle and 10 K-complex excerpts of 30 minutes | Read the licence on the Zenodo record itself and record it in `experiments/registry.yaml` before any use |
| 4 | MESA, MrOS (NSRR) | Scale, if leave-one-dataset-out needs more sources | NSRR agreement | Large | Only if phase 2 results justify it |

## Rules for every new dataset

1. Rebuild from the original files with a reader in `src/mist/data/`; never trust an inherited NPZ.
2. Write down the scoring standard, derivations, sampling rate, hardware filters if documented, and any overlap of people with datasets already in use.
3. Freeze subject-level splits before training anything that touches it.
4. A pretrained baseline is never tested on data from its own pretraining corpus.
5. Raw signals, per-epoch outputs and checkpoints stay off GitHub.

## Open questions about data

- What is in `/home/FA006/Desktop/Dimension/dataset/shhs` and in the `cap*` folders?
- How much free disk is on the workstation? HMC plus the Sleep-EDF store needs about 30 GB; CinC 2018 much more.
- Hardware phase response is the nuisance the M stream is most exposed to. For each dataset, look for documented analogue filter settings; they decide how hard the cohort shift is for morphology.
