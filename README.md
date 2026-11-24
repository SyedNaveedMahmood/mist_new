# mist_new: MIST-GRM

Single-channel sleep staging with waveform tokens split, exactly, into **gain**, **rhythm** and **morphology** (waveform shape), built to answer two questions (re-aimed at transfer on 2026-11-24, DECISIONS D9, **after** the interim H1 result of GPU session 1 was seen):

1. What in single-channel sleep EEG carries over to new recordings: gain, rhythm or waveform shape?
2. Does a model restricted to the part that carries over beat standard models on unseen datasets?

The pre-specified hypotheses H1 to H6 are unchanged (docs/EXPERIMENT_PLAN.md). H1 (shape improves transfer) is one tested factor; H3 (the factorisation without shape helps transfer) is the central claim to test.

The original questions, kept as history: (1) Can a model that understands EEG morphology transfer better? (2) Which morphology matters, and does better morphology for N1 raise every stage?

Phase 1 runs on Sleep-EDF Expanded only (within-database shifts). Current state: reports/STATUS.md.

## Where things stand (2026-11-22)

| Item | Status |
| --- | --- |
| Factorisation, model, baseline, training and evaluation code | Written; 25 unit and smoke tests pass on synthetic data |
| By-construction properties (gain invariance, phase-blind G/R, power-blind M, lossless) | VERIFIED by tests, synthetic signals |
| Invariance to smooth zero-phase filters | VERIFIED on synthetic 1/f noise (about 1% change against 17 to 33% for raw tokens); NOT TESTED on real EEG |
| Sleep-EDF reader | Tested on a synthetic EDF pair; NOT TESTED on the real files |
| Splits | Code ready; file not generated (needs the real manifest and decision D1) |
| Cascade topology (supervisor revision, 2026-11-22; docs/ARCHITECTURE.md section 8) | Implemented 2026-11-23; properties VERIFIED on synthetic signals; NOT TESTED on real EEG |
| Any result on RQ1 or RQ2 | NOT TESTED |
| Novelty | Screened, not guaranteed: see docs/NOVELTY_CHECK.md |

## Read in this order

1. `docs/ARCHITECTURE.md`: the design, module by module
2. `docs/EXPERIMENT_PLAN.md`: hypotheses, protocols, gates, statistics
3. `docs/NOVELTY_CHECK.md`: closest prior work and what can be claimed
4. `docs/DATASETS_AND_ACCESS.md`: what to download or apply for, and when
5. `docs/LESSONS_FROM_PROTO_SLEEP.md`: what the earlier project settled
6. `docs/DECISIONS.md`: choices only the owner can make
7. `docs/TASKS.md`: the ordered work list
8. `CLAUDE.md`: operating rules for Claude Code

## Quick start

```bash
pip install -e ".[data,dev]"
pytest                                             # about one minute on CPU
python scripts/sanity_phase_only.py                # synthetic checks of the two design premises
cp configs/paths.example.yaml configs/paths.yaml   # then edit
python scripts/prepare_sleepedf.py --cohorts SC ST # raw EDF -> store; compare totals with gate G0
python scripts/make_splits.py                      # once, after decision D1
python scripts/train.py --arm grm --name grm_cv0_s0 split.fold=0 train.seed=0
python scripts/evaluate.py --run runs/grm_cv0_s0                      # in-domain
python scripts/evaluate.py --run runs/grm_cv0_s0 --channel Pz-Oz      # montage shift
python scripts/evaluate.py --run runs/grm_cv0_s0 --protocol sc2st     # cohort shift
python scripts/compare.py runs/a/eval_test_cv_Fpz-Cz.json real runs/b/eval_test_cv_Fpz-Cz.json real
```

## Layout

```
src/mist/models/factorizer.py   night whitening and the G / R / M split (no parameters)
src/mist/models/mist_grm.py     encoders, shape codebook, critic, composer, context, head
src/mist/models/attnsleep.py    baseline, ported from Proto-sleep
src/mist/data/                  Sleep-EDF reader, store, splits, dataset, surrogates
src/mist/engine/loop.py         training and evaluation
src/mist/metrics.py             subject-level metrics and paired bootstrap
configs/                        base.yaml, arms/*.yaml, paths.example.yaml
scripts/                        prepare, split, train, evaluate, compare, checks
tests/                          properties the design depends on
experiments/registry.yaml       append-only record of runs
```

Data, checkpoints and per-epoch outputs are git-ignored and must stay off GitHub.
