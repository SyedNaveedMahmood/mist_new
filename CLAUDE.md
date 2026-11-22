# Instructions for Claude Code

## What this project is

A methods paper. The model (MIST-GRM) factorises single-channel sleep EEG tokens into gain, rhythm and morphology so that two questions can be answered by switching streams on and off: does morphology help transfer, and which morphology matters (N1 in particular). The owner is Syed Naveed Mahmood. An earlier attempt (repository `Proto-sleep`) produced careful data audits and several negative results; read `docs/LESSONS_FROM_PROTO_SLEEP.md` before proposing anything that resembles it.

Start with `README.md`, then `docs/TASKS.md`. Work the tasks in order.

## Commands

```bash
pip install -e ".[data,dev]"
pytest                                   # must pass before every commit
python scripts/train.py --arm <arm> --name <run> [key=value ...]
python scripts/evaluate.py --run runs/<run> [--channel Pz-Oz] [--protocol sc2st]
```

Configs merge `configs/base.yaml`, then `configs/arms/<arm>.yaml`, then `key=value` overrides. Local paths live in `configs/paths.yaml` (git-ignored).

## Rules that are not negotiable

1. **Lockbox.** The 20 lockbox people are evaluated once, in task T12, by the owner's instruction. No other code path may load them. Do not weaken the guard in `engine/loop.py` or `scripts/evaluate.py`.
2. **Splits are frozen.** Never regenerate or edit `splits/sleepedf_v1.json`. Splits are by person, never by night or epoch.
3. **No target information.** Checkpoints are chosen on source-validation people only. No hyperparameter is chosen by looking at a shift-protocol test score. ST is development-grade; say so wherever it is reported.
4. **Raw data only.** The legacy trimmed NPZ folders are for smoke tests. Any number that enters a report comes from the store built by `scripts/prepare_sleepedf.py`.
5. **Nothing private in git.** No EEG, labels per epoch, checkpoints, manifests with participant rows, or `configs/paths.yaml`.
6. **Do not weaken the invariance tests** in `tests/test_factorizer.py` to make a change pass. If the factorisation changes, the properties must be re-proved or the claim removed from `docs/ARCHITECTURE.md`.
7. **Stop and ask** when a choice belongs to the owner (`docs/DECISIONS.md`) or when a gate fails.

## How to report

- Use the labels VERIFIED, CONDITIONAL, BLOCKED, UNKNOWN, NOT TESTED, with scope.
- Every reported number needs a `runs/.../eval_*.json` and an entry in `experiments/registry.yaml` (append-only).
- Report null and negative results with the same care as positive ones. The owner prefers accuracy to agreement: point out errors, including in these documents.
- Shifts inside Sleep-EDF are "within-database shifts", not cross-dataset transfer.
- Do not invent citations. `docs/REFERENCES.md` separates confirmed entries from ones whose authors still need checking.

## Things known to be unverified

- `src/mist/data/sleepedf.py` has not read a real Sleep-EDF file. Task T1 checks it, including the spreadsheet columns and seven ST hypnograms with unusual headers.
- All evidence for the design so far is on synthetic signals.
- The cascade topology (`docs/ARCHITECTURE.md`, section 8) is the main model since 2026-11-22. Implemented 2026-11-23; its section 8.5 properties are tested on synthetic signals only (`tests/test_cascade.py`), not on real EEG. The reading of the supervisor's sketch is an assumption.
- The morphology stream is sensitive to phase distortion (see `docs/ARCHITECTURE.md`, section 6). Expect this to matter on the cohort shift.
- `src/mist/models/attnsleep.py` came from the earlier project; check the upstream licence before the repository is made public.

## Style

Python 3.10+, PyTorch. Small modules, one responsibility each, a test per module. No new dependency without a reason written in the commit message. Comments state constraints the code cannot show.
