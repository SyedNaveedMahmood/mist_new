# Decisions that belong to the project owner

Defaults are what the code does today. Change the answer here and in the config before T2.

| ID | Decision | Default | Alternative | Why it matters |
| --- | --- | --- | --- | --- |
| D1 | Release 35 of the 55 never-used SC people into development; keep 20 locked | **Yes** | Keep all 55 locked | Without the release, development has 19 people, all aged 25 to 34: no age-shift protocol and no stable 5-fold estimate. The release cannot be undone |
| D2 | Primary evaluation window | `std` (label-conditioned trim), with a label-free lights-off window as sensitivity analysis | Label-free window as primary | `std` is comparable with the literature but uses labels to pick the window; the earlier protocol barred it for zero-shot claims |
| D3 | Night reference computed from the test recording's own unlabelled signal | **Allowed**, with `grm_absolute` as the strict-inductive ablation | Source-only statistics everywhere | It is ordinary per-recording normalisation, but the earlier protocol forbade any target statistics |
| D4 | Keep SC 06, 23, 36, 74 quarantined | **Yes** | Re-audit and readmit | Carried over from the earlier audit; reasons were not re-checked here |
| D5 | ST role | Development-grade cohort-shift target | Hold out half of ST people untouched | ST was already examined in earlier analyses |
| D6 | Working name | MIST-GRM | any | Cosmetic |

Sign-off log (owner, date, decision IDs):

- Syed Naveed Mahmood, 2026-11-22: D1 = Yes (release 35, keep 20 locked; `make_splits.py` defaults n_lockbox=20, seed 20261122). D2 to D6 = the defaults in the table above. Conveyed in the owner's implementation brief and recorded by Claude Code.
- Syed Naveed Mahmood, 2026-11-22: recipe fix (no early stopping; class weighting chosen once on validation by a fixed rule; no other recipe change), G0b redefined as attnsleep 5-fold pooled test macro-F1 >= 0.70 under the frozen recipe, Sleep-EDF-20 literature replication, lockbox hardening. Conveyed in the owner's message, recorded by Claude Code.
- Syed Naveed Mahmood, 2026-11-22: **delegation while offline.** Claude Code continues without waiting and decides on its own, overriding CLAUDE.md rule 7 and earlier "stop and ask / wait" instructions, within the owner's hard limits (no lockbox / T12, splits frozen, no test-informed choices, no push to main or history rewrite, no deletion outside repo/runs/store/own dataset folders, no restricted data, disk floor 50 GB, nothing private in git, no weakened tests). Pre-decided answers for gate failures are in the owner's message and copied into docs/DECISION_LOG.md. Reserved for the owner: lockbox, anything breaking a hard limit, spending/accounts/applications/third-party contact, making the repository public or changing its licence (tracked in docs/OPEN_QUESTIONS.md).
