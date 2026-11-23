#!/usr/bin/env bash
# E1 order of work under the 2026-11-23 budget (DECISION_LOG D-014). Serial on one GPU; every step resumable.
# Decisive arms first, seed 0 across all folds before further seeds, so partial results always cover 5 folds.
# raw and attnsleep under cv are the E0 runs (D-006) and are not retrained here.
set -u
cd "$(dirname "$0")/.."
q=scripts/run_e1.sh
$q cv "0" cascade cascade_placebo cascade_absolute
$q cv "0 1 2" gr grm_placebo grm grm_absolute
$q cv "1 2" cascade cascade_placebo cascade_absolute
$q cv "0 1" grm_nocritic cascade_nocritic
$q cv "0" r m grm_loose cascade_bands cascade_broad
for b in age cvpz; do
  $q $b "0 1 2" raw gr grm_placebo grm grm_absolute cascade cascade_placebo
done
echo "e1 queue done"
