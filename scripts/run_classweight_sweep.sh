#!/usr/bin/env bash
# Recipe step 1b (owner decision 2026-11-22): attnsleep and raw, 5 folds, seed 0,
# class_weight null vs sqrt_inv. Training only: train.py never touches test people.
set -u
cd "$(dirname "$0")/.."
for cw in null sqrt_inv; do
  for arm in attnsleep raw; do
    for f in 0 1 2 3 4; do
      name="cw_${arm}_${cw}_cv${f}_s0"
      [ -f "runs/$name/train_summary.json" ] && continue
      .venv/bin/python scripts/train.py --arm "$arm" --name "$name" split.fold=$f train.seed=0 train.class_weight=$cw \
        > "runs/logs/$name.log" 2>&1 || echo "FAILED $name"
    done
  done
done
echo sweep done
