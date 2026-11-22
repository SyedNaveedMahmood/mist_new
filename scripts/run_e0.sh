#!/usr/bin/env bash
# E0 (T6): raw and attnsleep, 5 folds x 3 seeds, frozen recipe, then every evaluation.
# Seed 0 reuses the recipe-sweep runs trained with the frozen class weighting
# (identical config); seeds 1 and 2 are trained here. Usage: run_e0.sh <cw>
set -u
cd "$(dirname "$0")/.."
cw=$1
PY=.venv/bin/python
for arm in attnsleep raw; do
  for f in 0 1 2 3 4; do
    src="runs/cw_${arm}_${cw}_cv${f}_s0"; dst="runs/e0_${arm}_cv${f}_s0"
    [ -e "$dst" ] || ln -s "$(basename "$src")" "$dst"
    for s in 1 2; do
      name="e0_${arm}_cv${f}_s${s}"
      [ -f "runs/$name/train_summary.json" ] && continue
      $PY scripts/train.py --arm "$arm" --name "$name" split.fold=$f train.seed=$s > "runs/logs/$name.log" 2>&1 \
        || echo "FAILED train $name"
    done
  done
done
for arm in attnsleep raw; do
  for f in 0 1 2 3 4; do
    for s in 0 1 2; do
      r="runs/e0_${arm}_cv${f}_s${s}"
      [ -f "$r/eval_test_cv_Fpz-Cz.json" ]    || $PY scripts/evaluate.py --run "$r" >> runs/logs/e0_eval.log 2>&1 || echo "FAILED eval $r cv"
      [ -f "$r/eval_test_cv_Pz-Oz.json" ]     || $PY scripts/evaluate.py --run "$r" --channel Pz-Oz >> runs/logs/e0_eval.log 2>&1 || echo "FAILED eval $r Pz-Oz"
      [ -f "$r/eval_test_sc2st_Fpz-Cz.json" ] || $PY scripts/evaluate.py --run "$r" --protocol sc2st >> runs/logs/e0_eval.log 2>&1 || echo "FAILED eval $r sc2st"
      [ -f "$r/eval_test_cv_Fpz-Cz_lf.json" ] || $PY scripts/evaluate.py --run "$r" --view lf >> runs/logs/e0_eval.log 2>&1 || echo "FAILED eval $r cv lf"
      [ -f "$r/eval_test_sc2st_Fpz-Cz_lf.json" ] || $PY scripts/evaluate.py --run "$r" --protocol sc2st --view lf >> runs/logs/e0_eval.log 2>&1 || echo "FAILED eval $r sc2st lf"
    done
  done
done
echo "e0 done"
