#!/usr/bin/env bash
# E1 (T7). Usage: run_e1.sh <block> <seeds> <arm...>
#   block cv      : train under cv on Fpz-Cz, 5 folds; evaluate in-domain, Pz-Oz, sc2st, and lf (cv, sc2st)
#   block cvpz    : train under cv on Pz-Oz, 5 folds; evaluate in-domain (Pz-Oz) and Fpz-Cz (reverse montage)
#   block age     : train young2old and old2young (fold 0 as validation); evaluate the protocol's test band
# seeds: e.g. "0 1 2". Runs are skipped when already trained / evaluated (resumable).
set -u
cd "$(dirname "$0")/.."
block=$1; seeds=$2; shift 2
PY=.venv/bin/python
ev() { local r=$1 f=$2; shift 2; [ -f "$r/$f" ] || $PY scripts/evaluate.py --run "$r" "$@" >> runs/logs/e1_eval.log 2>&1 || echo "FAILED eval $r $*"; }
for arm in "$@"; do
  for s in $seeds; do
    if [ "$block" = cv ] || [ "$block" = cvpz ]; then
      for f in 0 1 2 3 4; do
        if [ "$block" = cv ]; then name="e1_${arm}_cv${f}_s${s}"; extra=""; else name="e1pz_${arm}_cv${f}_s${s}"; extra="data.channel=Pz-Oz"; fi
        r="runs/$name"
        [ -f "$r/train_summary.json" ] || $PY scripts/train.py --arm "$arm" --name "$name" split.fold=$f train.seed=$s $extra \
            > "runs/logs/$name.log" 2>&1 || { echo "FAILED train $name"; continue; }
        if [ "$block" = cv ]; then
          ev "$r" eval_test_cv_Fpz-Cz.json
          ev "$r" eval_test_cv_Pz-Oz.json --channel Pz-Oz
          ev "$r" eval_test_sc2st_Fpz-Cz.json --protocol sc2st
          ev "$r" eval_test_cv_Fpz-Cz_lf.json --view lf
          ev "$r" eval_test_sc2st_Fpz-Cz_lf.json --protocol sc2st --view lf
        else
          ev "$r" eval_test_cv_Pz-Oz.json
          ev "$r" eval_test_cv_Fpz-Cz.json --channel Fpz-Cz
        fi
      done
    else
      for p in young2old old2young; do
        name="e1_${arm}_${p}_s${s}"; r="runs/$name"
        [ -f "$r/train_summary.json" ] || $PY scripts/train.py --arm "$arm" --name "$name" split.protocol=$p split.fold=0 train.seed=$s \
            > "runs/logs/$name.log" 2>&1 || { echo "FAILED train $name"; continue; }
        ev "$r" "eval_test_${p}_Fpz-Cz.json"
      done
    fi
  done
done
echo "e1 block $block done"
