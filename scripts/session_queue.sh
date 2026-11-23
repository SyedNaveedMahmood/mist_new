#!/usr/bin/env bash
# GPU session queue (EXPERIMENT_PLAN section 12). Usage:
#   scripts/session_queue.sh <session number> "<YYYY-MM-DD HH:MM end>" [priority list, default experiments/priority.txt]
# Runs in the foreground; launch with nohup. Every GPU process ends 15 minutes before the end time.
# Outside sessions nothing here may run; CPU-only jobs set CUDA_VISIBLE_DEVICES= (empty).
set -u
cd "$(dirname "$0")/.."
n=$1; end=$2; plan=${3:-experiments/priority.txt}
mkdir -p runs/logs
.venv/bin/python scripts/session_queue.py --end "$end" --plan "$plan" 2>&1 | tee -a "runs/logs/session_${n}.log"
