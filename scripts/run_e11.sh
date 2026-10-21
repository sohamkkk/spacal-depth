#!/usr/bin/env bash
# E11 sparse-anchor conformal on every available (dataset, model) table, JOBS at a time.
#   bash scripts/run_e11.sh                                   # full sweep
#   E11_PAIRS="kitti:dav2-outdoor-large kitti:unidepthv2-large" KS="0 1 5 20" SEEDS=3 DRAWS=3 \
#     bash scripts/run_e11.sh                                 # go/no-go test
# Env: PY, E11_PAIRS (dataset:model ...), KS, SEEDS (default 10), DRAWS (default 5), JOBS (default 4).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SEEDS=${SEEDS:-10}
DRAWS=${DRAWS:-5}
JOBS=${JOBS:-4}
KS=${KS:-"0 1 3 5 9 20 50 200"}
E11_PAIRS=${E11_PAIRS:-"kitti:dav2-outdoor-large kitti:unidepthv2-large nyu:dav2-indoor-large nyu:unidepthv2-large diode_indoor:dav2-indoor-large diode_outdoor:dav2-outdoor-large"}
mkdir -p logs
threads=$(( $(nproc) / JOBS )); (( threads > 0 )) || threads=1

running=0
for pair in $E11_PAIRS; do
  ds=${pair%%:*}; model=${pair#*:}
  [[ -f "${SPACAL_PREPARED_ROOT:-prepared}/${ds}__${model}.npz" ]] || { echo "skip $ds/$model (no table)"; continue; }
  echo "=== $(date -Is) e11 $ds/$model"
  # shellcheck disable=SC2086
  OMP_NUM_THREADS=$threads $PY -W ignore -m experiments.e11_anchors --dataset "$ds" --model "$model" \
    --ks $KS --n-seeds "$SEEDS" --draws "$DRAWS" > "logs/e11_${ds}__${model}.log" 2>&1 &
  running=$((running + 1))
  if (( running % JOBS == 0 )); then wait; fi
done
wait
echo "done. Summaries: results/e11/*/summary.md"
