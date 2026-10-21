#!/usr/bin/env bash
# Every CPU experiment, after caches exist (run on the remote box's CPUs or a laptop).
#   bash scripts/run_all_cpu.sh                 # full study
#   SEEDS=3 E8_SEEDS=5 bash scripts/run_all_cpu.sh   # quick pass
# Env: PY (python, default python), SEEDS (main study seeds, default 20),
#      E8_SEEDS (default 50), E5_SEEDS (default 10), JOBS (parallel run_calibration jobs, default 4).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SEEDS=${SEEDS:-20}
E8_SEEDS=${E8_SEEDS:-50}
E5_SEEDS=${E5_SEEDS:-10}
JOBS=${JOBS:-4}
mkdir -p logs

echo "=== $(date -Is) prepare tables"
$PY -m experiments.prepare_tables --all

echo "=== $(date -Is) table diagnostics (split units, UniDepth confidence scale)"
$PY -m experiments.diag_tables > logs/diag_tables.log 2>&1 || echo "diag_tables failed (see logs/diag_tables.log)"

echo "=== $(date -Is) E0 base metrics + E1 synthetic"
$PY -m experiments.e0_base_metrics > logs/e0.log 2>&1 &
$PY -m experiments.e1_synthetic --reps "${E1_REPS:-100}" > logs/e1.log 2>&1 &
wait

echo "=== $(date -Is) main study (E2/E3/E4/E6/E9)"
PAIRS=(
  "nyu dav2-indoor-large"   "nyu unidepthv2-large"
  "kitti dav2-outdoor-large" "kitti unidepthv2-large"
  "diode_indoor dav2-indoor-large" "diode_outdoor dav2-outdoor-large"
)
running=0
for pair in "${PAIRS[@]}"; do
  read -r ds model <<< "$pair"
  [[ -f "${SPACAL_PREPARED_ROOT:-prepared}/${ds}__${model}.npz" ]] || { echo "skip $ds/$model (no table)"; continue; }
  # OMP threads split across concurrent jobs so HistGradientBoosting does not oversubscribe
  threads=$(( $(nproc) / JOBS )); (( threads > 0 )) || threads=1
  OMP_NUM_THREADS=$threads $PY -W ignore -m experiments.run_calibration \
    --dataset "$ds" --model "$model" --n-seeds "$SEEDS" > "logs/main_${ds}__${model}.log" 2>&1 &
  running=$((running + 1))
  if (( running % JOBS == 0 )); then wait; fi
done
wait

have() { [[ -f "${SPACAL_PREPARED_ROOT:-prepared}/$1__$2.npz" ]]; }

echo "=== $(date -Is) E5 shift + E8 cal size + qualitative"
export OMP_NUM_THREADS=$(( $(nproc) / 6 > 0 ? $(nproc) / 6 : 1 ))   # 6 concurrent jobs below
for spec in "nyu diode_indoor dav2-indoor-large" "kitti diode_outdoor dav2-outdoor-large" \
            "nyu diode_indoor unidepthv2-large" "kitti diode_outdoor unidepthv2-large"; do
  read -r src tgt model <<< "$spec"
  if have "$src" "$model" && have "$tgt" "$model"; then
    $PY -W ignore -m experiments.e5_shift --source "$src" --target "$tgt" --model "$model" \
      --n-seeds "$E5_SEEDS" > "logs/e5_${src}__${model}.log" 2>&1 &
  fi
done
for spec in "nyu dav2-indoor-large" "kitti dav2-outdoor-large"; do
  read -r ds model <<< "$spec"
  have "$ds" "$model" && $PY -W ignore -m experiments.e8_calsize --dataset "$ds" --model "$model" \
    --n-seeds "$E8_SEEDS" > "logs/e8_${ds}.log" 2>&1 &
done
wait
for spec in "nyu dav2-indoor-large" "kitti dav2-outdoor-large" "nyu unidepthv2-large"; do
  read -r ds model <<< "$spec"
  have "$ds" "$model" || continue
  $PY -W ignore -m experiments.qualitative --dataset "$ds" --model "$model" > "logs/qual_${ds}__${model}.log" 2>&1 \
    || echo "qualitative $ds/$model failed (see logs)"
done

echo "=== $(date -Is) summarize + cost + figures"
$PY -m experiments.summarize --exp-id main
$PY -m experiments.e10_cost
$PY -m experiments.make_figures
echo "done. Tables: results/main/_summary/summary.md   Figures: results/figures/"
