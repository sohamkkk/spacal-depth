#!/usr/bin/env bash
# Experiments added after the external review (CPU only):
#   E12 safety query, E13 structured anchors, and the Depth Pro pairs (main study, E5, E12, E11)
#   when their tables can be built (Depth Pro caches from slurm/cache.sbatch with MODEL_JOBS="depthpro --flip").
#   bash scripts/run_new.sh                    # everything available
#   PARTS="e12 e13" bash scripts/run_new.sh    # only the parts listed (e12 e13 depthpro)
#   PARTS="e12 e13" E13_PAIRS="kitti:depthpro nyu:depthpro" bash scripts/run_new.sh
#   PARTS=e14 bash scripts/run_new.sh          # A-dist (docs/ANCHOR_METHOD_PROPOSAL.md); not in the default PARTS
#   PARTS=e15 bash scripts/run_new.sh          # calibration vs test anchor layout, LSCP; not in the default PARTS
#   PARTS=e16 bash scripts/run_new.sh          # noisy and real-LiDAR anchors (lidar only on tables with the field,
#                                              # see scripts/add_kitti_lidar.py); not in the default PARTS
#   PARTS=metric3d bash scripts/run_new.sh     # Metric3D v2 tables (all five datasets), main study, E5, E11
#   PARTS=eth3d bash scripts/run_new.sh        # ETH3D tables and main study for UniDepthV2 and Depth Pro
# e12-e16 include the Depth Pro pairs whenever their tables exist; for Metric3D pass e.g.
#   E15_PAIRS="kitti:metric3d-large nyu:metric3d-large".
# Env: PY, JOBS (default 6: E11-style jobs need up to ~20 GB each), E12_SEEDS (20), E13_SEEDS (5),
#      E13_DRAWS (3), E13_PAIRS (all six), SEEDS (main, 20), E5_SEEDS (10),
#      E14_SEEDS (3), E14_DRAWS (2), E14_PAIRS (all six), E14_PATTERNS (all five),
#      E15_SEEDS (3), E15_DRAWS (2), E15_PAIRS (all six), E15_PATTERNS (all five test layouts),
#      E16_SEEDS (3), E16_DRAWS (2), E16_PAIRS (all six), E16_PATTERNS (uniform).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
JOBS=${JOBS:-6}
PARTS=${PARTS:-"e12 e13 depthpro"}
PREP=${SPACAL_PREPARED_ROOT:-prepared}
mkdir -p logs
have() { [[ -f "$PREP/$1__$2.npz" ]]; }
want() { [[ " $PARTS " == *" $1 "* ]]; }
threads=$(( $(nproc) / JOBS > 0 ? $(nproc) / JOBS : 1 ))
running=0
spawn() {  # log_name command...
  local log=$1; shift
  OMP_NUM_THREADS=$threads "$@" > "logs/$log.log" 2>&1 || echo "!!! FAILED: $log (see logs/$log.log)" &
  running=$((running + 1))
  if (( running % JOBS == 0 )); then wait; fi
}

BASE_PAIRS="kitti:dav2-outdoor-large kitti:unidepthv2-large nyu:dav2-indoor-large nyu:unidepthv2-large"
DP_PAIRS="kitti:depthpro nyu:depthpro"

EXTRA=()        # "dataset model" pairs whose tables are built and main study run here
EXTRA_MODELS=() # models that also get E5 and E11
if want depthpro; then
  for ds in nyu kitti diode_indoor diode_outdoor; do EXTRA+=("$ds depthpro"); done
  EXTRA_MODELS+=(depthpro)
fi
if want metric3d; then
  for ds in nyu kitti diode_indoor diode_outdoor eth3d; do EXTRA+=("$ds metric3d-large"); done
  EXTRA_MODELS+=(metric3d-large)
fi
if want eth3d; then
  for model in unidepthv2-large depthpro; do EXTRA+=("eth3d $model"); done
fi

if (( ${#EXTRA[@]} )); then
  echo "=== $(date -Is) tables: ${EXTRA[*]}"
  for spec in "${EXTRA[@]}"; do
    read -r ds model <<< "$spec"
    if have "$ds" "$model"; then echo "table $ds/$model exists"; continue; fi
    if [[ -d "${SPACAL_CACHE_ROOT:-cache}/$ds/$model" ]]; then
      $PY -m experiments.prepare_tables --dataset "$ds" --model "$model" || echo "!!! FAILED: table $ds/$model"
    else
      echo "skip $ds/$model (no cache)"
    fi
  done
fi

if want e12 || want depthpro; then
  echo "=== $(date -Is) E12 safety query"
  if want e12; then pairs="$BASE_PAIRS $DP_PAIRS"; else pairs="$DP_PAIRS"; fi
  for pair in $pairs; do
    ds=${pair%%:*}; model=${pair#*:}
    have "$ds" "$model" || { echo "skip e12 $ds/$model (no table)"; continue; }
    spawn "e12_${ds}__${model}" $PY -W ignore -m experiments.e12_safety --dataset "$ds" --model "$model" \
      --n-seeds "${E12_SEEDS:-20}"
  done
fi

if want e13; then
  echo "=== $(date -Is) E13 structured anchors"
  # uniform = the E11 placement with the same ks/seeds/draws, as the reference for the distance breakdown
  for pattern in uniform rows cluster edges near; do
    for pair in ${E13_PAIRS:-$BASE_PAIRS $DP_PAIRS}; do
      ds=${pair%%:*}; model=${pair#*:}
      have "$ds" "$model" || { echo "skip e13 $ds/$model (no table)"; continue; }
      spawn "e13_${ds}__${model}__${pattern}" $PY -W ignore -m experiments.e11_anchors --exp-id e13 \
        --pattern "$pattern" --dataset "$ds" --model "$model" --ks 20 50 200 --no-oracle --anchor-dist \
        --n-seeds "${E13_SEEDS:-5}" --draws "${E13_DRAWS:-3}"
    done
  done
fi

if want e14; then
  echo "=== $(date -Is) E14 A-dist"
  for pattern in ${E14_PATTERNS:-uniform rows cluster edges near}; do
    for pair in ${E14_PAIRS:-$BASE_PAIRS $DP_PAIRS}; do
      ds=${pair%%:*}; model=${pair#*:}
      have "$ds" "$model" || { echo "skip e14 $ds/$model (no table)"; continue; }
      spawn "e14_${ds}__${model}__${pattern}" $PY -W ignore -m experiments.e11_anchors --exp-id e14 \
        --pattern "$pattern" --dataset "$ds" --model "$model" --ks 5 10 20 50 200 --no-oracle --anchor-dist \
        --dist-kinds median knn kernel bilateral --n-seeds "${E14_SEEDS:-3}" --draws "${E14_DRAWS:-2}"
    done
  done
fi

if want e15; then
  echo "=== $(date -Is) E15 calibration vs test anchor layout, LSCP"
  for pattern in ${E15_PATTERNS:-uniform rows cluster edges near}; do
    for pair in ${E15_PAIRS:-$BASE_PAIRS $DP_PAIRS}; do
      ds=${pair%%:*}; model=${pair#*:}
      have "$ds" "$model" || { echo "skip e15 $ds/$model (no table)"; continue; }
      spawn "e15_${ds}__${model}__${pattern}" $PY -W ignore -m experiments.e15_layouts \
        --pattern "$pattern" --dataset "$ds" --model "$model" --ks 20 50 200 --alphas 0.1 0.05 \
        --n-seeds "${E15_SEEDS:-3}" --draws "${E15_DRAWS:-2}"
    done
  done
fi

if want e16; then
  echo "=== $(date -Is) E16 noisy and LiDAR anchors"
  for pattern in ${E16_PATTERNS:-uniform}; do
    for pair in ${E16_PAIRS:-$BASE_PAIRS $DP_PAIRS}; do
      ds=${pair%%:*}; model=${pair#*:}
      have "$ds" "$model" || { echo "skip e16 $ds/$model (no table)"; continue; }
      spawn "e16_${ds}__${model}__${pattern}" $PY -W ignore -m experiments.e16_noisy \
        --pattern "$pattern" --dataset "$ds" --model "$model" --ks 20 200 --alphas 0.1 \
        --n-seeds "${E16_SEEDS:-3}" --draws "${E16_DRAWS:-2}"
    done
  done
fi

if (( ${#EXTRA[@]} )); then
  echo "=== $(date -Is) main study: ${EXTRA[*]}; E5 and E11: ${EXTRA_MODELS[*]:-none}"
  for spec in "${EXTRA[@]}"; do
    read -r ds model <<< "$spec"
    have "$ds" "$model" || continue
    spawn "main_${ds}__${model}" $PY -W ignore -m experiments.run_calibration --dataset "$ds" --model "$model" \
      --n-seeds "${SEEDS:-20}"
  done
  for model in "${EXTRA_MODELS[@]}"; do
    for spec in "nyu diode_indoor" "kitti diode_outdoor"; do
      read -r src tgt <<< "$spec"
      have "$src" "$model" && have "$tgt" "$model" || continue
      spawn "e5_${src}__${model}" $PY -W ignore -m experiments.e5_shift --source "$src" --target "$tgt" \
        --model "$model" --n-seeds "${E5_SEEDS:-10}"
    done
    for ds in kitti nyu; do
      have "$ds" "$model" || continue
      spawn "e11_${ds}__${model}" $PY -W ignore -m experiments.e11_anchors --dataset "$ds" --model "$model" \
        --n-seeds 5 --draws 6
    done
  done
fi
wait

if (( ${#EXTRA[@]} )); then
  echo "=== $(date -Is) summarize main study (all models)"
  $PY -m experiments.summarize --exp-id main || echo "!!! FAILED: summarize"
fi
grep -l "Traceback" logs/e12_*.log logs/e13_*.log logs/e14_*.log logs/e15_*.log logs/e16_*.log logs/main_*.log \
  logs/e5_*.log logs/e11_*.log 2>/dev/null && echo "some jobs crashed (see above)" || true
echo "=== $(date -Is) done. results/e12 … results/e16, results/main/_summary"
