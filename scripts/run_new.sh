#!/usr/bin/env bash
# Experiments added after the external review (CPU only):
#   E12 safety query, E13 structured anchors, and the Depth Pro pairs (main study, E5, E12, E11)
#   when their tables can be built (Depth Pro caches from slurm/cache.sbatch with MODEL_JOBS="depthpro --flip").
#   bash scripts/run_new.sh                    # everything available
#   PARTS="e12 e13" bash scripts/run_new.sh    # only the parts listed (e12 e13 depthpro)
#   PARTS="e12 e13" E13_PAIRS="kitti:depthpro nyu:depthpro" bash scripts/run_new.sh
#   PARTS=e14 bash scripts/run_new.sh          # A-dist (docs/ANCHOR_METHOD_PROPOSAL.md); not in the default PARTS
# e12, e13 and e14 include the Depth Pro pairs whenever their tables exist.
# Env: PY, JOBS (default 6: E11-style jobs need up to ~20 GB each), E12_SEEDS (20), E13_SEEDS (5),
#      E13_DRAWS (3), E13_PAIRS (all six), SEEDS (main, 20), E5_SEEDS (10),
#      E14_SEEDS (3), E14_DRAWS (2), E14_PAIRS (all six), E14_PATTERNS (all five).
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

if want depthpro; then
  echo "=== $(date -Is) Depth Pro tables"
  for ds in nyu kitti diode_indoor diode_outdoor; do
    if have "$ds" depthpro; then echo "table $ds/depthpro exists"; continue; fi
    if [[ -d "${SPACAL_CACHE_ROOT:-cache}/$ds/depthpro" ]]; then
      $PY -m experiments.prepare_tables --dataset "$ds" --model depthpro || echo "!!! FAILED: table $ds/depthpro"
    else
      echo "skip $ds/depthpro (no cache)"
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

if want depthpro; then
  echo "=== $(date -Is) Depth Pro: main study, E5, E11"
  for ds in nyu kitti diode_indoor diode_outdoor; do
    have "$ds" depthpro || continue
    spawn "main_${ds}__depthpro" $PY -W ignore -m experiments.run_calibration --dataset "$ds" --model depthpro \
      --n-seeds "${SEEDS:-20}"
  done
  for spec in "nyu diode_indoor" "kitti diode_outdoor"; do
    read -r src tgt <<< "$spec"
    have "$src" depthpro && have "$tgt" depthpro || continue
    spawn "e5_${src}__depthpro" $PY -W ignore -m experiments.e5_shift --source "$src" --target "$tgt" \
      --model depthpro --n-seeds "${E5_SEEDS:-10}"
  done
  for ds in kitti nyu; do
    have "$ds" depthpro || continue
    spawn "e11_${ds}__depthpro" $PY -W ignore -m experiments.e11_anchors --dataset "$ds" --model depthpro \
      --n-seeds 5 --draws 6
  done
fi
wait

if want depthpro; then
  echo "=== $(date -Is) summarize main study (all models)"
  $PY -m experiments.summarize --exp-id main || echo "!!! FAILED: summarize"
fi
grep -l "Traceback" logs/e12_*.log logs/e13_*.log logs/e14_*.log logs/*depthpro*.log 2>/dev/null \
  && echo "some jobs crashed (see above)" || true
echo "=== $(date -Is) done. results/e12, results/e13, results/e14, results/main/_summary"
