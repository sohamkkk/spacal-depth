#!/usr/bin/env bash
# Rerun after the S6 bound, V4-scene/K2-scene and E11 A-local fixes:
#   main study (all scores x all calibrators), E5, E11, then summaries + figures.
# The previous results are copied to results/{main,e5,e11}_v1 first (once).
#   bash scripts/run_fixes.sh
# Env: PY, SEEDS (main, default 20), E5_SEEDS (default 10), E11_SEEDS (default 5), E11_DRAWS (default 6).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SEEDS=${SEEDS:-20}
E5_SEEDS=${E5_SEEDS:-10}
RES=${SPACAL_RESULTS_ROOT:-results}
mkdir -p logs
have() { [[ -f "${SPACAL_PREPARED_ROOT:-prepared}/$1__$2.npz" ]]; }

for d in main e5 e11; do
  if [[ -d "$RES/$d" && ! -d "$RES/${d}_v1" ]]; then
    cp -r "$RES/$d" "$RES/${d}_v1"
    echo "backed up $RES/$d -> ${d}_v1"
  fi
done

main_job() {  # dataset model threads
  have "$1" "$2" || { echo "skip $1/$2 (no table)"; return 0; }
  OMP_NUM_THREADS=$3 $PY -W ignore -m experiments.run_calibration --dataset "$1" --model "$2" \
    --n-seeds "$SEEDS" > "logs/main_$1__$2.log" 2>&1
}

echo "=== $(date -Is) main study: NYU + KITTI (4 jobs)"
t=$(( $(nproc) / 4 > 0 ? $(nproc) / 4 : 1 ))
main_job nyu dav2-indoor-large "$t" &
main_job nyu unidepthv2-large "$t" &
main_job kitti dav2-outdoor-large "$t" &
main_job kitti unidepthv2-large "$t" &
wait

echo "=== $(date -Is) main study: DIODE + E5 shift (6 jobs)"
t=$(( $(nproc) / 6 > 0 ? $(nproc) / 6 : 1 ))
main_job diode_indoor dav2-indoor-large "$t" &
main_job diode_outdoor dav2-outdoor-large "$t" &
for spec in "nyu diode_indoor dav2-indoor-large" "kitti diode_outdoor dav2-outdoor-large" \
            "nyu diode_indoor unidepthv2-large" "kitti diode_outdoor unidepthv2-large"; do
  read -r src tgt model <<< "$spec"
  if have "$src" "$model" && have "$tgt" "$model"; then
    OMP_NUM_THREADS=$t $PY -W ignore -m experiments.e5_shift --source "$src" --target "$tgt" --model "$model" \
      --n-seeds "$E5_SEEDS" > "logs/e5_${src}__${model}.log" 2>&1 &
  fi
done
wait

echo "=== $(date -Is) E11 sparse anchors (6 jobs)"
SEEDS=${E11_SEEDS:-5} DRAWS=${E11_DRAWS:-6} JOBS=6 PY=$PY bash scripts/run_e11.sh

echo "=== $(date -Is) summarize + figures"
$PY -m experiments.summarize --exp-id main
$PY -m experiments.make_figures || echo "make_figures failed (see above)"
echo "=== $(date -Is) done. results/main/_summary/summary.md, results/e5, results/e11"
