#!/usr/bin/env bash
# Confirmatory replication (docs/ANALYSIS_PLAN_2026-11-03.md), one seed shard per call (CPU only).
#   PART=e17 SEED_START=0 N_SEEDS=2 bash scripts/run_confirm.sh   # 6 NYU/KITTI pairs x 5 layouts, k 50 200
#   PART=e18 SEED_START=0 N_SEEDS=2 bash scripts/run_confirm.sh   # ARKitScenes UniDepthV2, DA-V2 x uniform, cluster
# Each shard writes results/<exp>/<dataset>__<model>[__<layout>]__seeds<a>_<b>/; merge with experiments.confirm_analysis.
# Env: PY, JOBS (parallel processes; default 6 for e17, 3 for e18: ARKitScenes tables are large), DRAWS (10),
#      E17_EXP (e17_conf), E18_EXP (e18_conf)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
PART=${PART:?PART=e17|e18}
SEED_START=${SEED_START:?}
N_SEEDS=${N_SEEDS:?}
DRAWS=${DRAWS:-10}
mkdir -p logs
if [[ $PART == e17 ]]; then JOBS=${JOBS:-6}; else JOBS=${JOBS:-3}; fi
threads=$(( $(nproc) / JOBS > 0 ? $(nproc) / JOBS : 1 ))
tag="seeds${SEED_START}_$((SEED_START + N_SEEDS - 1))"
running=0
spawn() {  # log_name command...
  local log=$1; shift
  OMP_NUM_THREADS=$threads "$@" > "logs/$log.log" 2>&1 || echo "!!! FAILED: $log (see logs/$log.log)" &
  running=$((running + 1))
  if (( running % JOBS == 0 )); then wait; fi
}

echo "=== $(date -Is) confirmatory $PART, $tag, $DRAWS draws, $JOBS jobs x $threads threads"
if [[ $PART == e17 ]]; then
  exp=${E17_EXP:-e17_conf}
  for pattern in uniform rows cluster edges near; do
    for pair in kitti:dav2-outdoor-large kitti:unidepthv2-large kitti:depthpro \
                nyu:dav2-indoor-large nyu:unidepthv2-large nyu:depthpro; do
      ds=${pair%%:*}; model=${pair#*:}
      spawn "${exp}_${ds}__${model}__${pattern}__$tag" $PY -W ignore -m experiments.e11_anchors --exp-id "$exp" \
        --pattern "$pattern" --dataset "$ds" --model "$model" --ks 50 200 --no-oracle --anchor-dist \
        --dual --dual-rmax --lean --dual-gamma 0.5 --dist-k-random 20 \
        --seed-start "$SEED_START" --n-seeds "$N_SEEDS" --draws "$DRAWS"
    done
  done
else
  exp=${E18_EXP:-e18_conf}
  for pattern in uniform cluster; do
    for model in unidepthv2-large dav2-indoor-large; do
      spawn "${exp}_arkitscenes__${model}__${pattern}__$tag" $PY -W ignore -m experiments.e18_sensor_dual \
        --exp-id "$exp" --pattern "$pattern" --dataset arkitscenes --model "$model" --ks 200 \
        --dist-k-random 20 --dual-gamma 0.5 --seed-start "$SEED_START" --n-seeds "$N_SEEDS" --draws "$DRAWS"
    done
  done
fi
wait
grep -l "Traceback" logs/${exp}_*__"$tag".log 2>/dev/null && echo "some jobs crashed (see above)" || true
echo "=== $(date -Is) done $PART $tag"
