#!/usr/bin/env bash
# Non-Slurm machines only (on the cluster use slurm/cache.sbatch instead).
# Caches every (dataset, model) pair, one dataset per GPU, in parallel.
#   GPUS="0 1 2 3" bash scripts/run_all_gpu.sh
#   LIMIT=5 GPUS="0" bash scripts/run_all_gpu.sh      # smoke test: 5 images per pair
set -uo pipefail
cd "$(dirname "$0")/.."
read -r -a GPU_LIST <<< "${GPUS:-0}"
mkdir -p logs

DATASETS=(nyu kitti diode_indoor diode_outdoor)
i=0
for ds in "${DATASETS[@]}"; do
  gpu=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}
  CUDA_VISIBLE_DEVICES=$gpu bash scripts/cache_dataset.sh "$ds" > "logs/cache_${ds}.log" 2>&1 &
  i=$((i + 1))
  # with fewer GPUs than datasets, wait before reusing a GPU
  if (( i % ${#GPU_LIST[@]} == 0 )); then wait; fi
done
wait
grep -h "FAILED" logs/cache_*.log && echo "Some jobs failed — see logs/" || echo "All caching jobs succeeded."
du -sh "${SPACAL_CACHE_ROOT:-cache}"
