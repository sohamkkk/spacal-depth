#!/usr/bin/env bash
#SBATCH --job-name=spacal-cache
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%A_%a.out
# Full caching: one array task (= one GPU) per dataset. Resumable: resubmitting skips
# images already cached, so a timeout just needs a resubmit.
# Submit:  sbatch -p <gpu_partition> [--gres=gpu:<a6000_type>:1] slurm/cache.sbatch
#          sbatch -p <gpu_partition> --array=1 slurm/cache.sbatch     # only KITTI
#          sbatch -p <gpu_partition> --array=4 slurm/cache.sbatch     # ETH3D (UniDepthV2, Depth Pro, Metric3D)
#          sbatch -p <gpu_partition> --export=ALL,MODEL_JOBS="metric3d-large --flip" slurm/cache.sbatch
set -uo pipefail
source slurm/env.sh

DATASETS=(nyu kitti diode_indoor diode_outdoor eth3d)
ds=${DATASETS[${SLURM_ARRAY_TASK_ID:-0}]}
bash scripts/cache_dataset.sh "$ds"
