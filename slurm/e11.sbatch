#!/usr/bin/env bash
#SBATCH --job-name=spacal-e11
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
# E11 sparse-anchor conformal (CPU only). Needs prepared/ tables.
# Go/no-go test (KITTI, ~30-60 min):
#   sbatch -p <cpu_partition> --cpus-per-task=8 --mem=48G \
#     --export=ALL,E11_PAIRS="kitti:dav2-outdoor-large kitti:unidepthv2-large",KS="0 1 5 20",SEEDS=3,DRAWS=3,JOBS=2 slurm/e11.sbatch
# Full sweep:
#   sbatch -p <cpu_partition> --export=ALL,JOBS=4 slurm/e11.sbatch
set -euo pipefail
source slurm/env.sh
PY=python bash scripts/run_e11.sh
