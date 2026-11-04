#!/usr/bin/env bash
#SBATCH --job-name=spacal-confirm
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm-%x-%j.out
# One seed shard of the confirmatory replication (scripts/run_confirm.sh). Submit all shards:
#   for s in 0 2 4 6 8 10 12 14 16 18; do
#     sbatch -p phd --export=ALL,PART=e17,SEED_START=$s,N_SEEDS=2 slurm/confirm.sbatch
#     sbatch -p phd --export=ALL,PART=e18,SEED_START=$s,N_SEEDS=2 slurm/confirm.sbatch
#   done
set -euo pipefail
source slurm/env.sh
PY=python bash scripts/run_confirm.sh
