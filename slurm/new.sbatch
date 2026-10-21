#!/usr/bin/env bash
#SBATCH --job-name=spacal-new
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm-%x-%j.out
# E12 safety query + E13 structured anchors + Depth Pro pairs (CPU only; scripts/run_new.sh).
# Submit now (E12, E13):   sbatch -p <cpu_partition> --export=ALL,PARTS="e12 e13" slurm/new.sbatch
# After Depth Pro caching:  sbatch -p <cpu_partition> --export=ALL,PARTS=depthpro slurm/new.sbatch
set -euo pipefail
source slurm/env.sh
PY=python bash scripts/run_new.sh
