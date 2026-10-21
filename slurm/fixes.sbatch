#!/usr/bin/env bash
#SBATCH --job-name=spacal-fixes
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm-%x-%j.out
# Main study + E5 + E11 rerun after the S6 / scene-level V4,K2 / A-local fixes (CPU only, ~7-8 h).
# Submit:  sbatch -p <cpu_partition> slurm/fixes.sbatch
set -euo pipefail
source slurm/env.sh
PY=python bash scripts/run_fixes.sh
