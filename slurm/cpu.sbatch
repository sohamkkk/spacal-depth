#!/usr/bin/env bash
#SBATCH --job-name=spacal-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
# All CPU experiments (tables, main study, E0/E1/E5/E8/E10, summaries, figures). No GPU.
# Submit:  sbatch -p <cpu_partition> slurm/cpu.sbatch
#          sbatch -p <cpu_partition> --export=ALL,SEEDS=3,E8_SEEDS=5,E5_SEEDS=2 slurm/cpu.sbatch   # quick pass
set -euo pipefail
source slurm/env.sh
PY=python bash scripts/run_all_cpu.sh
