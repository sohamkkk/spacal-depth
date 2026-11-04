#!/usr/bin/env bash
#SBATCH --job-name=spacal-omnidc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
# OMNI-DC forward passes on ARKitScenes (scripts/omnidc_forward.py). Resumable: resubmit after a timeout.
# One-time, on the login node (internet):
#   git clone --depth 1 https://github.com/princeton-vl/OMNI-DC third_party/OMNI-DC
#   source .venv-gpu/bin/activate && HF_HOME=$PWD/.hf_cache python -c \
#     "from huggingface_hub import hf_hub_download as d; d('zuoym15/OMNI-DC', 'model.safetensors')"
#   python scripts/fetch_arkitscenes.py --dst data/arkitscenes        # ~15 GB, skip if already there
# Smoke:  sbatch -p <gpu_partition> --time=00:30:00 --export=ALL,LIMIT=8,JOB=smoke slurm/omnidc.sbatch
# Run:    sbatch -p <gpu_partition> --export=ALL,JOB=s1,MASKS="uniform/k200_seed1_draw0 cluster/k200_seed1_draw0" slurm/omnidc.sbatch
set -uo pipefail
source slurm/env.sh
export SPACAL_DATA_ROOT=${SPACAL_DATA_ROOT:-$PWD/data}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -m scripts.omnidc_forward --omnidc "${OMNIDC:-$PWD/third_party/OMNI-DC}" \
  --anchors "${ANCHORS:-$PWD/omni_anchors/arkitscenes}" \
  --masks ${MASKS:-uniform/k200_seed0_draw0 cluster/k200_seed0_draw0} --sources clean lidar \
  --out "$PWD/results/omnidc_raw/${JOB:-s0d0}" --shard "${SHARD:-0}" --n-shards "${NSHARDS:-1}" \
  --workers "${WORKERS:-6}" ${LIMIT:+--limit $LIMIT --chunk $LIMIT}
