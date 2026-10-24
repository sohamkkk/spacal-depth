# Sourced by every sbatch script. Edit here once instead of in each job file.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
# module load python/3.11 cuda/12.8     # uncomment / adapt if your cluster needs modules
# shellcheck disable=SC1091
source .venv-gpu/bin/activate
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
export HF_HUB_OFFLINE=1                  # compute nodes usually lack internet; weights come from prefetch_models
export TORCH_HOME=${TORCH_HOME:-$PWD/.torch_cache}   # torch.hub repo + weights (Metric3D)
export PYTHONUNBUFFERED=1
mkdir -p logs
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-none} cpus=$(nproc)"
