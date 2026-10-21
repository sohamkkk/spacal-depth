#!/usr/bin/env bash
#SBATCH --job-name=spacal-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm-%x-%j.out
# 3 images per (dataset, model) pair on one GPU, E0 table, unit tests.
# Submit:  sbatch -p <gpu_partition> [--gres=gpu:<a6000_type>:1] slurm/smoke.sbatch
set -uo pipefail
source slurm/env.sh

nvidia-smi
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available in the job: driver too old for this torch build? (try CUDA_TAG=cu124)"
print("torch", torch.__version__, "| gpu:", torch.cuda.get_device_name(0))
EOF

status=0
for ds in nyu kitti diode_indoor diode_outdoor; do
  LIMIT=3 bash scripts/cache_dataset.sh "$ds" || status=1
done
python -m experiments.e0_base_metrics || status=1
python -m pytest -q || status=1
echo "SMOKE STATUS: $status (0 = all good)"
exit $status
