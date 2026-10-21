#!/usr/bin/env bash
# One-time environment setup. Works on a GPU-less Slurm login node; the GPU check
# is reported, not required (slurm/smoke.sbatch verifies CUDA on a compute node).
#   bash scripts/setup_remote.sh                       # creates .venv-gpu in the repo
#   CUDA_TAG=cu124 PYTHON=python3.11 bash scripts/setup_remote.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
VENV=${VENV:-.venv-gpu}
CUDA_TAG=${CUDA_TAG:-cu128}   # cu128: Blackwell + A6000; use cu124 if the node driver is < 570

"$PY" -c 'import sys; assert sys.version_info >= (3, 10), f"need Python >= 3.10, got {sys.version}"'
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,driver_version --format=csv \
  || echo "no GPU on this node (fine on a login node)"

"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip wheel
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install -r requirements.txt

# UniDepth pins its own torch/xformers; install the package only, deps are in requirements.txt
pip install --no-deps "git+https://github.com/lpiccinelli-eth/UniDepth.git"
pip install --no-deps xformers --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
  || echo "WARNING: xformers unavailable for ${CUDA_TAG}; UniDepth falls back to PyTorch attention if supported"

python - <<'EOF'
import torch
print("torch", torch.__version__, "built for cuda", torch.version.cuda, "| cuda visible here:", torch.cuda.is_available())
import transformers; print("transformers", transformers.__version__)
try:
    import unidepth  # noqa: F401
    print("unidepth import ok")
except Exception as e:
    print("WARNING: unidepth import failed:", repr(e))
EOF

pip freeze > requirements-lock-remote.txt
echo "Environment ready. Activate with: source $VENV/bin/activate"
