#!/usr/bin/env bash
# First thing to run on the remote box after setup_remote.sh + download_data.sh.
# Caches 3 images per (dataset, model) pair, prints E0 metrics, runs the tests.
# Catches model-API / data-layout problems in minutes instead of hours.
# Cached files are reused by the full run (cache_predictions.py resumes).
#   GPUS="0" bash scripts/smoke_remote.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(" gpu", i, torch.cuda.get_device_name(i))
print("transformers", transformers.__version__)
import unidepth  # noqa: F401
print("unidepth import ok")
EOF
LIMIT=3 GPUS="${GPUS:-0}" bash scripts/run_all_gpu.sh
python -m experiments.e0_base_metrics
python -m pytest -q
echo "Smoke test passed. Sanity-check the E0 table above (AbsRel / delta1 should be plausible for zero-shot metric depth)."
