#!/usr/bin/env bash
# Cache every model the experiments need for ONE dataset, on whatever GPU is visible
# (inside a Slurm job that is the allocated GPU).
#   bash scripts/cache_dataset.sh nyu
#   LIMIT=3 bash scripts/cache_dataset.sh kitti      # smoke test
#   AMP=1 bash scripts/cache_dataset.sh nyu          # fp16 forward passes (T4 / Colab)
#   NO_GRAD=1 bash scripts/cache_dataset.sh kitti    # skip gradient uncertainty (slowest part)
#   MODEL_JOBS="depthpro --flip" bash scripts/cache_dataset.sh nyu   # only these jobs (';'-separated)
# Default order: DA-V2 Large (+flip, +gradient uncertainty), Base, Small (size ensemble), UniDepthV2-Large (+flip).
set -uo pipefail
cd "$(dirname "$0")/.."
ds=$1
case "$ds" in
  nyu|diode_indoor) domain=indoor ;;
  kitti|diode_outdoor) domain=outdoor ;;
  *) echo "unknown dataset $ds"; exit 2 ;;
esac
python -c "from src.data.registry import build_dataset; print('$ds:', len(build_dataset('$ds')), 'samples')" 2>/dev/null \
  || { echo "!!! FAILED: dataset $ds not found under ${SPACAL_DATA_ROOT:-data} — run: bash scripts/download_data.sh"; exit 1; }
EXTRA="${LIMIT:+--limit $LIMIT} ${AMP:+--amp}"
GRAD=$([[ -n "${NO_GRAD:-}" ]] || echo "--grad")
if [[ -n "${MODEL_JOBS:-}" ]]; then
  IFS=';' read -r -a JOBS <<< "$MODEL_JOBS"
else
  JOBS=("dav2-${domain}-large --flip $GRAD" "dav2-${domain}-base" "dav2-${domain}-small" "unidepthv2-large --flip")
fi
status=0
for job in "${JOBS[@]}"; do
  echo "=== $(date -Is) $ds :: $job $EXTRA  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
  # shellcheck disable=SC2086
  SPACAL_DEVICE=cuda python -m scripts.cache_predictions --dataset "$ds" --model $job $EXTRA \
    || { echo "!!! FAILED: $ds :: $job"; status=1; }
done
exit $status
