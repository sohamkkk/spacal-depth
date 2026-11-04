#!/usr/bin/env bash
# Cache every model the experiments need for ONE dataset, on whatever GPU is visible
# (inside a Slurm job that is the allocated GPU).
#   bash scripts/cache_dataset.sh nyu
#   LIMIT=3 bash scripts/cache_dataset.sh kitti      # smoke test
#   AMP=1 bash scripts/cache_dataset.sh nyu          # fp16 forward passes (T4 / Colab)
#   NO_GRAD=1 bash scripts/cache_dataset.sh kitti    # skip gradient uncertainty (slowest part)
#   MODEL_JOBS="depthpro --flip" bash scripts/cache_dataset.sh nyu   # only these jobs (';'-separated)
#   MODEL_JOBS="metric3d-large --flip" bash scripts/cache_dataset.sh kitti
# Default order: DA-V2 Large (+flip, +gradient uncertainty), Base, Small (size ensemble), UniDepthV2-Large (+flip).
# ETH3D mixes indoor and outdoor scenes, so it gets the three domain-agnostic models instead of DA-V2.
set -uo pipefail
cd "$(dirname "$0")/.."
ds=$1
case "$ds" in
  nyu|diode_indoor|arkitscenes) domain=indoor ;;
  kitti|diode_outdoor) domain=outdoor ;;
  eth3d) domain= ;;
  *) echo "unknown dataset $ds"; exit 2 ;;
esac
if ! err=$(python -c "from src.data.registry import build_dataset; print('$ds:', len(build_dataset('$ds')), 'samples')" 2>&1); then
  echo "!!! FAILED: dataset $ds not loadable under ${SPACAL_DATA_ROOT:-data}: $(tail -1 <<< "$err")"
  echo "    run: bash scripts/download_data.sh"; exit 1
fi
echo "$err"
EXTRA="${LIMIT:+--limit $LIMIT} ${AMP:+--amp}"
GRAD=$([[ -n "${NO_GRAD:-}" ]] || echo "--grad")
if [[ -n "${MODEL_JOBS:-}" ]]; then
  IFS=';' read -r -a JOBS <<< "$MODEL_JOBS"
elif [[ -z "$domain" ]]; then
  JOBS=("unidepthv2-large --flip" "depthpro --flip" "metric3d-large --flip")
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
