#!/usr/bin/env bash
#SBATCH --job-name=spacal-confirm-pool
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
# Confirmatory replication as a memory-aware pool of single-threaded tasks (scripts/confirm_pool.py):
#   sbatch -p phd slurm/confirm_pool.sbatch            # one node
#   sbatch -p phd -N 2 slurm/confirm_pool.sbatch       # one job, one pool per node, sharing the queue
# Env: PARTS ("e18 e17"; add e20 for the sensor-error envelope), SEEDS ("0-19"), SLOTS (allocated CPUs), DRAWS (10),
#      CANCEL_JOBS ("46476 ..."): older confirm jobs to scancel once this one starts, so no seed runs twice.
#      PRIOR_GB ("8 7.5"): expected peak GB of an E18/E20 and an E17 task until this pool has measured its own.
# Resumable and shares done-markers with slurm/confirm_all.sbatch: resubmit the same command after a time-out.
# Pools share the queue through claim files; each runs tasks no other live pool holds.
# Memory budget per pool = 0.85 x --mem (per node; override on the sbatch command line, e.g. --mem=200G).
set -euo pipefail
source slurm/env.sh
for j in ${CANCEL_JOBS:-}; do
  scancel "$j" 2>/dev/null || true
  while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 10; done
  echo "cancelled job $j"
done
echo "nodes: ${SLURM_JOB_NODELIST} (${SLURM_JOB_NUM_NODES})"
# shellcheck disable=SC2086
srun --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --kill-on-bad-exit=0 --label \
  python -m scripts.confirm_pool --parts ${PARTS:-e18 e17} --seeds ${SEEDS:-0-19} --draws "${DRAWS:-10}" \
  ${SLOTS:+--slots "$SLOTS"} ${PRIOR_GB:+--prior-gb $PRIOR_GB}
