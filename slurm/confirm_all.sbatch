#!/usr/bin/env bash
#SBATCH --job-name=spacal-confirm-all
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
# Whole confirmatory replication (docs/ANALYSIS_PLAN_2026-11-03.md) in ONE job:
#   sbatch -p phd slurm/confirm_all.sbatch
# Runs E18c seeds 0..19, then E17c seeds 0..19, one seed per unit (scripts/run_confirm.sh).
# Resumable: a unit that finished without a Traceback writes logs/confirm_done/<part>_seed<s>; resubmitting the
# same command skips those, so if the time limit kills the job, just submit it again.
# Env: PARTS ("e18 e17"), SEEDS ("0 1 ... 19"), DRAWS (10), E17_JOBS (6), E18_JOBS (3: ARKitScenes tables are large),
#      CHAIN=1 queues one dependent copy of this job that starts only after this one ends (at most 2 jobs queued).
set -euo pipefail
source slurm/env.sh
PARTS=${PARTS:-"e18 e17"}
SEEDS=${SEEDS:-$(seq -s ' ' 0 19)}
export DRAWS=${DRAWS:-10}
done_dir=logs/confirm_done
mkdir -p "$done_dir"

pending=0
for part in $PARTS; do for s in $SEEDS; do [[ -e $done_dir/${part}_seed$s ]] || pending=$((pending + 1)); done; done
echo "=== $(date -Is) $pending units pending (parts: $PARTS)"
if (( pending == 0 )); then echo "=== all confirmatory units done"; exit 0; fi
if [[ ${CHAIN:-0} == 1 && -n ${SLURM_JOB_ID:-} ]]; then
  sbatch -p "${SLURM_JOB_PARTITION:-phd}" --dependency=afterany:"$SLURM_JOB_ID" \
    --export=ALL,CHAIN=1,PARTS="$PARTS",SEEDS="$SEEDS" slurm/confirm_all.sbatch
fi

for part in $PARTS; do
  if [[ $part == e17 ]]; then exp=${E17_EXP:-e17_conf}; jobs=${E17_JOBS:-6}; else exp=${E18_EXP:-e18_conf}; jobs=${E18_JOBS:-3}; fi
  for s in $SEEDS; do
    marker=$done_dir/${part}_seed$s
    [[ -e $marker ]] && continue
    t0=$(date +%s)
    unit_log=logs/confirm_unit_${part}_seed$s.txt
    PY=python PART=$part SEED_START=$s N_SEEDS=1 JOBS=$jobs bash scripts/run_confirm.sh | tee "$unit_log"
    shopt -s nullglob; logs_s=(logs/"${exp}"_*__seeds"${s}_${s}".log); shopt -u nullglob
    if (( ${#logs_s[@]} == 0 )) || grep -q "FAILED" "$unit_log" || grep -lq "Traceback" "${logs_s[@]}"; then
      echo "!!! $part seed $s had failures (see $unit_log); not marked done"
    else
      date -Is > "$marker"
      echo "=== $part seed $s finished in $(( ($(date +%s) - t0) / 60 )) min"
    fi
  done
done
echo "=== $(date -Is) pass complete"
