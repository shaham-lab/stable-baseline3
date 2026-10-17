#!/bin/bash
# Resubmit the runs listed in failed_runs.txt on L4-12h
# They were pending on generic with the old dqn1_sl_suft.sh interface and were cancelled on 2026-10-17
# --partition on the command line overrides the #SBATCH partition of each job script
# Each job is a single run, so it asks for 1 CPU and 55G instead of the 3-run #SBATCH defaults
# The job scripts don't checkpoint, so every run starts again from timestep 0 and overwrites its old log
#
# Usage:
#   ./submit_failed_runs.sh              submit all 4 runs
#   DRY_RUN=1 ./submit_failed_runs.sh    print the sbatch commands without submitting

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARTITION="shaham-L4"

# submit <job_name> <script> <script_args...>
submit() {
  local JOB_NAME=$1
  local SCRIPT=$2
  shift 2
  CMD=(sbatch --partition="$PARTITION" --job-name="$JOB_NAME" --cpus-per-task=1 --mem=55G "$SCRIPT_DIR/$SCRIPT" "$@")
  if [ -n "$DRY_RUN" ]; then
    echo "${CMD[@]}"
  else
    "${CMD[@]}"
  fi
}

# Cancelled on 2026-10-17 while pending on generic (4)
submit enduro_dqn1_sl_s4000_l1_f0.25_r1 dqn1_sl_suft.sh 4000 1 0.25 Enduro:1
submit krull_dqn1_sl_s4000_l1_f0.25_r1 dqn1_sl_suft.sh 4000 1 0.25 Krull:1
submit krull_dqn1_sl_s4000_l1_f0.25_r2 dqn1_sl_suft.sh 4000 1 0.25 Krull:2
submit krull_dqn1_sl_s4000_l1_f0.25_r4 dqn1_sl_suft.sh 4000 1 0.25 Krull:4
