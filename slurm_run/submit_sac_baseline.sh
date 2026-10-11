#!/bin/bash
# Submit sac_baseline.sh once per environment: one job runs all the runs of that environment in parallel
#
# Usage:
#   ./submit_sac_baseline.sh                      all 5 MuJoCo environments, runs 1-4 (5 jobs)
#   ./submit_sac_baseline.sh Ant Hopper           only these environments
#   RUNS="1 2" ./submit_sac_baseline.sh Ant       only these runs
#   DRY_RUN=1 ./submit_sac_baseline.sh            print the sbatch commands without submitting

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -ra RUN_LIST <<< "${RUNS:-1 2 3 4}"

# Each run gets one CPU core and 16G, so the job is sized to the number of runs
N_RUNS=${#RUN_LIST[@]}
MEM_PER_RUN_GB=16

# The MuJoCo (v5) environments
MUJOCO_ENVS="Ant HalfCheetah Hopper Humanoid Walker2d"

if [ $# -gt 0 ]; then
  ENVS="$*"
else
  ENVS=$MUJOCO_ENVS
fi

for ENV_NAME in $ENVS; do
  CMD=(sbatch --job-name="${ENV_NAME,,}_sac_base" --cpus-per-task="$N_RUNS" --mem="$((N_RUNS * MEM_PER_RUN_GB))G"
       "$SCRIPT_DIR/sac_baseline.sh" "$ENV_NAME" "${RUN_LIST[@]}")
  if [ -n "$DRY_RUN" ]; then
    echo "${CMD[@]}"
  else
    "${CMD[@]}"
  fi
done
