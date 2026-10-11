#!/bin/bash
# Submit sac_baseline.sh for every (environment, run) pair
#
# Usage:
#   ./submit_sac_baseline.sh                      all 5 MuJoCo environments, runs 1-4
#   ./submit_sac_baseline.sh Ant Hopper           only these environments
#   RUNS="1 2" ./submit_sac_baseline.sh Ant       only these runs
#   DRY_RUN=1 ./submit_sac_baseline.sh            print the sbatch commands without submitting

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNS=${RUNS:-"1 2 3 4"}

# The MuJoCo (v5) environments
MUJOCO_ENVS="Ant HalfCheetah Hopper Humanoid Walker2d"

if [ $# -gt 0 ]; then
  ENVS="$*"
else
  ENVS=$MUJOCO_ENVS
fi

for ENV_NAME in $ENVS; do
  for RUN_NUMBER in $RUNS; do
    CMD=(sbatch --job-name="${ENV_NAME,,}_sac_base_r${RUN_NUMBER}" "$SCRIPT_DIR/sac_baseline.sh" "$ENV_NAME" "$RUN_NUMBER")
    if [ -n "$DRY_RUN" ]; then
      echo "${CMD[@]}"
    else
      "${CMD[@]}"
    fi
  done
done
