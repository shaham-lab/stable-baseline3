#!/bin/bash
# Submit ddpg_sl_suft.sh once per (short buffer size, TF lambda, sampling fraction, environment) combination:
# one job runs all the runs of that combination in parallel
# The long term buffer size is 1M minus the short term buffer size (set in ddpg_sl_suft.sh)
#
# By default the short term buffer size, the TF lambda and the sampling fraction are all searched:
# 3 short buffer sizes * 4 TF lambdas * 4 sampling fractions * 5 environments = 240 jobs (4 runs each, 960 runs)
# Submission order is a nested loop: short buffer size -> TF lambda -> sampling fraction -> environment,
# so all the environments of short buffer 4000, TF lambda 1, fraction 0.5 are submitted first, then the next fraction, ...
#
# Usage:
#   ./submit_ddpg_sl_suft.sh                                  all 5 MuJoCo environments, runs 1-4, all sampling fractions
#   ./submit_ddpg_sl_suft.sh Ant Hopper                       only these environments
#   RUNS="1 2" ./submit_ddpg_sl_suft.sh Ant                   only these runs
#   SHORT_BUFFER_SIZES="4000" ./submit_ddpg_sl_suft.sh        only these short term buffer sizes
#   TF_LAMBDAS="1" ./submit_ddpg_sl_suft.sh                   only these TF lambdas
#   SAMPLING_FRACTIONS="0.5" ./submit_ddpg_sl_suft.sh         only these sampling fractions
#   DRY_RUN=1 ./submit_ddpg_sl_suft.sh                        print the sbatch commands without submitting

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -ra RUN_LIST <<< "${RUNS:-1 2 3 4}"
SHORT_BUFFER_SIZES=${SHORT_BUFFER_SIZES:-"4000 2000 8000"}
TF_LAMBDAS=${TF_LAMBDAS:-"1 0.5 2 4"}
SAMPLING_FRACTIONS=${SAMPLING_FRACTIONS:-"0.5 0.75 0.25 1"}

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

for SHORT_BUFFER_SIZE in $SHORT_BUFFER_SIZES; do
  for TF_LAMBDA in $TF_LAMBDAS; do
    for SAMPLING_FRACTION in $SAMPLING_FRACTIONS; do
      for ENV_NAME in $ENVS; do
        CMD=(sbatch --job-name="${ENV_NAME,,}_ddpg_sl_s${SHORT_BUFFER_SIZE}_l${TF_LAMBDA}_f${SAMPLING_FRACTION}" --cpus-per-task="$N_RUNS" --mem="$((N_RUNS * MEM_PER_RUN_GB))G"
             "$SCRIPT_DIR/ddpg_sl_suft.sh" "$ENV_NAME" "$SHORT_BUFFER_SIZE" "$TF_LAMBDA" "$SAMPLING_FRACTION" "${RUN_LIST[@]}")
        if [ -n "$DRY_RUN" ]; then
          echo "${CMD[@]}"
        else
          "${CMD[@]}"
        fi
      done
    done
  done
done
