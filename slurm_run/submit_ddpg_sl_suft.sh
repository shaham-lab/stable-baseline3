#!/bin/bash
# Submit ddpg_sl_suft.sh as a single job array, with one array task per
# (short buffer size, TF lambda, sampling fraction, batch size, train frequency, environment) combination:
# each task runs all the runs of its combination in parallel
# The long term buffer size is 1M minus the short term buffer size (set in ddpg_sl_suft.sh)
#
# By default the short term buffer size and the TF lambda are searched:
# 3 short buffer sizes * 2 TF lambdas * 1 sampling fraction * 1 batch size * 1 train freq * 5 environments = 30 array tasks (20 runs each, 600 runs)
# Task order (the task id) is a nested loop: short buffer size -> TF lambda -> sampling fraction -> batch size -> train freq -> environment,
# so the tasks of the first short buffer size, TF lambda and fraction come first; Slurm starts array tasks roughly in id order
#
# Usage:
#   ./submit_ddpg_sl_suft.sh                                  all 5 MuJoCo environments, runs 1-20, the full search
#   ./submit_ddpg_sl_suft.sh Ant Hopper                       only these environments
#   RUNS="1 2" ./submit_ddpg_sl_suft.sh Ant                   only these runs
#   SHORT_BUFFER_SIZES="1000" ./submit_ddpg_sl_suft.sh        only these short term buffer sizes
#   TF_LAMBDAS="1" ./submit_ddpg_sl_suft.sh                   only these TF lambdas
#   SAMPLING_FRACTIONS="0.5" ./submit_ddpg_sl_suft.sh         only these sampling fractions
#   BATCH_SIZES="32 256" ./submit_ddpg_sl_suft.sh             these minibatch sizes (default 32)
#   TRAIN_FREQS="4" ./submit_ddpg_sl_suft.sh                  these train frequencies (default 4)
#   DRY_RUN=1 ./submit_ddpg_sl_suft.sh                        print the sbatch command and every task's combination without submitting
# The task log is slurm-<job id>_<task id>.out and its first line names the task's combination

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -ra RUN_LIST <<< "${RUNS:-$(seq -s " " 1 20)}"
read -ra SHORT_BUFFER_LIST <<< "${SHORT_BUFFER_SIZES:-4000 2000 8000}"
read -ra TF_LAMBDA_LIST <<< "${TF_LAMBDAS:-1 0}"
read -ra SAMPLING_FRACTION_LIST <<< "${SAMPLING_FRACTIONS:-0.5}"
read -ra BATCH_SIZE_LIST <<< "${BATCH_SIZES:-32}"
read -ra TRAIN_FREQ_LIST <<< "${TRAIN_FREQS:-4}"

# Each run gets one CPU core and 7G (Humanoid peaks at ~5.8G per run), so every array task is sized to the number of runs
N_RUNS=${#RUN_LIST[@]}
MEM_PER_RUN_GB=7

# The MuJoCo (v5) environments
MUJOCO_ENVS="Ant HalfCheetah Hopper Humanoid Walker2d"

if [ $# -gt 0 ]; then
  read -ra ENV_LIST <<< "$*"
else
  read -ra ENV_LIST <<< "$MUJOCO_ENVS"
fi

N_TASKS=$((${#SHORT_BUFFER_LIST[@]} * ${#TF_LAMBDA_LIST[@]} * ${#SAMPLING_FRACTION_LIST[@]} * ${#BATCH_SIZE_LIST[@]} * ${#TRAIN_FREQ_LIST[@]} * ${#ENV_LIST[@]}))
# ddpg_sl_suft.sh takes every hyperparameter as a comma separated list and picks one combination per array task
csv() { local IFS=','; echo "$*"; }

CMD=(sbatch --array="0-$((N_TASKS - 1))" --job-name="ddpg_sl_suft"
     --cpus-per-task="$N_RUNS" --mem="$((N_RUNS * MEM_PER_RUN_GB))G"
     "$SCRIPT_DIR/ddpg_sl_suft.sh" "$(csv "${ENV_LIST[@]}")" "$(csv "${SHORT_BUFFER_LIST[@]}")" "$(csv "${TF_LAMBDA_LIST[@]}")"
     "$(csv "${SAMPLING_FRACTION_LIST[@]}")" "$(csv "${BATCH_SIZE_LIST[@]}")" "$(csv "${TRAIN_FREQ_LIST[@]}")" "${RUN_LIST[@]}")
if [ -n "$DRY_RUN" ]; then
  echo "${CMD[@]}"
  # Same decoding as ddpg_sl_suft.sh: the environment varies fastest
  TASK=0
  for SHORT_BUFFER_SIZE in "${SHORT_BUFFER_LIST[@]}"; do
    for TF_LAMBDA in "${TF_LAMBDA_LIST[@]}"; do
      for SAMPLING_FRACTION in "${SAMPLING_FRACTION_LIST[@]}"; do
        for BATCH_SIZE in "${BATCH_SIZE_LIST[@]}"; do
          for TRAIN_FREQ in "${TRAIN_FREQ_LIST[@]}"; do
            for ENV_NAME in "${ENV_LIST[@]}"; do
              echo "  task $TASK: $ENV_NAME, short buffer $SHORT_BUFFER_SIZE, TF lambda $TF_LAMBDA, fraction $SAMPLING_FRACTION, batch $BATCH_SIZE, train_freq $TRAIN_FREQ"
              TASK=$((TASK + 1))
            done
          done
        done
      done
    done
  done
else
  "${CMD[@]}"
fi
