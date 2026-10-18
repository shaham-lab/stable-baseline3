#!/bin/bash
# Submit ddpg_sl_suft.sh as one job array per
# (short buffer size, TF lambda, sampling fraction, batch size, train frequency, environment) configuration:
# the runs of the configuration are split into chunks of PARALLEL runs (default 5), one array task per chunk,
# and each task runs its chunk in parallel, one CPU core and 7G per run
# (runs 1-20 with PARALLEL=5 -> 4 tasks per array: runs 1-5, 6-10, 11-15, 16-20)
# The long term buffer size is 1M minus the short term buffer size (set in ddpg_sl_suft.sh)
#
# By default the short term buffer size is searched:
# 2 short buffer sizes (1000, 500) * 1 TF lambda (1) * 1 sampling fraction * 1 batch size (256) * 1 train freq (1) * 1 environment (Swimmer)
# = 2 job arrays of 4 tasks (5 runs each, 40 runs)
# The arrays are submitted in a nested loop: short buffer size -> TF lambda -> sampling fraction -> batch size -> train freq -> environment
#
# Usage:
#   ./submit_ddpg_sl_suft2.sh                                  Swimmer, runs 1-20, the full search
#   ./submit_ddpg_sl_suft2.sh Ant Hopper                       only these environments
#   RUNS="1 2" ./submit_ddpg_sl_suft2.sh Ant                   only these runs
#   SHORT_BUFFER_SIZES="1000" ./submit_ddpg_sl_suft2.sh        only these short term buffer sizes
#   TF_LAMBDAS="1 0" ./submit_ddpg_sl_suft2.sh                 these TF lambdas (default 1)
#   SAMPLING_FRACTIONS="0.5" ./submit_ddpg_sl_suft2.sh         only these sampling fractions
#   BATCH_SIZES="32 256" ./submit_ddpg_sl_suft2.sh             these minibatch sizes (default 256)
#   TRAIN_FREQS="4" ./submit_ddpg_sl_suft2.sh                  these train frequencies (default 1)
#   PARALLEL=10 ./submit_ddpg_sl_suft2.sh                      runs per array task, all in parallel (default 5)
#   DRY_RUN=1 ./submit_ddpg_sl_suft2.sh                        print the sbatch commands and every task's runs without submitting
#   PARTITION=L4-12h ./submit_ddpg_sl_suft2.sh                             another partition (default shaham-L4)
# The task log is slurm-<job id>_<task id>.out and its first line names the task's configuration and runs

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -ra RUN_LIST <<< "${RUNS:-$(seq -s " " 1 20)}"
read -ra SHORT_BUFFER_LIST <<< "${SHORT_BUFFER_SIZES:-1000 500}"
read -ra TF_LAMBDA_LIST <<< "${TF_LAMBDAS:-1}"
read -ra SAMPLING_FRACTION_LIST <<< "${SAMPLING_FRACTIONS:-0.5}"
read -ra BATCH_SIZE_LIST <<< "${BATCH_SIZES:-256}"
read -ra TRAIN_FREQ_LIST <<< "${TRAIN_FREQS:-1}"

# Each run gets one CPU core and 7G (Humanoid peaks at ~5.8G per run), so every array task is sized to PARALLEL runs
N_RUNS=${#RUN_LIST[@]}
PARALLEL=${PARALLEL:-5}
(( PARALLEL > N_RUNS )) && PARALLEL=$N_RUNS
N_CHUNKS=$(( (N_RUNS + PARALLEL - 1) / PARALLEL ))
MEM_PER_RUN_GB=7
PARTITION=${PARTITION:-shaham-L4}

# The MuJoCo (v5) environments
MUJOCO_ENVS="Ant HalfCheetah Hopper Humanoid Walker2d Swimmer"
# MUJOCO_ENVS="Swimmer"

if [ $# -gt 0 ]; then
  read -ra ENV_LIST <<< "$*"
else
  read -ra ENV_LIST <<< "$MUJOCO_ENVS"
fi

for SHORT_BUFFER_SIZE in "${SHORT_BUFFER_LIST[@]}"; do
  for TF_LAMBDA in "${TF_LAMBDA_LIST[@]}"; do
    for SAMPLING_FRACTION in "${SAMPLING_FRACTION_LIST[@]}"; do
      for BATCH_SIZE in "${BATCH_SIZE_LIST[@]}"; do
        for TRAIN_FREQ in "${TRAIN_FREQ_LIST[@]}"; do
          for ENV_NAME in "${ENV_LIST[@]}"; do
            # RUNS_PER_TASK makes ddpg_sl_suft.sh give array task i the i-th chunk of PARALLEL runs
            CMD=(sbatch --partition="$PARTITION" --array="0-$((N_CHUNKS - 1))"
                 --job-name="ddpg_sl_${ENV_NAME}_s${SHORT_BUFFER_SIZE}_l${TF_LAMBDA}_f${SAMPLING_FRACTION}_b${BATCH_SIZE}_tf${TRAIN_FREQ}"
                 --cpus-per-task="$PARALLEL" --mem="$((PARALLEL * MEM_PER_RUN_GB))G" --export="ALL,RUNS_PER_TASK=$PARALLEL"
                 "$SCRIPT_DIR/ddpg_sl_suft.sh" "$ENV_NAME" "$SHORT_BUFFER_SIZE" "$TF_LAMBDA" "$SAMPLING_FRACTION" "$BATCH_SIZE" "$TRAIN_FREQ"
                 "${RUN_LIST[@]}")
            if [ -n "$DRY_RUN" ]; then
              echo "${CMD[@]}"
              for ((i = 0; i < N_CHUNKS; i++)); do
                echo "  task $i: $ENV_NAME, short buffer $SHORT_BUFFER_SIZE, TF lambda $TF_LAMBDA, fraction $SAMPLING_FRACTION, batch $BATCH_SIZE, train_freq $TRAIN_FREQ, runs ${RUN_LIST[*]:$((i * PARALLEL)):PARALLEL}"
              done
            else
              "${CMD[@]}"
            fi
          done
        done
      done
    done
  done
done
