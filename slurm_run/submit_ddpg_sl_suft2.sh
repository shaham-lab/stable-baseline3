#!/bin/bash
# Submit ddpg_sl_suft.sh as a single job array over every
# (short buffer size, TF lambda, sampling fraction, batch size, train frequency, environment) configuration:
# the runs of each configuration are split into chunks of PARALLEL runs (default 5), one array task per chunk,
# and each task runs its chunk in parallel, one CPU core and 7G per run
# The long term buffer size is 1M minus the short term buffer size (set in ddpg_sl_suft.sh)
#
# By default:
# 1 short buffer size (1000) * 1 TF lambda (1) * 1 sampling fraction (0.5) * 1 batch size (256) * 1 train freq (1) * 5 environments
# = 5 configurations, runs 1-20 with PARALLEL=5 -> 4 tasks per configuration, 20 tasks, 100 runs
# Task order (the task id) is a nested loop: short buffer size -> TF lambda -> sampling fraction -> batch size -> train freq
# -> environment -> chunk of runs (see ddpg_sl_suft.sh)
# One email is sent when the whole job array ends (Slurm mails per array, not per task, without ARRAY_TASKS)
#
# Usage:
#   ./submit_ddpg_sl_suft2.sh                                  the 5 MuJoCo environments, runs 1-20, the defaults above
#   ./submit_ddpg_sl_suft2.sh Ant Hopper                       only these environments
#   RUNS="1 2" ./submit_ddpg_sl_suft2.sh Ant                   only these runs
#   SHORT_BUFFER_SIZES="1000 500" ./submit_ddpg_sl_suft2.sh    these short term buffer sizes (default 1000)
#   TF_LAMBDAS="1 0" ./submit_ddpg_sl_suft2.sh                 these TF lambdas (default 1)
#   SAMPLING_FRACTIONS="0.5" ./submit_ddpg_sl_suft2.sh         these sampling fractions (default 0.5)
#   BATCH_SIZES="32 256" ./submit_ddpg_sl_suft2.sh             these minibatch sizes (default 256)
#   TRAIN_FREQS="4" ./submit_ddpg_sl_suft2.sh                  these train frequencies (default 1)
#   PARALLEL=10 ./submit_ddpg_sl_suft2.sh                      runs per array task, all in parallel (default 5)
#   DRY_RUN=1 ./submit_ddpg_sl_suft2.sh                        print the sbatch command and every task's runs without submitting
#   PARTITION=L4-12h ./submit_ddpg_sl_suft2.sh                 another partition (default cpu192G-48h)
#   MAIL_USER=fiskust@biu.ac.il ./submit_ddpg_sl_suft2.sh         another address for the end of array email
# The task log is slurm-<job id>_<task id>.out and its first line names the task's configuration and runs

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -ra RUN_LIST <<< "${RUNS:-$(seq -s " " 1 20)}"
read -ra SHORT_BUFFER_LIST <<< "${SHORT_BUFFER_SIZES:-1000}"
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
PARTITION=${PARTITION:-cpu192G-48h}
MAIL_USER=${MAIL_USER:-fiskust@biu.ac.il}

# The MuJoCo (v5) environments
MUJOCO_ENVS="Ant HalfCheetah Hopper Humanoid Walker2d"

if [ $# -gt 0 ]; then
  read -ra ENV_LIST <<< "$*"
else
  read -ra ENV_LIST <<< "$MUJOCO_ENVS"
fi

N_TASKS=$((N_CHUNKS * ${#SHORT_BUFFER_LIST[@]} * ${#TF_LAMBDA_LIST[@]} * ${#SAMPLING_FRACTION_LIST[@]} * ${#BATCH_SIZE_LIST[@]}
           * ${#TRAIN_FREQ_LIST[@]} * ${#ENV_LIST[@]}))
# ddpg_sl_suft.sh takes every hyperparameter as a comma separated list and picks one combination per array task
csv() { local IFS=','; echo "$*"; }

# RUNS_PER_TASK makes ddpg_sl_suft.sh give each array task one chunk of PARALLEL runs
CMD=(sbatch --partition="$PARTITION" --array="0-$((N_TASKS - 1))" --job-name="ddpg_sl_suft"
     --cpus-per-task="$PARALLEL" --mem="$((PARALLEL * MEM_PER_RUN_GB))G" --export="ALL,RUNS_PER_TASK=$PARALLEL"
     --mail-type=END --mail-user="$MAIL_USER"
     "$SCRIPT_DIR/ddpg_sl_suft.sh" "$(csv "${ENV_LIST[@]}")" "$(csv "${SHORT_BUFFER_LIST[@]}")" "$(csv "${TF_LAMBDA_LIST[@]}")"
     "$(csv "${SAMPLING_FRACTION_LIST[@]}")" "$(csv "${BATCH_SIZE_LIST[@]}")" "$(csv "${TRAIN_FREQ_LIST[@]}")" "${RUN_LIST[@]}")
if [ -n "$DRY_RUN" ]; then
  echo "${CMD[@]}"
  # Same decoding as ddpg_sl_suft.sh: the chunk of runs varies fastest, then the environment
  TASK=0
  for SHORT_BUFFER_SIZE in "${SHORT_BUFFER_LIST[@]}"; do
    for TF_LAMBDA in "${TF_LAMBDA_LIST[@]}"; do
      for SAMPLING_FRACTION in "${SAMPLING_FRACTION_LIST[@]}"; do
        for BATCH_SIZE in "${BATCH_SIZE_LIST[@]}"; do
          for TRAIN_FREQ in "${TRAIN_FREQ_LIST[@]}"; do
            for ENV_NAME in "${ENV_LIST[@]}"; do
              for ((i = 0; i < N_CHUNKS; i++)); do
                echo "  task $TASK: $ENV_NAME, short buffer $SHORT_BUFFER_SIZE, TF lambda $TF_LAMBDA, fraction $SAMPLING_FRACTION, batch $BATCH_SIZE, train_freq $TRAIN_FREQ, runs ${RUN_LIST[*]:$((i * PARALLEL)):PARALLEL}"
                TASK=$((TASK + 1))
              done
            done
          done
        done
      done
    done
  done
else
  "${CMD[@]}"
fi
