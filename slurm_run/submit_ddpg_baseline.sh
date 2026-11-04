#!/bin/bash
# Submit ddpg_baseline.sh as a single job array over every (train frequency, environment) configuration:
# the runs of each configuration are split into chunks of PARALLEL runs (default 5), one array task per chunk,
# and each task runs its chunk in parallel, one CPU core and 7G per run
# (5 environments, runs 1-20 with PARALLEL=5 -> 4 tasks per environment, 20 tasks, 100 runs)
# Task order (the task id) is a nested loop: train freq -> environment -> chunk of runs (see ddpg_baseline.sh)
# One email is sent when the whole job array ends (Slurm mails per array, not per task, without ARRAY_TASKS)
#
# Usage:
#   ./submit_ddpg_baseline.sh                      the 5 MuJoCo environments, runs 1-20, batch 256, train freq 1
#   ./submit_ddpg_baseline.sh Ant Hopper           only these environments
#   RUNS="1 2" ./submit_ddpg_baseline.sh Ant       only these runs
#   BATCH_SIZE=32 TRAIN_FREQ="4 1" ./submit_ddpg_baseline.sh   other minibatch size / train frequencies (default 256 / 1)
#   PARALLEL=10 ./submit_ddpg_baseline.sh          runs per array task, all in parallel (default 5)
#   DRY_RUN=1 ./submit_ddpg_baseline.sh            print the sbatch command and every task's runs without submitting
#   PARTITION=L4-12h ./submit_ddpg_baseline.sh     another partition (default cpu192G-48h)
#   MAIL_USER=fiskust@biu.ac.il ./submit_ddpg_baseline.sh         another address for the end of array email
# The task log is slurm-<job id>_<task id>.out and its first line names the task's configuration and runs

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -ra RUN_LIST <<< "${RUNS:-$(seq -s " " 1 20)}"
BATCH_SIZE=${BATCH_SIZE:-256}
read -ra TRAIN_FREQ_LIST <<< "${TRAIN_FREQ:-1}"

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

N_TASKS=$((N_CHUNKS * ${#ENV_LIST[@]} * ${#TRAIN_FREQ_LIST[@]}))
# ddpg_baseline.sh takes the environments and train frequencies as comma separated lists and picks one per array task
csv() { local IFS=','; echo "$*"; }

# RUNS_PER_TASK makes ddpg_baseline.sh give each array task one chunk of PARALLEL runs
CMD=(sbatch --partition="$PARTITION" --array="0-$((N_TASKS - 1))" --job-name="ddpg_base_b${BATCH_SIZE}"
     --cpus-per-task="$PARALLEL" --mem="$((PARALLEL * MEM_PER_RUN_GB))G" --export="ALL,RUNS_PER_TASK=$PARALLEL"
     --mail-type=END --mail-user="$MAIL_USER"
     "$SCRIPT_DIR/ddpg_baseline.sh" "$(csv "${ENV_LIST[@]}")" "$BATCH_SIZE" "$(csv "${TRAIN_FREQ_LIST[@]}")" "${RUN_LIST[@]}")
if [ -n "$DRY_RUN" ]; then
  echo "${CMD[@]}"
  # Same decoding as ddpg_baseline.sh: the chunk of runs varies fastest, then the environment
  TASK=0
  for TRAIN_FREQ in "${TRAIN_FREQ_LIST[@]}"; do
    for ENV_NAME in "${ENV_LIST[@]}"; do
      for ((i = 0; i < N_CHUNKS; i++)); do
        echo "  task $TASK: $ENV_NAME, train_freq $TRAIN_FREQ, runs ${RUN_LIST[*]:$((i * PARALLEL)):PARALLEL}"
        TASK=$((TASK + 1))
      done
    done
  done
else
  "${CMD[@]}"
fi
