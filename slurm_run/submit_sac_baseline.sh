#!/bin/bash
# Submit sac_baseline.sh as one job array per (train frequency, environment) configuration:
# the runs of the configuration are split into chunks of PARALLEL runs (default 5), one array task per chunk,
# and each task runs its chunk in parallel, one CPU core and 7G per run
# (runs 1-20 with PARALLEL=5 -> 4 tasks per array: runs 1-5, 6-10, 11-15, 16-20)
#
# Usage:
#   ./submit_sac_baseline.sh                      Swimmer, runs 1-20, batch 256, train freq 1 (1 job array, 4 tasks)
#   ./submit_sac_baseline.sh Ant Hopper           only these environments (one job array each)
#   RUNS="1 2" ./submit_sac_baseline.sh Ant       only these runs
#   BATCH_SIZE=32 TRAIN_FREQ="4 1" ./submit_sac_baseline.sh   other minibatch size / train frequencies (default 256 / 1)
#   PARALLEL=10 ./submit_sac_baseline.sh          runs per array task, all in parallel (default 5)
#   DRY_RUN=1 ./submit_sac_baseline.sh            print the sbatch commands and every task's runs without submitting
#   PARTITION=L4-12h ./submit_sac_baseline.sh                 another partition (default shaham-L4)
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
PARTITION=${PARTITION:-shaham-L4}

# The MuJoCo (v5) environments
MUJOCO_ENVS="Ant HalfCheetah Hopper Humanoid Walker2d Swimmer"
# MUJOCO_ENVS="Swimmer"

if [ $# -gt 0 ]; then
  read -ra ENV_LIST <<< "$*"
else
  read -ra ENV_LIST <<< "$MUJOCO_ENVS"
fi

for TRAIN_FREQ in "${TRAIN_FREQ_LIST[@]}"; do
  for ENV_NAME in "${ENV_LIST[@]}"; do
    # RUNS_PER_TASK makes sac_baseline.sh give array task i the i-th chunk of PARALLEL runs
    CMD=(sbatch --partition="$PARTITION" --array="0-$((N_CHUNKS - 1))" --job-name="sac_base_${ENV_NAME}_b${BATCH_SIZE}_tf${TRAIN_FREQ}"
         --cpus-per-task="$PARALLEL" --mem="$((PARALLEL * MEM_PER_RUN_GB))G" --export="ALL,RUNS_PER_TASK=$PARALLEL"
         "$SCRIPT_DIR/sac_baseline.sh" "$ENV_NAME" "$BATCH_SIZE" "$TRAIN_FREQ" "${RUN_LIST[@]}")
    if [ -n "$DRY_RUN" ]; then
      echo "${CMD[@]}"
      for ((i = 0; i < N_CHUNKS; i++)); do
        echo "  task $i: $ENV_NAME, train_freq $TRAIN_FREQ, runs ${RUN_LIST[*]:$((i * PARALLEL)):PARALLEL}"
      done
    else
      "${CMD[@]}"
    fi
  done
done
