#!/bin/bash
# Submit ddpg_baseline.sh as a single job array, with one array task per (train frequency, environment) pair:
# each task runs all the runs of its pair in parallel
#
# Usage:
#   ./submit_ddpg_baseline.sh                      all 5 MuJoCo environments, runs 1-20, train freqs 4 and 1 (1 job array, 10 tasks)
#   ./submit_ddpg_baseline.sh Ant Hopper           only these environments
#   RUNS="1 2" ./submit_ddpg_baseline.sh Ant       only these runs
#   BATCH_SIZE=256 TRAIN_FREQ=1 ./submit_ddpg_baseline.sh   other minibatch size / train frequencies (default 32 / "4 1")
#   DRY_RUN=1 ./submit_ddpg_baseline.sh            print the sbatch command without submitting
# Array task i runs environment i % N_ENVS with train frequency i / N_ENVS (0-based); the task log is slurm-<job id>_<task id>.out

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -ra RUN_LIST <<< "${RUNS:-$(seq -s " " 1 20)}"
BATCH_SIZE=${BATCH_SIZE:-32}
read -ra TRAIN_FREQ_LIST <<< "${TRAIN_FREQ:-4 1}"

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
N_ENVS=${#ENV_LIST[@]}
N_TASKS=$((N_ENVS * ${#TRAIN_FREQ_LIST[@]}))
# ddpg_baseline.sh takes the environments and train frequencies as comma separated lists and picks one pair per array task
ENVS_CSV=$(IFS=','; echo "${ENV_LIST[*]}")
TRAIN_FREQS_CSV=$(IFS=','; echo "${TRAIN_FREQ_LIST[*]}")

CMD=(sbatch --array="0-$((N_TASKS - 1))" --job-name="ddpg_base_b${BATCH_SIZE}"
     --cpus-per-task="$N_RUNS" --mem="$((N_RUNS * MEM_PER_RUN_GB))G"
     "$SCRIPT_DIR/ddpg_baseline.sh" "$ENVS_CSV" "$BATCH_SIZE" "$TRAIN_FREQS_CSV" "${RUN_LIST[@]}")
if [ -n "$DRY_RUN" ]; then
  echo "${CMD[@]}"
  for ((i = 0; i < N_TASKS; i++)); do
    echo "  task $i: ${ENV_LIST[$((i % N_ENVS))]}, train_freq ${TRAIN_FREQ_LIST[$((i / N_ENVS))]}"
  done
else
  "${CMD[@]}"
fi
