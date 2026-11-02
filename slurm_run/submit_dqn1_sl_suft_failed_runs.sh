#!/bin/bash
# Resubmit the 2 DQN1SL runs (short buffer 4000, TF lambda 1, sampling fraction 1)
# that stopped before 1M timesteps, as a single job array of dqn1_sl_suft.sh with one run per array task:
#   Enduro:1 (stopped at 884,197), Enduro:4 (stopped at 947,396)
# Every run gets its own task (1 CPU, MEM_PER_RUN_GB) on a partition without a time limit
# The job scripts don't checkpoint, so every run starts again from timestep 0 and overwrites its old log
#
# Usage:
#   ./submit_dqn1_sl_suft_failed_runs.sh                      submit both runs
#   PARTITION=shaham-L40 ./submit_dqn1_sl_suft_failed_runs.sh another partition (default shaham-L4)
#   MEM_PER_RUN_GB=70 ./submit_dqn1_sl_suft_failed_runs.sh    another memory per run (default 55)
#   DRY_RUN=1 ./submit_dqn1_sl_suft_failed_runs.sh            print the sbatch command and every task's run without submitting
# The task log is slurm-<job id>_<task id>.out and its first line names the task's run

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SHORT_BUFFER_SIZE=4000
TF_LAMBDA=1
SAMPLING_FRACTION=1
PARTITION=${PARTITION:-shaham-L4}
# 55G per run: the 1M transitions peak at ~53.7GB
MEM_PER_RUN_GB=${MEM_PER_RUN_GB:-55}

# The failed runs as <Game>:<run_number>, one per array task
RUN_PAIRS=(
  Enduro:1
  Enduro:4
)
N_TASKS=${#RUN_PAIRS[@]}

CMD=(sbatch --array="0-$((N_TASKS - 1))" --partition="$PARTITION" --job-name="dqn1_sl_failed"
     --cpus-per-task=1 --mem="${MEM_PER_RUN_GB}G"
     "$SCRIPT_DIR/dqn1_sl_suft.sh" "$SHORT_BUFFER_SIZE" "$TF_LAMBDA" "$SAMPLING_FRACTION" "${RUN_PAIRS[@]}")
if [ -n "$DRY_RUN" ]; then
  echo "${CMD[@]}"
  for ((i = 0; i < N_TASKS; i++)); do
    echo "  task $i: ${RUN_PAIRS[$i]}"
  done
else
  "${CMD[@]}"
fi
