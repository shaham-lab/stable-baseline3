#!/bin/bash
# Resubmit the 6 DQNSLNStep runs (short buffer 4000, TF lambda 1, sampling fraction 0.5, n_steps 3)
# that stopped before 1M timesteps, as a single job array of dqn_sl_n_step_suft.sh with one run per array task:
#   DoubleDunk:4, Enduro:1-4, FishingDerby:4
#     cancelled on the 4h partition (job 35662699) and Enduro again by the time limit on the reruns (jobs 37647325, 37647326)
# Every run gets its own task (1 CPU, MEM_PER_RUN_GB) on a partition without a time limit
# The job scripts don't checkpoint, so every run starts again from timestep 0 and overwrites its old log
#
# Usage:
#   ./submit_dqn_sl_n_step_suft_failed_runs.sh                      submit all 6 runs
#   PARTITION=shaham-L40 ./submit_dqn_sl_n_step_suft_failed_runs.sh another partition (default shaham-L4)
#   MEM_PER_RUN_GB=70 ./submit_dqn_sl_n_step_suft_failed_runs.sh    another memory per run (default 55)
#   DRY_RUN=1 ./submit_dqn_sl_n_step_suft_failed_runs.sh            print the sbatch command and every task's run without submitting
# The task log is slurm-<job id>_<task id>.out and its first line names the task's run

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SHORT_BUFFER_SIZE=4000
TF_LAMBDA=1
SAMPLING_FRACTION=0.5
N_STEPS=3
PARTITION=${PARTITION:-shaham-L4}
# The runs hit the time limit, not the memory limit, so 55G per run (the 1M transitions peak at ~53.7GB) is kept
MEM_PER_RUN_GB=${MEM_PER_RUN_GB:-55}

# The failed runs as <Game>:<run_number>, one per array task
RUN_PAIRS=(
  DoubleDunk:4
  Enduro:1
  Enduro:2
  Enduro:3
  Enduro:4
  FishingDerby:4
)
N_TASKS=${#RUN_PAIRS[@]}

CMD=(sbatch --array="0-$((N_TASKS - 1))" --partition="$PARTITION" --job-name="dqn_sl_nstep_failed"
     --cpus-per-task=1 --mem="${MEM_PER_RUN_GB}G"
     "$SCRIPT_DIR/dqn_sl_n_step_suft.sh" "$SHORT_BUFFER_SIZE" "$TF_LAMBDA" "$SAMPLING_FRACTION" "$N_STEPS" "${RUN_PAIRS[@]}")
if [ -n "$DRY_RUN" ]; then
  echo "${CMD[@]}"
  for ((i = 0; i < N_TASKS; i++)); do
    echo "  task $i: ${RUN_PAIRS[$i]}"
  done
else
  "${CMD[@]}"
fi
