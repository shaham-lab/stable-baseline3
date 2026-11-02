#!/bin/bash
# Resubmit the 10 DQNSLCER runs (short buffer 4000, TF lambda 1, sampling fraction 0.5, CER latest train_freq)
# that stopped before 1M timesteps, as a single job array of dqnsl_cer_suft.sh with one run per array task:
#   Phoenix:1, Tennis:2, MontezumaRevenge:2, Pitfall:2, Solaris:2, Tutankham:2
#     OOM killed on shaham-L40 (jobs 10723_3, 10723_4: 3 runs sharing 165G)
#   Enduro:1-4
#     cancelled on L40s-4h (job 35663422) and again by the time limit on the rerun (job 37647327)
# Every run gets its own task (1 CPU, MEM_PER_RUN_GB) on a partition without a time limit
# The job scripts don't checkpoint, so every run starts again from timestep 0 and overwrites its old log
#
# Usage:
#   ./submit_dqnsl_cer_suft_failed_runs.sh                     submit all 10 runs
#   PARTITION=shaham-L4 ./submit_dqnsl_cer_suft_failed_runs.sh another partition (default shaham-L40)
#   MEM_PER_RUN_GB=80 ./submit_dqnsl_cer_suft_failed_runs.sh   another memory per run (default 70)
#   DRY_RUN=1 ./submit_dqnsl_cer_suft_failed_runs.sh           print the sbatch command and every task's run without submitting
# The task log is slurm-<job id>_<task id>.out and its first line names the task's run

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SHORT_BUFFER_SIZE=4000
TF_LAMBDA=1
SAMPLING_FRACTION=0.5
CER_N_LATEST=train_freq
PARTITION=${PARTITION:-shaham-L4}
# 55G per run was not enough with 3 runs in one job, so every run gets more and a job of its own
MEM_PER_RUN_GB=${MEM_PER_RUN_GB:-70}

# The failed runs as <Game>:<run_number>, one per array task
RUN_PAIRS=(
  Enduro:1
  Enduro:2
  Enduro:3
  Enduro:4
  MontezumaRevenge:2
  Phoenix:1
  Pitfall:2
  Solaris:2
  Tennis:2
  Tutankham:2
)
N_TASKS=${#RUN_PAIRS[@]}

CMD=(sbatch --export=ALL,CER_N_LATEST="$CER_N_LATEST" --array="0-$((N_TASKS - 1))" --partition="$PARTITION" --job-name="dqnsl_cer_failed"
     --cpus-per-task=1 --mem="${MEM_PER_RUN_GB}G"
     "$SCRIPT_DIR/dqnsl_cer_suft.sh" "$SHORT_BUFFER_SIZE" "$TF_LAMBDA" "$SAMPLING_FRACTION" "${RUN_PAIRS[@]}")
if [ -n "$DRY_RUN" ]; then
  echo "${CMD[@]}"
  for ((i = 0; i < N_TASKS; i++)); do
    echo "  task $i: ${RUN_PAIRS[$i]}"
  done
else
  "${CMD[@]}"
fi
