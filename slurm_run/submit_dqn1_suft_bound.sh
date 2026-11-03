#!/bin/bash
# Submit the SUFT bound runs as 2 job arrays, all on Alien, 1M timesteps, a single run per configuration
#
# DQN1SB (dqn1_suft_bound.sh), one job array of 4 tasks:
#   buffer 1M and 4K, each with TF lambda 0 and 1
# DQN1SLSB (dqn1_sl_suft_bound.sh), one job array of 12 tasks, long term buffer = 1M - short term buffer:
#   short term buffer 500, 1K, 2K, 4K, 8K, 16K, each with TF lambda 0 and 1
# Both job arrays run on the generic partition
# Every task gets the same resources (#SBATCH lines of the job script: 1 GPU, 1 CPU core, 64G),
# which the 1M transitions need; the 4K buffer DQN1SB tasks need much less but share the array's request
#
# Usage:
#   ./submit_dqn1_suft_bound.sh                     both job arrays
#   ONLY=SB ./submit_dqn1_suft_bound.sh             only the DQN1SB job array (ONLY=SLSB: only the DQN1SLSB one)
#   GAME=Pong RUN_NUMBER=2 ./submit_dqn1_suft_bound.sh   another game / run number (default Alien, run 1)
#   SAMPLING_FRACTION=0.25 ./submit_dqn1_suft_bound.sh   DQN1SLSB fraction of each minibatch from the short term buffer (default 0.5)
#   DRY_RUN=1 ./submit_dqn1_suft_bound.sh           print the sbatch commands and every task's configuration without submitting
# The task log is slurm-<job id>_<task id>.out and its first line names the task's configuration
# Every run prints one line per gradient update (~250K lines, ~110MB per log for 1M timesteps)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GAME=${GAME:-Alien}
RUN_NUMBER=${RUN_NUMBER:-1}
SAMPLING_FRACTION=${SAMPLING_FRACTION:-0.5}
ONLY=${ONLY:-}

SB_BUFFER_SIZES="1000000 4000"
SLSB_SHORT_BUFFER_SIZES="500 1000 2000 4000 8000 16000"
TF_LAMBDAS="0 1"

# Submit one job array, task i runs the i-th configuration
submit_array() {
  local NAME=$1 SCRIPT=$2
  shift 2
  local N_FIXED=$1
  shift
  local FIXED_ARGS=("${@:1:N_FIXED}")
  local CONFIGS=("${@:N_FIXED+1}")
  local CMD=(sbatch --array="0-$((${#CONFIGS[@]} - 1))" --partition=generic --job-name="$NAME"
             "$SCRIPT_DIR/$SCRIPT" "${FIXED_ARGS[@]}" "${CONFIGS[@]}")
  if [ -n "$DRY_RUN" ]; then
    echo "${CMD[@]}"
    for ((i = 0; i < ${#CONFIGS[@]}; i++)); do
      echo "  task $i: ${CONFIGS[$i]}"
    done
  else
    "${CMD[@]}"
  fi
}

if [ -z "$ONLY" ] || [ "$ONLY" = "SB" ]; then
  # <buffer_size>:<tf_lambda>
  SB_CONFIGS=()
  for BUFFER_SIZE in $SB_BUFFER_SIZES; do
    for TF_LAMBDA in $TF_LAMBDAS; do
      SB_CONFIGS+=("$BUFFER_SIZE:$TF_LAMBDA")
    done
  done
  submit_array dqn1_sb dqn1_suft_bound.sh 2 "$GAME" "$RUN_NUMBER" "${SB_CONFIGS[@]}"
fi

if [ -z "$ONLY" ] || [ "$ONLY" = "SLSB" ]; then
  # <short_buffer_size>:<tf_lambda>
  SLSB_CONFIGS=()
  for TF_LAMBDA in $TF_LAMBDAS; do
    for SHORT_BUFFER_SIZE in $SLSB_SHORT_BUFFER_SIZES; do
      SLSB_CONFIGS+=("$SHORT_BUFFER_SIZE:$TF_LAMBDA")
    done
  done
  submit_array dqn1_slsb dqn1_sl_suft_bound.sh 3 "$SAMPLING_FRACTION" "$GAME" "$RUN_NUMBER" "${SLSB_CONFIGS[@]}"
fi
