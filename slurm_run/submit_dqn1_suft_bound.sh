#!/bin/bash
# Submit all the SUFT bound runs as a single job array (dqn1_all_suft_bound.sh), all on Alien, 1M timesteps,
# a single run per configuration, every configuration once per replay ratio 1 and 2:
#
# DQN1SB (dqn1_suft_bound.sh), 4 configurations per replay ratio:
#   buffer 1M and 4K, each with TF lambda 0 and 1
# DQN1SLSB (dqn1_sl_suft_bound.sh), 12 configurations per replay ratio, long term buffer = 1M - short term buffer:
#   short term buffer 500, 1K, 2K, 4K, 8K, 16K, each with TF lambda 0 and 1
# (4 + 12) configurations x 2 replay ratios = 32 tasks
# The job array runs on the generic partition
# Every task gets the same resources (#SBATCH lines of the job script: 1 GPU, 1 CPU core, 64G),
# which the 1M transitions need; the 4K buffer DQN1SB tasks need much less but share the array's request
#
# Usage:
#   ./submit_dqn1_suft_bound.sh                     the job array of both agents
#   ONLY=SB ./submit_dqn1_suft_bound.sh             only the DQN1SB tasks (ONLY=SLSB: only the DQN1SLSB ones)
#   GAME=Pong RUN_NUMBER=2 ./submit_dqn1_suft_bound.sh   another game / run number (default Alien, run 1)
#   SAMPLING_FRACTION=0.25 ./submit_dqn1_suft_bound.sh   DQN1SLSB fraction of each minibatch from the short term buffer (default 0.5)
#   REPLAY_RATIOS="1" ./submit_dqn1_suft_bound.sh      gradient updates per environment step, one task per ratio
#                                                        (default "1 2", 0.25 is the original 1 update every 4 steps)
#   DRY_RUN=1 ./submit_dqn1_suft_bound.sh           print the sbatch command and every task's configuration without submitting
# The task log is slurm-<job id>_<task id>.out and its first line names the task's configuration
# Every run prints one line per gradient update, ~1M lines per log for 1M timesteps at replay ratio 1 (x2 at 2),
# ~410 bytes per line for DQN1SB and ~780 for DQN1SLSB, and runs about that many times longer than at replay ratio 1
# Log size per run (replay ratio 1 / 2): DQN1SB ~0.4 / 0.8 GB, DQN1SLSB ~0.8 / 1.6 GB
# All 32 runs: DQN1SB 4 x 3M lines ~4.9 GB + DQN1SLSB 12 x 3M lines ~28.1 GB = ~33 GB in $SB3_DIR/results

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GAME=${GAME:-Alien}
RUN_NUMBER=${RUN_NUMBER:-1}
SAMPLING_FRACTION=${SAMPLING_FRACTION:-0.5}
ONLY=${ONLY:-}
REPLAY_RATIOS=${REPLAY_RATIOS:-1 2}

SB_BUFFER_SIZES="1000000 4000"
SLSB_SHORT_BUFFER_SIZES="500 1000 2000 4000 8000 16000"
TF_LAMBDAS="0 1"

TASKS=()
if [ -z "$ONLY" ] || [ "$ONLY" = "SB" ]; then
  # SB:<buffer_size>:<tf_lambda>:<track_behavior_distance>:<replay_ratio>
  for REPLAY_RATIO in $REPLAY_RATIOS; do
    for BUFFER_SIZE in $SB_BUFFER_SIZES; do
      for TF_LAMBDA in $TF_LAMBDAS; do
        TASKS+=("SB:$BUFFER_SIZE:$TF_LAMBDA:false:$REPLAY_RATIO")
      done
    done
  done
fi
if [ -z "$ONLY" ] || [ "$ONLY" = "SLSB" ]; then
  # SLSB:<sampling_fraction>:<short_buffer_size>:<tf_lambda>:<track_behavior_distance>:<replay_ratio>
  for REPLAY_RATIO in $REPLAY_RATIOS; do
    for TF_LAMBDA in $TF_LAMBDAS; do
      for SHORT_BUFFER_SIZE in $SLSB_SHORT_BUFFER_SIZES; do
        TASKS+=("SLSB:$SAMPLING_FRACTION:$SHORT_BUFFER_SIZE:$TF_LAMBDA:false:$REPLAY_RATIO")
      done
    done
  done
fi
if [ ${#TASKS[@]} -eq 0 ]; then
  echo "No task to submit (ONLY='$ONLY', expected SB, SLSB or empty)" >&2
  exit 1
fi

# Task i runs the i-th <task>
CMD=(sbatch --array="0-$((${#TASKS[@]} - 1))" --partition=generic --job-name=dqn1_bound
     --export="ALL,SUFT_SCRIPT_DIR=$SCRIPT_DIR"
     "$SCRIPT_DIR/dqn1_all_suft_bound.sh" "$GAME" "$RUN_NUMBER" "${TASKS[@]}")
if [ -n "$DRY_RUN" ]; then
  echo "${CMD[@]}"
  echo "${#TASKS[@]} tasks"
  for ((i = 0; i < ${#TASKS[@]}; i++)); do
    echo "  task $i: ${TASKS[$i]}"
  done
else
  "${CMD[@]}"
fi
