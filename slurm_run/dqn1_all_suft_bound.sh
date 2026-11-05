#!/bin/bash
#SBATCH --job-name=dqn1_bound
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

# DQN1SB and DQN1SLSB SUFT bound runs in a single job array: every array task picks its <task> and runs
# dqn1_suft_bound.sh (DQN1SB) or dqn1_sl_suft_bound.sh (DQN1SLSB) with it, as a single (non array) run
# The resources are the same as the ones of those 2 scripts (1 GPU, 1 CPU core, 64G)
#
# Usage: sbatch --array=0-<N-1> dqn1_all_suft_bound.sh <Game> <run_number> <task> [<task> ...]
# <task> is one of
#   SB:<buffer_size>:<tf_lambda>[:<track_behavior_distance>[:<replay_ratio>]]
#   SLSB:<sampling_fraction>:<short_buffer_size>:<tf_lambda>[:<track_behavior_distance>[:<replay_ratio>]]
# (see dqn1_suft_bound.sh / dqn1_sl_suft_bound.sh for the fields)
#        e.g. sbatch --array=0-1 dqn1_all_suft_bound.sh Alien 1 SB:4000:1:false:4 SLSB:0.5:4000:1:false:4
# Slurm runs a copy of this script, so the directory of the 2 scripts comes from SUFT_SCRIPT_DIR
# (submit_dqn1_suft_bound.sh exports it), default the slurm_run directory of the cluster repo
# To submit all the SUFT bound configurations at once use submit_dqn1_suft_bound.sh

GAME=$1
RUN_NUMBER=$2
if [ -z "$GAME" ] || [ -z "$RUN_NUMBER" ] || [ -z "$3" ]; then
  echo "Usage: sbatch --array=... dqn1_all_suft_bound.sh <Game> <run_number> <task> [<task> ...]" >&2
  exit 1
fi
shift 2
TASKS=("$@")

# Job array: pick this task's configuration
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
  if [ -z "$TASK" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID is past the last of the ${#TASKS[@]} given <task>" >&2
    exit 1
  fi
elif [ ${#TASKS[@]} -gt 1 ]; then
  echo "Several <task> arguments are only allowed in a job array (sbatch --array=...)" >&2
  exit 1
else
  TASK=${TASKS[0]}
fi

SCRIPT_DIR=${SUFT_SCRIPT_DIR:-/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3/slurm_run}
AGENT=${TASK%%:*}
REST=${TASK#*:}
echo "${SLURM_ARRAY_TASK_ID:+Array task $SLURM_ARRAY_TASK_ID: }$TASK"
# The run script gets a single <config>, without the array task id it would use to pick one
case "$AGENT" in
  SB)
    exec env -u SLURM_ARRAY_TASK_ID bash "$SCRIPT_DIR/dqn1_suft_bound.sh" "$GAME" "$RUN_NUMBER" "$REST"
    ;;
  SLSB)
    SAMPLING_FRACTION=${REST%%:*}
    CONFIG=${REST#*:}
    exec env -u SLURM_ARRAY_TASK_ID bash "$SCRIPT_DIR/dqn1_sl_suft_bound.sh" "$SAMPLING_FRACTION" "$GAME" "$RUN_NUMBER" "$CONFIG"
    ;;
  *)
    echo "Bad <task> '$TASK', expected SB:<config> or SLSB:<sampling_fraction>:<config>" >&2
    exit 1
    ;;
esac
