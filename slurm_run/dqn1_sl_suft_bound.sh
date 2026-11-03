#!/bin/bash
#SBATCH --job-name=dqn1_slsb
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

# Short-Long SUFT bound: a single DQN1SLSB run on one Atari game on one CPU core
# After every gradient update it prints the SUFT term of the cold buffer (the first experiences, collected
# before any gradient update) and of the short / long term samples of the minibatch, the distance of the weights
# from the initial ones and the gradient updates / actions taken since the oldest sample was collected
# The memory is split between a short term and a long term buffer, 1M transitions in total:
# long term buffer size = 1M - short term buffer size
# RAM: the 1M transitions take ~57GB (observations + next observations), plus the process overhead
#
# Usage: sbatch dqn1_sl_suft_bound.sh <sampling_fraction> <Game> <run_number> <config>
#        e.g. sbatch dqn1_sl_suft_bound.sh 0.5 Alien 1 4000:1
# <sampling_fraction> is the fraction of each minibatch taken from the short term buffer
# <Game> is the ALE name without prefix/suffix (Alien, KungFuMaster, ...)
# <config> is <short_buffer_size>:<tf_lambda>[:<track_behavior_distance>]
# <track_behavior_distance> true/false (default false): also print the distance from the weights each sample
#   was collected with; it keeps a copy of the weights (~6.7MB) per gradient update still referenced by
#   the replay buffers, so it doesn't fit with a 1M long term buffer
# As a job array pass one <config> per array task: task i runs the i-th <config> (0-based)
#        e.g. sbatch --array=0-1 dqn1_sl_suft_bound.sh 0.5 Alien 1 500:0 500:1
# To submit all the SUFT bound configurations at once use submit_dqn1_suft_bound.sh

SAMPLING_FRACTION=$1
GAME=$2
RUN_NUMBER=$3
if [ -z "$SAMPLING_FRACTION" ] || [ -z "$GAME" ] || [ -z "$RUN_NUMBER" ] || [ -z "$4" ]; then
  echo "Usage: sbatch dqn1_sl_suft_bound.sh <sampling_fraction> <Game> <run_number> <config> [<config> ...]" >&2
  exit 1
fi
shift 3
CONFIGS=("$@")

# Job array: pick this task's configuration
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
  if [ -z "$CONFIG" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID is past the last of the ${#CONFIGS[@]} given <config>" >&2
    exit 1
  fi
elif [ ${#CONFIGS[@]} -gt 1 ]; then
  echo "Several <config> arguments are only allowed in a job array (sbatch --array=...)" >&2
  exit 1
else
  CONFIG=${CONFIGS[0]}
fi
IFS=':' read -r SHORT_BUFFER_SIZE TF_LAMBDA TRACK_BEHAVIOR_DISTANCE <<< "$CONFIG"
TRACK_BEHAVIOR_DISTANCE=${TRACK_BEHAVIOR_DISTANCE:-false}
if [ -z "$SHORT_BUFFER_SIZE" ] || [ -z "$TF_LAMBDA" ]; then
  echo "Bad <config> '$CONFIG', expected <short_buffer_size>:<tf_lambda>[:<track_behavior_distance>]" >&2
  exit 1
fi

SB3_DIR=/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3
PYTHON=~/miniconda3/envs/stable-baselines3/bin/python

DEVICE="cuda:0"
AGENT_NAME="DQN1SLSB"
RESULTS_DIR=$SB3_DIR/results/$AGENT_NAME
TIMESTEPS=1000000
TOTAL_BUFFER_SIZE=1000000
LONG_BUFFER_SIZE=$((TOTAL_BUFFER_SIZE - SHORT_BUFFER_SIZE))
GYM_ENV_NAME="ALE/${GAME}-v5"
echo "${SLURM_ARRAY_TASK_ID:+Array task $SLURM_ARRAY_TASK_ID: }$GAME run $RUN_NUMBER, short buffer $SHORT_BUFFER_SIZE, long buffer $LONG_BUFFER_SIZE, TF lambda $TF_LAMBDA, fraction $SAMPLING_FRACTION, track behavior distance $TRACK_BEHAVIOR_DISTANCE"

# One core for the run: stop PyTorch / NumPy from spawning extra threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Import stable_baselines3 from this repo (atari100k.py appends the old repo to sys.path)
export PYTHONPATH="$SB3_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RESULTS_DIR"
cd "$SB3_DIR/stable_baselines3" || exit 1

LOG_FILE="$RESULTS_DIR/${GAME}-v5_${AGENT_NAME}_suft_tf_lambda_${TF_LAMBDA}_timesteps_${TIMESTEPS}_short_buffer_${SHORT_BUFFER_SIZE}_long_buffer_${LONG_BUFFER_SIZE}_fraction_${SAMPLING_FRACTION}_track_${TRACK_BEHAVIOR_DISTANCE}_run_${RUN_NUMBER}.log"
$PYTHON "$SB3_DIR/stable_baselines3/atari100k.py" "$DEVICE" "$TF_LAMBDA" "$AGENT_NAME" "$TIMESTEPS" "$RUN_NUMBER" "$GYM_ENV_NAME" "$LONG_BUFFER_SIZE" \
  "$SHORT_BUFFER_SIZE" "$SAMPLING_FRACTION" "$TRACK_BEHAVIOR_DISTANCE" \
  > "$LOG_FILE" 2>&1
