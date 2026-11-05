#!/bin/bash
#SBATCH --job-name=dqn1_sb
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

# SUFT bound: a single DQN1SB run on one Atari game on one CPU core
# After every gradient update it prints the SUFT term of the cold buffer (the first experiences, collected
# before any gradient update) and of the minibatch, the distance of the weights from the initial ones
# and the gradient updates / actions taken since the oldest sample of the minibatch was collected
# RAM: a 1M replay buffer takes ~57GB (observations + next observations), plus the process overhead
#
# Usage: sbatch dqn1_suft_bound.sh <Game> <run_number> <config>
#        e.g. sbatch dqn1_suft_bound.sh Alien 1 1000000:1
#             sbatch dqn1_suft_bound.sh Alien 1 4000:1:false:4      (replay ratio 4)
# <Game> is the ALE name without prefix/suffix (Alien, KungFuMaster, ...)
# <config> is <buffer_size>:<tf_lambda>[:<track_behavior_distance>[:<replay_ratio>]]
# <track_behavior_distance> true/false (default false): also print the distance from the weights each sample
#   was collected with; it keeps a copy of the weights (~6.7MB) per gradient update still referenced by
#   the replay buffer, so it only fits with a small buffer
# <replay_ratio> (default 0.25): gradient updates per environment step, 1, 2, 4, 8, 16, ... (that many updates
#   after every step) or 0.25, 0.5, ... (one update every 4, 2, ... steps); the default 0.25 is the original
#   1 update every 4 steps. A higher ratio is that many times more updates and log lines (and run time)
# As a job array pass one <config> per array task: task i runs the i-th <config> (0-based)
#        e.g. sbatch --array=0-3 dqn1_suft_bound.sh Alien 1 1000000:0 1000000:1 4000:0 4000:1
# To submit all the SUFT bound configurations at once use submit_dqn1_suft_bound.sh

GAME=$1
RUN_NUMBER=$2
if [ -z "$GAME" ] || [ -z "$RUN_NUMBER" ] || [ -z "$3" ]; then
  echo "Usage: sbatch dqn1_suft_bound.sh <Game> <run_number> <config> [<config> ...]" >&2
  exit 1
fi
shift 2
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
IFS=':' read -r BUFFER_SIZE TF_LAMBDA TRACK_BEHAVIOR_DISTANCE REPLAY_RATIO <<< "$CONFIG"
TRACK_BEHAVIOR_DISTANCE=${TRACK_BEHAVIOR_DISTANCE:-false}
REPLAY_RATIO=${REPLAY_RATIO:-0.25}
if [ -z "$BUFFER_SIZE" ] || [ -z "$TF_LAMBDA" ]; then
  echo "Bad <config> '$CONFIG', expected <buffer_size>:<tf_lambda>[:<track_behavior_distance>[:<replay_ratio>]]" >&2
  exit 1
fi
echo "${SLURM_ARRAY_TASK_ID:+Array task $SLURM_ARRAY_TASK_ID: }$GAME run $RUN_NUMBER, buffer $BUFFER_SIZE, TF lambda $TF_LAMBDA, track behavior distance $TRACK_BEHAVIOR_DISTANCE, replay ratio $REPLAY_RATIO"

SB3_DIR=/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3
PYTHON=~/miniconda3/envs/stable-baselines3/bin/python

DEVICE="cuda:0"
AGENT_NAME="DQN1SB"
RESULTS_DIR=$SB3_DIR/results/$AGENT_NAME
TIMESTEPS=1000000
GYM_ENV_NAME="ALE/${GAME}-v5"

# One core for the run: stop PyTorch / NumPy from spawning extra threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Import stable_baselines3 from this repo (atari100k.py appends the old repo to sys.path)
export PYTHONPATH="$SB3_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RESULTS_DIR"
cd "$SB3_DIR/stable_baselines3" || exit 1

LOG_FILE="$RESULTS_DIR/${GAME}-v5_${AGENT_NAME}_suft_tf_lambda_${TF_LAMBDA}_timesteps_${TIMESTEPS}_buffer_${BUFFER_SIZE}_track_${TRACK_BEHAVIOR_DISTANCE}_replay_ratio_${REPLAY_RATIO}_run_${RUN_NUMBER}.log"
$PYTHON "$SB3_DIR/stable_baselines3/atari100k.py" "$DEVICE" "$TF_LAMBDA" "$AGENT_NAME" "$TIMESTEPS" "$RUN_NUMBER" "$GYM_ENV_NAME" "$BUFFER_SIZE" \
  "$TRACK_BEHAVIOR_DISTANCE" "$REPLAY_RATIO" \
  > "$LOG_FILE" 2>&1
