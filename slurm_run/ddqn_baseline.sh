#!/bin/bash
#SBATCH --job-name=ddqn_base
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

# Baseline: a single DDQN run on one Atari game (1M buffer) on one CPU core
# RAM: the 1M replay buffer takes ~57GB (observations + next observations), plus the process overhead
#
# Usage: sbatch ddqn_baseline.sh <Game> <run_number>     e.g. sbatch ddqn_baseline.sh Alien 1
# <Game> is the ALE name without prefix/suffix (Alien, KungFuMaster, ...)
# To submit many games / runs at once use submit_ddqn_baseline_long_games.sh / submit_ddqn_baseline_other_games.sh

GAME=$1
RUN_NUMBER=$2
if [ -z "$GAME" ] || [ -z "$RUN_NUMBER" ]; then
  echo "Usage: sbatch ddqn_baseline.sh <Game> <run_number>" >&2
  exit 1
fi

SB3_DIR=/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3
PYTHON=~/miniconda3/envs/stable-baselines3/bin/python

DEVICE="cuda:0"
TF_LAMBDA=0
AGENT_NAME="DDQN"
RESULTS_DIR=$SB3_DIR/results/$AGENT_NAME
TIMESTEPS=1000000
GYM_ENV_NAME="ALE/${GAME}-v5"
BUFFER_SIZE=1000000

# One core for the run: stop PyTorch / NumPy from spawning extra threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Import stable_baselines3 from this repo (atari100k.py appends the old repo to sys.path)
export PYTHONPATH="$SB3_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RESULTS_DIR"
cd "$SB3_DIR/stable_baselines3" || exit 1

LOG_FILE="$RESULTS_DIR/${GAME}-v5_${AGENT_NAME}_baseline_tf_lambda_${TF_LAMBDA}_timesteps_${TIMESTEPS}_buffer_${BUFFER_SIZE}_run_${RUN_NUMBER}.log"
$PYTHON "$SB3_DIR/stable_baselines3/atari100k.py" "$DEVICE" "$TF_LAMBDA" "$AGENT_NAME" "$TIMESTEPS" "$RUN_NUMBER" "$GYM_ENV_NAME" "$BUFFER_SIZE" \
  > "$LOG_FILE" 2>&1
