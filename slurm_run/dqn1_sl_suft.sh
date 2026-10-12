#!/bin/bash
#SBATCH --job-name=dqn1_sl_suft
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

# Short-Long SUFT: a single DQN1SL run on one Atari game on one CPU core
# The memory is split between a short term and a long term buffer, 1M transitions in total:
# long term buffer size = 1M - short term buffer size
# RAM: the 1M transitions take ~57GB (observations + next observations), plus the process overhead
#
# Usage: sbatch dqn1_sl_suft.sh <Game> <run_number> <short_buffer_size> <tf_lambda> <sampling_fraction>
#        e.g. sbatch dqn1_sl_suft.sh Alien 1 4000 1 0.5
# <Game> is the ALE name without prefix/suffix (Alien, KungFuMaster, ...)
# <sampling_fraction> is the fraction of each minibatch taken from the short term buffer
# To submit many games / runs / hyperparameters at once use submit_dqn1_sl_suft.sh

GAME=$1
RUN_NUMBER=$2
SHORT_BUFFER_SIZE=$3
TF_LAMBDA=$4
SAMPLING_FRACTION=$5
if [ -z "$GAME" ] || [ -z "$RUN_NUMBER" ] || [ -z "$SHORT_BUFFER_SIZE" ] || [ -z "$TF_LAMBDA" ] || [ -z "$SAMPLING_FRACTION" ]; then
  echo "Usage: sbatch dqn1_sl_suft.sh <Game> <run_number> <short_buffer_size> <tf_lambda> <sampling_fraction>" >&2
  exit 1
fi

SB3_DIR=/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3
PYTHON=~/miniconda3/envs/stable-baselines3/bin/python

DEVICE="cuda:0"
AGENT_NAME="DQN1SL"
RESULTS_DIR=$SB3_DIR/results/$AGENT_NAME
TIMESTEPS=1000000
GYM_ENV_NAME="ALE/${GAME}-v5"
TOTAL_BUFFER_SIZE=1000000
LONG_BUFFER_SIZE=$((TOTAL_BUFFER_SIZE - SHORT_BUFFER_SIZE))

# One core for the run: stop PyTorch / NumPy from spawning extra threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Import stable_baselines3 from this repo (atari100k.py appends the old repo to sys.path)
export PYTHONPATH="$SB3_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RESULTS_DIR"
cd "$SB3_DIR/stable_baselines3" || exit 1

LOG_FILE="$RESULTS_DIR/${GAME}-v5_${AGENT_NAME}_suft_tf_lambda_${TF_LAMBDA}_timesteps_${TIMESTEPS}_short_buffer_${SHORT_BUFFER_SIZE}_long_buffer_${LONG_BUFFER_SIZE}_fraction_${SAMPLING_FRACTION}_run_${RUN_NUMBER}.log"
$PYTHON "$SB3_DIR/stable_baselines3/atari100k.py" "$DEVICE" "$TF_LAMBDA" "$AGENT_NAME" "$TIMESTEPS" "$RUN_NUMBER" "$GYM_ENV_NAME" "$LONG_BUFFER_SIZE" \
  "$SHORT_BUFFER_SIZE" "$SAMPLING_FRACTION" \
  > "$LOG_FILE" 2>&1
