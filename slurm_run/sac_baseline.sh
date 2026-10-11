#!/bin/bash
#SBATCH --job-name=sac_base
#SBATCH --partition=shaham-L4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-user=fiskust@biu.ac.il
#SBATCH --mail-type=END

# Baseline: a single SAC run on one MuJoCo v5 environment (1M buffer) on one CPU core
# RAM: the 1M replay buffer takes ~6GB for Humanoid (observations + next observations), less for the others
#
# Usage: sbatch sac_baseline.sh <Env> <run_number>     e.g. sbatch sac_baseline.sh Ant 1
# <Env> is the MuJoCo name without the version suffix (Ant, HalfCheetah, Hopper, Humanoid, Walker2d)
# To submit many environments / runs at once use submit_sac_baseline.sh

ENV_NAME=$1
RUN_NUMBER=$2
if [ -z "$ENV_NAME" ] || [ -z "$RUN_NUMBER" ]; then
  echo "Usage: sbatch sac_baseline.sh <Env> <run_number>" >&2
  exit 1
fi

SB3_DIR=/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3
PYTHON=~/miniconda3/envs/stable-baselines3/bin/python

DEVICE="cpu"
TF_LAMBDA=0
AGENT_NAME="SAC"
RESULTS_DIR=$SB3_DIR/results/$AGENT_NAME
TIMESTEPS=10000000
GYM_ENV_NAME="${ENV_NAME}-v5"
BUFFER_SIZE=1000000

# One core for the run: stop PyTorch / NumPy from spawning extra threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Import stable_baselines3 from this repo (mujoco_cf.py appends the old repo to sys.path)
export PYTHONPATH="$SB3_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RESULTS_DIR"
cd "$SB3_DIR/stable_baselines3" || exit 1

LOG_FILE="$RESULTS_DIR/${GYM_ENV_NAME}_${AGENT_NAME}_baseline_tf_lambda_${TF_LAMBDA}_timesteps_${TIMESTEPS}_buffer_${BUFFER_SIZE}_run_${RUN_NUMBER}.log"
$PYTHON "$SB3_DIR/stable_baselines3/mujoco_cf.py" "$DEVICE" "$TF_LAMBDA" "$AGENT_NAME" "$TIMESTEPS" "$RUN_NUMBER" "$GYM_ENV_NAME" "$BUFFER_SIZE" \
  > "$LOG_FILE" 2>&1
