#!/bin/bash
#SBATCH --job-name=ddpg_sl_suft
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Short-Long SUFT: DDPGSL on one MuJoCo v5 environment, all the runs in parallel in one job, one CPU core per run
# The memory is split between a short term and a long term buffer, 1M transitions in total:
# long term buffer size = 1M - short term buffer size
# RAM: 16G per run - the 1M transitions take ~6GB for Humanoid (observations + next observations), less for the others
#
# Usage: sbatch ddpg_sl_suft.sh <Env> <short_buffer_size> <tf_lambda> <sampling_fraction> [run_number ...]
#        e.g. sbatch ddpg_sl_suft.sh Ant 4000 1 0.5        (runs 1-4)
#             sbatch ddpg_sl_suft.sh Ant 4000 1 0.5 1 2    (only these runs)
# <Env> is the MuJoCo name without the version suffix (Ant, HalfCheetah, Hopper, Humanoid, Walker2d)
# <sampling_fraction> is the fraction of each minibatch taken from the short term buffer
# The #SBATCH lines above are sized for 4 runs; for another number of runs pass --cpus-per-task / --mem to sbatch
# To submit many environments / hyperparameters at once use submit_ddpg_sl_suft.sh (it sizes the job to the number of runs)

ENV_NAME=$1
SHORT_BUFFER_SIZE=$2
TF_LAMBDA=$3
SAMPLING_FRACTION=$4
if [ -z "$ENV_NAME" ] || [ -z "$SHORT_BUFFER_SIZE" ] || [ -z "$TF_LAMBDA" ] || [ -z "$SAMPLING_FRACTION" ]; then
  echo "Usage: sbatch ddpg_sl_suft.sh <Env> <short_buffer_size> <tf_lambda> <sampling_fraction> [run_number ...]" >&2
  exit 1
fi
shift 4
RUNS=${*:-"1 2 3 4"}

SB3_DIR=/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3
PYTHON=~/miniconda3/envs/stable-baselines3-mujoco/bin/python

DEVICE="cpu"
AGENT_NAME="DDPGSL"
RESULTS_DIR=$SB3_DIR/results/$AGENT_NAME
TIMESTEPS=1000000
GYM_ENV_NAME="${ENV_NAME}-v5"
TOTAL_BUFFER_SIZE=1000000
LONG_BUFFER_SIZE=$((TOTAL_BUFFER_SIZE - SHORT_BUFFER_SIZE))

# One core per run: stop PyTorch / NumPy from spawning extra threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Import stable_baselines3 from this repo (mujoco_10M.py appends the old repo to sys.path)
export PYTHONPATH="$SB3_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RESULTS_DIR"
cd "$SB3_DIR/stable_baselines3" || exit 1

# Start every run in the background, each with its own log file
PIDS=()
for RUN_NUMBER in $RUNS; do
  LOG_FILE="$RESULTS_DIR/${GYM_ENV_NAME}_${AGENT_NAME}_suft_tf_lambda_${TF_LAMBDA}_timesteps_${TIMESTEPS}_short_buffer_${SHORT_BUFFER_SIZE}_long_buffer_${LONG_BUFFER_SIZE}_fraction_${SAMPLING_FRACTION}_run_${RUN_NUMBER}.log"
  $PYTHON "$SB3_DIR/stable_baselines3/mujoco_10M.py" "$DEVICE" "$TF_LAMBDA" "$AGENT_NAME" "$TIMESTEPS" "$RUN_NUMBER" "$GYM_ENV_NAME" "$LONG_BUFFER_SIZE" \
    "$SHORT_BUFFER_SIZE" "$SAMPLING_FRACTION" \
    > "$LOG_FILE" 2>&1 &
  PIDS+=($!)
done

# Wait for all the runs; the job fails if any of them failed
STATUS=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || STATUS=1
done
exit $STATUS
