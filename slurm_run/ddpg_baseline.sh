#!/bin/bash
#SBATCH --job-name=ddpg_base
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=fiskust@biu.ac.il
#SBATCH --mail-type=END

# Baseline: DDPG on one MuJoCo v5 environment (1M buffer), all the runs in parallel in one job, one CPU core per run
# RAM: 16G per run - the 1M replay buffer takes ~6GB for Humanoid (observations + next observations), less for the others
#
# Usage: sbatch ddpg_baseline.sh <Env> [run_number ...]     e.g. sbatch ddpg_baseline.sh Ant        (runs 1-4)
#                                                               sbatch ddpg_baseline.sh Ant 1 2    (only these runs)
# <Env> is the MuJoCo name without the version suffix (Ant, HalfCheetah, Hopper, Humanoid, Walker2d)
# The #SBATCH lines above are sized for 4 runs; for another number of runs pass --cpus-per-task / --mem to sbatch
# To submit many environments at once use submit_ddpg_baseline.sh (it sizes the job to the number of runs)

ENV_NAME=$1
if [ -z "$ENV_NAME" ]; then
  echo "Usage: sbatch ddpg_baseline.sh <Env> [run_number ...]" >&2
  exit 1
fi
shift
RUNS=${*:-"1 2 3 4"}

SB3_DIR=/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3
PYTHON=~/miniconda3/envs/stable-baselines3-mujoco/bin/python

DEVICE="cpu"
TF_LAMBDA=0
AGENT_NAME="DDPG"
RESULTS_DIR=$SB3_DIR/results/$AGENT_NAME
TIMESTEPS=1000000
GYM_ENV_NAME="${ENV_NAME}-v5"
BUFFER_SIZE=1000000

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
  LOG_FILE="$RESULTS_DIR/${GYM_ENV_NAME}_${AGENT_NAME}_baseline_tf_lambda_${TF_LAMBDA}_timesteps_${TIMESTEPS}_buffer_${BUFFER_SIZE}_run_${RUN_NUMBER}.log"
  $PYTHON "$SB3_DIR/stable_baselines3/mujoco_10M.py" "$DEVICE" "$TF_LAMBDA" "$AGENT_NAME" "$TIMESTEPS" "$RUN_NUMBER" "$GYM_ENV_NAME" "$BUFFER_SIZE" \
    > "$LOG_FILE" 2>&1 &
  PIDS+=($!)
done

# Wait for all the runs; the job fails if any of them failed
STATUS=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || STATUS=1
done
exit $STATUS
