#!/bin/bash
#SBATCH --job-name=td3_sl_suft
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=20
#SBATCH --mem=140G

# Short-Long SUFT: TD3SL on one MuJoCo v5 environment, all the runs in parallel in one job, one CPU core per run
# The memory is split between a short term and a long term buffer, 1M transitions in total:
# long term buffer size = 1M - short term buffer size
# RAM: 7G per run - the 1M transitions take ~6GB for Humanoid (observations + next observations), less for the others
#
# Usage: sbatch td3_sl_suft.sh <Env> <short_buffer_size> <tf_lambda> <sampling_fraction> <batch_size> <train_freq> [run_number ...]
#        e.g. sbatch td3_sl_suft.sh Ant 4000 1 0.5 32 4        (runs 1-20)
#             sbatch td3_sl_suft.sh Ant 4000 1 0.5 32 4 1 2    (only these runs)
# <Env> is the MuJoCo name without the version suffix (Ant, HalfCheetah, Hopper, Humanoid, Walker2d)
# <sampling_fraction> is the fraction of each minibatch taken from the short term buffer
# <batch_size> is the minibatch size, <train_freq> is the number of environment steps between gradient updates
# As a job array every argument before the run numbers is a comma separated list and each array task runs one combination:
# the task id is decoded with <Env> varying fastest, then <train_freq>, <batch_size>, <sampling_fraction>, <tf_lambda>
# and <short_buffer_size> slowest, so the array is 0..(product of the list lengths)-1
#        e.g. sbatch --array=0-3 td3_sl_suft.sh Ant,Hopper 4000 1 0.5 32 4,1   (tasks 0-1 train_freq 4, tasks 2-3 train_freq 1)
# The #SBATCH lines above are sized for 20 runs; for another number of runs pass --cpus-per-task / --mem to sbatch
# To submit many environments / hyperparameters at once use submit_td3_sl_suft2.sh (one job array, sized to the number of runs)

ENV_NAME=$1
SHORT_BUFFER_SIZE=$2
TF_LAMBDA=$3
SAMPLING_FRACTION=$4
BATCH_SIZE=$5
TRAIN_FREQ=$6
if [ -z "$ENV_NAME" ] || [ -z "$SHORT_BUFFER_SIZE" ] || [ -z "$TF_LAMBDA" ] || [ -z "$SAMPLING_FRACTION" ] || [ -z "$BATCH_SIZE" ] || [ -z "$TRAIN_FREQ" ]; then
  echo "Usage: sbatch td3_sl_suft.sh <Env> <short_buffer_size> <tf_lambda> <sampling_fraction> <batch_size> <train_freq> [run_number ...]" >&2
  exit 1
fi
shift 6
RUNS=${*:-"$(seq -s " " 1 20)"}

# Job array: pick this task's combination from the comma separated lists
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  REST=$SLURM_ARRAY_TASK_ID
  # pick <variable>: replace the comma separated list in <variable> by its element for this task
  pick() {
    local -n VALUE=$1
    local LIST
    IFS=',' read -ra LIST <<< "$VALUE"
    VALUE=${LIST[$((REST % ${#LIST[@]}))]}
    REST=$((REST / ${#LIST[@]}))
  }
  pick ENV_NAME
  pick TRAIN_FREQ
  pick BATCH_SIZE
  pick SAMPLING_FRACTION
  pick TF_LAMBDA
  pick SHORT_BUFFER_SIZE
  if [ "$REST" -ne 0 ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID is past the last combination of the given lists" >&2
    exit 1
  fi
  echo "Array task $SLURM_ARRAY_TASK_ID: $ENV_NAME, short buffer $SHORT_BUFFER_SIZE, TF lambda $TF_LAMBDA, fraction $SAMPLING_FRACTION, batch $BATCH_SIZE, train_freq $TRAIN_FREQ"
elif [[ "$ENV_NAME$SHORT_BUFFER_SIZE$TF_LAMBDA$SAMPLING_FRACTION$BATCH_SIZE$TRAIN_FREQ" == *,* ]]; then
  echo "Comma separated lists are only allowed in a job array (sbatch --array=...)" >&2
  exit 1
fi

SB3_DIR=/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3
PYTHON=~/miniconda3/envs/stable-baselines3-mujoco/bin/python

DEVICE="cpu"
AGENT_NAME="TD3SL"
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
  LOG_FILE="$RESULTS_DIR/${GYM_ENV_NAME}_${AGENT_NAME}_suft_tf_lambda_${TF_LAMBDA}_timesteps_${TIMESTEPS}_short_buffer_${SHORT_BUFFER_SIZE}_long_buffer_${LONG_BUFFER_SIZE}_fraction_${SAMPLING_FRACTION}_batch_${BATCH_SIZE}_train_freq_${TRAIN_FREQ}_run_${RUN_NUMBER}.log"
  $PYTHON "$SB3_DIR/stable_baselines3/mujoco_10M.py" "$DEVICE" "$TF_LAMBDA" "$AGENT_NAME" "$TIMESTEPS" "$RUN_NUMBER" "$GYM_ENV_NAME" "$LONG_BUFFER_SIZE" \
    "$BATCH_SIZE" "$TRAIN_FREQ" "$SHORT_BUFFER_SIZE" "$SAMPLING_FRACTION" \
    > "$LOG_FILE" 2>&1 &
  PIDS+=($!)
done

# Wait for all the runs; the job fails if any of them failed
STATUS=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || STATUS=1
done
exit $STATUS
