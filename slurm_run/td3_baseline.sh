#!/bin/bash
#SBATCH --job-name=td3_base
#SBATCH --partition=shaham-L4
#SBATCH --cpus-per-task=5
#SBATCH --mem=35G
#SBATCH --mail-user=fiskust@biu.ac.il
#SBATCH --mail-type=END

# Baseline: TD3 on one MuJoCo v5 environment (1M buffer), the runs in parallel in one job, one CPU core per run (at most one run per allocated core at a time)
# RAM: 7G per run - the 1M replay buffer takes ~6GB for Humanoid (observations + next observations), less for the others
#
# Usage: sbatch td3_baseline.sh <Env> <batch_size> <train_freq> [run_number ...]
#        e.g. sbatch td3_baseline.sh Ant 32 4        (runs 1-20)
#             sbatch td3_baseline.sh Ant 32 4 1 2    (only these runs)
# <Env> is the MuJoCo name without the version suffix (Ant, HalfCheetah, Hopper, Humanoid, Walker2d)
# As a job array <Env> and <train_freq> are comma separated lists and each array task runs one (train_freq, Env) pair:
# task i runs Env i % N_ENVS with train_freq i / N_ENVS, so the array is 0..N_ENVS*N_TRAIN_FREQS-1
#        e.g. sbatch --array=0-1 td3_baseline.sh Ant,Hopper 32 4        (Ant / Hopper, train_freq 4)
#             sbatch --array=0-3 td3_baseline.sh Ant,Hopper 32 4,1      (tasks 0-1 train_freq 4, tasks 2-3 train_freq 1)
# <batch_size> is the minibatch size, <train_freq> is the number of environment steps between gradient updates
# With RUNS_PER_TASK=<n> exported (sbatch --export=ALL,RUNS_PER_TASK=<n>) the runs are split into chunks of n runs:
# array task i runs chunk i % N_CHUNKS of the combination picked by i / N_CHUNKS
# The #SBATCH lines above are sized for 5 runs at once (7G each); for another number at once pass --cpus-per-task / --mem to sbatch:
# the job then runs that many runs at a time and starts the next run as soon as one finishes
# To submit many environments at once use submit_td3_baseline.sh (one job array per configuration, one task per chunk of runs)

ENV_NAME=$1
BATCH_SIZE=$2
TRAIN_FREQ=$3
if [ -z "$ENV_NAME" ] || [ -z "$BATCH_SIZE" ] || [ -z "$TRAIN_FREQ" ]; then
  echo "Usage: sbatch td3_baseline.sh <Env> <batch_size> <train_freq> [run_number ...]" >&2
  exit 1
fi
shift 3
RUNS=${*:-"$(seq -s " " 1 20)"}

# Job array with RUNS_PER_TASK set (the submit scripts set it): split the runs into chunks of RUNS_PER_TASK runs
# and give each array task one chunk; the chunk varies fastest in the task id
TASK_ID=$SLURM_ARRAY_TASK_ID
if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ -n "$RUNS_PER_TASK" ]; then
  read -ra RUN_LIST <<< "$RUNS"
  N_CHUNKS=$(( (${#RUN_LIST[@]} + RUNS_PER_TASK - 1) / RUNS_PER_TASK ))
  RUNS="${RUN_LIST[*]:$((TASK_ID % N_CHUNKS * RUNS_PER_TASK)):RUNS_PER_TASK}"
  TASK_ID=$((TASK_ID / N_CHUNKS))
fi

# Job array: pick this task's (train_freq, environment) pair from the comma separated lists
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  IFS=',' read -ra ENV_LIST <<< "$ENV_NAME"
  IFS=',' read -ra TRAIN_FREQ_LIST <<< "$TRAIN_FREQ"
  N_ENVS=${#ENV_LIST[@]}
  ENV_NAME=${ENV_LIST[$((TASK_ID % N_ENVS))]}
  TRAIN_FREQ=${TRAIN_FREQ_LIST[$((TASK_ID / N_ENVS))]}
  if [ -z "$ENV_NAME" ] || [ -z "$TRAIN_FREQ" ]; then
    echo "No (train_freq, environment) for array task $SLURM_ARRAY_TASK_ID in train_freqs (${TRAIN_FREQ_LIST[*]}) x environments (${ENV_LIST[*]})" >&2
    exit 1
  fi
  echo "Array task $SLURM_ARRAY_TASK_ID: $ENV_NAME, train_freq $TRAIN_FREQ, runs $RUNS"
fi

SB3_DIR=/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3
PYTHON=~/miniconda3/envs/stable-baselines3-mujoco/bin/python

DEVICE="cpu"
TF_LAMBDA=0
AGENT_NAME="TD3"
RESULTS_DIR=$SB3_DIR/results/$AGENT_NAME
TIMESTEPS=10000000
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

# Start the runs in the background, each with its own log file, at most MAX_PARALLEL at a time
# (one per allocated CPU core unless MAX_PARALLEL is set)
MAX_PARALLEL=${MAX_PARALLEL:-${SLURM_CPUS_PER_TASK:-5}}
PIDS=()
for RUN_NUMBER in $RUNS; do
  # Wait for a free slot before starting the next run
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n; done
  LOG_FILE="$RESULTS_DIR/${GYM_ENV_NAME}_${AGENT_NAME}_baseline_tf_lambda_${TF_LAMBDA}_timesteps_${TIMESTEPS}_buffer_${BUFFER_SIZE}_batch_${BATCH_SIZE}_train_freq_${TRAIN_FREQ}_run_${RUN_NUMBER}.log"
  $PYTHON "$SB3_DIR/stable_baselines3/mujoco_10M.py" "$DEVICE" "$TF_LAMBDA" "$AGENT_NAME" "$TIMESTEPS" "$RUN_NUMBER" "$GYM_ENV_NAME" "$BUFFER_SIZE" \
    "$BATCH_SIZE" "$TRAIN_FREQ" \
    > "$LOG_FILE" 2>&1 &
  PIDS+=($!)
done

# Wait for all the runs (wait PID still returns the status of a run already reaped by wait -n); the job fails if any of them failed
STATUS=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || STATUS=1
done
exit $STATUS
