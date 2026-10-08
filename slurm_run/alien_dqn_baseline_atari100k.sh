#!/bin/bash
#SBATCH --job-name=alien
#SBATCH --partition=shaham-L4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --mail-user=fiskust@biu.ac.il
#SBATCH --mail-type=END

# Baseline: 8 concurrent DQN runs on Alien (Atari 100K), one CPU core each, sharing the single GPU

SB3_DIR=/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3
PYTHON=~/miniconda3/envs/stable-baselines3/bin/python

DEVICE="cuda:0"
TF_LAMBDA=0
AGENT_NAME="DQN"
RESULTS_DIR=$SB3_DIR/results/$AGENT_NAME
TIMESTEPS=100000
GYM_ENV_NAME="ALE/Alien-v5"
BUFFER_SIZE=100000

# One core per run: stop PyTorch / NumPy from spawning extra threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Import stable_baselines3 from this repo (atari100k.py appends the old repo to sys.path)
export PYTHONPATH="$SB3_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RESULTS_DIR"
cd "$SB3_DIR/stable_baselines3" || exit 1

pids=()
for RUN_NUMBER in 1 2 3 4 5 6 7 8; do
  LOG_FILE="$RESULTS_DIR/Alien-v5_${AGENT_NAME}_baseline_tf_lambda_${TF_LAMBDA}_timesteps_${TIMESTEPS}_buffer_${BUFFER_SIZE}_run_${RUN_NUMBER}.log"
  $PYTHON "$SB3_DIR/stable_baselines3/atari100k.py" "$DEVICE" "$TF_LAMBDA" "$AGENT_NAME" "$TIMESTEPS" "$RUN_NUMBER" "$GYM_ENV_NAME" "$BUFFER_SIZE" \
    > "$LOG_FILE" 2>&1 &
  pids+=($!)
done

# Wait for all the runs, fail the job if one of them failed
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit $status
