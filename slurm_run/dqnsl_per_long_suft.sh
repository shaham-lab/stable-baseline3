#!/bin/bash
#SBATCH --job-name=dqnsl_per_long_suft
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=165G

# Short-Long SUFT with PER: up to 3 DQNSLPERLong runs (Atari game, run number) in parallel in one job sharing one GPU, one CPU core per run
# short term buffer sampled uniformly, long term buffer prioritized (PER, Dopamine version)
# The memory is split between a short term and a long term buffer, 1M transitions in total:
# long term buffer size = 1M - short term buffer size
# RAM: 55G per run - the 1M transitions peak at ~53.7GB (observations + next observations) including the process overhead
#
# Usage: sbatch dqnsl_per_long_suft.sh <short_buffer_size> <tf_lambda> <sampling_fraction> <runs>
#        e.g. sbatch dqnsl_per_long_suft.sh 4000 1 0.5 Alien:1,Pong:1,Qbert:1
# <runs> is a comma separated list of <Game>:<run_number> pairs that run in parallel
# <Game> is the ALE name without prefix/suffix (Alien, KungFuMaster, ...)
# <sampling_fraction> is the fraction of each minibatch taken from the short term buffer
# As a job array pass one <runs> argument per array task: task i runs the i-th <runs> (0-based)
#        e.g. sbatch --array=0-1 dqnsl_per_long_suft.sh 4000 1 0.5 Alien:1,Pong:1,Qbert:1 Alien:2,Pong:2,Qbert:2
# The #SBATCH lines above are sized for 3 runs; for another number of runs pass --cpus-per-task / --mem to sbatch
# To submit many games at once use submit_dqnsl_per_long_suft_long_games.sh / submit_dqnsl_per_long_suft_other_games.sh

SHORT_BUFFER_SIZE=$1
TF_LAMBDA=$2
SAMPLING_FRACTION=$3
if [ -z "$SHORT_BUFFER_SIZE" ] || [ -z "$TF_LAMBDA" ] || [ -z "$SAMPLING_FRACTION" ] || [ -z "$4" ]; then
  echo "Usage: sbatch dqnsl_per_long_suft.sh <short_buffer_size> <tf_lambda> <sampling_fraction> <runs> [<runs> ...]" >&2
  exit 1
fi
shift 3
RUN_GROUPS=("$@")

# Job array: pick this task's runs
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  RUN_GROUP=${RUN_GROUPS[$SLURM_ARRAY_TASK_ID]}
  if [ -z "$RUN_GROUP" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID is past the last of the ${#RUN_GROUPS[@]} given <runs>" >&2
    exit 1
  fi
  echo "Array task $SLURM_ARRAY_TASK_ID: $RUN_GROUP, short buffer $SHORT_BUFFER_SIZE, TF lambda $TF_LAMBDA, fraction $SAMPLING_FRACTION"
elif [ ${#RUN_GROUPS[@]} -gt 1 ]; then
  echo "Several <runs> arguments are only allowed in a job array (sbatch --array=...)" >&2
  exit 1
else
  RUN_GROUP=${RUN_GROUPS[0]}
fi

SB3_DIR=/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3
PYTHON=~/miniconda3/envs/stable-baselines3/bin/python

DEVICE="cuda:0"
AGENT_NAME="DQNSLPERLong"
RESULTS_DIR=$SB3_DIR/results/$AGENT_NAME
TIMESTEPS=1000000
TOTAL_BUFFER_SIZE=1000000
LONG_BUFFER_SIZE=$((TOTAL_BUFFER_SIZE - SHORT_BUFFER_SIZE))

# One core per run: stop PyTorch / NumPy from spawning extra threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Import stable_baselines3 from this repo (atari100k.py appends the old repo to sys.path)
export PYTHONPATH="$SB3_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RESULTS_DIR"
cd "$SB3_DIR/stable_baselines3" || exit 1

# Start every run in the background on the shared GPU, each with its own log file
IFS=',' read -ra RUN_PAIRS <<< "$RUN_GROUP"
PIDS=()
for RUN_PAIR in "${RUN_PAIRS[@]}"; do
  GAME=${RUN_PAIR%%:*}
  RUN_NUMBER=${RUN_PAIR##*:}
  GYM_ENV_NAME="ALE/${GAME}-v5"
  LOG_FILE="$RESULTS_DIR/${GAME}-v5_${AGENT_NAME}_suft_tf_lambda_${TF_LAMBDA}_timesteps_${TIMESTEPS}_short_buffer_${SHORT_BUFFER_SIZE}_long_buffer_${LONG_BUFFER_SIZE}_fraction_${SAMPLING_FRACTION}_run_${RUN_NUMBER}.log"
  $PYTHON "$SB3_DIR/stable_baselines3/atari100k.py" "$DEVICE" "$TF_LAMBDA" "$AGENT_NAME" "$TIMESTEPS" "$RUN_NUMBER" "$GYM_ENV_NAME" "$LONG_BUFFER_SIZE" \
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
