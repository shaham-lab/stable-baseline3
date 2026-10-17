#!/bin/bash
#SBATCH --job-name=dqncer_base
#SBATCH --partition=shaham-L4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=165G

# Baseline: up to 3 DQNCER runs (DQN with combined experience replay, TF lambda 0)
# (Atari game, run number, 1M buffer) in parallel in one job sharing one GPU, one CPU core per run
# RAM: 55G per run - the 1M transitions peak at ~53.7GB (observations + next observations) including the process overhead
#
# Usage: sbatch dqncer_baseline.sh <runs>     e.g. sbatch dqncer_baseline.sh Alien:1,Pong:1,Qbert:1
# <runs> is a comma separated list of <Game>:<run_number> pairs that run in parallel
# <Game> is the ALE name without prefix/suffix (Alien, KungFuMaster, ...)
# CER_N_LATEST (environment variable) is the number of latest transitions in every minibatch: train_freq (default,
# all the transitions collected since the previous update) or a number (1 = original CER),
#        e.g. CER_N_LATEST=1 sbatch ...
# As a job array pass one <runs> argument per array task: task i runs the i-th <runs> (0-based)
#        e.g. sbatch --array=0-1 dqncer_baseline.sh Alien:1,Pong:1,Qbert:1 Alien:2,Pong:2,Qbert:2
# The #SBATCH lines above are sized for 3 runs; for another number of runs pass --cpus-per-task / --mem to sbatch
# To submit many games / runs at once use submit_dqncer_baseline_long_games.sh / submit_dqncer_baseline_other_games.sh

if [ $# -eq 0 ]; then
  echo "Usage: sbatch dqncer_baseline.sh <runs> [<runs> ...]" >&2
  exit 1
fi
RUN_GROUPS=("$@")

# Job array: pick this task's runs
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  RUN_GROUP=${RUN_GROUPS[$SLURM_ARRAY_TASK_ID]}
  if [ -z "$RUN_GROUP" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID is past the last of the ${#RUN_GROUPS[@]} given <runs>" >&2
    exit 1
  fi
  echo "Array task $SLURM_ARRAY_TASK_ID: $RUN_GROUP, CER latest ${CER_N_LATEST:-train_freq}"
elif [ ${#RUN_GROUPS[@]} -gt 1 ]; then
  echo "Several <runs> arguments are only allowed in a job array (sbatch --array=...)" >&2
  exit 1
else
  RUN_GROUP=${RUN_GROUPS[0]}
fi

SB3_DIR=/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3
PYTHON=~/miniconda3/envs/stable-baselines3/bin/python

DEVICE="cuda:0"
TF_LAMBDA=0
AGENT_NAME="DQNCER"
# Combined experience replay: the latest CER_N_LATEST transitions in every minibatch
CER_N_LATEST=${CER_N_LATEST:-train_freq}
# atari100k.py uses train_freq when <cer_n_latest> is not passed
CER_ARGS=()
[ "$CER_N_LATEST" != "train_freq" ] && CER_ARGS=("$CER_N_LATEST")
RESULTS_DIR=$SB3_DIR/results/$AGENT_NAME
TIMESTEPS=1000000
BUFFER_SIZE=1000000

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
  if [ -z "$GAME" ] || [ -z "$RUN_NUMBER" ] || [ "$GAME" = "$RUN_PAIR" ]; then
    echo "Bad run '$RUN_PAIR': expected <Game>:<run_number>" >&2
    exit 1
  fi
  GYM_ENV_NAME="ALE/${GAME}-v5"
  LOG_FILE="$RESULTS_DIR/${GAME}-v5_${AGENT_NAME}_baseline_tf_lambda_${TF_LAMBDA}_timesteps_${TIMESTEPS}_buffer_${BUFFER_SIZE}_cer_latest_${CER_N_LATEST}_run_${RUN_NUMBER}.log"
  $PYTHON "$SB3_DIR/stable_baselines3/atari100k.py" "$DEVICE" "$TF_LAMBDA" "$AGENT_NAME" "$TIMESTEPS" "$RUN_NUMBER" "$GYM_ENV_NAME" "$BUFFER_SIZE" \
    "${CER_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &
  PIDS+=($!)
done

# Wait for all the runs; the job fails if any of them failed
STATUS=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || STATUS=1
done
exit $STATUS
