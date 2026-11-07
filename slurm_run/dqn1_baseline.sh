#!/bin/bash
#SBATCH --job-name=dqn1_base
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

# Baseline: a single DQN1 run on one Atari game (1M buffer) on one CPU core
# RAM: the 1M replay buffer takes ~57GB (observations + next observations), plus the process overhead
#
# Usage: sbatch dqn1_baseline.sh <Game> <run_number>     e.g. sbatch dqn1_baseline.sh Alien 1
#    or: sbatch dqn1_baseline.sh <Game>:<run_number>     e.g. sbatch dqn1_baseline.sh Alien:1
# <Game> is the ALE name without prefix/suffix (Alien, KungFuMaster, ...)
# As a job array pass one <Game>:<run_number> per array task: task i runs the i-th pair (0-based)
#        e.g. sbatch --array=0-1 dqn1_baseline.sh Alien:1 Pong:1
# To submit many games / runs at once use submit_dqn1_baseline_long_games.sh / submit_dqn1_baseline_other_games.sh

usage() {
  echo "Usage: sbatch dqn1_baseline.sh <Game> <run_number>  |  sbatch [--array=...] dqn1_baseline.sh <Game>:<run_number> [...]" >&2
  exit 1
}
if [ $# -eq 2 ] && [[ $1 != *:* ]] && [[ $2 =~ ^[0-9]+$ ]]; then
  # Old form: <Game> <run_number>
  PAIRS=("$1:$2")
else
  PAIRS=("$@")
fi
[ ${#PAIRS[@]} -gt 0 ] || usage

# Job array: pick this task's (game, run) pair
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  PAIR=${PAIRS[$SLURM_ARRAY_TASK_ID]}
  if [ -z "$PAIR" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID is past the last of the ${#PAIRS[@]} given <Game>:<run_number>" >&2
    exit 1
  fi
elif [ ${#PAIRS[@]} -gt 1 ]; then
  echo "Several <Game>:<run_number> arguments are only allowed in a job array (sbatch --array=...)" >&2
  exit 1
else
  PAIR=${PAIRS[0]}
fi
IFS=':' read -r GAME RUN_NUMBER <<< "$PAIR"
if [ -z "$GAME" ] || ! [[ $RUN_NUMBER =~ ^[0-9]+$ ]]; then
  echo "Bad <Game>:<run_number> '$PAIR'" >&2
  usage
fi
echo "${SLURM_ARRAY_TASK_ID:+Array task $SLURM_ARRAY_TASK_ID: }$GAME run $RUN_NUMBER, DQN1 baseline"

SB3_DIR=/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3
PYTHON=~/miniconda3/envs/stable-baselines3/bin/python

DEVICE="cuda:0"
TF_LAMBDA=1
AGENT_NAME="DQN1"
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
