#!/bin/bash
#SBATCH --job-name=dqn_agents_suft
#SBATCH --partition=shaham-L4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=55G

# SUFT with a single 1M replay buffer: one run (Atari game, run number) of DQN / DDQN / DQNNStep / DQNPER on one GPU and one CPU core
# RAM: 55G - the 1M transitions peak at ~53.7GB (observations + next observations) including the process overhead
# DQNNStep uses the Rainbow n-step length 3
#
# Usage: sbatch dqn_agents_suft.sh <agent> <tf_lambda> <Game>:<run_number>
#        e.g. sbatch dqn_agents_suft.sh DDQN 1 Alien:1
# <agent> is DQN, DDQN, DQNNStep or DQNPER
# <Game> is the ALE name without prefix/suffix (Alien, KungFuMaster, ...)
# As a job array pass one <Game>:<run_number> per array task: task i runs the i-th pair (0-based)
#        e.g. sbatch --array=0-1 dqn_agents_suft.sh DQN 1 Alien:1 Pong:1
# To submit all the games at once use submit_dqn_agents_suft_shaham.sh

AGENT_NAME=$1
TF_LAMBDA=$2
if [ -z "$AGENT_NAME" ] || [ -z "$TF_LAMBDA" ] || [ -z "$3" ]; then
  echo "Usage: sbatch [--array=...] dqn_agents_suft.sh <agent> <tf_lambda> <Game>:<run_number> [...]" >&2
  exit 1
fi
shift 2
PAIRS=("$@")

N_STEPS=3
EXTRA_ARGS=()
LOG_SUFFIX=""
case $AGENT_NAME in
  DQN|DDQN|DQNPER) ;;
  DQNNStep) EXTRA_ARGS=("$N_STEPS"); LOG_SUFFIX="_n_steps_${N_STEPS}" ;;
  *) echo "Unknown agent '$AGENT_NAME' (DQN, DDQN, DQNNStep, DQNPER)" >&2; exit 1 ;;
esac

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
  exit 1
fi
echo "${SLURM_ARRAY_TASK_ID:+Array task $SLURM_ARRAY_TASK_ID: }$GAME run $RUN_NUMBER, $AGENT_NAME, TF lambda $TF_LAMBDA${LOG_SUFFIX:+, n_steps $N_STEPS}"

SB3_DIR=/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3
PYTHON=~/miniconda3/envs/stable-baselines3/bin/python

DEVICE="cuda:0"
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

LOG_FILE="$RESULTS_DIR/${GAME}-v5_${AGENT_NAME}_suft_tf_lambda_${TF_LAMBDA}_timesteps_${TIMESTEPS}_buffer_${BUFFER_SIZE}${LOG_SUFFIX}_run_${RUN_NUMBER}.log"
$PYTHON "$SB3_DIR/stable_baselines3/atari100k.py" "$DEVICE" "$TF_LAMBDA" "$AGENT_NAME" "$TIMESTEPS" "$RUN_NUMBER" "$GYM_ENV_NAME" "$BUFFER_SIZE" \
  "${EXTRA_ARGS[@]}" \
  > "$LOG_FILE" 2>&1
