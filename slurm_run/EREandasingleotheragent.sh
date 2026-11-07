#!/bin/bash
#SBATCH --job-name=ere_and_rainbow_sl
#SBATCH --partition=shaham-L4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=55G

# One run (Atari game, run number) of one of these agents on one GPU and one CPU core:
#   DDQNSLNStepPERDueling  Double DQN + n-step 3 + PER (alpha 0.5) + dueling, short-long:
#                          short term buffer sampled UNIFORMLY, long term buffer prioritized
#   DQNERE                 ERE, single buffer (1M)
#   DQNSLERE               ERE on the long term buffer, uniform short term buffer
# Both ERE agents do the usual DQN updates (every 4 steps, replay ratio 0.25, SUFT term on the short term samples for
# DQNSLERE) plus the ERE updates after every episode (ere_continuous_updates true)
# RAM: 55G - the 1M transitions peak at ~53.7GB (observations + next observations) including the process overhead
# (every configuration holds 1M transitions in total: single buffer 1M, short-long: long = 1M - short)
#
# Usage: sbatch EREandasingleotheragent.sh <agent> <tf_lambda> <short_buffer_size> <sampling_fraction> <Game>:<run_number>
#        e.g. sbatch EREandasingleotheragent.sh DQNERE 0 0 0 Alien:1
#             sbatch EREandasingleotheragent.sh DQNSLERE 1 4000 0.5 Alien:1
# <short_buffer_size> and <sampling_fraction> are only used by the short-long agents (pass 0 0 for DQNERE)
# <Game> is the ALE name without prefix/suffix (Alien, KungFuMaster, ...)
# As a job array pass one <Game>:<run_number> per array task: task i runs the i-th pair (0-based)
#        e.g. sbatch --array=0-1 EREandasingleotheragent.sh DQNERE 0 0 0 Alien:1 Pong:1
# To submit all the games at once use submit_EREandasingleotheragent.sh

AGENT_NAME=$1
TF_LAMBDA=$2
SHORT_BUFFER_SIZE=$3
SAMPLING_FRACTION=$4
if [ -z "$AGENT_NAME" ] || [ -z "$TF_LAMBDA" ] || [ -z "$SHORT_BUFFER_SIZE" ] || [ -z "$SAMPLING_FRACTION" ] || [ -z "$5" ]; then
  echo "Usage: sbatch [--array=...] EREandasingleotheragent.sh <agent> <tf_lambda> <short_buffer_size> <sampling_fraction> <Game>:<run_number> [...]" >&2
  exit 1
fi
shift 4
PAIRS=("$@")

N_STEPS=3
# The fixes: uniform short term buffer for DDQNSLNStepPERDueling, usual DQN updates + ERE updates for the ERE agents
PRIORITIZE_SHORT_TERM=false
ERE_CONTINUOUS=true
# SHORT_LONG: 1 for the short-long agents; AGENT_ARGS: the agent's optional atari100k.py arguments; LOG_SUFFIX: in the log name
case $AGENT_NAME in
  DDQNSLNStepPERDueling) SHORT_LONG=1; AGENT_ARGS=("$N_STEPS" "$PRIORITIZE_SHORT_TERM"); LOG_SUFFIX="_n_steps_${N_STEPS}_per_short_${PRIORITIZE_SHORT_TERM}" ;;
  DQNERE)                SHORT_LONG=0; AGENT_ARGS=("$ERE_CONTINUOUS");                   LOG_SUFFIX="_ere_continuous_${ERE_CONTINUOUS}" ;;
  DQNSLERE)              SHORT_LONG=1; AGENT_ARGS=("$ERE_CONTINUOUS");                   LOG_SUFFIX="_ere_continuous_${ERE_CONTINUOUS}" ;;
  *) echo "Unknown agent '$AGENT_NAME' (DDQNSLNStepPERDueling, DQNERE, DQNSLERE)" >&2; exit 1 ;;
esac
if [ "$SHORT_LONG" = 1 ] && ! [[ $SHORT_BUFFER_SIZE =~ ^[1-9][0-9]*$ ]]; then
  echo "$AGENT_NAME needs a <short_buffer_size> > 0, got '$SHORT_BUFFER_SIZE'" >&2
  exit 1
fi
# tf_lambda 0: baseline, otherwise SUFT (the kind in the log name)
case $TF_LAMBDA in
  0|0.0|0.00) KIND=baseline ;;
  *) KIND=suft ;;
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

SB3_DIR=${SB3_DIR:-/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3}
PYTHON=${PYTHON:-~/miniconda3/envs/stable-baselines3/bin/python}

DEVICE="cuda:0"
RESULTS_DIR=$SB3_DIR/results/$AGENT_NAME
TIMESTEPS=1000000
GYM_ENV_NAME="ALE/${GAME}-v5"
TOTAL_BUFFER_SIZE=1000000

if [ "$SHORT_LONG" = 1 ]; then
  # The memory is split between the short term and the long term buffer: long = 1M - short
  LONG_BUFFER_SIZE=$((TOTAL_BUFFER_SIZE - SHORT_BUFFER_SIZE))
  BUFFER_ARGS=("$LONG_BUFFER_SIZE" "$SHORT_BUFFER_SIZE" "$SAMPLING_FRACTION")
  BUFFER_NAME="short_buffer_${SHORT_BUFFER_SIZE}_long_buffer_${LONG_BUFFER_SIZE}_fraction_${SAMPLING_FRACTION}"
  echo "${SLURM_ARRAY_TASK_ID:+Array task $SLURM_ARRAY_TASK_ID: }$GAME run $RUN_NUMBER, $AGENT_NAME, TF lambda $TF_LAMBDA, short buffer $SHORT_BUFFER_SIZE, long buffer $LONG_BUFFER_SIZE, fraction $SAMPLING_FRACTION, args ${AGENT_ARGS[*]}"
else
  BUFFER_ARGS=("$TOTAL_BUFFER_SIZE")
  BUFFER_NAME="buffer_${TOTAL_BUFFER_SIZE}"
  echo "${SLURM_ARRAY_TASK_ID:+Array task $SLURM_ARRAY_TASK_ID: }$GAME run $RUN_NUMBER, $AGENT_NAME, TF lambda $TF_LAMBDA, buffer $TOTAL_BUFFER_SIZE, args ${AGENT_ARGS[*]}"
fi

# One core for the run: stop PyTorch / NumPy from spawning extra threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Import stable_baselines3 from this repo (atari100k.py appends the old repo to sys.path)
export PYTHONPATH="$SB3_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RESULTS_DIR"
cd "$SB3_DIR/stable_baselines3" || exit 1

LOG_FILE="$RESULTS_DIR/${GAME}-v5_${AGENT_NAME}_${KIND}_tf_lambda_${TF_LAMBDA}_timesteps_${TIMESTEPS}_${BUFFER_NAME}${LOG_SUFFIX}_run_${RUN_NUMBER}.log"
$PYTHON "$SB3_DIR/stable_baselines3/atari100k.py" "$DEVICE" "$TF_LAMBDA" "$AGENT_NAME" "$TIMESTEPS" "$RUN_NUMBER" "$GYM_ENV_NAME" \
  "${BUFFER_ARGS[@]}" "${AGENT_ARGS[@]}" \
  > "$LOG_FILE" 2>&1
