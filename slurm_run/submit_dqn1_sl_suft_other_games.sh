#!/bin/bash
# Submit dqn1_sl_suft.sh for the 46 Atari games that are not long as a single job array:
# every (game, run) pair is one run, and each array task runs RUNS_PER_TASK (default 3) of them in parallel on one GPU
# The long term buffer size is 1M minus the short term buffer size (set in dqn1_sl_suft.sh)
#
# SHORT_BUFFER_SIZE may hold several sizes: every (game, run) pair is run once per size, in the same job array,
# all the pairs of the first size first, then all the pairs of the next size, ...
#
# By default: short term buffer sizes 16000 then 8000, TF lambda 1, sampling fraction 0.5, runs 1-4, one run per task:
# 2 sizes * 46 games * 4 runs = 368 array tasks (tasks 0-183 short buffer 16000, tasks 184-367 short buffer 8000)
# Within a size the runs are ordered by run number then game (all the games of run 1 first)
# The 11 long games (more than 4h) are submitted by submit_dqn1_sl_suft_long_games.sh
# One email is sent to MAIL_USER when the whole job array has ended (Slurm --mail-type=END applies to the array as a whole)
#
# Usage:
#   ./submit_dqn1_sl_suft_other_games.sh                         the 46 other games, runs 1-4
#   ./submit_dqn1_sl_suft_other_games.sh Alien Pong              only these games
#   RUNS="1 2" ./submit_dqn1_sl_suft_other_games.sh              only these runs
#   SHORT_BUFFER_SIZE=2000 TF_LAMBDA=0.5 SAMPLING_FRACTION=0.25 ./submit_dqn1_sl_suft_other_games.sh   other hyperparameters
#   SHORT_BUFFER_SIZE="4000 2000" ./submit_dqn1_sl_suft_other_games.sh   other short term buffer sizes (space or comma separated)
#   RUNS_PER_TASK=2 ./submit_dqn1_sl_suft_other_games.sh         another number of parallel runs per task (and GPU)
#   PARTITION=L4-12h ./submit_dqn1_sl_suft_other_games.sh        another partition (default generic)
#   MAIL_USER=me@example.com ./submit_dqn1_sl_suft_other_games.sh   another email address (default fiskust@biu.ac.il, MAIL_USER= for no email)
#   DRY_RUN=1 ./submit_dqn1_sl_suft_other_games.sh               print the sbatch command and every task's runs without submitting
# The task log is slurm-<job id>_<task id>.out and its first line names the task's runs

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -ra RUN_LIST <<< "${RUNS:-1 2 3 4}"
read -ra SHORT_BUFFER_SIZE_LIST <<< "${SHORT_BUFFER_SIZE:-16000 8000}"
SHORT_BUFFER_SIZE_LIST=(${SHORT_BUFFER_SIZE_LIST[@]//,/ })
TF_LAMBDA=${TF_LAMBDA:-1}
SAMPLING_FRACTION=${SAMPLING_FRACTION:-0.5}
RUNS_PER_TASK=${RUNS_PER_TASK:-1}
PARTITION=${PARTITION:-generic}
MAIL_USER=${MAIL_USER-fiskust@biu.ac.il}

# All the runs of a task share one GPU, each run gets one CPU core and 55G
# (the 1M transitions peak at ~53.7GB per run), so every array task is sized to RUNS_PER_TASK
MEM_PER_RUN_GB=55

# The 57 games of the Atari benchmark without the 11 long ones (submit_dqn1_sl_suft_long_games.sh)
GAMES="Alien Amidar Assault Asterix Asteroids Atlantis BankHeist BeamRider Berzerk Bowling
Boxing Breakout Centipede ChopperCommand CrazyClimber DoubleDunk FishingDerby Freeway Frostbite
Gopher Gravitar Hero IceHockey Jamesbond Kangaroo Krull KungFuMaster MsPacman NameThisGame
Pong PrivateEye Qbert Riverraid RoadRunner Robotank Seaquest Skiing SpaceInvaders StarGunner
Surround TimePilot UpNDown Venture WizardOfWor YarsRevenge Zaxxon"

if [ $# -gt 0 ]; then
  read -ra GAME_LIST <<< "$*"
else
  read -ra GAME_LIST <<< "$(echo $GAMES)"
fi

# Every (game, run) pair as <Game>:<run_number>, all the games of the first run first
RUN_PAIRS=()
for RUN_NUMBER in "${RUN_LIST[@]}"; do
  for GAME in "${GAME_LIST[@]}"; do
    RUN_PAIRS+=("$GAME:$RUN_NUMBER")
  done
done
# Split them into the array tasks, RUNS_PER_TASK pairs per task as a comma separated list,
# once per short term buffer size; TASK_SIZES holds every task's short term buffer size
csv() { local IFS=','; echo "$*"; }
RUN_GROUPS=()
TASK_SIZES=()
for SIZE in "${SHORT_BUFFER_SIZE_LIST[@]}"; do
  for ((i = 0; i < ${#RUN_PAIRS[@]}; i += RUNS_PER_TASK)); do
    RUN_GROUPS+=("$(csv "${RUN_PAIRS[@]:i:RUNS_PER_TASK}")")
    TASK_SIZES+=("$SIZE")
  done
done
N_TASKS=${#RUN_GROUPS[@]}
# A single size is passed as is, several as one size per task (dqn1_sl_suft.sh picks the task's one)
if [ ${#SHORT_BUFFER_SIZE_LIST[@]} -eq 1 ]; then
  SIZE_ARG=${SHORT_BUFFER_SIZE_LIST[0]}
else
  SIZE_ARG=$(csv "${TASK_SIZES[@]}")
fi

MAIL_ARGS=()
if [ -n "$MAIL_USER" ]; then
  MAIL_ARGS=(--mail-type=END --mail-user="$MAIL_USER")
fi

CMD=(sbatch --array="0-$((N_TASKS - 1))" --partition="$PARTITION" --job-name="dqn1_sl_other" "${MAIL_ARGS[@]}"
     --cpus-per-task="$RUNS_PER_TASK" --mem="$((RUNS_PER_TASK * MEM_PER_RUN_GB))G"
     "$SCRIPT_DIR/dqn1_sl_suft.sh" "$SIZE_ARG" "$TF_LAMBDA" "$SAMPLING_FRACTION" "${RUN_GROUPS[@]}")
if [ -n "$DRY_RUN" ]; then
  echo "${CMD[@]}"
  for ((i = 0; i < N_TASKS; i++)); do
    echo "  task $i: ${RUN_GROUPS[$i]}, short buffer ${TASK_SIZES[$i]}"
  done
else
  "${CMD[@]}"
fi
