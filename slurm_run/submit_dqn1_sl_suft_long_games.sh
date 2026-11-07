#!/bin/bash
# Submit dqn1_sl_suft.sh for the 11 long Atari games (the ones that took more than 4h) as a single job array
# over every short term buffer size: every (buffer size, game, run) is one run, and each array task runs
# RUNS_PER_TASK (default 1) runs of the same buffer size in parallel on one GPU
# The long term buffer size is 1M minus the short term buffer size (set in dqn1_sl_suft.sh)
# One email is sent when the whole job array ends (Slurm mails per array, not per task, without ARRAY_TASKS)
#
# By default: short term buffer sizes 16000 and 8000, TF lambda 1, sampling fraction 0.5, runs 1-4:
# 2 buffer sizes * 11 games * 4 runs = 88 runs in 88 array tasks of 1 run
# The tasks are ordered by buffer size, then run number, then game (all the games of run 1 first)
# The other 46 games are submitted by submit_dqn1_sl_suft_other_games.sh
#
# Usage:
#   ./submit_dqn1_sl_suft_long_games.sh                         the 11 long games, runs 1-4
#   ./submit_dqn1_sl_suft_long_games.sh Tennis Pitfall          only these games
#   RUNS="1 2" ./submit_dqn1_sl_suft_long_games.sh              only these runs
#   SHORT_BUFFER_SIZE="4000 2000" TF_LAMBDA=0.5 SAMPLING_FRACTION=0.25 ./submit_dqn1_sl_suft_long_games.sh   other hyperparameters
#   RUNS_PER_TASK=2 ./submit_dqn1_sl_suft_long_games.sh         another number of parallel runs per task (and GPU)
#   PARTITION=L4-12h ./submit_dqn1_sl_suft_long_games.sh        another partition (default shaham-L4)
#   MAIL_USER=someone@biu.ac.il ./submit_dqn1_sl_suft_long_games.sh   another address for the end of array email
#   DRY_RUN=1 ./submit_dqn1_sl_suft_long_games.sh               print the sbatch command and every task's runs without submitting
# The task log is slurm-<job id>_<task id>.out and its first line names the task's runs and buffer size

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -ra RUN_LIST <<< "${RUNS:-1 2 3 4}"
read -ra SHORT_BUFFER_LIST <<< "${SHORT_BUFFER_SIZE:-16000 8000}"
TF_LAMBDA=${TF_LAMBDA:-1}
SAMPLING_FRACTION=${SAMPLING_FRACTION:-0.5}
RUNS_PER_TASK=${RUNS_PER_TASK:-1}
PARTITION=${PARTITION:-shaham-L4}
MAIL_USER=${MAIL_USER:-fiskust@biu.ac.il}

# All the runs of a task share one GPU, each run gets one CPU core and 55G
# (the 1M transitions peak at ~53.7GB per run), so every array task is sized to RUNS_PER_TASK
MEM_PER_RUN_GB=55

# The long games: they took more than 4h
GAMES="Tennis MontezumaRevenge Pitfall Solaris Tutankham DemonAttack Defender VideoPinball BattleZone Phoenix Enduro"

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
# For every buffer size split the pairs into array tasks, RUNS_PER_TASK pairs per task as a comma separated list;
# TASK_BUFFERS holds each task's short buffer size (dqn1_sl_suft.sh takes them as a comma separated list)
csv() { local IFS=','; echo "$*"; }
RUN_GROUPS=()
TASK_BUFFERS=()
for BUFFER in "${SHORT_BUFFER_LIST[@]}"; do
  for ((i = 0; i < ${#RUN_PAIRS[@]}; i += RUNS_PER_TASK)); do
    RUN_GROUPS+=("$(csv "${RUN_PAIRS[@]:i:RUNS_PER_TASK}")")
    TASK_BUFFERS+=("$BUFFER")
  done
done
N_TASKS=${#RUN_GROUPS[@]}

CMD=(sbatch --array="0-$((N_TASKS - 1))" --partition="$PARTITION" --job-name="dqn1_sl_long"
     --cpus-per-task="$RUNS_PER_TASK" --mem="$((RUNS_PER_TASK * MEM_PER_RUN_GB))G"
     --mail-type=END --mail-user="$MAIL_USER"
     "$SCRIPT_DIR/dqn1_sl_suft.sh" "$(csv "${TASK_BUFFERS[@]}")" "$TF_LAMBDA" "$SAMPLING_FRACTION" "${RUN_GROUPS[@]}")
if [ -n "$DRY_RUN" ]; then
  echo "${CMD[@]}"
  for ((i = 0; i < N_TASKS; i++)); do
    echo "  task $i: ${RUN_GROUPS[$i]}, short buffer ${TASK_BUFFERS[$i]}"
  done
else
  "${CMD[@]}"
fi
