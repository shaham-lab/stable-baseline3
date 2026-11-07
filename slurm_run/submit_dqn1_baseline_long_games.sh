#!/bin/bash
# Submit dqn1_baseline.sh for the 10 long Atari games (the ones that took more than 4h) as a single job array
# The other 47 games are submitted as another job array by submit_dqn1_baseline_other_games.sh
#
# By default: runs 1-4, partition L4-12h, 10 games * 4 runs = 40 array tasks, one (game, run) pair per task
# Each task gets the #SBATCH resources of dqn1_baseline.sh (1 GPU, 1 CPU core, 64G)
# The runs are ordered by run number then game (all the games of run 1 first)
# One email is sent to MAIL_USER when the whole job array has ended (Slurm --mail-type=END applies to the array as a whole)
#
# Usage:
#   ./submit_dqn1_baseline_long_games.sh                      the 10 long games, runs 1-4
#   ./submit_dqn1_baseline_long_games.sh Tennis Pitfall     only these games
#   RUNS="1 2" ./submit_dqn1_baseline_long_games.sh           only these runs
#   PARTITION=L4-12h ./submit_dqn1_baseline_long_games.sh     another partition (default L4-12h)
#   MAIL_USER=me@example.com ./submit_dqn1_baseline_long_games.sh   another email address (default fiskust@biu.ac.il, MAIL_USER= for no email)
#   DRY_RUN=1 ./submit_dqn1_baseline_long_games.sh            print the sbatch command and every task's (game, run) pair without submitting
# The task log is slurm-<job id>_<task id>.out and its first line names the task's game and run

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -ra RUN_LIST <<< "${RUNS:-1 2 3 4}"
PARTITION=${PARTITION:-L4-12h}
MAIL_USER=${MAIL_USER-fiskust@biu.ac.il}

GAMES="Tennis MontezumaRevenge Pitfall Solaris Tutankham DemonAttack Defender VideoPinball BattleZone Phoenix"

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
N_TASKS=${#RUN_PAIRS[@]}

MAIL_ARGS=()
if [ -n "$MAIL_USER" ]; then
  MAIL_ARGS=(--mail-type=END --mail-user="$MAIL_USER")
fi

CMD=(sbatch --array="0-$((N_TASKS - 1))" --partition="$PARTITION" --job-name="dqn1_base_long" "${MAIL_ARGS[@]}"
     "$SCRIPT_DIR/dqn1_baseline.sh" "${RUN_PAIRS[@]}")
if [ -n "$DRY_RUN" ]; then
  echo "${CMD[@]}"
  for ((i = 0; i < N_TASKS; i++)); do
    echo "  task $i: ${RUN_PAIRS[$i]}"
  done
else
  "${CMD[@]}"
fi
