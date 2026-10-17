#!/bin/bash
# Submit dqn_n_step_baseline.sh for the 10 long Atari games (the ones that took more than 4h): one job per (game, run) pair
# The other 47 games are submitted by submit_dqn_n_step_baseline_other_games.sh
#
# By default: runs 1-4, n_steps 3, partition shaham-L4
# 10 games * 4 runs = 40 jobs
#
# Usage:
#   ./submit_dqn_n_step_baseline_long_games.sh                      the 10 long games, runs 1-4
#   ./submit_dqn_n_step_baseline_long_games.sh Tennis Pitfall     only these games
#   RUNS="1 2" ./submit_dqn_n_step_baseline_long_games.sh           only these runs
#   N_STEPS=5 ./submit_dqn_n_step_baseline_long_games.sh            another n-step return length (default 3, as in Rainbow)
#   PARTITION=L4-12h ./submit_dqn_n_step_baseline_long_games.sh     another partition (default shaham-L4)
#   DRY_RUN=1 ./submit_dqn_n_step_baseline_long_games.sh            print the sbatch commands without submitting

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNS=${RUNS:-"1 2 3 4"}
N_STEPS=${N_STEPS:-3}
PARTITION=${PARTITION:-shaham-L4}

# The long games: they took more than 4h
GAMES="Tennis MontezumaRevenge Pitfall Solaris Tutankham DemonAttack Defender VideoPinball BattleZone Phoenix"

if [ $# -gt 0 ]; then
  GAMES="$*"
fi

for GAME in $GAMES; do
  for RUN_NUMBER in $RUNS; do
    CMD=(sbatch --partition="$PARTITION" --job-name="${GAME,,}_nstep${N_STEPS}_r${RUN_NUMBER}" "$SCRIPT_DIR/dqn_n_step_baseline.sh" "$GAME" "$RUN_NUMBER" "$N_STEPS")
    if [ -n "$DRY_RUN" ]; then
      echo "${CMD[@]}"
    else
      "${CMD[@]}"
    fi
  done
done
