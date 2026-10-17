#!/bin/bash
# Submit dqnper_baseline.sh for the 47 Atari games that are not long: one job per (game, run) pair
# The 10 long games (more than 4h) are submitted by submit_dqnper_baseline_long_games.sh
#
# By default: runs 1-4, partition L4-4h
# 47 games * 4 runs = 188 jobs
#
# Usage:
#   ./submit_dqnper_baseline_other_games.sh                      the 47 other games, runs 1-4
#   ./submit_dqnper_baseline_other_games.sh Alien Pong     only these games
#   RUNS="1 2" ./submit_dqnper_baseline_other_games.sh           only these runs
#   PARTITION=L4-12h ./submit_dqnper_baseline_other_games.sh     another partition (default L4-4h)
#   DRY_RUN=1 ./submit_dqnper_baseline_other_games.sh            print the sbatch commands without submitting

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNS=${RUNS:-"1 2 3 4"}
PARTITION=${PARTITION:-L4-4h}

# The 57 games of the Atari benchmark without the 10 long ones (submit_dqnper_baseline_long_games.sh)
GAMES="Alien Amidar Assault Asterix Asteroids Atlantis BankHeist BeamRider Berzerk Bowling
Boxing Breakout Centipede ChopperCommand CrazyClimber DoubleDunk Enduro FishingDerby Freeway Frostbite
Gopher Gravitar Hero IceHockey Jamesbond Kangaroo Krull KungFuMaster MsPacman NameThisGame
Pong PrivateEye Qbert Riverraid RoadRunner Robotank Seaquest Skiing SpaceInvaders StarGunner
Surround TimePilot UpNDown Venture WizardOfWor YarsRevenge Zaxxon"

if [ $# -gt 0 ]; then
  GAMES="$*"
fi

for GAME in $GAMES; do
  for RUN_NUMBER in $RUNS; do
    CMD=(sbatch --partition="$PARTITION" --job-name="${GAME,,}_per_base_r${RUN_NUMBER}" "$SCRIPT_DIR/dqnper_baseline.sh" "$GAME" "$RUN_NUMBER")
    if [ -n "$DRY_RUN" ]; then
      echo "${CMD[@]}"
    else
      "${CMD[@]}"
    fi
  done
done
