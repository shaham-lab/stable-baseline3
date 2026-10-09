#!/bin/bash
# Submit dqn_baseline.sh for every (game, run) pair
#
# Usage:
#   ./submit_dqn_baseline.sh                      all 57 Atari games, runs 1-4
#   ./submit_dqn_baseline.sh Alien Pong           only these games
#   RUNS="1 2" ./submit_dqn_baseline.sh Alien     only these runs
#   DRY_RUN=1 ./submit_dqn_baseline.sh            print the sbatch commands without submitting

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNS=${RUNS:-"1 2 3 4"}

# The 57 games of the Atari benchmark
ATARI_57="Alien Amidar Assault Asterix Asteroids Atlantis BankHeist BattleZone BeamRider Berzerk
Bowling Boxing Breakout Centipede ChopperCommand CrazyClimber Defender DemonAttack DoubleDunk
Enduro FishingDerby Freeway Frostbite Gopher Gravitar Hero IceHockey Jamesbond Kangaroo Krull
KungFuMaster MontezumaRevenge MsPacman NameThisGame Phoenix Pitfall Pong PrivateEye Qbert
Riverraid RoadRunner Robotank Seaquest Skiing Solaris SpaceInvaders StarGunner Surround Tennis
TimePilot Tutankham UpNDown Venture VideoPinball WizardOfWor YarsRevenge Zaxxon"

if [ $# -gt 0 ]; then
  GAMES="$*"
else
  GAMES=$ATARI_57
fi

for GAME in $GAMES; do
  for RUN_NUMBER in $RUNS; do
    CMD=(sbatch --job-name="${GAME,,}_base_r${RUN_NUMBER}" "$SCRIPT_DIR/dqn_baseline.sh" "$GAME" "$RUN_NUMBER")
    if [ -n "$DRY_RUN" ]; then
      echo "${CMD[@]}"
    else
      "${CMD[@]}"
    fi
  done
done
