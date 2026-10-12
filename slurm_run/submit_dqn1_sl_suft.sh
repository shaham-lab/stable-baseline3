#!/bin/bash
# Submit dqn1_sl_suft.sh for every (short buffer size, TF lambda, sampling fraction, game, run) combination
# The long term buffer size is 1M minus the short term buffer size (set in dqn1_sl_suft.sh)
#
# By default only the sampling fraction is searched (short term buffer size 4000, TF lambda 1):
# 4 sampling fractions * 57 games * 4 runs = 912 runs
# Submission order is a nested loop: short buffer size -> TF lambda -> sampling fraction -> game -> run,
# so all the games / runs of fraction 0.5 are submitted first, then the next fraction, ...
#
# Usage:
#   ./submit_dqn1_sl_suft.sh                                  all 57 Atari games, runs 1-4, all sampling fractions
#   ./submit_dqn1_sl_suft.sh Alien Pong                       only these games
#   RUNS="1 2" ./submit_dqn1_sl_suft.sh Alien                 only these runs
#   SHORT_BUFFER_SIZES="4000 2000" ./submit_dqn1_sl_suft.sh   also search these short term buffer sizes
#   TF_LAMBDAS="1 2 0.5" ./submit_dqn1_sl_suft.sh             also search these TF lambdas
#   SAMPLING_FRACTIONS="0.5" ./submit_dqn1_sl_suft.sh         only these sampling fractions
#   DRY_RUN=1 ./submit_dqn1_sl_suft.sh                        print the sbatch commands without submitting

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNS=${RUNS:-"1 2 3 4"}
SHORT_BUFFER_SIZES=${SHORT_BUFFER_SIZES:-"4000"}
TF_LAMBDAS=${TF_LAMBDAS:-"1"}
SAMPLING_FRACTIONS=${SAMPLING_FRACTIONS:-"0.5 0.75 0.25 1"}

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

for SHORT_BUFFER_SIZE in $SHORT_BUFFER_SIZES; do
  for TF_LAMBDA in $TF_LAMBDAS; do
    for SAMPLING_FRACTION in $SAMPLING_FRACTIONS; do
      for GAME in $GAMES; do
        for RUN_NUMBER in $RUNS; do
          CMD=(sbatch --job-name="${GAME,,}_dqn1_sl_s${SHORT_BUFFER_SIZE}_l${TF_LAMBDA}_f${SAMPLING_FRACTION}_r${RUN_NUMBER}" "$SCRIPT_DIR/dqn1_sl_suft.sh" "$GAME" "$RUN_NUMBER" "$SHORT_BUFFER_SIZE" "$TF_LAMBDA" "$SAMPLING_FRACTION")
          if [ -n "$DRY_RUN" ]; then
            echo "${CMD[@]}"
          else
            "${CMD[@]}"
          fi
        done
      done
    done
  done
done
