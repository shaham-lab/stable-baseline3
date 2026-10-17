#!/bin/bash
# Submit dqncer_baseline.sh for the 47 Atari games that are not long as a single job array:
# every (game, run) pair is one run, and each array task runs RUNS_PER_TASK (default 3) of them in parallel on one GPU
#
# By default: runs 1-4, partition L40s-4h:
# 47 games * 4 runs = 188 runs in 63 array tasks of 3 runs (the last one has 2)
# The runs are ordered by run number then game (all the games of run 1 first), so a task runs the same run of 3 games
# The 10 long games (more than 4h) are submitted by submit_dqncer_baseline_long_games.sh
#
# Usage:
#   ./submit_dqncer_baseline_other_games.sh                      the 47 other games, runs 1-4
#   ./submit_dqncer_baseline_other_games.sh Alien Pong           only these games
#   RUNS="1 2" ./submit_dqncer_baseline_other_games.sh           only these runs
#   RUNS_PER_TASK=2 ./submit_dqncer_baseline_other_games.sh      another number of parallel runs per task (and GPU)
#   PARTITION=L4-12h ./submit_dqncer_baseline_other_games.sh     another partition (default L40s-4h)
#   CER_N_LATEST=1 ./submit_dqncer_baseline_other_games.sh   original CER: only the latest transition (default train_freq)
#   DRY_RUN=1 ./submit_dqncer_baseline_other_games.sh            print the sbatch command and every task's runs without submitting
# The task log is slurm-<job id>_<task id>.out and its first line names the task's runs

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -ra RUN_LIST <<< "${RUNS:-1 2 3 4}"
RUNS_PER_TASK=${RUNS_PER_TASK:-3}
CER_N_LATEST=${CER_N_LATEST:-train_freq}
PARTITION=${PARTITION:-L40s-4h}

# All the runs of a task share one GPU, each run gets one CPU core and 55G
# (the 1M transitions peak at ~53.7GB per run), so every array task is sized to RUNS_PER_TASK
MEM_PER_RUN_GB=55

# The 57 games of the Atari benchmark without the 10 long ones (submit_dqncer_baseline_long_games.sh)
GAMES="Alien Amidar Assault Asterix Asteroids Atlantis BankHeist BeamRider Berzerk Bowling
Boxing Breakout Centipede ChopperCommand CrazyClimber DoubleDunk Enduro FishingDerby Freeway Frostbite
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
# Split them into the array tasks, RUNS_PER_TASK pairs per task as a comma separated list
csv() { local IFS=','; echo "$*"; }
RUN_GROUPS=()
for ((i = 0; i < ${#RUN_PAIRS[@]}; i += RUNS_PER_TASK)); do
  RUN_GROUPS+=("$(csv "${RUN_PAIRS[@]:i:RUNS_PER_TASK}")")
done
N_TASKS=${#RUN_GROUPS[@]}

CMD=(sbatch --export=ALL,CER_N_LATEST="$CER_N_LATEST" --array="0-$((N_TASKS - 1))" --partition="$PARTITION" --job-name="dqncer_base_other"
     --cpus-per-task="$RUNS_PER_TASK" --mem="$((RUNS_PER_TASK * MEM_PER_RUN_GB))G"
     "$SCRIPT_DIR/dqncer_baseline.sh" "${RUN_GROUPS[@]}")
if [ -n "$DRY_RUN" ]; then
  echo "${CMD[@]}"
  for ((i = 0; i < N_TASKS; i++)); do
    echo "  task $i: ${RUN_GROUPS[$i]}"
  done
else
  "${CMD[@]}"
fi
