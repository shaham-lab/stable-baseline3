#!/bin/bash
# Submit dqn_n_step_baseline.sh (DQNNStep) for the 10 long Atari games (the ones that took more than 4h) as a single job array:
# every (game, run) pair is one run, and each array task runs RUNS_PER_TASK (default 3) of them in parallel on one GPU
#
# By default: n_steps 3, runs 1-4, partition L4-4h:
# 10 games * 4 runs = 40 runs in 14 array tasks of 3 runs (the last one has 1)
# The runs are ordered by run number then game (all the games of run 1 first), so a task runs the same run of 3 games
# The other 47 games are submitted by submit_dqn_n_step_baseline_other_games.sh
#
# Usage:
#   ./submit_dqn_n_step_baseline_long_games.sh                         the 10 long games, runs 1-4
#   ./submit_dqn_n_step_baseline_long_games.sh Tennis Pitfall          only these games
#   RUNS="1 2" ./submit_dqn_n_step_baseline_long_games.sh              only these runs
#   N_STEPS=5 ./submit_dqn_n_step_baseline_long_games.sh               another n-step return length (default 3, as in Rainbow)
#   RUNS_PER_TASK=2 ./submit_dqn_n_step_baseline_long_games.sh         another number of parallel runs per task (and GPU)
#   PARTITION=shaham-L4 ./submit_dqn_n_step_baseline_long_games.sh     another partition (default L4-4h)
#   DRY_RUN=1 ./submit_dqn_n_step_baseline_long_games.sh               print the sbatch command and every task's runs without submitting
# The task log is slurm-<job id>_<task id>.out and its first line names the task's runs

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -ra RUN_LIST <<< "${RUNS:-1 2 3 4}"
N_STEPS=${N_STEPS:-3}
RUNS_PER_TASK=${RUNS_PER_TASK:-3}
PARTITION=${PARTITION:-shaham-L4}

# All the runs of a task share one GPU, each run gets one CPU core and 55G
# (the 1M transitions peak at ~53.7GB per run), so every array task is sized to RUNS_PER_TASK
MEM_PER_RUN_GB=55

# The long games: they took more than 4h
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
# Split them into the array tasks, RUNS_PER_TASK pairs per task as a comma separated list
csv() { local IFS=','; echo "$*"; }
RUN_GROUPS=()
for ((i = 0; i < ${#RUN_PAIRS[@]}; i += RUNS_PER_TASK)); do
  RUN_GROUPS+=("$(csv "${RUN_PAIRS[@]:i:RUNS_PER_TASK}")")
done
N_TASKS=${#RUN_GROUPS[@]}

CMD=(sbatch --array="0-$((N_TASKS - 1))" --partition="$PARTITION" --job-name="dqn_nstep_long"
     --cpus-per-task="$RUNS_PER_TASK" --mem="$((RUNS_PER_TASK * MEM_PER_RUN_GB))G"
     "$SCRIPT_DIR/dqn_n_step_baseline.sh" "$N_STEPS" "${RUN_GROUPS[@]}")
if [ -n "$DRY_RUN" ]; then
  echo "${CMD[@]}"
  for ((i = 0; i < N_TASKS; i++)); do
    echo "  task $i: ${RUN_GROUPS[$i]}"
  done
else
  "${CMD[@]}"
fi
