#!/bin/bash
# Submit dqn_agents_suft.sh (single replay buffer) on the shaham-L4 partition as 4 job arrays, one array per agent:
#   DQN, DDQN, DQNNStep (n_steps 3), DQNPER
# Configuration: buffer 1M (BUFFER_SIZE), TF lambda 1, runs 1-4
# Each array: all 57 Atari games * 4 runs = 228 tasks, ONE run per task (1 GPU, 1 CPU core,
# memory sized to the buffer: 55G for 1M, 8G for small buffers such as 4K)
# Total: 4 arrays * 228 tasks = 912 runs
# The tasks are ordered by run number then game (all the games of run 1 first)
# One email is sent to MAIL_USER when each job array has ended (Slurm --mail-type=END applies to the array as a whole)
#
# Usage:
#   ./submit_dqn_agents_suft_shaham.sh                       the 4 job arrays
#   ONLY="DQN DDQN" ./submit_dqn_agents_suft_shaham.sh       only these agents
#   RUNS="1 2" ./submit_dqn_agents_suft_shaham.sh            only these runs
#   TF_LAMBDA=0.5 ./submit_dqn_agents_suft_shaham.sh         another TF lambda (default 1)
#   BUFFER_SIZE=4000 ./submit_dqn_agents_suft_shaham.sh      another buffer size (default 1M)
#   MEM_GB=16 ./submit_dqn_agents_suft_shaham.sh             another memory per task (default sized to the buffer)
#   PARTITION=shaham-L40 ./submit_dqn_agents_suft_shaham.sh  another partition (default shaham-L4)
#   MAIL_USER= ./submit_dqn_agents_suft_shaham.sh            no email (default fiskust@biu.ac.il)
#   DRY_RUN=1 ./submit_dqn_agents_suft_shaham.sh             print the sbatch commands and every task's run without submitting
# The task log is slurm-<job id>_<task id>.out and its first line names the task's run

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -ra RUN_LIST <<< "${RUNS:-1 2 3 4}"
read -ra AGENT_LIST <<< "${ONLY:-DQN DDQN DQNNStep DQNPER}"
TF_LAMBDA=${TF_LAMBDA:-1}
BUFFER_SIZE=${BUFFER_SIZE:-1000000}
# 1M transitions peak at ~53.7GB including the process overhead (55G), at least 8G for the process itself
DEFAULT_MEM_GB=$(( (51 * BUFFER_SIZE + 999999) / 1000000 + 4 ))
[ $DEFAULT_MEM_GB -lt 8 ] && DEFAULT_MEM_GB=8
MEM_GB=${MEM_GB:-$DEFAULT_MEM_GB}
PARTITION=${PARTITION:-shaham-L4}
MAIL_USER=${MAIL_USER-fiskust@biu.ac.il}

# All the 57 games of the Atari benchmark (the 47 "other" games + the 10 long ones)
GAMES="Alien Amidar Assault Asterix Asteroids Atlantis BankHeist BeamRider Berzerk Bowling
Boxing Breakout Centipede ChopperCommand CrazyClimber DoubleDunk Enduro FishingDerby Freeway Frostbite
Gopher Gravitar Hero IceHockey Jamesbond Kangaroo Krull KungFuMaster MsPacman NameThisGame
Pong PrivateEye Qbert Riverraid RoadRunner Robotank Seaquest Skiing SpaceInvaders StarGunner
Surround TimePilot UpNDown Venture WizardOfWor YarsRevenge Zaxxon
Tennis MontezumaRevenge Pitfall Solaris Tutankham DemonAttack Defender VideoPinball BattleZone Phoenix"
read -ra GAME_LIST <<< "$(echo $GAMES)"

# Every (game, run) pair as <Game>:<run_number>, one per array task
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

for AGENT in "${AGENT_LIST[@]}"; do
  case $AGENT in
    DQN)      JOB_NAME=dqn_suft ;;
    DDQN)     JOB_NAME=ddqn_suft ;;
    DQNNStep) JOB_NAME=dqn_nstep_suft ;;
    DQNPER)   JOB_NAME=dqnper_suft ;;
    *) echo "Unknown agent $AGENT (DQN, DDQN, DQNNStep, DQNPER)" >&2; exit 1 ;;
  esac
  [ "$BUFFER_SIZE" -ne 1000000 ] && JOB_NAME="${JOB_NAME}_b${BUFFER_SIZE}"
  CMD=(sbatch --array="0-$((N_TASKS - 1))" --partition="$PARTITION" --job-name="$JOB_NAME" "${MAIL_ARGS[@]}"
       --mem="${MEM_GB}G"
       "$SCRIPT_DIR/dqn_agents_suft.sh" "$AGENT" "$TF_LAMBDA" "$BUFFER_SIZE" "${RUN_PAIRS[@]}")
  if [ -n "$DRY_RUN" ]; then
    echo "$AGENT: ${CMD[*]:0:${#CMD[@]}-N_TASKS} <$N_TASKS runs>"
    for ((i = 0; i < N_TASKS; i++)); do
      echo "  task $i: ${RUN_PAIRS[$i]}"
    done
  else
    echo -n "$AGENT ($N_TASKS tasks): "
    "${CMD[@]}" || exit 1
  fi
done
