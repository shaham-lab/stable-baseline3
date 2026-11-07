#!/bin/bash
# Submit the 4 Short-Long SUFT agents on the shaham-L4 partition as 4 job arrays, one array per agent:
#   DQNSL        (dqn_sl_suft.sh)
#   DDQNSL       (ddqn_sl_suft.sh)
#   DQNSLNStep   (dqn_sl_n_step_suft.sh, n_steps 3)
#   DQNSLPERLong (dqnsl_per_long_suft.sh)
# Configuration: short term buffer 4000 (long term buffer = 1M - 4000), sampling fraction 1
# (every minibatch is taken from the short term buffer only), TF lambda 1, runs 1-4
# Each array: all 57 Atari games * 4 runs = 228 tasks, ONE run per task (1 GPU, 1 CPU core, 55G)
# Total: 4 arrays * 228 tasks = 912 runs
# The tasks are ordered by run number then game (all the games of run 1 first)
# One email is sent to MAIL_USER when each job array has ended (Slurm --mail-type=END applies to the array as a whole)
#
# Usage:
#   ./submit_sl_suft_fraction1_shaham.sh                    the 4 job arrays
#   ONLY="DQNSL DDQNSL" ./submit_sl_suft_fraction1_shaham.sh   only these agents
#   RUNS="1 2" ./submit_sl_suft_fraction1_shaham.sh         only these runs
#   PARTITION=shaham-L40 ./submit_sl_suft_fraction1_shaham.sh   another partition (default shaham-L4)
#   MAIL_USER= ./submit_sl_suft_fraction1_shaham.sh         no email (default fiskust@biu.ac.il)
#   DRY_RUN=1 ./submit_sl_suft_fraction1_shaham.sh          print the sbatch commands and every task's run without submitting
# The task log is slurm-<job id>_<task id>.out and its first line names the task's run

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -ra RUN_LIST <<< "${RUNS:-1 2 3 4}"
read -ra AGENT_LIST <<< "${ONLY:-DQNSL DDQNSL DQNSLNStep DQNSLPERLong}"
SHORT_BUFFER_SIZE=4000
TF_LAMBDA=1
SAMPLING_FRACTION=1
N_STEPS=3
PARTITION=${PARTITION:-shaham-L4}
MAIL_USER=${MAIL_USER-fiskust@biu.ac.il}
# One run per task: 1 CPU core and 55G (the 1M transitions peak at ~53.7GB), the GPU comes from the job script
MEM_GB=55

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
    DQNSL)        JOB_SCRIPT=dqn_sl_suft.sh;         JOB_NAME=dqn_sl_f1;        HPARAMS=("$SHORT_BUFFER_SIZE" "$TF_LAMBDA" "$SAMPLING_FRACTION") ;;
    DDQNSL)       JOB_SCRIPT=ddqn_sl_suft.sh;        JOB_NAME=ddqn_sl_f1;       HPARAMS=("$SHORT_BUFFER_SIZE" "$TF_LAMBDA" "$SAMPLING_FRACTION") ;;
    DQNSLNStep)   JOB_SCRIPT=dqn_sl_n_step_suft.sh;  JOB_NAME=dqn_sl_nstep_f1;  HPARAMS=("$SHORT_BUFFER_SIZE" "$TF_LAMBDA" "$SAMPLING_FRACTION" "$N_STEPS") ;;
    DQNSLPERLong) JOB_SCRIPT=dqnsl_per_long_suft.sh; JOB_NAME=dqnsl_per_long_f1; HPARAMS=("$SHORT_BUFFER_SIZE" "$TF_LAMBDA" "$SAMPLING_FRACTION") ;;
    *) echo "Unknown agent $AGENT (DQNSL, DDQNSL, DQNSLNStep, DQNSLPERLong)" >&2; exit 1 ;;
  esac
  CMD=(sbatch --array="0-$((N_TASKS - 1))" --partition="$PARTITION" --job-name="$JOB_NAME" "${MAIL_ARGS[@]}"
       --cpus-per-task=1 --mem="${MEM_GB}G"
       "$SCRIPT_DIR/$JOB_SCRIPT" "${HPARAMS[@]}" "${RUN_PAIRS[@]}")
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
