#!/bin/bash
# Submit EREandasingleotheragent.sh on the shaham-L4 partition as 3 job arrays, one per agent configuration:
#
#   DDQNSLNStepPERDueling  short term buffer 4000 sampled UNIFORMLY, long term buffer 1M - 4000 prioritized,
#                          n-step 3, PER alpha 0.5, TF lambda 1, sampling fraction 0.5   (replaces job array 22611)
#   DQNERE                 single buffer 1M, TF lambda 0
#   DQNSLERE               short term buffer 4000 (uniform), long term buffer 1M - 4000 (ERE),
#                          TF lambda 1, sampling fraction 0.5
#   Both ERE agents: the usual DQN updates every 4 steps (replay ratio 0.25) plus the ERE updates after every episode
#
# Each array: all 57 Atari games * 4 runs = 228 tasks, ONE run per task (1 GPU, 1 CPU core, 55G)
# Total: 3 arrays * 228 tasks = 684 runs
# The tasks are ordered by run number then game (all the games of run 1 first)
# One email is sent to MAIL_USER when each job array has ended (Slurm --mail-type=END applies to the array as a whole)
# Log names: ..._n_steps_3_per_short_false_run_<run>.log (DDQNSLNStepPERDueling),
#            ..._ere_continuous_true_run_<run>.log (DQNERE, DQNSLERE)
#
# Usage:
#   ./submit_EREandasingleotheragent.sh                                 the 3 job arrays
#   ONLY="DQNERE DQNSLERE" ./submit_EREandasingleotheragent.sh          only these agents
#   RUNS="1 2" ./submit_EREandasingleotheragent.sh                      only these runs
#   PARTITION=shaham-L40 ./submit_EREandasingleotheragent.sh            another partition (default shaham-L4)
#   MAIL_USER= ./submit_EREandasingleotheragent.sh                      no email (default fiskust@biu.ac.il)
#   DRY_RUN=1 ./submit_EREandasingleotheragent.sh                       print the sbatch commands and every task's run without submitting
# The task log is slurm-<job id>_<task id>.out and its first line names the task's run

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -ra RUN_LIST <<< "${RUNS:-1 2 3 4}"
read -ra AGENT_LIST <<< "${ONLY:-DDQNSLNStepPERDueling DQNERE DQNSLERE}"
PARTITION=${PARTITION:-shaham-L4}
MAIL_USER=${MAIL_USER-fiskust@biu.ac.il}
# Short-long configuration
SHORT_BUFFER_SIZE=4000
SL_TF_LAMBDA=1
SAMPLING_FRACTION=0.5
# Single buffer configuration (the buffer size, 1M, is set by the job script)
SINGLE_TF_LAMBDA=0
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

for AGENT_NAME in "${AGENT_LIST[@]}"; do
  case $AGENT_NAME in
    DDQNSLNStepPERDueling) JOB_NAME=rainbow_sl_uniform_short; HPARAMS=("$SL_TF_LAMBDA" "$SHORT_BUFFER_SIZE" "$SAMPLING_FRACTION") ;;
    DQNERE)                JOB_NAME=ere_single;               HPARAMS=("$SINGLE_TF_LAMBDA" 0 0) ;;
    DQNSLERE)              JOB_NAME=ere_sl;                   HPARAMS=("$SL_TF_LAMBDA" "$SHORT_BUFFER_SIZE" "$SAMPLING_FRACTION") ;;
    *) echo "Unknown agent $AGENT_NAME (DDQNSLNStepPERDueling, DQNERE, DQNSLERE)" >&2; exit 1 ;;
  esac
  CMD=(sbatch --array="0-$((N_TASKS - 1))" --partition="$PARTITION" --job-name="$JOB_NAME" "${MAIL_ARGS[@]}"
       --cpus-per-task=1 --mem="${MEM_GB}G"
       "$SCRIPT_DIR/EREandasingleotheragent.sh" "$AGENT_NAME" "${HPARAMS[@]}" "${RUN_PAIRS[@]}")
  if [ -n "$DRY_RUN" ]; then
    echo "$AGENT_NAME: ${CMD[*]:0:${#CMD[@]}-N_TASKS} <$N_TASKS runs>"
    for ((i = 0; i < N_TASKS; i++)); do
      echo "  task $i: ${RUN_PAIRS[$i]}"
    done
  else
    echo -n "$AGENT_NAME ($N_TASKS tasks): "
    "${CMD[@]}" || exit 1
  fi
done
