#!/bin/bash
# Submit the 4 new agents (new_agents_suft.sh) on the shaham-L4 partition, 2 configurations each, one job array per
# agent configuration (8 job arrays):
#
#   agent    single buffer (CONFIGS single)                 short-long (CONFIGS sl)
#   Dueling  DQNDueling                                     DQNSLDueling
#   Rainbow  DDQNNStepPERDueling (n-step 3, PER 0.5)        DDQNSLNStepPERDueling
#   DER      DQNDER (standard DER, H1 + H2 budget 1M)       DQNSLDER (short + DER, long DER budget 1M - short)
#   ERE      DQNERE                                         DQNSLERE (ERE on the long term buffer)
#
#   single buffer: buffer 1M, TF lambda 0
#   short-long:    short term buffer 4000, long term buffer 1M - 4000, TF lambda 1, sampling fraction 0.5
#
# Each array: all 57 Atari games * 4 runs = 228 tasks, ONE run per task (1 GPU, 1 CPU core, 55G)
# Total: 8 arrays * 228 tasks = 1824 runs
# The tasks are ordered by run number then game (all the games of run 1 first)
# One email is sent to MAIL_USER when each job array has ended (Slurm --mail-type=END applies to the array as a whole)
#
# Usage:
#   ./submit_new_agents_shaham.sh                          the 8 job arrays
#   ONLY="Dueling DER" ./submit_new_agents_shaham.sh       only these agents (Dueling, Rainbow, DER, ERE)
#   CONFIGS=sl ./submit_new_agents_shaham.sh               only these configurations (single, sl)
#   RUNS="1 2" ./submit_new_agents_shaham.sh               only these runs
#   ERE_CONTINUOUS=false ./submit_new_agents_shaham.sh     ERE updates only (paper; default: the usual DQN updates every
#                                                          4 steps plus the ERE updates after every episode)
#   PARTITION=shaham-L40 ./submit_new_agents_shaham.sh     another partition (default shaham-L4)
#   MAIL_USER= ./submit_new_agents_shaham.sh               no email (default fiskust@biu.ac.il)
#   DRY_RUN=1 ./submit_new_agents_shaham.sh                print the sbatch commands and every task's run without submitting
# The task log is slurm-<job id>_<task id>.out and its first line names the task's run

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -ra RUN_LIST <<< "${RUNS:-1 2 3 4}"
read -ra AGENT_LIST <<< "${ONLY:-Dueling Rainbow DER ERE}"
read -ra CONFIG_LIST <<< "${CONFIGS:-single sl}"
PARTITION=${PARTITION:-shaham-L4}
MAIL_USER=${MAIL_USER-fiskust@biu.ac.il}
# Exported to the job script (sbatch passes the environment)
export ERE_CONTINUOUS=${ERE_CONTINUOUS:-true}
# Single buffer configuration (the buffer size, 1M, is set by the job script)
SINGLE_TF_LAMBDA=0
# Short-long configuration
SHORT_BUFFER_SIZE=4000
SL_TF_LAMBDA=1
SAMPLING_FRACTION=0.5
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
    Dueling) SINGLE_AGENT=DQNDueling;          SL_AGENT=DQNSLDueling;          JOB_PREFIX=dueling ;;
    Rainbow) SINGLE_AGENT=DDQNNStepPERDueling; SL_AGENT=DDQNSLNStepPERDueling; JOB_PREFIX=rainbow ;;
    DER)     SINGLE_AGENT=DQNDER;              SL_AGENT=DQNSLDER;              JOB_PREFIX=der ;;
    ERE)     SINGLE_AGENT=DQNERE;              SL_AGENT=DQNSLERE;              JOB_PREFIX=ere ;;
    *) echo "Unknown agent $AGENT (Dueling, Rainbow, DER, ERE)" >&2; exit 1 ;;
  esac
  for CONFIG in "${CONFIG_LIST[@]}"; do
    case $CONFIG in
      single) AGENT_NAME=$SINGLE_AGENT; JOB_NAME=${JOB_PREFIX}_single; HPARAMS=("$SINGLE_TF_LAMBDA" 0 0) ;;
      sl)     AGENT_NAME=$SL_AGENT;     JOB_NAME=${JOB_PREFIX}_sl;     HPARAMS=("$SL_TF_LAMBDA" "$SHORT_BUFFER_SIZE" "$SAMPLING_FRACTION") ;;
      *) echo "Unknown configuration $CONFIG (single, sl)" >&2; exit 1 ;;
    esac
    CMD=(sbatch --array="0-$((N_TASKS - 1))" --partition="$PARTITION" --job-name="$JOB_NAME" "${MAIL_ARGS[@]}"
         --cpus-per-task=1 --mem="${MEM_GB}G"
         "$SCRIPT_DIR/new_agents_suft.sh" "$AGENT_NAME" "${HPARAMS[@]}" "${RUN_PAIRS[@]}")
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
done
