#!/bin/bash
# Submit DDQNSLNStepPERDueling (new_agents_suft.sh) on the shaham-L4 partition as ONE job array:
#   Double DQN + n-step 3 + PER (alpha 0.5) + dueling, short-long SUFT,
#   short term buffer 4000 sampled UNIFORMLY, long term buffer 1M - 4000 prioritized,
#   TF lambda 1, sampling fraction 0.5, runs 1-4
# Replaces job array 22611, which prioritized the short term buffer too: cancel it first (scancel 22611)
# and remove its logs (results/DDQNSLNStepPERDueling/*_per_short_true_run_*.log);
# the logs of this array end with _per_short_false_run_<run>.log
# The array: all 57 Atari games * 4 runs = 228 tasks, ONE run per task (1 GPU, 1 CPU core, 55G)
# The tasks are ordered by run number then game (all the games of run 1 first)
# One email is sent to MAIL_USER when the job array has ended (Slurm --mail-type=END applies to the array as a whole)
#
# Usage:
#   ./submit_ddqnsl_nstep_per_dueling_shaham.sh                         the job array
#   RUNS="1 2" ./submit_ddqnsl_nstep_per_dueling_shaham.sh              only these runs
#   PARTITION=shaham-L40 ./submit_ddqnsl_nstep_per_dueling_shaham.sh    another partition (default shaham-L4)
#   MAIL_USER= ./submit_ddqnsl_nstep_per_dueling_shaham.sh              no email (default fiskust@biu.ac.il)
#   DRY_RUN=1 ./submit_ddqnsl_nstep_per_dueling_shaham.sh               print the sbatch command and every task's run without submitting
# The task log is slurm-<job id>_<task id>.out and its first line names the task's run

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -ra RUN_LIST <<< "${RUNS:-1 2 3 4}"
PARTITION=${PARTITION:-shaham-L4}
MAIL_USER=${MAIL_USER-fiskust@biu.ac.il}
AGENT_NAME=DDQNSLNStepPERDueling
JOB_NAME=rainbow_sl
SHORT_BUFFER_SIZE=4000
TF_LAMBDA=1
SAMPLING_FRACTION=0.5
# One run per task: 1 CPU core and 55G (the 1M transitions peak at ~53.7GB), the GPU comes from the job script
MEM_GB=55

# The job script must pass a uniform short term buffer (prioritize_short_term false)
if ! grep -q '^ *DDQNSLNStepPERDueling).*AGENT_ARGS=("$N_STEPS" false)' "$SCRIPT_DIR/new_agents_suft.sh"; then
  echo "new_agents_suft.sh does not pass a uniform short term buffer for $AGENT_NAME (pull the fix first)" >&2
  exit 1
fi

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

CMD=(sbatch --array="0-$((N_TASKS - 1))" --partition="$PARTITION" --job-name="$JOB_NAME" "${MAIL_ARGS[@]}"
     --cpus-per-task=1 --mem="${MEM_GB}G"
     "$SCRIPT_DIR/new_agents_suft.sh" "$AGENT_NAME" "$TF_LAMBDA" "$SHORT_BUFFER_SIZE" "$SAMPLING_FRACTION" "${RUN_PAIRS[@]}")
if [ -n "$DRY_RUN" ]; then
  echo "$AGENT_NAME: ${CMD[*]:0:${#CMD[@]}-N_TASKS} <$N_TASKS runs>"
  for ((i = 0; i < N_TASKS; i++)); do
    echo "  task $i: ${RUN_PAIRS[$i]}"
  done
else
  echo -n "$AGENT_NAME ($N_TASKS tasks): "
  "${CMD[@]}" || exit 1
fi
