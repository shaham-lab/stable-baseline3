#!/bin/bash
# Resubmit the runs listed in failed_runs.txt (hit the 4h time limit before finishing) on shaham-L4
# shaham-L4 has no time limit; --partition on the command line overrides the #SBATCH partition of each job script
# The job scripts don't checkpoint, so every run starts again from timestep 0 and overwrites its old log
#
# Usage:
#   ./submit_failed_runs.sh              submit all 47 runs
#   DRY_RUN=1 ./submit_failed_runs.sh    print the sbatch commands without submitting

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARTITION="shaham-L4"

# submit <job_name> <script> <script_args...>
submit() {
  local JOB_NAME=$1
  local SCRIPT=$2
  shift 2
  CMD=(sbatch --partition="$PARTITION" --job-name="$JOB_NAME" "$SCRIPT_DIR/$SCRIPT" "$@")
  if [ -n "$DRY_RUN" ]; then
    echo "${CMD[@]}"
  else
    "${CMD[@]}"
  fi
}

# Cancelled on 2026-10-13 while pending after being requeued at the 4h limit (29 jobs)
submit montezumarevenge_ddqn_base_r4 ddqn_baseline.sh MontezumaRevenge 4
submit pitfall_ddqn_base_r2 ddqn_baseline.sh Pitfall 2
submit pitfall_ddqn_base_r4 ddqn_baseline.sh Pitfall 4
submit battlezone_dqn1_base_r2 dqn1_baseline.sh BattleZone 2
submit battlezone_dqn1_base_r3 dqn1_baseline.sh BattleZone 3
submit tennis_dqn1_base_r1 dqn1_baseline.sh Tennis 1
submit tennis_dqn1_base_r2 dqn1_baseline.sh Tennis 2
submit tennis_dqn1_base_r3 dqn1_baseline.sh Tennis 3
submit tennis_dqn1_base_r4 dqn1_baseline.sh Tennis 4
submit tutankham_dqn1_base_r1 dqn1_baseline.sh Tutankham 1
submit tutankham_dqn1_base_r2 dqn1_baseline.sh Tutankham 2
submit tutankham_dqn1_base_r3 dqn1_baseline.sh Tutankham 3
submit tutankham_dqn1_base_r4 dqn1_baseline.sh Tutankham 4
submit videopinball_dqn1_base_r2 dqn1_baseline.sh VideoPinball 2
submit videopinball_dqn1_base_r3 dqn1_baseline.sh VideoPinball 3
submit battlezone_dqn1_sl_s4000_l1_f0.5_r1 dqn1_sl_suft.sh BattleZone 1 4000 1 0.5
submit defender_dqn1_sl_s4000_l1_f0.5_r1 dqn1_sl_suft.sh Defender 1 4000 1 0.5
submit defender_dqn1_sl_s4000_l1_f0.5_r2 dqn1_sl_suft.sh Defender 2 4000 1 0.5
submit defender_dqn1_sl_s4000_l1_f0.5_r3 dqn1_sl_suft.sh Defender 3 4000 1 0.5
submit demonattack_dqn1_sl_s4000_l1_f0.5_r1 dqn1_sl_suft.sh DemonAttack 1 4000 1 0.5
submit demonattack_dqn1_sl_s4000_l1_f0.5_r2 dqn1_sl_suft.sh DemonAttack 2 4000 1 0.5
submit demonattack_dqn1_sl_s4000_l1_f0.5_r3 dqn1_sl_suft.sh DemonAttack 3 4000 1 0.5
submit demonattack_dqn1_sl_s4000_l1_f0.5_r4 dqn1_sl_suft.sh DemonAttack 4 4000 1 0.5
submit montezumarevenge_dqn1_sl_s4000_l1_f0.5_r1 dqn1_sl_suft.sh MontezumaRevenge 1 4000 1 0.5
submit montezumarevenge_dqn1_sl_s4000_l1_f0.5_r2 dqn1_sl_suft.sh MontezumaRevenge 2 4000 1 0.5
submit montezumarevenge_dqn1_sl_s4000_l1_f0.5_r3 dqn1_sl_suft.sh MontezumaRevenge 3 4000 1 0.5
submit montezumarevenge_dqn1_sl_s4000_l1_f0.5_r4 dqn1_sl_suft.sh MontezumaRevenge 4 4000 1 0.5
submit phoenix_dqn1_sl_s4000_l1_f0.5_r2 dqn1_sl_suft.sh Phoenix 2 4000 1 0.5

# Timed out at 4h and not requeued (18 jobs)
submit defender_dqn1_base_r1 dqn1_baseline.sh Defender 1
submit defender_dqn1_base_r2 dqn1_baseline.sh Defender 2
submit defender_dqn1_base_r3 dqn1_baseline.sh Defender 3
submit defender_dqn1_base_r4 dqn1_baseline.sh Defender 4
submit demonattack_dqn1_base_r1 dqn1_baseline.sh DemonAttack 1
submit demonattack_dqn1_base_r2 dqn1_baseline.sh DemonAttack 2
submit demonattack_dqn1_base_r3 dqn1_baseline.sh DemonAttack 3
submit demonattack_dqn1_base_r4 dqn1_baseline.sh DemonAttack 4
submit montezumarevenge_dqn1_base_r2 dqn1_baseline.sh MontezumaRevenge 2
submit montezumarevenge_dqn1_base_r3 dqn1_baseline.sh MontezumaRevenge 3
submit montezumarevenge_dqn1_base_r4 dqn1_baseline.sh MontezumaRevenge 4
submit pitfall_dqn1_base_r1 dqn1_baseline.sh Pitfall 1
submit pitfall_dqn1_base_r2 dqn1_baseline.sh Pitfall 2
submit pitfall_dqn1_base_r4 dqn1_baseline.sh Pitfall 4
submit solaris_dqn1_base_r1 dqn1_baseline.sh Solaris 1
submit solaris_dqn1_base_r2 dqn1_baseline.sh Solaris 2
submit solaris_dqn1_base_r3 dqn1_baseline.sh Solaris 3
submit solaris_dqn1_base_r4 dqn1_baseline.sh Solaris 4
