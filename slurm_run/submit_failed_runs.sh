#!/bin/bash
# Resubmit the runs listed in failed_runs.txt on shaham-L4
# They hit the 4h time limit on generic / L4-4h, or were cancelled because their game times out at 4h
# shaham-L4 has no time limit; --partition on the command line overrides the #SBATCH partition of each job script
# The job scripts don't checkpoint, so every run starts again from timestep 0 and overwrites its old log
#
# Usage:
#   ./submit_failed_runs.sh              submit all 132 runs
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

# Timed out at the 4h limit (16)
submit solaris_dqn1_sl_s4000_l1_f0.5_r1 dqn1_sl_suft.sh Solaris 1 4000 1 0.5
submit solaris_dqn1_sl_s4000_l1_f0.5_r2 dqn1_sl_suft.sh Solaris 2 4000 1 0.5
submit solaris_dqn1_sl_s4000_l1_f0.5_r3 dqn1_sl_suft.sh Solaris 3 4000 1 0.5
submit solaris_dqn1_sl_s4000_l1_f0.5_r4 dqn1_sl_suft.sh Solaris 4 4000 1 0.5
submit tennis_ddqn_base_r1 ddqn_baseline.sh Tennis 1
submit tennis_ddqn_base_r2 ddqn_baseline.sh Tennis 2
submit tennis_ddqn_base_r3 ddqn_baseline.sh Tennis 3
submit tennis_ddqn_base_r4 ddqn_baseline.sh Tennis 4
submit tennis_dqn1_sl_s4000_l1_f0.5_r1 dqn1_sl_suft.sh Tennis 1 4000 1 0.5
submit tennis_dqn1_sl_s4000_l1_f0.5_r2 dqn1_sl_suft.sh Tennis 2 4000 1 0.5
submit tennis_dqn1_sl_s4000_l1_f0.5_r3 dqn1_sl_suft.sh Tennis 3 4000 1 0.5
submit tennis_dqn1_sl_s4000_l1_f0.5_r4 dqn1_sl_suft.sh Tennis 4 4000 1 0.5
submit tutankham_dqn1_sl_s4000_l1_f0.5_r1 dqn1_sl_suft.sh Tutankham 1 4000 1 0.5
submit tutankham_dqn1_sl_s4000_l1_f0.5_r2 dqn1_sl_suft.sh Tutankham 2 4000 1 0.5
submit tutankham_dqn1_sl_s4000_l1_f0.5_r3 dqn1_sl_suft.sh Tutankham 3 4000 1 0.5
submit tutankham_dqn1_sl_s4000_l1_f0.5_r4 dqn1_sl_suft.sh Tutankham 4 4000 1 0.5

# Timed out at 4h, requeued, then cancelled on 2026-10-13 (1)
submit montezumarevenge_ddqn_base_r2 ddqn_baseline.sh MontezumaRevenge 2

# Timed out at 4h, requeued, then cancelled on 2026-10-15 (35)
submit battlezone_dqn1_sl_s4000_l1_f0.75_r1 dqn1_sl_suft.sh BattleZone 1 4000 1 0.75
submit defender_dqn1_sl_s4000_l1_f0.75_r4 dqn1_sl_suft.sh Defender 4 4000 1 0.75
submit demonattack_dqn1_sl_s4000_l1_f0.75_r1 dqn1_sl_suft.sh DemonAttack 1 4000 1 0.75
submit demonattack_dqn1_sl_s4000_l1_f0.75_r2 dqn1_sl_suft.sh DemonAttack 2 4000 1 0.75
submit demonattack_dqn1_sl_s4000_l1_f0.75_r3 dqn1_sl_suft.sh DemonAttack 3 4000 1 0.75
submit montezumarevenge_dqn1_sl_s4000_l1_f0.75_r1 dqn1_sl_suft.sh MontezumaRevenge 1 4000 1 0.75
submit montezumarevenge_dqn1_sl_s4000_l1_f0.75_r2 dqn1_sl_suft.sh MontezumaRevenge 2 4000 1 0.75
submit montezumarevenge_dqn1_sl_s4000_l1_f0.75_r3 dqn1_sl_suft.sh MontezumaRevenge 3 4000 1 0.75
submit montezumarevenge_dqn1_sl_s4000_l1_f0.75_r4 dqn1_sl_suft.sh MontezumaRevenge 4 4000 1 0.75
submit phoenix_dqn1_sl_s4000_l1_f0.75_r1 dqn1_sl_suft.sh Phoenix 1 4000 1 0.75
submit phoenix_dqn1_sl_s4000_l1_f0.75_r3 dqn1_sl_suft.sh Phoenix 3 4000 1 0.75
submit phoenix_dqn1_sl_s4000_l1_f0.75_r4 dqn1_sl_suft.sh Phoenix 4 4000 1 0.75
submit pitfall_dqn1_sl_s4000_l1_f0.5_r1 dqn1_sl_suft.sh Pitfall 1 4000 1 0.5
submit pitfall_dqn1_sl_s4000_l1_f0.5_r2 dqn1_sl_suft.sh Pitfall 2 4000 1 0.5
submit pitfall_dqn1_sl_s4000_l1_f0.5_r3 dqn1_sl_suft.sh Pitfall 3 4000 1 0.5
submit pitfall_dqn1_sl_s4000_l1_f0.5_r4 dqn1_sl_suft.sh Pitfall 4 4000 1 0.5
submit pitfall_dqn1_sl_s4000_l1_f0.75_r1 dqn1_sl_suft.sh Pitfall 1 4000 1 0.75
submit pitfall_dqn1_sl_s4000_l1_f0.75_r2 dqn1_sl_suft.sh Pitfall 2 4000 1 0.75
submit pitfall_dqn1_sl_s4000_l1_f0.75_r3 dqn1_sl_suft.sh Pitfall 3 4000 1 0.75
submit pitfall_dqn1_sl_s4000_l1_f0.75_r4 dqn1_sl_suft.sh Pitfall 4 4000 1 0.75
submit solaris_dqn1_sl_s4000_l1_f0.75_r1 dqn1_sl_suft.sh Solaris 1 4000 1 0.75
submit solaris_dqn1_sl_s4000_l1_f0.75_r2 dqn1_sl_suft.sh Solaris 2 4000 1 0.75
submit solaris_dqn1_sl_s4000_l1_f0.75_r3 dqn1_sl_suft.sh Solaris 3 4000 1 0.75
submit solaris_dqn1_sl_s4000_l1_f0.75_r4 dqn1_sl_suft.sh Solaris 4 4000 1 0.75
submit tennis_dqn1_sl_s4000_l1_f0.75_r1 dqn1_sl_suft.sh Tennis 1 4000 1 0.75
submit tennis_dqn1_sl_s4000_l1_f0.75_r2 dqn1_sl_suft.sh Tennis 2 4000 1 0.75
submit tennis_dqn1_sl_s4000_l1_f0.75_r3 dqn1_sl_suft.sh Tennis 3 4000 1 0.75
submit tennis_dqn1_sl_s4000_l1_f0.75_r4 dqn1_sl_suft.sh Tennis 4 4000 1 0.75
submit tutankham_dqn1_sl_s4000_l1_f0.75_r1 dqn1_sl_suft.sh Tutankham 1 4000 1 0.75
submit tutankham_dqn1_sl_s4000_l1_f0.75_r2 dqn1_sl_suft.sh Tutankham 2 4000 1 0.75
submit tutankham_dqn1_sl_s4000_l1_f0.75_r3 dqn1_sl_suft.sh Tutankham 3 4000 1 0.75
submit tutankham_dqn1_sl_s4000_l1_f0.75_r4 dqn1_sl_suft.sh Tutankham 4 4000 1 0.75
submit videopinball_dqn1_sl_s4000_l1_f0.5_r1 dqn1_sl_suft.sh VideoPinball 1 4000 1 0.5
submit videopinball_dqn1_sl_s4000_l1_f0.5_r3 dqn1_sl_suft.sh VideoPinball 3 4000 1 0.5
submit videopinball_dqn1_sl_s4000_l1_f0.5_r4 dqn1_sl_suft.sh VideoPinball 4 4000 1 0.5

# Cancelled on 2026-10-15 while running, before the 4h limit; game times out on other configs (4)
submit battlezone_dqn1_sl_s4000_l1_f0.25_r1 dqn1_sl_suft.sh BattleZone 1 4000 1 0.25
submit battlezone_dqn1_sl_s4000_l1_f0.25_r2 dqn1_sl_suft.sh BattleZone 2 4000 1 0.25
submit battlezone_dqn1_sl_s4000_l1_f0.25_r3 dqn1_sl_suft.sh BattleZone 3 4000 1 0.25
submit battlezone_dqn1_sl_s4000_l1_f0.25_r4 dqn1_sl_suft.sh BattleZone 4 4000 1 0.25

# Cancelled on 2026-10-15 while pending; game times out on other configs (76)
submit battlezone_dqn1_sl_s4000_l1_f1_r1 dqn1_sl_suft.sh BattleZone 1 4000 1 1
submit battlezone_dqn1_sl_s4000_l1_f1_r2 dqn1_sl_suft.sh BattleZone 2 4000 1 1
submit battlezone_dqn1_sl_s4000_l1_f1_r3 dqn1_sl_suft.sh BattleZone 3 4000 1 1
submit battlezone_dqn1_sl_s4000_l1_f1_r4 dqn1_sl_suft.sh BattleZone 4 4000 1 1
submit defender_dqn1_sl_s4000_l1_f0.25_r1 dqn1_sl_suft.sh Defender 1 4000 1 0.25
submit defender_dqn1_sl_s4000_l1_f0.25_r2 dqn1_sl_suft.sh Defender 2 4000 1 0.25
submit defender_dqn1_sl_s4000_l1_f0.25_r3 dqn1_sl_suft.sh Defender 3 4000 1 0.25
submit defender_dqn1_sl_s4000_l1_f0.25_r4 dqn1_sl_suft.sh Defender 4 4000 1 0.25
submit defender_dqn1_sl_s4000_l1_f1_r1 dqn1_sl_suft.sh Defender 1 4000 1 1
submit defender_dqn1_sl_s4000_l1_f1_r2 dqn1_sl_suft.sh Defender 2 4000 1 1
submit defender_dqn1_sl_s4000_l1_f1_r3 dqn1_sl_suft.sh Defender 3 4000 1 1
submit defender_dqn1_sl_s4000_l1_f1_r4 dqn1_sl_suft.sh Defender 4 4000 1 1
submit demonattack_dqn1_sl_s4000_l1_f0.25_r1 dqn1_sl_suft.sh DemonAttack 1 4000 1 0.25
submit demonattack_dqn1_sl_s4000_l1_f0.25_r2 dqn1_sl_suft.sh DemonAttack 2 4000 1 0.25
submit demonattack_dqn1_sl_s4000_l1_f0.25_r3 dqn1_sl_suft.sh DemonAttack 3 4000 1 0.25
submit demonattack_dqn1_sl_s4000_l1_f0.25_r4 dqn1_sl_suft.sh DemonAttack 4 4000 1 0.25
submit demonattack_dqn1_sl_s4000_l1_f1_r1 dqn1_sl_suft.sh DemonAttack 1 4000 1 1
submit demonattack_dqn1_sl_s4000_l1_f1_r2 dqn1_sl_suft.sh DemonAttack 2 4000 1 1
submit demonattack_dqn1_sl_s4000_l1_f1_r3 dqn1_sl_suft.sh DemonAttack 3 4000 1 1
submit demonattack_dqn1_sl_s4000_l1_f1_r4 dqn1_sl_suft.sh DemonAttack 4 4000 1 1
submit montezumarevenge_dqn1_sl_s4000_l1_f0.25_r1 dqn1_sl_suft.sh MontezumaRevenge 1 4000 1 0.25
submit montezumarevenge_dqn1_sl_s4000_l1_f0.25_r2 dqn1_sl_suft.sh MontezumaRevenge 2 4000 1 0.25
submit montezumarevenge_dqn1_sl_s4000_l1_f0.25_r3 dqn1_sl_suft.sh MontezumaRevenge 3 4000 1 0.25
submit montezumarevenge_dqn1_sl_s4000_l1_f0.25_r4 dqn1_sl_suft.sh MontezumaRevenge 4 4000 1 0.25
submit montezumarevenge_dqn1_sl_s4000_l1_f1_r1 dqn1_sl_suft.sh MontezumaRevenge 1 4000 1 1
submit montezumarevenge_dqn1_sl_s4000_l1_f1_r2 dqn1_sl_suft.sh MontezumaRevenge 2 4000 1 1
submit montezumarevenge_dqn1_sl_s4000_l1_f1_r3 dqn1_sl_suft.sh MontezumaRevenge 3 4000 1 1
submit montezumarevenge_dqn1_sl_s4000_l1_f1_r4 dqn1_sl_suft.sh MontezumaRevenge 4 4000 1 1
submit phoenix_dqn1_sl_s4000_l1_f0.25_r1 dqn1_sl_suft.sh Phoenix 1 4000 1 0.25
submit phoenix_dqn1_sl_s4000_l1_f0.25_r2 dqn1_sl_suft.sh Phoenix 2 4000 1 0.25
submit phoenix_dqn1_sl_s4000_l1_f0.25_r3 dqn1_sl_suft.sh Phoenix 3 4000 1 0.25
submit phoenix_dqn1_sl_s4000_l1_f0.25_r4 dqn1_sl_suft.sh Phoenix 4 4000 1 0.25
submit phoenix_dqn1_sl_s4000_l1_f1_r1 dqn1_sl_suft.sh Phoenix 1 4000 1 1
submit phoenix_dqn1_sl_s4000_l1_f1_r2 dqn1_sl_suft.sh Phoenix 2 4000 1 1
submit phoenix_dqn1_sl_s4000_l1_f1_r3 dqn1_sl_suft.sh Phoenix 3 4000 1 1
submit phoenix_dqn1_sl_s4000_l1_f1_r4 dqn1_sl_suft.sh Phoenix 4 4000 1 1
submit pitfall_dqn1_sl_s4000_l1_f0.25_r1 dqn1_sl_suft.sh Pitfall 1 4000 1 0.25
submit pitfall_dqn1_sl_s4000_l1_f0.25_r2 dqn1_sl_suft.sh Pitfall 2 4000 1 0.25
submit pitfall_dqn1_sl_s4000_l1_f0.25_r3 dqn1_sl_suft.sh Pitfall 3 4000 1 0.25
submit pitfall_dqn1_sl_s4000_l1_f0.25_r4 dqn1_sl_suft.sh Pitfall 4 4000 1 0.25
submit pitfall_dqn1_sl_s4000_l1_f1_r1 dqn1_sl_suft.sh Pitfall 1 4000 1 1
submit pitfall_dqn1_sl_s4000_l1_f1_r2 dqn1_sl_suft.sh Pitfall 2 4000 1 1
submit pitfall_dqn1_sl_s4000_l1_f1_r3 dqn1_sl_suft.sh Pitfall 3 4000 1 1
submit pitfall_dqn1_sl_s4000_l1_f1_r4 dqn1_sl_suft.sh Pitfall 4 4000 1 1
submit solaris_dqn1_sl_s4000_l1_f0.25_r1 dqn1_sl_suft.sh Solaris 1 4000 1 0.25
submit solaris_dqn1_sl_s4000_l1_f0.25_r2 dqn1_sl_suft.sh Solaris 2 4000 1 0.25
submit solaris_dqn1_sl_s4000_l1_f0.25_r3 dqn1_sl_suft.sh Solaris 3 4000 1 0.25
submit solaris_dqn1_sl_s4000_l1_f0.25_r4 dqn1_sl_suft.sh Solaris 4 4000 1 0.25
submit solaris_dqn1_sl_s4000_l1_f1_r1 dqn1_sl_suft.sh Solaris 1 4000 1 1
submit solaris_dqn1_sl_s4000_l1_f1_r2 dqn1_sl_suft.sh Solaris 2 4000 1 1
submit solaris_dqn1_sl_s4000_l1_f1_r3 dqn1_sl_suft.sh Solaris 3 4000 1 1
submit solaris_dqn1_sl_s4000_l1_f1_r4 dqn1_sl_suft.sh Solaris 4 4000 1 1
submit tennis_dqn1_sl_s4000_l1_f0.25_r1 dqn1_sl_suft.sh Tennis 1 4000 1 0.25
submit tennis_dqn1_sl_s4000_l1_f0.25_r2 dqn1_sl_suft.sh Tennis 2 4000 1 0.25
submit tennis_dqn1_sl_s4000_l1_f0.25_r3 dqn1_sl_suft.sh Tennis 3 4000 1 0.25
submit tennis_dqn1_sl_s4000_l1_f0.25_r4 dqn1_sl_suft.sh Tennis 4 4000 1 0.25
submit tennis_dqn1_sl_s4000_l1_f1_r1 dqn1_sl_suft.sh Tennis 1 4000 1 1
submit tennis_dqn1_sl_s4000_l1_f1_r2 dqn1_sl_suft.sh Tennis 2 4000 1 1
submit tennis_dqn1_sl_s4000_l1_f1_r3 dqn1_sl_suft.sh Tennis 3 4000 1 1
submit tennis_dqn1_sl_s4000_l1_f1_r4 dqn1_sl_suft.sh Tennis 4 4000 1 1
submit tutankham_dqn1_sl_s4000_l1_f0.25_r1 dqn1_sl_suft.sh Tutankham 1 4000 1 0.25
submit tutankham_dqn1_sl_s4000_l1_f0.25_r2 dqn1_sl_suft.sh Tutankham 2 4000 1 0.25
submit tutankham_dqn1_sl_s4000_l1_f0.25_r3 dqn1_sl_suft.sh Tutankham 3 4000 1 0.25
submit tutankham_dqn1_sl_s4000_l1_f0.25_r4 dqn1_sl_suft.sh Tutankham 4 4000 1 0.25
submit tutankham_dqn1_sl_s4000_l1_f1_r1 dqn1_sl_suft.sh Tutankham 1 4000 1 1
submit tutankham_dqn1_sl_s4000_l1_f1_r2 dqn1_sl_suft.sh Tutankham 2 4000 1 1
submit tutankham_dqn1_sl_s4000_l1_f1_r3 dqn1_sl_suft.sh Tutankham 3 4000 1 1
submit tutankham_dqn1_sl_s4000_l1_f1_r4 dqn1_sl_suft.sh Tutankham 4 4000 1 1
submit videopinball_dqn1_sl_s4000_l1_f0.25_r1 dqn1_sl_suft.sh VideoPinball 1 4000 1 0.25
submit videopinball_dqn1_sl_s4000_l1_f0.25_r2 dqn1_sl_suft.sh VideoPinball 2 4000 1 0.25
submit videopinball_dqn1_sl_s4000_l1_f0.25_r3 dqn1_sl_suft.sh VideoPinball 3 4000 1 0.25
submit videopinball_dqn1_sl_s4000_l1_f0.25_r4 dqn1_sl_suft.sh VideoPinball 4 4000 1 0.25
submit videopinball_dqn1_sl_s4000_l1_f1_r1 dqn1_sl_suft.sh VideoPinball 1 4000 1 1
submit videopinball_dqn1_sl_s4000_l1_f1_r2 dqn1_sl_suft.sh VideoPinball 2 4000 1 1
submit videopinball_dqn1_sl_s4000_l1_f1_r3 dqn1_sl_suft.sh VideoPinball 3 4000 1 1
submit videopinball_dqn1_sl_s4000_l1_f1_r4 dqn1_sl_suft.sh VideoPinball 4 4000 1 1
