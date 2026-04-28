#!/bin/bash
# run_hybrid.sh -- Train FAME with the hybrid (cascade + log-evidence) detector.
#
# Hybrid detector (Approach 3): a 3-state FSM over two p-values.
#   NEUTRAL -> SUSPECT  when implicit p_imp < tau_imp_loose (fast alert)
#   SUSPECT -> CONFIRMED when any of:
#     • p_stat  < tau_stat_strict           (SWOKS confirms)
#     • p_imp   < tau_imp_strict            (implicit extreme)
#     • -log(p_imp) - log(p_stat) > tau_combined  (joint evidence)
# Plus two parallel failsafes from NEUTRAL for extreme signals.
#
# Both inner detectors (implicit TSN + SWOKS) run every step; the
# hybrid FSM reads their p-values but overrides their self-fire signals.
# On a confirmed fire it resets both inner detectors so their stable-
# phase guard protects us from immediately re-firing.
#
# CHECKPOINT / RESUME
# -------------------
# The checkpoint captures the full hybrid FSM state (state, streaks,
# last_fire_ts, suspect_entered_at) PLUS the complete state_dict of
# both inner detectors, so the TSN, the SWOKS ring buffers, and the
# FSM all resume exactly from where the kernel timed out.
#
# Usage:
#   bash kaggle_experiments/run_hybrid.sh
#   SEQ=1 SEED=2 bash kaggle_experiments/run_hybrid.sh

set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"

echo "==================================================================="
echo " FAME hybrid (cascade + log-evidence)  |  seq=${SEQ}  seed=${SEED}"
echo " schedule: ${TSTEPS} steps / ${SWITCH} per task"
echo " tau_imp_loose=1e-2  tau_combined=15.0  horizon=480"
echo " checkpoint every ${CKPT_EVERY} steps -> ${CKPT_DIR}"
echo "==================================================================="

run_detector hybrid "${COMMON_DET_ARGS[@]}"

echo "==================================================================="
echo " hybrid training complete -- pkl at: $(expected_pkl hybrid)"
echo "==================================================================="
