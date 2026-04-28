#!/bin/bash
# run_swoks.sh -- Train FAME with the SWOKS detector on Kaggle.
#
# SWOKS (Sliced-Wasserstein + KS test) is Approach 1: a purely
# statistical detector that operates on the latent action-reward
# feature distribution.  It is the most reliable of the three
# boundary-free methods and produces the lowest false-positive rate.
#
# CHECKPOINT / RESUME
# -------------------
# Every CKPT_EVERY steps the full detector state (ring buffers,
# SWD history, KS bookkeeping) is checkpointed alongside the model
# weights.  A timed-out kernel can resume from the last checkpoint;
# FAME.py detects the file automatically on re-run.
#
# Usage:
#   bash kaggle_experiments/run_swoks.sh
#   SEQ=1 SEED=2 bash kaggle_experiments/run_swoks.sh

set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"

echo "==================================================================="
echo " FAME swoks   |  seq=${SEQ}  seed=${SEED}  gpu=${GPU}"
echo " schedule: ${TSTEPS} steps / ${SWITCH} per task"
echo " alpha=1e-3  beta=2.0  L_D=1200  L_W=30"
echo "==================================================================="

run_detector swoks "${COMMON_DET_ARGS[@]}"

echo "==================================================================="
echo " swoks training complete -- pkl at: $(expected_pkl swoks)"
echo "==================================================================="
