#!/bin/bash
# run_oracle.sh -- Train FAME with the oracle detector on Kaggle.
#
# Oracle mode: the agent is told the true task boundary at every switch.
# This gives the upper bound on FAME performance and serves as the
# baseline for Forward Transfer (FT) computation in experiment.py.
#
# CHECKPOINT / RESUME
# -------------------
# Every CKPT_EVERY steps (default 50k) a checkpoint is written to
# $CKPT_DIR.  Re-running this script after a Kaggle timeout resumes
# exactly from the last checkpoint -- no steps are redone.
# On clean completion the checkpoint is deleted and the results pkl
# is left as the permanent artefact.
#
# Usage:
#   bash kaggle_experiments/run_oracle.sh
#   SEQ=1 SEED=2 bash kaggle_experiments/run_oracle.sh

set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"

echo "==================================================================="
echo " FAME oracle  |  seq=${SEQ}  seed=${SEED}  gpu=${GPU}"
echo " schedule: ${TSTEPS} steps / ${SWITCH} per task"
echo "==================================================================="

# Oracle mode takes no detector hyperparameters.
run_detector oracle

echo "==================================================================="
echo " oracle training complete -- pkl at: $(expected_pkl oracle)"
echo "==================================================================="
