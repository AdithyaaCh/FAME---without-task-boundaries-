#!/bin/bash
# run_implicit.sh -- Train FAME with the implicit (TSN) detector on Kaggle.
#
# Implicit detector (Approach 2): a dual-head Task-Signature Network
# (TSN) predicts both reward and forward dynamics from the fast
# learner's latent feature phi.  A Welch t-test on the window of
# standardised prediction errors detects distribution shift.
#
# The TSN is trained online alongside the policy, so its weights
# evolve throughout the run.  The checkpoint therefore includes the
# full TSN model + optimiser so training resumes exactly from where
# the kernel timed out -- the network picks up its learning curve
# without any cold-start gap.
#
# CHECKPOINT / RESUME
# -------------------
# Mid-training checkpoints include:
#   • Fast / Meta / Target network weights
#   • TSN (implicit) network weights + Adam optimiser state
#   • Welford running stats (per-head error normalisation)
#   • Score window + replay buffer for the TSN
#   • Replay buffers (fast, fast2meta, meta)
#   • All loop bookkeeping (step, gameid, boundaries, flag history)
#   • Torch / NumPy / Python RNG states
#
# On resume FAME.py reads the checkpoint file, restores everything
# listed above, and re-enters the training loop at the saved step.
# The TSN continues training from its saved weights with correct
# error statistics -- no re-calibration period needed.
#
# Usage:
#   bash kaggle_experiments/run_implicit.sh
#   SEQ=1 SEED=2 bash kaggle_experiments/run_implicit.sh
#   CKPT_EVERY=25000 bash kaggle_experiments/run_implicit.sh  # finer checkpoints

set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"

echo "==================================================================="
echo " FAME implicit (TSN)  |  seq=${SEQ}  seed=${SEED}  gpu=${GPU}"
echo " schedule: ${TSTEPS} steps / ${SWITCH} per task"
echo " imp_alpha=1e-3  L_D=1200  lr=1e-4  update_every=16"
echo " checkpoint every ${CKPT_EVERY} steps -> ${CKPT_DIR}"
echo "==================================================================="

run_detector implicit "${COMMON_DET_ARGS[@]}"

echo "==================================================================="
echo " implicit training complete -- pkl at: $(expected_pkl implicit)"
echo "==================================================================="
