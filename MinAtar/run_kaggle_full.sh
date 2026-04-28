#!/bin/bash
# run_kaggle_full.sh -- Full FAME + TSDM sweep on a single Kaggle GPU session.
#
# Runs all 4 detectors (oracle / swoks / implicit / hybrid) for ONE seed
# SEQUENTIALLY, then invokes experiment.py to produce the full analysis
# bundle (LaTeX table, 13 plots, markdown report, CSVs, pickle).
#
# CHECKPOINT DESIGN
# -----------------
# FAME.py emits a deterministic results pickle on completion:
#   $RESULTS_DIR/FAME_{det}_steps_{TSTEPS}_switch_{SWITCH}_seq_{SEQ}
#               _warmstep_{WARMSTEP}_lambda_{LAMBDA}_seed_{SEED}_returns.pkl
#
# Before starting each detector this script checks for that file.  If it
# already exists the training phase is skipped entirely.  Re-run the
# script after a Kaggle timeout -- only the unfinished detector(s) will
# re-execute; all previously completed ones are fast-skipped.
#
# EXECUTION ORDER (why sequential, not parallel)
# -----------------------------------------------
# The GPU script runs detectors in parallel because it targets a
# multi-hour server.  On a single Kaggle GPU, parallel execution
# would OOM and thrash memory.  Sequential keeps peak VRAM low and
# lets each run reach completion before the next begins.
#
# TYPICAL RUNTIME (T4 / P100 GPU)
#   oracle   ~40-50 min    (no detector overhead, pure RL)
#   swoks    ~50-65 min    (KS tests every 240 steps)
#   implicit ~55-70 min    (TSN forward passes every 240 steps)
#   hybrid   ~55-70 min    (cascade; early fire saves a few steps)
#   analysis ~5-10 min     (experiment.py, 30 posthoc eps per game)
#   ─────────────────────
#   TOTAL    ~3.5 – 4.5 h  well within a 9 h Kaggle session
#
# Env vars (all optional):
#   SEQ       -- sequence id, default 0
#   SEED      -- single seed, default 1
#   GPU       -- CUDA device id, default 0
#   TSTEPS    -- total env steps, default 3500000 (paper protocol)
#   SWITCH    -- steps per task, default 500000 (7 tasks)
#   DETECTORS -- space-separated subset, default "oracle swoks implicit hybrid"
#   SKIP_ANALYSIS=1 -- skip experiment.py even if all training is done
#
# Usage:
#   python -c "import subprocess; subprocess.run(['bash','run_kaggle_full.sh'])"
#   # or directly if bash is available:
#   bash run_kaggle_full.sh
#   SEQ=2 SEED=2 bash run_kaggle_full.sh     # different sequence / seed
#   DETECTORS="oracle swoks" bash run_kaggle_full.sh  # subset

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PY="${PY:-python}"
SEQ="${SEQ:-0}"
SEED="${SEED:-1}"
GPU="${GPU:-0}"
TSTEPS="${TSTEPS:-3500000}"
SWITCH="${SWITCH:-500000}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"
DETECTORS=(${DETECTORS:-oracle swoks implicit hybrid})

# FAME paper hyperparameters (Appendix F.1)
WARMSTEP=50000
LAMBDA_REG=1.0
EPOCH_META=200
LR1=1e-3
LR2=1e-5
SIZE_FAST2META=12000
DETECTION_STEP=600

# Kaggle persistent output directories
RESULTS_DIR="/kaggle/working/results"
MODELS_DIR="/kaggle/working/models"
LOGS_DIR="/kaggle/working/logs"

mkdir -p "$RESULTS_DIR" "$MODELS_DIR" "$LOGS_DIR"

# ---------------------------------------------------------------------------
# Detector hyperparameters (Bonferroni-corrected for 3.5M / 240 ~ 14.5k tests)
# ---------------------------------------------------------------------------
COMMON_DET=(
    --swoks_L_D 1200 --swoks_L_W 30
    --swoks_alpha 1e-3 --swoks_beta 2.0
    --swoks_stable_phase 36000 --swoks_interval 240
    --swoks_warmup 5000
    --swoks_snapshot 1 --swoks_snapshot_interval 6000
    --imp_L_D 1200 --imp_alpha 1e-3
    --imp_stable_phase 36000 --imp_interval 240
    --imp_warmup 5000 --imp_lr 1e-4
    --imp_update_every 16
    --hyb_tau_imp_loose 1e-2
    --hyb_tau_imp_strict 1e-6
    --hyb_tau_stat_strict 1e-3
    --hyb_tau_combined 15.0
    --hyb_horizon 480
    --hyb_persistence 2
)

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

# Returns the expected results pkl path for a given detector.
expected_pkl() {
    local det="$1"
    echo "${RESULTS_DIR}/FAME_${det}_steps_${TSTEPS}_switch_${SWITCH}_seq_${SEQ}_warmstep_${WARMSTEP}_lambda_${LAMBDA_REG}_seed_${SEED}_returns.pkl"
}

# Prints a green-ish status line.
status() { echo ">>> $*"; }

# ---------------------------------------------------------------------------
# Training phase (checkpoint-aware)
# ---------------------------------------------------------------------------
echo "==================================================================="
echo " FAME + TSDM  |  seq=${SEQ}  seed=${SEED}  gpu=${GPU}"
echo "   detectors : ${DETECTORS[*]}"
echo "   schedule  : ${TSTEPS} steps / ${SWITCH} per task"
echo " Checkpoint dir: ${RESULTS_DIR}"
echo "==================================================================="

for det in "${DETECTORS[@]}"; do
    pkl="$(expected_pkl "$det")"

    if [ -f "$pkl" ]; then
        status "SKIP $det -- checkpoint found: $(basename "$pkl")"
        continue
    fi

    status "START $det  ($(date '+%H:%M:%S'))"

    if [ "$det" = "oracle" ]; then
        extra=()
    else
        extra=("${COMMON_DET[@]}")
    fi

    # Run training.  tee to a per-detector log; also echoes to notebook cell.
    "$PY" FAME.py \
        --detector "$det" \
        ${extra[@]+"${extra[@]}"} \
        --lr1 "$LR1" --lr2 "$LR2" \
        --size_fast2meta "$SIZE_FAST2META" \
        --detection_step "$DETECTION_STEP" \
        --warmstep "$WARMSTEP" \
        --lambda_reg "$LAMBDA_REG" \
        --epoch_meta "$EPOCH_META" \
        --t-steps "$TSTEPS" --switch "$SWITCH" \
        --seq "$SEQ" --seed "$SEED" --gpu "$GPU" \
        --save --save-model \
        --results_dir "$RESULTS_DIR" \
        --models_dir "$MODELS_DIR" \
        2>&1 | tee "$LOGS_DIR/${det}_seq${SEQ}_seed${SEED}.log"

    # Verify the checkpoint was actually written before moving on.
    if [ ! -f "$pkl" ]; then
        echo "ERROR: expected pkl not found after training: $pkl" >&2
        echo "       Check the log: $LOGS_DIR/${det}_seq${SEQ}_seed${SEED}.log" >&2
        exit 1
    fi

    status "DONE  $det  ($(date '+%H:%M:%S'))"
    echo
done

echo "==================================================================="
echo " All training phases finished at $(date)"
echo "==================================================================="

# ---------------------------------------------------------------------------
# Analysis phase (only runs when ALL selected detectors have their pkl)
# ---------------------------------------------------------------------------
if [ "${SKIP_ANALYSIS}" = "1" ]; then
    status "SKIP_ANALYSIS=1 -- skipping experiment.py"
    exit 0
fi

# Confirm every requested detector is checkpointed before analysis.
missing=0
for det in "${DETECTORS[@]}"; do
    pkl="$(expected_pkl "$det")"
    if [ ! -f "$pkl" ]; then
        echo "WARNING: missing pkl for $det -- analysis skipped" >&2
        missing=1
    fi
done
if [ "$missing" = "1" ]; then
    echo "Re-run this script to complete the missing training runs." >&2
    exit 1
fi

status "Running experiment.py analysis  ($(date '+%H:%M:%S'))"

"$PY" experiment.py \
    --results_dir "$RESULTS_DIR" \
    --models_dir  "$MODELS_DIR" \
    --seq         "$SEQ" \
    --seeds       "$SEED" \
    --tolerance      60000 \
    --tolerance_sweep 5000 15000 60000 120000 240000 500000 \
    --smooth      5000 \
    --tag         "kaggle_seq${SEQ}_seed${SEED}" \
    --eval-posthoc \
    --posthoc_episodes  30 \
    --posthoc_max_steps 3000 \
    --device "cuda:${GPU}"

echo
status "All artifacts written to ${RESULTS_DIR}/experiment_kaggle_seq${SEQ}_seed${SEED}/"
echo "==================================================================="
echo " Run complete at $(date)"
echo "==================================================================="
