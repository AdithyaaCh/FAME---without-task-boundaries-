#!/bin/bash
# _common.sh -- Shared configuration sourced by all kaggle_experiments scripts.
# Source this file, never run it directly.

# ---------------------------------------------------------------------------
# Paths (Kaggle /kaggle/working is the only persistent volume)
# ---------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"

RESULTS_DIR="${RESULTS_DIR:-/kaggle/working/results}"
MODELS_DIR="${MODELS_DIR:-/kaggle/working/models}"
LOGS_DIR="${LOGS_DIR:-/kaggle/working/logs}"
CKPT_DIR="${CKPT_DIR:-/kaggle/working/checkpoints}"

mkdir -p "$RESULTS_DIR" "$MODELS_DIR" "$LOGS_DIR" "$CKPT_DIR"

SEQ="${SEQ:-0}"
SEED="${SEED:-1}"
GPU="${GPU:-0}"

TSTEPS="${TSTEPS:-3500000}"   # paper: 7 tasks × 500k steps
SWITCH="${SWITCH:-500000}"

# FAME paper hyperparameters (Appendix F.1)
WARMSTEP=50000
LAMBDA_REG=1.0
EPOCH_META=200
LR1=1e-3
LR2=1e-5
SIZE_FAST2META=12000
DETECTION_STEP=600

# Checkpoint every 50k steps (~6 min on T4) so a timeout loses < 50k work.
CKPT_EVERY="${CKPT_EVERY:-50000}"

# ---------------------------------------------------------------------------
# Bonferroni-corrected detector hyperparameters
# (3.5M / 240 ≈ 14.5k tests → α = 7e-5, rounded conservatively to 1e-3)
# ---------------------------------------------------------------------------
COMMON_DET_ARGS=(
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


expected_pkl() {
    local det="$1"
    echo "${RESULTS_DIR}/FAME_${det}_steps_${TSTEPS}_switch_${SWITCH}_seq_${SEQ}_warmstep_${WARMSTEP}_lambda_${LAMBDA_REG}_seed_${SEED}_returns.pkl"
}

run_detector() {
    local det="$1"; shift
    local extra=("$@")   

    local pkl
    pkl="$(expected_pkl "$det")"

    if [ -f "$pkl" ]; then
        echo ">>> [SKIP] $det -- results pkl already exists: $(basename "$pkl")"
        return 0
    fi

    echo ">>> [START] $det  seq=${SEQ}  seed=${SEED}  $(date '+%H:%M:%S')"

    (cd "$REPO_DIR" && "$PY" FAME.py \
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
        --models_dir  "$MODELS_DIR" \
        --checkpoint_every "$CKPT_EVERY" \
        --checkpoint_dir   "$CKPT_DIR" \
        2>&1 | tee "$LOGS_DIR/${det}_seq${SEQ}_seed${SEED}.log")

    # Verify the results pkl was actually written.
    if [ ! -f "$pkl" ]; then
        echo "ERROR: training completed but pkl not found: $pkl" >&2
        echo "       Check log: $LOGS_DIR/${det}_seq${SEQ}_seed${SEED}.log" >&2
        exit 1
    fi

    echo ">>> [DONE]  $det  $(date '+%H:%M:%S')"
}
