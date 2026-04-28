#!/bin/bash
# run_analysis_local.sh -- Run the full experiment.py analysis locally.
#
# Drop your four pkl files into RESULTS_DIR (default: results/) then run:
#
#   bash run_analysis_local.sh
#
# Expected pkl names (seq=0, seed=1, full paper schedule):
#   FAME_oracle_steps_3500000_switch_500000_seq_0_warmstep_50000_lambda_1.0_seed_1_returns.pkl
#   FAME_swoks_steps_3500000_switch_500000_seq_0_warmstep_50000_lambda_1.0_seed_1_returns.pkl
#   FAME_implicit_steps_3500000_switch_500000_seq_0_warmstep_50000_lambda_1.0_seed_1_returns.pkl
#   FAME_hybrid_steps_3500000_switch_500000_seq_0_warmstep_50000_lambda_1.0_seed_1_returns.pkl
#
# Env-var overrides (all optional):
#   RESULTS_DIR   path that contains the four pkl files  (default: results)
#   MODELS_DIR    path with MetaFinal.pt weights         (default: models)
#   SEQ           sequence id used during training       (default: 0)
#   SEED          seed used during training              (default: 1)
#   TSTEPS        total steps used during training       (default: 3500000)
#   SWITCH        steps-per-task used during training    (default: 500000)
#   WARMSTEP      warmstep used during training          (default: 50000)
#   LAMBDA        lambda_reg used during training        (default: 1.0)
#   SKIP_POSTHOC  set to 1 to skip post-hoc rollouts    (default: 1)
#   PY            python binary                          (default: ../FAMEenv/bin/python)

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PY="${PY:-../FAMEenv/bin/python}"

RESULTS_DIR="${RESULTS_DIR:-results}"
MODELS_DIR="${MODELS_DIR:-models}"
LOGS_DIR="${LOGS_DIR:-logs}"

SEQ="${SEQ:-0}"
SEED="${SEED:-1}"

# Must match the values used when training (they appear in the pkl filename).
TSTEPS="${TSTEPS:-3500000}"
SWITCH="${SWITCH:-500000}"
WARMSTEP="${WARMSTEP:-50000}"
LAMBDA="${LAMBDA:-1.0}"

# Post-hoc rollouts require MetaFinal.pt files saved by FAME.py --save-model.
# Default is off because Kaggle pkl-only exports usually don't include them.
SKIP_POSTHOC="${SKIP_POSTHOC:-1}"
POSTHOC_EPS="${POSTHOC_EPS:-30}"
POSTHOC_STEPS="${POSTHOC_STEPS:-3000}"

mkdir -p "$LOGS_DIR"

# ---------------------------------------------------------------------------
# Helper: expected pkl path for a given detector
# ---------------------------------------------------------------------------
expected_pkl() {
    local det="$1"
    echo "${RESULTS_DIR}/FAME_${det}_steps_${TSTEPS}_switch_${SWITCH}_seq_${SEQ}_warmstep_${WARMSTEP}_lambda_${LAMBDA}_seed_${SEED}_returns.pkl"
}

# ---------------------------------------------------------------------------
# Pre-flight: all four pkl files must exist
# ---------------------------------------------------------------------------
echo "==================================================================="
echo " FAME + TSDM local analysis"
echo "   seq=${SEQ}  seed=${SEED}  steps=${TSTEPS}  switch=${SWITCH}"
echo "   results dir : ${RESULTS_DIR}"
echo "   models dir  : ${MODELS_DIR}"
echo "==================================================================="

missing=0
for det in oracle swoks implicit hybrid; do
    pkl="$(expected_pkl "$det")"
    if [ -f "$pkl" ]; then
        echo "  [OK]      $det  ->  $(basename "$pkl")"
    else
        echo "  [MISSING] $det  ->  $pkl"
        missing=1
    fi
done

if [ "$missing" = "1" ]; then
    echo
    echo "ERROR: one or more pkl files are missing." >&2
    echo "       Place all four files in: ${RESULTS_DIR}/" >&2
    echo "       If your files use different hyperparameter values, override" >&2
    echo "       TSTEPS / SWITCH / WARMSTEP / LAMBDA / SEQ / SEED." >&2
    echo "       Example: SEED=2 bash run_analysis_local.sh" >&2
    exit 1
fi

echo
echo "All pkl files found -- running experiment.py ..."
echo

# ---------------------------------------------------------------------------
# Build experiment.py argument list
# ---------------------------------------------------------------------------
ANALYSIS_ARGS=(
    --results_dir "$RESULTS_DIR"
    --models_dir  "$MODELS_DIR"
    --seq         "$SEQ"
    --seeds       "$SEED"
    --tolerance      60000
    --tolerance_sweep 5000 15000 60000 120000 240000 500000
    --smooth      5000
    --tag         "local_seq${SEQ}_seed${SEED}"
    --device      "cpu"
)

if [ "$SKIP_POSTHOC" = "1" ]; then
    echo "  post-hoc rollouts : SKIPPED (set SKIP_POSTHOC=0 to enable)"
    echo "  (requires MetaFinal.pt files in ${MODELS_DIR}/)"
else
    echo "  post-hoc rollouts : ${POSTHOC_EPS} episodes / ${POSTHOC_STEPS} max steps"
    ANALYSIS_ARGS+=(
        --eval-posthoc
        --posthoc_episodes  "$POSTHOC_EPS"
        --posthoc_max_steps "$POSTHOC_STEPS"
    )
fi

echo

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
"$PY" experiment.py "${ANALYSIS_ARGS[@]}" \
    2>&1 | tee "$LOGS_DIR/analysis_local_seq${SEQ}_seed${SEED}.log"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
OUT_DIR="${RESULTS_DIR}/experiment_local_seq${SEQ}_seed${SEED}"
echo
echo "==================================================================="
echo " Analysis complete."
echo "   Output dir : ${OUT_DIR}/"
echo
echo "   Key files:"
echo "     paper_table.tex        <- paste into LaTeX"
echo "     paper_table.txt        <- plain-text quick view"
echo "     report.md              <- narrative summary"
echo "     metrics_per_seed.csv"
echo "     metrics_aggregate.csv"
echo "     plots/                 <- 13 PNGs"
echo "==================================================================="
