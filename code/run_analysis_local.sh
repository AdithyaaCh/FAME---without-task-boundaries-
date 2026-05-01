set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-../FAMEenv/bin/python}"

RESULTS_DIR="${RESULTS_DIR:-results}"
MODELS_DIR="${MODELS_DIR:-models}"
LOGS_DIR="${LOGS_DIR:-logs}"

SEQ="${SEQ:-0}"
SEED="${SEED:-1}"

TSTEPS="${TSTEPS:-3500000}"
SWITCH="${SWITCH:-500000}"
WARMSTEP="${WARMSTEP:-50000}"
LAMBDA="${LAMBDA:-1.0}"


SKIP_POSTHOC="${SKIP_POSTHOC:-1}"
POSTHOC_EPS="${POSTHOC_EPS:-30}"
POSTHOC_STEPS="${POSTHOC_STEPS:-3000}"

mkdir -p "$LOGS_DIR"

expected_pkl() {
    local det="$1"
    echo "${RESULTS_DIR}/FAME_${det}_steps_${TSTEPS}_switch_${SWITCH}_seq_${SEQ}_warmstep_${WARMSTEP}_lambda_${LAMBDA}_seed_${SEED}_returns.pkl"
}


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

"$PY" experiment.py "${ANALYSIS_ARGS[@]}" \
    2>&1 | tee "$LOGS_DIR/analysis_local_seq${SEQ}_seed${SEED}.log"


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
