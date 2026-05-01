
set -euo pipefail

source "$(dirname "$0")/_common.sh"

SKIP_POSTHOC="${SKIP_POSTHOC:-0}"
POSTHOC_EPS="${POSTHOC_EPS:-30}"
POSTHOC_STEPS="${POSTHOC_STEPS:-3000}"


echo "==================================================================="
echo " FAME + TSDM analysis  |  seq=${SEQ}  seed=${SEED}"
echo "==================================================================="

missing=0
for det in oracle swoks implicit hybrid; do
    pkl="$(expected_pkl "$det")"
    if [ -f "$pkl" ]; then
        echo "  [OK]  $det  $(basename "$pkl")"
    else
        echo "  [MISSING]  $det  -- run kaggle_experiments/run_${det}.sh first"
        missing=1
    fi
done

if [ "$missing" = "1" ]; then
    echo
    echo "ERROR: one or more training runs are incomplete.  Run the missing" >&2
    echo "       detector scripts above and re-run this analysis." >&2
    exit 1
fi

echo
echo "All training artefacts found -- running experiment.py ..."
echo

ANALYSIS_ARGS=(
    --results_dir "$RESULTS_DIR"
    --models_dir  "$MODELS_DIR"
    --seq         "$SEQ"
    --seeds       "$SEED"
    --tolerance      60000
    --tolerance_sweep 5000 15000 60000 120000 240000 500000
    --smooth      5000
    --tag         "kaggle_seq${SEQ}_seed${SEED}"
    --device      "cuda:${GPU}"
)

if [ "$SKIP_POSTHOC" != "1" ]; then
    ANALYSIS_ARGS+=(
        --eval-posthoc
        --posthoc_episodes  "$POSTHOC_EPS"
        --posthoc_max_steps "$POSTHOC_STEPS"
    )
    echo "  post-hoc rollouts: ${POSTHOC_EPS} episodes / ${POSTHOC_STEPS} max steps"
else
    echo "  SKIP_POSTHOC=1 -- skipping post-hoc policy evaluation"
fi

(cd "$REPO_DIR" && "$PY" experiment.py "${ANALYSIS_ARGS[@]}" \
    2>&1 | tee "$LOGS_DIR/analysis_seq${SEQ}_seed${SEED}.log")

OUT_DIR="${RESULTS_DIR}/experiment_kaggle_seq${SEQ}_seed${SEED}"
echo
echo "==================================================================="
echo " Analysis complete.  Artefacts written to:"
echo "   $OUT_DIR/"
echo
echo "   Key files:"
echo "   • paper_table.tex      (paste into LaTeX)"
echo "   • report.md            (narrative summary)"
echo "   • plots/*.png          (13 visualisations)"
echo "   • metrics_per_seed.csv"
echo "==================================================================="
