#!/bin/bash
# run_analysis.sh -- Run the full experiment.py analysis over all 4 detectors.
#
# This script is meant to run AFTER all four training scripts have
# completed (i.e. after oracle, swoks, implicit, and hybrid all have
# their results pkl).  It will refuse to run if any pkl is missing,
# printing exactly which detector still needs training.
#
# What experiment.py produces
# ---------------------------
#   paper_table.tex          LaTeX table (ready to paste into your paper)
#   paper_table.txt          Plain-text version for quick inspection
#   metrics_per_seed.csv     Per-seed raw numbers
#   metrics_aggregate.csv    Mean ± SE aggregated across seeds
#   report.md                Narrative Markdown report
#   summaries.pkl            All RunSummary objects (for programmatic access)
#   plots/
#     learning_curves.png         Smoothed return traces, all 4 modes
#     detection_timeline.png      Detected vs oracle boundaries over time
#     detection_delay_hist.png    Histogram of detection delays
#     detection_quality.png       F1 / precision / recall bar chart
#     per_task_auc.png            Per-task AUC (normalised) by mode
#     forgetting_heatmap.png      Forgetting matrix (task × mode)
#     warmup_flag_ratio.png       Fraction of steps in warmup (BC-reg) phase
#     hybrid_reason_pie.png       Breakdown of hybrid fire reasons
#     ratio_to_oracle.png         AvgPerf / oracle AvgPerf ratio
#     tolerance_sweep.png         F1 vs detection-tolerance window
#     hybrid_vs_implicit_timeline.png  Side-by-side p-value traces
#     pvalue_trace_swoks.png      SWOKS p-value over training
#     pvalue_trace_implicit.png   Implicit p-value over training
#
# Usage:
#   bash kaggle_experiments/run_analysis.sh
#   SKIP_POSTHOC=1 bash kaggle_experiments/run_analysis.sh  # skip rollouts

set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"

SKIP_POSTHOC="${SKIP_POSTHOC:-0}"
POSTHOC_EPS="${POSTHOC_EPS:-30}"
POSTHOC_STEPS="${POSTHOC_STEPS:-3000}"

# ---------------------------------------------------------------------------
# Pre-flight: confirm all 4 detectors have their results pkl.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# experiment.py analysis
# ---------------------------------------------------------------------------
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
