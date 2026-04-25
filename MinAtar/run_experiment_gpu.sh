#!/bin/bash
# run_experiment_gpu.sh -- paper-matching FAME + TSDM sweep on GPU.
#
# Reproduces the MinAtar benchmarking protocol from the FAME paper (Sun
# et al., ICLR 2026, Table 1) with the full 3.5M-step / 500k-per-task
# schedule, and extends it with the three task-shift detectors proposed
# in our term paper.  Produces the complete analysis bundle via
# experiment.py at the end, including LaTeX table + plot gallery +
# markdown report.
#
# Paper's 10-sequence x 3-seed protocol is supported natively -- set
# SEQS and SEEDS to pick any subset.  Defaults run the first sequence
# with 3 seeds (30x cheaper than the full sweep, still statistically
# meaningful for a class presentation).
#
# Env vars:
#   SEQS      -- space-separated seq ids, e.g. "0 1 2" (default "0")
#   SEEDS     -- seeds per sequence (default "1 2 3")
#   DETECTORS -- which methods to run (default all four)
#   GPU       -- GPU id (default 0)
#   FULL=1    -- convenience flag: SEQS="0..9" SEEDS="1 2 3" (30 seeds total)
#
# Usage:
#   ./run_experiment_gpu.sh                         # 3 seeds x 1 seq
#   FULL=1 ./run_experiment_gpu.sh                  # paper's full 30-seed sweep
#   DETECTORS="swoks implicit" ./run_experiment_gpu.sh   # detector subset
#
# All runs are saved with --save-model so experiment.py --eval-posthoc can
# reload the final meta learner and evaluate it in each MinAtar game.

set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-../FAMEenv/bin/python}"

if [ "${FULL:-0}" = "1" ]; then
    SEQS_DEFAULT="0 1 2 3 4 5 6 7 8 9"
    SEEDS_DEFAULT="1 2 3"
else
    SEQS_DEFAULT="0"
    SEEDS_DEFAULT="1 2 3"
fi
SEQS=(${SEQS:-$SEQS_DEFAULT})
SEEDS=(${SEEDS:-$SEEDS_DEFAULT})
DETECTORS=(${DETECTORS:-oracle swoks implicit hybrid})
GPU="${GPU:-0}"

# -- FAME paper's MinAtar hyperparameters (Appendix F.1) ---------------
# (3.5M steps, 500k per task -> 7 tasks per sequence)
TSTEPS="${TSTEPS:-3500000}"
SWITCH="${SWITCH:-500000}"
SIZE_FAST2META=12000       # N (meta buffer insertion size)
DETECTION_STEP=600         # n (policy-evaluation step for hypothesis test)
WARMSTEP=50000             # L (BC warm-up step)
EPOCH_META=200             # meta training epochs
LR1=1e-3                   # meta learning rate
LR2=1e-5                   # fast learner learning rate
LAMBDA_REG=1.0             # beta (BC regularisation weight)

# -- Detector hyperparameters ------------------------------------------
# Bonferroni-corrected alpha: 3.5M / 240 ~ 14.5k tests -> alpha = 7e-5
# (see detector_heuristics.py).  Tighten beta to 2 per SWOKS paper.
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

mkdir -p results models logs
echo "==================================================="
echo " FAME + TSDM paper sweep on GPU $GPU"
echo "   SEQS=${SEQS[*]}"
echo "   SEEDS=${SEEDS[*]}"
echo "   DETECTORS=${DETECTORS[*]}"
echo "   steps/task=$SWITCH  total=$TSTEPS  ->  $((TSTEPS / SWITCH)) tasks"
echo "==================================================="

for seq in "${SEQS[@]}"; do
    for s in "${SEEDS[@]}"; do
        pids=()
        for det in "${DETECTORS[@]}"; do
            if [ "$det" = "oracle" ]; then
                extra=()
            else
                extra=("${COMMON_DET[@]}")
            fi
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
                --seq "$seq" --seed "$s" --gpu "$GPU" \
                --save --save-model \
                2>&1 | tee "logs/exp_gpu_${det}_seq${seq}_seed${s}.log" &
            pids+=($!)
        done
        # Wait for all detector variants of (seq, seed) before next pair.
        for pid in "${pids[@]}"; do wait "$pid"; done
    done
done

echo "All training runs finished at $(date)"
echo "==================================================="
echo " Experiment analysis (experiment.py) -- all sequences"
echo "==================================================="

# experiment.py runs per sequence; the paper aggregates across all 10
# sequences, so we glue the per-seq tables with a tiny Python one-liner.
for seq in "${SEQS[@]}"; do
    "$PY" experiment.py \
        --results_dir results \
        --models_dir models \
        --seq "$seq" \
        --seeds "${SEEDS[@]}" \
        --tolerance 60000 \
        --tolerance_sweep 5000 15000 60000 120000 240000 500000 \
        --smooth 5000 \
        --tag "gpu_seq${seq}" \
        --eval-posthoc \
        --posthoc_episodes 30 \
        --posthoc_max_steps 3000 \
        --device "cuda:${GPU}"
done

if [ "${#SEQS[@]}" -gt 1 ]; then
    echo "==================================================="
    echo " Cross-sequence aggregation (mean over SEQS=${SEQS[*]})"
    echo "==================================================="
    SEQS_CSV=$(IFS=,; echo "${SEQS[*]}")
    SEQS_CSV="$SEQS_CSV" "$PY" - <<'PYEOF'
import os, pickle, numpy as np
seqs = [int(x) for x in os.environ["SEQS_CSV"].split(",") if x]
print("seqs=", seqs)
rows_by_mode = {}
for seq in seqs:
    bp = f"results/experiment_gpu_seq{seq}/summaries.pkl"
    if not os.path.exists(bp):
        print(" missing:", bp); continue
    with open(bp, "rb") as f: blob = pickle.load(f)
    for mode, runs in blob["summaries"].items():
        rows_by_mode.setdefault(mode, []).extend(runs)
print({m: len(v) for m, v in rows_by_mode.items()})
for m, runs in rows_by_mode.items():
    ap = [r.avg_perf_proxy for r in runs]
    po = [r.posthoc_avg_perf for r in runs if r.posthoc_avg_perf is not None]
    n = max(1, len(ap))
    se = float(np.std(ap, ddof=0) / n**0.5) if n > 1 else 0.0
    mu_po = float(np.mean(po)) if po else float("nan")
    print(f"  {m:<9s}: avg_perf={np.mean(ap):.2f}+/-{se:.2f} "
          f"posthoc={mu_po:.2f}  n={len(runs)}")
PYEOF
fi

echo
echo "Artifacts written under results/experiment_gpu_seq*/"
