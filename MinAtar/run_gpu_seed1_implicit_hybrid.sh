#!/bin/bash
# Focused FAME + TSDM sweep on GPU for seed 1 only.
#
# Runs only the implicit and hybrid detector variants side by side for the
# same sequence, then invokes the post-hoc comparison.
# Uses FAMEenv (see requirements.txt).
#
# Set DETECTORS env var to subset, e.g.
#   DETECTORS="implicit" ./run_gpu_seed1_implicit_hybrid.sh

set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-../FAMEenv/bin/python}"
SEQ="${SEQ:-0}"
SEEDS=(${SEEDS:-1})
GPU="${GPU:-0}"
TSTEPS="${TSTEPS:-3500000}"   # full protocol: 7 tasks x 500k
SWITCH="${SWITCH:-500000}"
DETECTORS=(${DETECTORS:-implicit hybrid})

# Detector params for the full-scale protocol. Shared between implicit and
# hybrid so they can be compared apples-to-apples.
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

for s in "${SEEDS[@]}"; do
    pids=()
    for det in "${DETECTORS[@]}"; do
        "$PY" FAME.py \
            --detector "$det" \
            "${COMMON_DET[@]}" \
            --lr1 1e-3 --lr2 1e-5 \
            --size_fast2meta 12000 \
            --detection_step 600 \
            --warmstep 50000 \
            --lambda_reg 1.0 \
            --t-steps "$TSTEPS" --switch "$SWITCH" \
            --seq "$SEQ" --seed "$s" --gpu "$GPU" \
            --save --save-model \
            2>&1 | tee "logs/${det}_seq${SEQ}_seed${s}.log" &
        pids+=($!)
    done
    # Wait for all detector variants for this seed to finish before exiting.
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
done

echo "All training runs finished at $(date)"
"$PY" compare_oracle_vs_swoks.py \
    --results_dir results \
    --seq "$SEQ" \
    --seeds "${SEEDS[@]}" \
    --tolerance 60000
