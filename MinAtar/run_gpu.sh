#!/bin/bash
# Full FAME + TSDM sweep on GPU.
#
# Runs the oracle ceiling, SWOKS, implicit, and hybrid variants side by
# side for the same seeds/sequences, then invokes the post-hoc comparison.
# Uses FAMEenv (see requirements.txt).
#
# Set DETECTORS env var to subset, e.g.  DETECTORS="implicit hybrid" ./run_gpu.sh

set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-../FAMEenv/bin/python}"
SEQ="${SEQ:-0}"
SEEDS=(${SEEDS:-1 2 3})
GPU="${GPU:-0}"
TSTEPS="${TSTEPS:-3500000}"   # full protocol: 7 tasks x 500k
SWITCH="${SWITCH:-500000}"
DETECTORS=(${DETECTORS:-oracle swoks implicit hybrid})

# Detector params for the full-scale protocol.  Shared between swoks,
# implicit and hybrid variants so they can be compared apples-to-apples.
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
        if [ "$det" = "oracle" ]; then
            extra=()
        else
            extra=("${COMMON_DET[@]}")
        fi
        "$PY" FAME.py \
            --detector "$det" \
            "${extra[@]}" \
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
    # Wait for all detector variants for this seed to finish before next seed.
    for pid in "${pids[@]}"; do wait "$pid"; done
done

echo "All training runs finished at $(date)"
"$PY" compare_oracle_vs_swoks.py \
    --results_dir results \
    --seq "$SEQ" \
    --seeds "${SEEDS[@]}" \
    --tolerance 60000
