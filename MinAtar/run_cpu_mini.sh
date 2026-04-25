#!/bin/bash
# Mini local smoke test for FAME + TSDM.
#
# Drastically shortens the schedule so you can sanity-check the full pipeline
# (oracle + detected, plus the comparison harness) on a CPU-only laptop in
# a few minutes.  NOT intended for scientific results -- use run_gpu.sh for
# those.
#
# Per user direction (SWOKS alone has already been validated), this script
# runs the IMPLICIT and HYBRID detectors by default.  Set DETECTORS to
# override, e.g.  DETECTORS="oracle swoks implicit hybrid" ./run_cpu_mini.sh

set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-../FAMEenv/bin/python}"
SEQ="${SEQ:-0}"
SEEDS=(${SEEDS:-1 2})
DETECTORS=(${DETECTORS:-implicit hybrid})

# Mini schedule: 4 tasks x 8k steps.  Windows scaled proportionally
# (L_D=200, stable_phase=2000) so detectors can fire within the mini budget.
TSTEPS=32000
SWITCH=8000
SIZE_FAST2META=1500
DETECTION_STEP=150
WARMSTEP=800
EPOCH_META=20

# Common boundary-free detector args (shared by swoks, implicit, hybrid).
COMMON_DET=(
    --swoks_L_D 200 --swoks_L_W 10
    --swoks_alpha 5e-2 --swoks_beta 1.2
    --swoks_stable_phase 2000 --swoks_interval 50
    --swoks_warmup 500
    --swoks_snapshot 1 --swoks_snapshot_interval 600
    --imp_L_D 200 --imp_alpha 5e-2
    --imp_stable_phase 2000 --imp_interval 50
    --imp_warmup 500 --imp_lr 1e-3
    --imp_update_every 8
    --hyb_tau_imp_loose 5e-2
    --hyb_tau_imp_strict 1e-4
    --hyb_tau_stat_strict 5e-2
    --hyb_tau_combined 10.0
    --hyb_horizon 150
    --hyb_persistence 2
)

# Force CPU with a fake gpu flag (the script falls back to cpu automatically).
export CUDA_VISIBLE_DEVICES=""

mkdir -p results models logs

for s in "${SEEDS[@]}"; do
    for det in "${DETECTORS[@]}"; do
        echo "==================================================="
        echo " Mini run: detector=$det seq=$SEQ seed=$s"
        echo "==================================================="
        if [ "$det" = "oracle" ]; then
            extra=()
        else
            extra=("${COMMON_DET[@]}")
        fi
        "$PY" FAME.py \
            --detector "$det" \
            "${extra[@]}" \
            --lr1 1e-3 --lr2 1e-4 \
            --size_fast2meta "$SIZE_FAST2META" \
            --detection_step "$DETECTION_STEP" \
            --warmstep "$WARMSTEP" \
            --lambda_reg 1.0 \
            --epoch_meta "$EPOCH_META" \
            --t-steps "$TSTEPS" --switch "$SWITCH" \
            --seq "$SEQ" --seed "$s" \
            --save \
            2>&1 | tee "logs/mini_${det}_seq${SEQ}_seed${s}.log"
    done
done

echo "==================================================="
echo " Mini comparison"
echo "==================================================="
"$PY" compare_oracle_vs_swoks.py \
    --results_dir results \
    --seq "$SEQ" \
    --seeds "${SEEDS[@]}" \
    --tolerance 2000
