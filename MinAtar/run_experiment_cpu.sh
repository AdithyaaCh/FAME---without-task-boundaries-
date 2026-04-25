#!/bin/bash
# run_experiment_cpu.sh -- end-to-end experiment sanity check on CPU.
#
# Drives four-method training (oracle + swoks + implicit + hybrid) on an
# ultra-compressed MinAtar schedule (4 tasks x 8k steps), then invokes
# experiment.py to generate the full analysis bundle -- tables, plots,
# LaTeX table, markdown report, and optional post-hoc rollouts.
#
# Mirrors the full-scale `run_experiment_gpu.sh` one-to-one so you can
# validate the pipeline end-to-end in a few minutes before queueing the
# expensive GPU job.
#
# Env vars (all optional):
#   SEQ       -- sequence id (default 0)
#   SEEDS     -- space-separated seeds (default "1 2")
#   DETECTORS -- which methods to run (default all four)
#   EVAL_POSTHOC=1 -- also run post-hoc rollouts after training
#
#   TSTEPS, SWITCH -- override the schedule for ablation runs
#
# Usage:
#   ./run_experiment_cpu.sh
#   SEEDS="1 2 3" EVAL_POSTHOC=1 ./run_experiment_cpu.sh

set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-../FAMEenv/bin/python}"
SEQ="${SEQ:-0}"
SEEDS=(${SEEDS:-1 2})
DETECTORS=(${DETECTORS:-oracle swoks implicit hybrid})
EVAL_POSTHOC="${EVAL_POSTHOC:-0}"

# Mini schedule -- 4 tasks x 8k = 32k steps
TSTEPS="${TSTEPS:-32000}"
SWITCH="${SWITCH:-8000}"
SIZE_FAST2META=1500
DETECTION_STEP=150
WARMSTEP=800
EPOCH_META=20

# Tight paper-aligned detector settings, rescaled to the mini horizon.
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

export CUDA_VISIBLE_DEVICES=""
mkdir -p results models logs

echo "==================================================="
echo " Experiment (CPU sanity): seq=$SEQ seeds=${SEEDS[*]} dets=${DETECTORS[*]}"
echo "==================================================="

for s in "${SEEDS[@]}"; do
    for det in "${DETECTORS[@]}"; do
        echo "--- train: detector=$det seq=$SEQ seed=$s ---"
        if [ "$det" = "oracle" ]; then
            extra=()
        else
            extra=("${COMMON_DET[@]}")
        fi
        "$PY" FAME.py \
            --detector "$det" \
            ${extra[@]+"${extra[@]}"} \
            --lr1 1e-3 --lr2 1e-4 \
            --size_fast2meta "$SIZE_FAST2META" \
            --detection_step "$DETECTION_STEP" \
            --warmstep "$WARMSTEP" \
            --lambda_reg 1.0 \
            --epoch_meta "$EPOCH_META" \
            --t-steps "$TSTEPS" --switch "$SWITCH" \
            --seq "$SEQ" --seed "$s" \
            --save --save-model \
            2>&1 | tee "logs/exp_cpu_${det}_seq${SEQ}_seed${s}.log"
    done
done

echo "==================================================="
echo " Experiment analysis (experiment.py)"
echo "==================================================="

ARGS=(
    --results_dir results
    --models_dir models
    --seq "$SEQ"
    --seeds "${SEEDS[@]}"
    --tolerance 2000
    --tolerance_sweep 500 1000 2000 5000 10000
    --smooth 500
    --tag "cpu_sanity_seq${SEQ}"
)
if [ "$EVAL_POSTHOC" = "1" ]; then
    ARGS+=(--eval-posthoc --posthoc_episodes 10 --posthoc_max_steps 1000)
fi

"$PY" experiment.py "${ARGS[@]}"

echo
echo "Artifacts written under results/experiment_cpu_sanity_seq${SEQ}/"
