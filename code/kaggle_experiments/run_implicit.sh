set -euo pipefail

source "$(dirname "$0")/_common.sh"

echo "==================================================================="
echo " FAME implicit (TSN)  |  seq=${SEQ}  seed=${SEED}  gpu=${GPU}"
echo " schedule: ${TSTEPS} steps / ${SWITCH} per task"
echo " imp_alpha=1e-3  L_D=1200  lr=1e-4  update_every=16"
echo " checkpoint every ${CKPT_EVERY} steps -> ${CKPT_DIR}"
echo "==================================================================="

run_detector implicit "${COMMON_DET_ARGS[@]}"

echo "==================================================================="
echo " implicit training complete -- pkl at: $(expected_pkl implicit)"
echo "==================================================================="
