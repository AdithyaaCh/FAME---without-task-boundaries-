

set -euo pipefail

source "$(dirname "$0")/_common.sh"

echo "==================================================================="
echo " FAME hybrid (cascade + log-evidence)  |  seq=${SEQ}  seed=${SEED}"
echo " schedule: ${TSTEPS} steps / ${SWITCH} per task"
echo " tau_imp_loose=1e-2  tau_combined=15.0  horizon=480"
echo " checkpoint every ${CKPT_EVERY} steps -> ${CKPT_DIR}"
echo "==================================================================="

run_detector hybrid "${COMMON_DET_ARGS[@]}"

echo "==================================================================="
echo " hybrid training complete -- pkl at: $(expected_pkl hybrid)"
echo "==================================================================="
