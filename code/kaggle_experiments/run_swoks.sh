

set -euo pipefail
source "$(dirname "$0")/_common.sh"

echo "==================================================================="
echo " FAME swoks   |  seq=${SEQ}  seed=${SEED}  gpu=${GPU}"
echo " schedule: ${TSTEPS} steps / ${SWITCH} per task"
echo " alpha=1e-3  beta=2.0  L_D=1200  L_W=30"
echo "==================================================================="

run_detector swoks "${COMMON_DET_ARGS[@]}"

echo "==================================================================="
echo " swoks training complete -- pkl at: $(expected_pkl swoks)"
echo "==================================================================="
