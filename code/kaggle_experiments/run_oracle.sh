

set -euo pipefail

source "$(dirname "$0")/_common.sh"

echo "==================================================================="
echo " FAME oracle  |  seq=${SEQ}  seed=${SEED}  gpu=${GPU}"
echo " schedule: ${TSTEPS} steps / ${SWITCH} per task"
echo "==================================================================="

run_detector oracle

echo "==================================================================="
echo " oracle training complete -- pkl at: $(expected_pkl oracle)"
echo "==================================================================="
