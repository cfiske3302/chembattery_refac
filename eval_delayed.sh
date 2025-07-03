#!/usr/bin/env bash
# eval_delayed.sh
# Waits until the directory ./runs/c_rate_1/model exists, then runs a command.
#
# Usage:
#   ./eval_delayed.sh [INTERVAL_MINUTES]
#
#   INTERVAL_MINUTES – how many minutes to wait between checks (default: 1)
#
# The script prints timestamped status messages while waiting.  Replace the
# placeholder `echo` with your real evaluation command.

set -euo pipefail

TARGET_DIR="./runs/c_rate_1/model"
TARGET_DIR2="./runs/reproduced_results_1/model"
INTERVAL_MIN=${1:-1}   # Minutes between checks

echo "$(date '+%F %T')  Waiting for ${TARGET_DIR} (checking every ${INTERVAL_MIN} minute(s))..."

# Loop until the directory appears
until [ -d "$TARGET_DIR" ]; do
    sleep "${INTERVAL_MIN}m"
done

# -------------------------------------------------------------------
# TODO: Replace the line below with your actual evaluation command
python main.py --eval --config ./configs/train_lfp_by_crate.yaml
# -------------------------------------------------------------------

until [ -d "$TARGET_DIR2" ]; do
    sleep "${INTERVAL_MIN}m"
done

python main.py --eval --config ./configs/train_lfp_reproduce.yaml