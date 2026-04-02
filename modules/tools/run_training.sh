#!/bin/bash
# Wrapper to launch training with safety checks and automatic logging

set -e

# Check if main.py is already running
if pgrep -f "python.*main.py" > /dev/null; then
    echo "ERROR: main.py already running. Kill existing process first."
    exit 1
fi

# Navigate to project root
cd "$(dirname "$0")/../.."
mkdir -p runs

EXISTING_RUNS=$(ls -d runs/run_[0-9][0-9][0-9] 2>/dev/null | wc -l)
RUN_NUM=$((EXISTING_RUNS + 1))
LOG_FILE="runs/run_$(printf "%03d" $RUN_NUM).log"

# Launch training in background
nohup python main.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "Training launched: PID=$PID, log=$LOG_FILE"
echo $PID
