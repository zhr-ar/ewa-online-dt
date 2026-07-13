#!/bin/bash

SCRIPT_NAME=${1:-"run_ewa.py"}
EXP_NAME=${2:-"direct_exp_$(date +%Y%m%d_%H%M%S)"}

echo "=== DIRECT EXPERIMENT RUNNER ==="
echo "Script: $SCRIPT_NAME"
echo "Experiment: $EXP_NAME"
echo "Start time: $(date)"
echo "================================"

# Use correct conda path and activate environment
source /opt/conda/etc/profile.d/conda.sh
conda activate odt

# Check if script exists
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "❌ Error: Script '$SCRIPT_NAME' not found!"
    exit 1
fi

# Create log directory for backup
LOG_DIR="./experiment_logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Logging to: $LOG_FILE"
echo "🚀 Starting experiment..."
echo "================================"

# Run experiment with both terminal output and logging
# This way you see output in real-time AND have a backup log
python $SCRIPT_NAME 2>&1 | tee $LOG_FILE

echo ""
echo "================================"
echo "✅ Experiment completed!"
echo "📁 Log saved to: $LOG_FILE"
echo "⏰ End time: $(date)"
