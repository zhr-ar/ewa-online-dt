#!/bin/bash

SCRIPT_NAME=${1:-"run_ewa.py"}
EXP_NAME=${2:-"screen_exp_$(date +%Y%m%d_%H%M%S)"}

# Create unique screen session name
SESSION_NAME="${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"

echo "=== SCREEN-BASED EXPERIMENT RUNNER ==="
echo "Script: $SCRIPT_NAME"
echo "Session: $SESSION_NAME"
echo "================================"

# Use correct conda path and activate environment
source /opt/conda/etc/profile.d/conda.sh
conda activate odt

# Create screen session that shows output in real-time
screen -dmS $SESSION_NAME bash -c "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate odt
    cd /home/ubuntu/online_decision_transformer/ewa-online-dt-main
    echo '=== EXPERIMENT STARTED IN SCREEN ==='
    echo 'Script: $SCRIPT_NAME'
    echo 'Session: $SESSION_NAME'
    echo 'Start time: \$(date)'
    echo '=============================='
    echo 'Starting experiment...'
    python $SCRIPT_NAME
    echo ''
    echo '=== EXPERIMENT COMPLETED ==='
    echo 'End time: \$(date)'
    echo 'Press any key to exit...'
    read -n 1
"

echo ""
echo "✅ Screen session created: $SESSION_NAME"
echo "🔗 To attach and see output: screen -r $SESSION_NAME"
echo "📋 To list all sessions: screen -ls"
echo "❌ To kill session: screen -S $SESSION_NAME -X quit"
echo ""
echo "💡 Your experiment will survive ANY disconnection!"
echo "📺 To see real-time output, attach to the session with: screen -r $SESSION_NAME"
echo "🚪 To detach (keep running): Press Ctrl+A, then D"