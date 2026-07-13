#!/bin/bash

# Robust experiment runner with automatic reconnection handling
SCRIPT_NAME=${1:-"run_ewa.py"}
EXP_NAME=${2:-"robust_exp_$(date +%Y%m%d_%H%M%S)"}

# Create log directory
LOG_DIR="./experiment_logs"
mkdir -p $LOG_DIR

# Set up logging
LOG_FILE="$LOG_DIR/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
ERROR_LOG="$LOG_DIR/${EXP_NAME}_errors.log"

echo "=== ROBUST EXPERIMENT RUNNER ===" | tee -a $LOG_FILE
echo "Script: $SCRIPT_NAME" | tee -a $LOG_FILE
echo "Experiment: $EXP_NAME" | tee -a $LOG_FILE
echo "Start time: $(date)" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "================================" | tee -a $LOG_FILE

# Function to check if experiment is still running
check_experiment() {
    if pgrep -f "python $SCRIPT_NAME" > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to restart experiment if needed
restart_experiment() {
    echo "$(date): Experiment stopped, attempting restart..." | tee -a $LOG_FILE
    echo "$(date): Restarting $SCRIPT_NAME..." | tee -a $LOG_FILE
    
    # Activate conda and restart
    source /opt/conda/etc/profile.d/conda.sh
    conda activate odt
    
    nohup python $SCRIPT_NAME >> $LOG_FILE 2>> $ERROR_LOG &
    
    echo "$(date): Restart completed with PID $(pgrep -f 'python $SCRIPT_NAME')" | tee -a $LOG_FILE
}

# Main execution
source /opt/conda/etc/profile.d/conda.sh
conda activate odt

# Start the experiment
echo "$(date): Starting experiment..." | tee -a $LOG_FILE
nohup python $SCRIPT_NAME >> $LOG_FILE 2>> $ERROR_LOG &
EXPERIMENT_PID=$!

echo "$(date): Experiment started with PID: $EXPERIMENT_PID" | tee -a $LOG_FILE

# Monitor and auto-restart if needed
while true; do
    if check_experiment; then
        echo "$(date): Experiment running normally..." | tee -a $LOG_FILE
        sleep 300  # Check every 5 minutes
    else
        echo "$(date): Experiment not found, restarting..." | tee -a $LOG_FILE
        restart_experiment
        sleep 60  # Wait before next check
    fi
done