# Experiment Runner Guide

## 🚀 Three Ways to Run Your Experiments

### 1. **Screen Method (RECOMMENDED for persistence)**
```bash
# Start experiment in background screen session
./run_exp_screen.sh run_ewa.py

# To see output in real-time:
screen -r screen_exp_[session_name]

# To detach (keep running): Press Ctrl+A, then D
# To list all sessions: screen -ls
# To kill session: screen -S [session_name] -X quit
```

### 2. **Direct Method (see output immediately)**
```bash
# Run directly in terminal with logging
./run_exp_direct.sh run_ewa.py

# Output shows in terminal AND saves to log file
# Good for short experiments or when you want immediate feedback
```

### 3. **Robust Method (auto-restart on failure)**
```bash
# Run with automatic monitoring and restart
./run_exp_robust.sh run_ewa.py

# Runs in background, monitors for failures, auto-restarts
# Check logs in ./experiment_logs/
```

## 📋 Quick Commands

```bash
# Make scripts executable (one time setup)
chmod +x run_exp_screen.sh run_exp_direct.sh run_exp_robust.sh

# Run any experiment
./run_exp_screen.sh run_ewa.py
./run_exp_screen.sh run_quick.py
./run_exp_screen.sh main.py

# Check running sessions
screen -ls

# Attach to see output
screen -r [session_name]

# Detach (keep running)
# Press: Ctrl+A, then D
```

## 🔧 What Each Method Does

### Screen Method:
- ✅ **Survives SSH disconnections**
- ✅ **Survives VSCode reconnections**
- ✅ **Survives instance restarts**
- ✅ **Shows real-time output when attached**
- ❌ **Requires screen -r to see output**

### Direct Method:
- ✅ **Shows output immediately**
- ✅ **Saves backup log file**
- ❌ **Stops if terminal disconnects**
- ❌ **Stops if VSCode reconnects**

### Robust Method:
- ✅ **Auto-restarts on failure**
- ✅ **Runs in background**
- ✅ **Comprehensive logging**
- ❌ **No real-time output**
- ❌ **Complex setup**

## 🎯 **RECOMMENDATION: Use Screen Method**

1. **Start your experiment:**
   ```bash
   ./run_exp_screen.sh run_ewa.py
   ```

2. **Attach to see output:**
   ```bash
   screen -r [session_name]
   ```

3. **Detach to keep running:**
   - Press: `Ctrl+A`, then `D`

4. **Your experiment continues running even if:**
   - VSCode disconnects
   - SSH connection drops
   - Mac goes to sleep
   - Instance has network issues

## 🚨 Troubleshooting

### If screen command not found:
```bash
sudo apt-get update
sudo apt-get install screen
```

### If conda activation fails:
```bash
# Check conda path
which conda
conda info --base

# Update script with correct path
```

### If experiment stops unexpectedly:
```bash
# Check if it's still running
screen -ls

# Reattach to see what happened
screen -r [session_name]
```

## 📁 File Structure
```
your_project/
├── run_exp_screen.sh      # Screen-based runner (RECOMMENDED)
├── run_exp_direct.sh      # Direct terminal runner
├── run_exp_robust.sh      # Auto-restart runner
├── experiment_logs/       # Backup log files
└── EXPERIMENT_RUNNER_GUIDE.md
```

## 🎉 **You're All Set!**

Your experiments will now survive any disconnection and continue running until completion!
