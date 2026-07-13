# GPU Monitoring Guide for g4dn.2xlarge

## Current GPU Status
Your Tesla T4 GPU has:
- **Total Memory**: 15.0GB
- **Currently Used**: ~4.3GB (28.6%)
- **Available**: ~10.7GB
- **GPU Utilization**: 0% (currently idle)

## Quick Monitoring Commands

### 1. Single GPU Status Check
```bash
nvidia-smi
```

### 2. Continuous Monitoring (every 2 seconds)
```bash
nvidia-smi -l 2
```

### 3. Interactive GPU Monitor (like htop for GPU)
```bash
nvtop
```

### 4. Custom GPU Monitor Script
```bash
# Single check
python monitor_gpu.py

# Continuous monitoring (every 5 seconds)
python monitor_gpu.py --continuous 5
```

## Memory Usage Analysis

### Current Experiment
- Main process (PID 380877): ~4.2GB GPU memory
- Multiple subprocesses: ~300MB each for parallel environments

### Memory Requirements Per Experiment
Based on your current usage:
- **Full experiment**: ~4-5GB
- **Small experiment** (reduced params): ~2-3GB
- **Quick test**: ~1-2GB

## Can You Run Parallel Experiments?

### ✅ YES - You have sufficient capacity!

With 10.7GB free memory, you can run:
- **1 additional full experiment** alongside current one
- **2-3 small experiments** with reduced parameters
- **4-5 quick test experiments**

## Parallel Execution Options

### Option 1: Manual Parallel Execution
Open multiple terminals and run experiments with different seeds/environments:

```bash
# Terminal 1
python main.py --env hopper-medium-v2 --seed 2 --batch_size 64 --max_online_iters 10

# Terminal 2  
python main.py --env walker2d-medium-replay-v2 --seed 1 --batch_size 64 --max_online_iters 10
```

### Option 2: Using the Parallel Runner Script
```bash
# Run predefined parallel experiments
python run_parallel.py

# Quick test experiments
python run_parallel.py quick

# Custom experiments
python run_parallel.py custom
```

## Recommended Parallel Configuration

To run 2 experiments simultaneously:

```bash
# Experiment 1 (reduce memory usage)
python main.py \
  --env hopper-medium-v2 \
  --seed 2 \
  --batch_size 64 \
  --max_online_iters 10 \
  --num_updates_per_online_iter 50 \
  --trajectory_length 500 \
  --K 10

# Experiment 2
python main.py \
  --env walker2d-medium-replay-v2 \
  --seed 1 \
  --batch_size 64 \
  --max_online_iters 10 \
  --num_updates_per_online_iter 50 \
  --trajectory_length 500 \
  --K 10
```

## Memory Optimization Tips

1. **Reduce batch_size**: From 256 to 64-128
2. **Reduce K (context length)**: From 20 to 10-15
3. **Reduce trajectory_length**: From 1000 to 500-750
4. **Reduce num_updates_per_online_iter**: From 100 to 50-75
5. **Use shorter experiments**: Reduce max_online_iters

## Monitoring During Parallel Execution

### Real-time GPU monitoring
```bash
# In separate terminal
watch -n 2 nvidia-smi
```

### Process monitoring
```bash
# Check running Python processes
ps aux | grep python | grep main.py
```

### Log monitoring
```bash
# Monitor experiment logs
tail -f exp_logs/*.log
```

## Performance Considerations

### Expected Speed Impact
- **Single experiment**: 100% speed
- **2 parallel experiments**: ~60-70% speed each (due to GPU sharing)
- **3+ parallel experiments**: May cause memory pressure

### Optimization Strategy
1. Start with 2 experiments to test performance
2. Monitor GPU utilization and memory
3. Adjust parameters if needed
4. Scale up if performance is acceptable

## Warning Signs to Watch

🚨 **Stop adding experiments if you see:**
- GPU memory usage > 90%
- Frequent CUDA out of memory errors
- System becoming unresponsive
- GPU temperature > 80°C

## Quick Commands Summary

```bash
# Check GPU status
nvidia-smi

# Monitor continuously  
python monitor_gpu.py --continuous

# Run parallel experiments
python run_parallel.py

# Check running experiments
ps aux | grep python | grep main.py

# Interactive GPU monitor
nvtop
```