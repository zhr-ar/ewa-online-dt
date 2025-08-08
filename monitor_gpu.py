#!/usr/bin/env python3
"""
GPU monitoring script for checking capacity during experiments.
"""
import subprocess
import time
import sys
import os

def get_gpu_stats():
    """Get GPU memory and utilization stats."""
    try:
        # Get GPU memory info
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            stats = result.stdout.strip().split(', ')
            return {
                'memory_used': int(stats[0]),
                'memory_total': int(stats[1]),
                'gpu_util': int(stats[2]),
                'temperature': int(stats[3]),
                'power_used': float(stats[4]),
                'power_limit': float(stats[5])
            }
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return None

def format_memory(mb):
    """Format memory in MB to human readable."""
    if mb >= 1024:
        return f"{mb/1024:.1f}GB"
    return f"{mb}MB"

def print_gpu_status():
    """Print current GPU status."""
    stats = get_gpu_stats()
    if not stats:
        print("Failed to get GPU stats")
        return
    
    memory_percent = (stats['memory_used'] / stats['memory_total']) * 100
    power_percent = (stats['power_used'] / stats['power_limit']) * 100
    
    print(f"\n{'='*60}")
    print(f"GPU Status - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Memory:      {format_memory(stats['memory_used'])} / {format_memory(stats['memory_total'])} ({memory_percent:.1f}%)")
    print(f"GPU Util:    {stats['gpu_util']}%")
    print(f"Temperature: {stats['temperature']}°C")
    print(f"Power:       {stats['power_used']:.1f}W / {stats['power_limit']:.1f}W ({power_percent:.1f}%)")
    
    # Memory availability assessment
    memory_free = stats['memory_total'] - stats['memory_used']
    print(f"\nFree Memory: {format_memory(memory_free)}")
    
    if memory_free > 4000:  # More than 4GB free
        print("✅ Sufficient memory for another experiment")
    elif memory_free > 2000:  # 2-4GB free
        print("⚠️  Might be able to run another smaller experiment")
    else:  # Less than 2GB free
        print("❌ Insufficient memory for another experiment")
    
    # GPU utilization assessment
    if stats['gpu_util'] < 50:
        print("✅ GPU utilization is low - good for parallel experiments")
    elif stats['gpu_util'] < 80:
        print("⚠️  GPU utilization is moderate")
    else:
        print("❌ GPU utilization is high")

def monitor_continuous(interval=5):
    """Monitor GPU continuously."""
    print("Starting continuous GPU monitoring (Press Ctrl+C to stop)")
    try:
        while True:
            print_gpu_status()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        monitor_continuous(interval)
    else:
        print_gpu_status()
        print(f"\nUsage:")
        print(f"  python {sys.argv[0]}              # Single check")
        print(f"  python {sys.argv[0]} --continuous [interval]  # Continuous monitoring")

if __name__ == "__main__":
    main()