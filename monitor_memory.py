#!/usr/bin/env python3
"""
Memory Monitoring Script for EWA Training
Helps debug memory issues that cause instance crashes
"""

import psutil
import torch
import time
import os
from datetime import datetime

def get_memory_info():
    """Get comprehensive memory information"""
    # System memory
    memory = psutil.virtual_memory()
    
    # GPU memory (if available)
    gpu_memory = {}
    if torch.cuda.is_available():
        gpu_memory = {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
            'max_reserved': torch.cuda.max_memory_reserved() / 1024**3,    # GB
        }
    
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'system_memory': {
            'total_gb': memory.total / 1024**3,
            'available_gb': memory.available / 1024**3,
            'used_gb': memory.used / 1024**3,
            'percent_used': memory.percent
        },
        'gpu_memory': gpu_memory,
        'process_memory': {
            'rss_gb': psutil.Process().memory_info().rss / 1024**3,
            'vms_gb': psutil.Process().memory_info().vms / 1024**3
        }
    }

def print_memory_info(memory_info):
    """Print memory information in a readable format"""
    print(f"\n{'='*60}")
    print(f"MEMORY STATUS - {memory_info['timestamp']}")
    print(f"{'='*60}")
    
    # System memory
    sys_mem = memory_info['system_memory']
    print(f"💾 SYSTEM MEMORY:")
    print(f"   Total: {sys_mem['total_gb']:.2f} GB")
    print(f"   Available: {sys_mem['available_gb']:.2f} GB")
    print(f"   Used: {sys_mem['used_gb']:.2f} GB ({sys_mem['percent_used']:.1f}%)")
    
    # GPU memory
    if memory_info['gpu_memory']:
        gpu_mem = memory_info['gpu_memory']
        print(f"🚀 GPU MEMORY:")
        print(f"   Allocated: {gpu_mem['allocated']:.3f} GB")
        print(f"   Reserved: {gpu_mem['reserved']:.3f} GB")
        print(f"   Max Allocated: {gpu_mem['max_allocated']:.3f} GB")
        print(f"   Max Reserved: {gpu_mem['max_reserved']:.3f} GB")
    
    # Process memory
    proc_mem = memory_info['process_memory']
    print(f"🔍 PROCESS MEMORY:")
    print(f"   RSS: {proc_mem['rss_gb']:.3f} GB")
    print(f"   VMS: {proc_mem['vms_gb']:.3f} GB")
    
    # Warnings
    warnings = []
    if sys_mem['percent_used'] > 90:
        warnings.append(f"⚠️  System memory usage is {sys_mem['percent_used']:.1f}% - CRITICAL!")
    elif sys_mem['percent_used'] > 80:
        warnings.append(f"⚠️  System memory usage is {sys_mem['percent_used']:.1f}% - HIGH!")
    
    if memory_info['gpu_memory'] and memory_info['gpu_memory']['allocated'] > 14:
        warnings.append(f"⚠️  GPU memory usage is {memory_info['gpu_memory']['allocated']:.3f} GB - HIGH!")
    
    if warnings:
        print(f"\n🚨 WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
    
    print(f"{'='*60}")

def monitor_memory_continuously(interval=30, max_iterations=None):
    """Monitor memory continuously"""
    print(f"Starting memory monitoring every {interval} seconds...")
    print("Press Ctrl+C to stop")
    
    iteration = 0
    try:
        while True:
            if max_iterations and iteration >= max_iterations:
                break
                
            memory_info = get_memory_info()
            print_memory_info(memory_info)
            
            iteration += 1
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMemory monitoring stopped by user")
    except Exception as e:
        print(f"\nError during monitoring: {e}")

def save_memory_log(filename=None):
    """Save current memory status to file"""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"memory_log_{timestamp}.txt"
    
    memory_info = get_memory_info()
    
    with open(filename, 'w') as f:
        f.write(f"MEMORY LOG - {memory_info['timestamp']}\n")
        f.write("="*60 + "\n")
        
        # System memory
        sys_mem = memory_info['system_memory']
        f.write(f"SYSTEM MEMORY:\n")
        f.write(f"  Total: {sys_mem['total_gb']:.2f} GB\n")
        f.write(f"  Available: {sys_mem['available_gb']:.2f} GB\n")
        f.write(f"  Used: {sys_mem['used_gb']:.2f} GB ({sys_mem['percent_used']:.1f}%)\n")
        
        # GPU memory
        if memory_info['gpu_memory']:
            gpu_mem = memory_info['gpu_memory']
            f.write(f"GPU MEMORY:\n")
            f.write(f"  Allocated: {gpu_mem['allocated']:.3f} GB\n")
            f.write(f"  Reserved: {gpu_mem['reserved']:.3f} GB\n")
            f.write(f"  Max Allocated: {gpu_mem['max_allocated']:.3f} GB\n")
            f.write(f"  Max Reserved: {gpu_mem['max_reserved']:.3f} GB\n")
        
        # Process memory
        proc_mem = memory_info['process_memory']
        f.write(f"PROCESS MEMORY:\n")
        f.write(f"  RSS: {proc_mem['rss_gb']:.3f} GB\n")
        f.write(f"  VMS: {proc_mem['vms_gb']:.3f} GB\n")
    
    print(f"Memory log saved to: {filename}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory monitoring for EWA training")
    parser.add_argument("--monitor", action="store_true", help="Monitor memory continuously")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--max-iterations", type=int, help="Maximum monitoring iterations")
    parser.add_argument("--save-log", action="store_true", help="Save memory log to file")
    parser.add_argument("--log-file", type=str, help="Custom log filename")
    
    args = parser.parse_args()
    
    if args.monitor:
        monitor_memory_continuously(args.interval, args.max_iterations)
    elif args.save_log:
        save_memory_log(args.log_file)
    else:
        # Default: just show current memory status
        memory_info = get_memory_info()
        print_memory_info(memory_info)
