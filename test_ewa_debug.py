#!/usr/bin/env python3
"""
Test script to run EWA VQ PQ with debug prints for the first two batches.
"""

import subprocess
import sys

def run_ewa_debug_test():
    """Run the EWA VQ PQ experiment with debug prints."""
    
    cmd = [
        "python", "main.py",
        "--env", "hopper-medium-v2",
        "--max_pretrain_iters", "1",
        "--max_online_iters", "1",
        "--eval_interval", "1",
        "--seed", "1",
        "--online_rtg", "1800",
        "--eval_rtg", "1800",
        "--eval_context_length", "5",
        "--ordering", "1",
        "--num_updates_per_pretrain_iter", "1",
        "--batch_size", "2",
        "--K", "10",
        "--exp_name", "ewa",
        "--beta", "0.05",
        "--phi", "0.05",
        "--delta", "0.8",
        "--grid_bins", "3",
        "--max_subspaces", "4",
        "--codes_per_subspace", "27"
    ]
    
    print("🚀 Running EWA VQ PQ Debug Test...")
    print(f"Command: {' '.join(cmd)}")
    print("\n" + "="*80)
    
    try:
        # Run the experiment
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ EWA VQ PQ Debug Test completed successfully!")
        else:
            print(f"\n❌ EWA VQ PQ Debug Test failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"\n❌ Error running EWA VQ PQ Debug Test: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(run_ewa_debug_test())
