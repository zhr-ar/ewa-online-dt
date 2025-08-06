#!/usr/bin/env python3
"""
Test script to debug the actual grid lookup process during runtime.
This will help identify if route_action_fast is working correctly.
"""

import torch
import numpy as np
from decision_transformer.models.ewa_vq import EWAVQ

def test_grid_lookup_process():
    """Test the actual grid lookup process step by step."""
    print("=== Testing Grid Lookup Process ===")
    
    # Create EWAVQ instance with hopper parameters
    ewa = EWAVQ(
        num_heads=8,
        tuple_seq_length=1000,
        phi=0.05,
        delta=0.8,
        num_codes=16,
        grid_bins=3  # hopper: 1.0 * 3 = 3 bins
    )
    
    # Set variant for environment name
    ewa.variant = {"env": "hopper-medium-v2"}
    
    # Load grid table
    print("Loading grid table...")
    ewa._load_grid_table_from_cache(action_dim=3, env_name="hopper-medium-v2")
    
    if ewa.grid_table is None:
        print("❌ Grid table not loaded!")
        return False
    
    print(f"✅ Grid table loaded: {ewa.grid_table.shape}")
    print(f"Grid table range: [{ewa.grid_table.min().item()}, {ewa.grid_table.max().item()}]")
    
    # Test action to grid index conversion
    print("\n=== Testing Action to Grid Index ===")
    
    # Create some test actions (3D for hopper)
    test_actions = [
        torch.tensor([0.0, 0.0, 0.0]),  # Center
        torch.tensor([1.0, 1.0, 1.0]),  # Max
        torch.tensor([-1.0, -1.0, -1.0]),  # Min
        torch.tensor([0.5, -0.3, 0.8]),  # Random
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\nTest action {i+1}: {action.tolist()}")
        
        # Test _action_to_grid_idx
        grid_idx = ewa._action_to_grid_idx(action)
        print(f"  Grid index: {grid_idx}")
        print(f"  Grid index valid: {0 <= grid_idx < len(ewa.grid_table)}")
        
        if 0 <= grid_idx < len(ewa.grid_table):
            code_idx = ewa.grid_table[grid_idx].item()
            print(f"  Code index from grid: {code_idx}")
        else:
            print(f"  ❌ Grid index out of bounds!")
    
    # Test route_action_fast
    print("\n=== Testing route_action_fast ===")
    
    # Initialize codes (random for testing)
    ewa.codes = torch.randn(16, 3, device=ewa.device, dtype=torch.float32)
    ewa.codes_initialized = True
    
    for i, action in enumerate(test_actions):
        print(f"\nTest action {i+1}: {action.tolist()}")
        
        # Test route_action_fast
        code_idx = ewa.route_action_fast(action)
        print(f"  Routed code index: {code_idx}")
        print(f"  Code index valid: {0 <= code_idx < 16}")
        
        # Also test direct computation for comparison
        distances = torch.norm(ewa.codes - action, dim=1)
        direct_code_idx = torch.argmin(distances).item()
        print(f"  Direct computation code index: {direct_code_idx}")
        
        if code_idx == direct_code_idx:
            print(f"  ✅ Grid lookup matches direct computation!")
        else:
            print(f"  ⚠ Grid lookup differs from direct computation")
    
    return True

def test_runtime_performance():
    """Test runtime performance of grid lookup vs direct computation."""
    print("\n=== Testing Runtime Performance ===")
    
    # Create EWAVQ instance
    ewa = EWAVQ(
        num_heads=8,
        tuple_seq_length=1000,
        phi=0.05,
        delta=0.8,
        num_codes=16,
        grid_bins=3
    )
    
    ewa.variant = {"env": "hopper-medium-v2"}
    ewa._load_grid_table_from_cache(action_dim=3, env_name="hopper-medium-v2")
    ewa.codes = torch.randn(16, 3, device=ewa.device, dtype=torch.float32)
    ewa.codes_initialized = True
    
    # Generate random actions
    num_actions = 1000
    actions = torch.randn(num_actions, 3, device=ewa.device) * 2 - 1  # [-1, 1]
    
    print(f"Testing {num_actions} random actions...")
    
    # Test grid lookup performance
    import time
    
    # Grid lookup
    start_time = time.time()
    grid_results = []
    for action in actions:
        code_idx = ewa.route_action_fast(action)
        grid_results.append(code_idx)
    grid_time = time.time() - start_time
    
    # Direct computation
    start_time = time.time()
    direct_results = []
    for action in actions:
        distances = torch.norm(ewa.codes - action, dim=1)
        code_idx = torch.argmin(distances).item()
        direct_results.append(code_idx)
    direct_time = time.time() - start_time
    
    print(f"Grid lookup time: {grid_time:.4f}s ({num_actions/grid_time:.0f} actions/sec)")
    print(f"Direct computation time: {direct_time:.4f}s ({num_actions/direct_time:.0f} actions/sec)")
    print(f"Speedup: {direct_time/grid_time:.2f}x")
    
    # Check if results match
    matches = sum(1 for g, d in zip(grid_results, direct_results) if g == d)
    print(f"Results match: {matches}/{num_actions} ({matches/num_actions*100:.1f}%)")
    
    return matches == num_actions

def test_batch_processing():
    """Test batch processing to see if that's where the issue is."""
    print("\n=== Testing Batch Processing ===")
    
    # Create EWAVQ instance
    ewa = EWAVQ(
        num_heads=8,
        tuple_seq_length=1000,
        phi=0.05,
        delta=0.8,
        num_codes=16,
        grid_bins=3
    )
    
    ewa.variant = {"env": "hopper-medium-v2"}
    ewa._load_grid_table_from_cache(action_dim=3, env_name="hopper-medium-v2")
    ewa.codes = torch.randn(16, 3, device=ewa.device, dtype=torch.float32)
    ewa.codes_initialized = True
    
    # Create a small batch (like in your test)
    batch_size = 4
    seq_length = 10
    actions = torch.randn(batch_size, seq_length, 3, device=ewa.device) * 2 - 1
    rewards = torch.randn(batch_size, seq_length, device=ewa.device)
    
    print(f"Processing batch: {actions.shape}")
    
    try:
        # Test process_batch
        D_kernels, D_trajectories = ewa.process_batch(actions, rewards)
        print(f"✅ Batch processing completed!")
        print(f"  - D_kernels: {len(D_kernels)}")
        print(f"  - D_trajectories: {len(D_trajectories)}")
        
        # Test get_attraction
        attr = ewa.get_attraction(D_trajectories, rewards)
        print(f"✅ Attraction matrix generated: {attr.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Grid Lookup Debug Test")
    print("=" * 50)
    
    # Test 1: Grid lookup process
    lookup_ok = test_grid_lookup_process()
    
    # Test 2: Runtime performance
    perf_ok = test_runtime_performance()
    
    # Test 3: Batch processing
    batch_ok = test_batch_processing()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  - Grid lookup process: {'✅ PASS' if lookup_ok else '❌ FAIL'}")
    print(f"  - Runtime performance: {'✅ PASS' if perf_ok else '❌ FAIL'}")
    print(f"  - Batch processing: {'✅ PASS' if batch_ok else '❌ FAIL'}")
    
    if lookup_ok and perf_ok and batch_ok:
        print("\n✅ Grid lookup is working correctly!")
        print("The issue with run.py getting stuck is likely due to scale, not lookup.")
    else:
        print("\n❌ Issues found in grid lookup process.")
        print("This may be causing run.py to get stuck.") 