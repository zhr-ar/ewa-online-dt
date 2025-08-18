#!/usr/bin/env python3
"""
Test script to demonstrate the performance improvements of the optimized EWAVQ implementation.
"""

import torch
import time
import numpy as np
from decision_transformer.models.ewa_vq import EWAVQ
from decision_transformer.models.ewa_vq_optimized import EWAVQOptimized

def create_test_data(batch_size=4, seq_length=20, action_dim=6, num_trajectories=10):
    """Create test data for performance comparison."""
    actions = torch.randn(batch_size, seq_length, action_dim)
    rewards = torch.randn(batch_size, seq_length)
    
    # Create multiple trajectories for testing
    trajectories = []
    for _ in range(num_trajectories):
        traj_actions = torch.randn(seq_length, action_dim)
        traj_rewards = torch.randn(seq_length)
        trajectories.append((traj_actions, traj_rewards))
    
    return actions, rewards, trajectories

def test_original_ewavq(trajectories, num_heads=4, tuple_seq_length=60):
    """Test the original EWAVQ implementation."""
    print("\n" + "="*50)
    print("TESTING ORIGINAL EWAVQ")
    print("="*50)
    
    ewa = EWAVQ(
        num_heads=num_heads,
        tuple_seq_length=tuple_seq_length,
        phi=0.05,
        delta=0.8,
        num_codes=36,
        grid_bins=3
    )
    
    # Warm up
    for _ in range(3):
        for actions, rewards in trajectories[:2]:
            _ = ewa.process_trajectory(actions, rewards)
    
    # Performance test
    start_time = time.time()
    for actions, rewards in trajectories:
        D_codebook, D_trajectory = ewa.process_trajectory(actions, rewards)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Original EWAVQ:")
    print(f"  - Processed {len(trajectories)} trajectories")
    print(f"  - Total time: {total_time:.4f} seconds")
    print(f"  - Average time per trajectory: {total_time/len(trajectories):.4f} seconds")
    
    return total_time

def test_optimized_ewavq(trajectories, num_heads=4, tuple_seq_length=60):
    """Test the optimized EWAVQ implementation."""
    print("\n" + "="*50)
    print("TESTING OPTIMIZED EWAVQ")
    print("="*50)
    
    ewa = EWAVQOptimized(
        num_heads=num_heads,
        tuple_seq_length=tuple_seq_length,
        phi=0.05,
        delta=0.8,
        num_codes=36,
        grid_bins=3
    )
    
    # Warm up
    for _ in range(3):
        for actions, rewards in trajectories[:2]:
            _ = ewa.process_trajectory(actions, rewards)
    
    # Performance test
    start_time = time.time()
    for actions, rewards in trajectories:
        D_codebook, D_trajectory = ewa.process_trajectory(actions, rewards)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Get cache statistics
    cache_stats = ewa.get_cache_stats()
    
    print(f"Optimized EWAVQ:")
    print(f"  - Processed {len(trajectories)} trajectories")
    print(f"  - Total time: {total_time:.4f} seconds")
    print(f"  - Average time per trajectory: {total_time/len(trajectories):.4f} seconds")
    print(f"  - Cache hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"  - Cache hits: {cache_stats['cache_hits']}")
    print(f"  - Cache misses: {cache_stats['cache_misses']}")
    
    return total_time

def test_batch_processing(actions, rewards, num_heads=4, tuple_seq_length=60):
    """Test batch processing performance."""
    print("\n" + "="*50)
    print("TESTING BATCH PROCESSING")
    print("="*50)
    
    # Test original EWAVQ batch processing
    ewa_original = EWAVQ(
        num_heads=num_heads,
        tuple_seq_length=tuple_seq_length,
        phi=0.05,
        delta=0.8,
        num_codes=36,
        grid_bins=3
    )
    
    start_time = time.time()
    D_codebooks, D_trajectories = ewa_original.process_batch(actions, rewards)
    end_time = time.time()
    original_batch_time = end_time - start_time
    
    print(f"Original EWAVQ batch processing:")
    print(f"  - Batch shape: {actions.shape}")
    print(f"  - Time: {original_batch_time:.4f} seconds")
    
    # Test optimized EWAVQ batch processing
    ewa_optimized = EWAVQOptimized(
        num_heads=num_heads,
        tuple_seq_length=tuple_seq_length,
        phi=0.05,
        delta=0.8,
        num_codes=36,
        grid_bins=3
    )
    
    start_time = time.time()
    D_codebooks, D_trajectories = ewa_optimized.process_batch(actions, rewards)
    end_time = time.time()
    optimized_batch_time = end_time - start_time
    
    print(f"Optimized EWAVQ batch processing:")
    print(f"  - Batch shape: {actions.shape}")
    print(f"  - Time: {optimized_batch_time:.4f} seconds")
    print(f"  - Speedup: {original_batch_time/optimized_batch_time:.2f}x")
    
    return original_batch_time, optimized_batch_time

def test_memory_management():
    """Test memory management features."""
    print("\n" + "="*50)
    print("TESTING MEMORY MANAGEMENT")
    print("="*50)
    
    ewa = EWAVQOptimized(
        num_heads=4,
        tuple_seq_length=60,
        phi=0.05,
        delta=0.8,
        num_codes=36,
        grid_bins=3
    )
    
    # Create many trajectories to test cache cleanup
    trajectories = []
    for i in range(1500):  # More than max_cache_size (1000)
        actions = torch.randn(20, 6)
        rewards = torch.randn(20)
        trajectories.append((actions, rewards))
    
    print(f"Processing {len(trajectories)} trajectories to test cache management...")
    
    start_time = time.time()
    for i, (actions, rewards) in enumerate(trajectories):
        if i % 200 == 0:
            print(f"  Processed {i}/{len(trajectories)} trajectories...")
        D_codebook, D_trajectory = ewa.process_trajectory(actions, rewards)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Get final cache statistics
    cache_stats = ewa.get_cache_stats()
    
    print(f"Memory management test completed:")
    print(f"  - Total time: {total_time:.4f} seconds")
    print(f"  - Final cache size: {cache_stats['cache_size']}")
    print(f"  - Max cache size: {cache_stats['max_cache_size']}")
    print(f"  - Cache hit rate: {cache_stats['hit_rate']:.2%}")
    
    # Test cache clearing
    print("\nTesting cache clearing...")
    ewa.clear_cache()
    cache_stats_after_clear = ewa.get_cache_stats()
    print(f"  - Cache size after clearing: {cache_stats_after_clear['cache_size']}")
    
    return total_time

def main():
    """Main test function."""
    print("EWA-VQ OPTIMIZATION PERFORMANCE TEST")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create test data
    print("Creating test data...")
    actions, rewards, trajectories = create_test_data(
        batch_size=4, 
        seq_length=20, 
        action_dim=6, 
        num_trajectories=100
    )
    
    # Test individual trajectory processing
    original_time = test_original_ewavq(trajectories)
    optimized_time = test_optimized_ewavq(trajectories)
    
    # Calculate speedup
    speedup = original_time / optimized_time
    print(f"\nOVERALL SPEEDUP: {speedup:.2f}x")
    
    # Test batch processing
    test_batch_processing(actions, rewards)
    
    # Test memory management
    test_memory_management()
    
    print("\n" + "="*60)
    print("PERFORMANCE TEST COMPLETED")
    print("="*60)
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"  - Original EWAVQ time: {original_time:.4f}s")
    print(f"  - Optimized EWAVQ time: {optimized_time:.4f}s")
    print(f"  - Performance improvement: {speedup:.2f}x")
    print(f"  - Time saved: {original_time - optimized_time:.4f}s")
    
    if speedup > 2.0:
        print(f"  ✓ SIGNIFICANT PERFORMANCE IMPROVEMENT ACHIEVED!")
    elif speedup > 1.5:
        print(f"  ✓ GOOD PERFORMANCE IMPROVEMENT ACHIEVED!")
    else:
        print(f"  ⚠ MODEST PERFORMANCE IMPROVEMENT")

if __name__ == "__main__":
    main()
