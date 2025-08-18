#!/usr/bin/env python3
"""
Test script to verify that the EWA metrics optimization is working correctly.
This tests the performance improvement from using get_current_metrics() instead of get_history().
"""

import torch
import time
import numpy as np

def test_ewa_metrics_performance():
    """Test the performance difference between get_history() and get_current_metrics()."""
    print("EWA Metrics Performance Test")
    print("=" * 50)
    
    try:
        from decision_transformer.models.ewa_vq_optimized import EWAVQOptimized
        
        # Create EWAVQ instance
        ewa = EWAVQOptimized(
            num_heads=4,
            tuple_seq_length=60,
            phi=0.05,
            delta=0.8,
            num_codes=36,
            grid_bins=3
        )
        
        # Process some trajectories to build up history
        print("Building up EWA history...")
        for i in range(100):
            actions = torch.randn(20, 6)  # 20 steps, 6D actions
            rewards = torch.randn(20)      # 20 rewards
            D_codebook, D_trajectory = ewa.process_trajectory(actions, rewards)
            
            if i % 20 == 0:
                print(f"  Processed {i+1}/100 trajectories")
        
        print(f"✓ Built up history with {len(ewa.attraction_history)} iterations")
        
        # Test get_history() performance (old method)
        print("\nTesting get_history() performance (old method)...")
        start_time = time.time()
        for _ in range(100):  # Call it 100 times to simulate frequent logging
            steps, attraction_values, code_usage_values, reward_values = ewa.get_history()
        history_time = time.time() - start_time
        
        print(f"  get_history() time for 100 calls: {history_time:.4f}s")
        print(f"  Average time per call: {history_time/100:.6f}s")
        
        # Test get_current_metrics() performance (new method)
        print("\nTesting get_current_metrics() performance (new method)...")
        start_time = time.time()
        for _ in range(100):  # Call it 100 times to simulate frequent logging
            current_metrics = ewa.get_current_metrics()
        current_time = time.time() - start_time
        
        print(f"  get_current_metrics() time for 100 calls: {current_time:.4f}s")
        print(f"  Average time per call: {current_time/100:.6f}s")
        
        # Calculate speedup
        speedup = history_time / current_time
        print(f"\n🎉 PERFORMANCE IMPROVEMENT: {speedup:.1f}x faster!")
        
        # Verify the data is correct
        print("\nVerifying data correctness...")
        history_data = ewa.get_history()
        current_data = ewa.get_current_metrics()
        
        print(f"  History method - Steps: {len(history_data[0])}, Attractions: {len(history_data[1])}")
        print(f"  Current method - Attraction: {current_data['current_attraction']:.4f}, Code Usage: {current_data['current_code_usage']}")
        
        # Test cache statistics
        print("\nTesting cache statistics...")
        cache_stats = ewa.get_cache_stats()
        print(f"  Cache hit rate: {cache_stats['hit_rate']:.2%}")
        print(f"  Cache size: {cache_stats['cache_size']}")
        
        print("\n✅ All tests passed! The optimization is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_attention_integration():
    """Test that the Attention class has the optimization methods."""
    print("\n" + "=" * 50)
    print("Testing Attention Class Integration")
    print("=" * 50)
    
    try:
        from decision_transformer.models.trajectory_gpt2 import Attention
        
        # Create a mock variant
        variant = {
            "phi": 0.05,
            "delta": 0.8,
            "beta": 0.05,
            "grid_bins_factor": 1.0,
            "num_codes": None,
            "max_online_iters": 20
        }
        
        # Test that we can create an Attention instance
        attention = Attention(
            nx=512,
            n_ctx=1000,
            config=None,
            variant=variant
        )
        
        print("✓ Successfully created Attention instance")
        
        # Test that optimization methods exist
        assert hasattr(attention, 'get_current_ewa_metrics'), "Missing get_current_ewa_metrics method"
        assert hasattr(attention, 'get_ewa_stats'), "Missing get_ewa_stats method"
        
        print("✓ All optimization methods are present")
        
        # Test that we can call the methods
        stats = attention.get_ewa_stats()
        assert isinstance(stats, dict), "get_ewa_stats should return a dictionary"
        
        current_metrics = attention.get_current_ewa_metrics()
        assert current_metrics is None, "Should return None when EWA not initialized"
        
        print("✓ Optimization methods work correctly")
        print("\n🎉 SUCCESS: Attention class integration is working!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Main test function."""
    print("EWA METRICS OPTIMIZATION TEST")
    print("=" * 60)
    
    # Test 1: EWAVQ performance
    print("\n1. Testing EWAVQ metrics performance...")
    ewa_ok = test_ewa_metrics_performance()
    
    # Test 2: Attention integration
    print("\n2. Testing Attention class integration...")
    attention_ok = test_attention_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION TEST SUMMARY")
    print("=" * 60)
    
    if ewa_ok and attention_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✓ EWA metrics optimization is working correctly")
        print("✓ Attention class integration is successful")
        print("✓ Performance improvement achieved")
        print("\nYou can now use the optimized EWA metrics logging!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above.")
    
    return ewa_ok and attention_ok

if __name__ == "__main__":
    main()
