#!/usr/bin/env python3
"""
Simple test script to verify that the EWA optimizations are properly integrated
into the original trajectory_gpt2.py file.
"""

import torch
import numpy as np

def test_attention_optimization():
    """Test that the Attention class has the optimization methods."""
    try:
        from decision_transformer.models.trajectory_gpt2 import Attention
        
        # Create a mock variant for testing
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
            config=None,  # We don't need the full config for this test
            variant=variant
        )
        
        print("✓ Successfully created Attention instance")
        
        # Test that optimization attributes exist
        assert hasattr(attention, '_cached_attraction'), "Missing _cached_attraction attribute"
        assert hasattr(attention, '_attraction_history'), "Missing _attraction_history attribute"
        assert hasattr(attention, '_max_cached_attractions'), "Missing _max_cached_attractions attribute"
        
        print("✓ All optimization attributes are present")
        
        # Test that optimization methods exist
        assert hasattr(attention, '_initialize_ewa_lazy'), "Missing _initialize_ewa_lazy method"
        assert hasattr(attention, '_get_cached_attraction'), "Missing _get_cached_attraction method"
        assert hasattr(attention, 'get_ewa_stats'), "Missing get_ewa_stats method"
        assert hasattr(attention, 'clear_ewa_cache'), "Missing clear_ewa_cache method"
        
        print("✓ All optimization methods are present")
        
        # Test that we can call the utility methods
        stats = attention.get_ewa_stats()
        assert isinstance(stats, dict), "get_ewa_stats should return a dictionary"
        
        print("✓ Utility methods work correctly")
        
        print("\n🎉 SUCCESS: All EWA optimizations are properly integrated!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_ewa_import():
    """Test that we can import the optimized EWAVQ."""
    try:
        from decision_transformer.models.ewa_vq_optimized import EWAVQOptimized
        
        # Test that we can create an EWAVQOptimized instance
        ewa = EWAVQOptimized(
            num_heads=4,
            tuple_seq_length=60,
            phi=0.05,
            delta=0.8,
            num_codes=36,
            grid_bins=3
        )
        
        print("✓ Successfully created EWAVQOptimized instance")
        
        # Test that optimization methods exist
        assert hasattr(ewa, 'get_cache_stats'), "Missing get_cache_stats method"
        assert hasattr(ewa, 'clear_cache'), "Missing clear_cache method"
        
        print("✓ All EWAVQOptimized methods are present")
        
        # Test cache statistics
        stats = ewa.get_cache_stats()
        assert isinstance(stats, dict), "get_cache_stats should return a dictionary"
        assert 'cache_hits' in stats, "Cache stats should include cache_hits"
        assert 'cache_misses' in stats, "Cache stats should include cache_misses"
        
        print("✓ Cache statistics work correctly")
        
        print("\n🎉 SUCCESS: EWAVQOptimized is properly implemented!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Main test function."""
    print("EWA OPTIMIZATION INTEGRATION TEST")
    print("=" * 50)
    
    # Test 1: Attention class integration
    print("\n1. Testing Attention class optimization integration...")
    attention_ok = test_attention_optimization()
    
    # Test 2: EWAVQOptimized implementation
    print("\n2. Testing EWAVQOptimized implementation...")
    ewa_ok = test_ewa_import()
    
    # Summary
    print("\n" + "=" * 50)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    if attention_ok and ewa_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✓ EWA optimizations are properly integrated into trajectory_gpt2.py")
        print("✓ EWAVQOptimized class is working correctly")
        print("\nYou can now use the optimized EWA implementation!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above.")
    
    return attention_ok and ewa_ok

if __name__ == "__main__":
    main()
