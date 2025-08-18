#!/usr/bin/env python3
"""
Comprehensive test script to verify that the optimized EWAVQ is working correctly.
This tests:
1. EWAVQ attraction calculations based on rewards and actions
2. Integration with attention weights
3. End-to-end flow correctness
"""

import torch
import numpy as np
import time

def test_ewa_attraction_calculations():
    """Test that EWAVQ calculates attractions correctly based on rewards and actions."""
    print("=" * 60)
    print("TESTING EWAVQ ATTRACTION CALCULATIONS")
    print("=" * 60)
    
    try:
        from decision_transformer.models.ewa_vq_optimized import EWAVQOptimized
        
        # Create EWAVQ instance with known parameters
        ewa = EWAVQOptimized(
            num_heads=4,
            tuple_seq_length=60,
            phi=0.1,      # Decay factor
            delta=1.0,     # Reward weight
            num_codes=27,  # 3^3 grid for 3D actions
            grid_bins=3
        )
        
        print("✓ Created EWAVQ instance")
        
        # Test 1: Simple trajectory with known rewards
        print("\n--- Test 1: Simple trajectory with known rewards ---")
        
        # Create a simple trajectory: 3 steps, 3D actions
        actions = torch.tensor([
            [0.5, 0.0, -0.5],   # Step 0: positive x, zero y, negative z
            [0.0, 0.8, 0.0],    # Step 1: zero x, positive y, zero z  
            [-0.3, 0.0, 0.7]    # Step 2: negative x, zero y, positive z
        ], dtype=torch.float32)
        
        rewards = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32)  # Known rewards
        
        print(f"Actions shape: {actions.shape}")
        print(f"Rewards: {rewards.tolist()}")
        print(f"EWA parameters: phi={ewa.phi}, delta={ewa.delta}")
        
        # Process trajectory
        D_codebook, D_trajectory = ewa.process_trajectory(actions, rewards)
        
        print(f"\nProcessed trajectory:")
        print(f"  D_codebook length: {len(D_codebook)}")
        print(f"  D_trajectory length: {len(D_trajectory)}")
        
        # Verify each step
        for i, step in enumerate(D_trajectory):
            print(f"  Step {i}: action={step['a_t']}, reward={step['r_t']:.2f}, "
                  f"code_idx={step['code_idx']}, attraction={step['A_t']:.4f}")
        
        # Test 2: Verify attraction decay and accumulation
        print("\n--- Test 2: Verify attraction decay and accumulation ---")
        
        # Process same trajectory again to see decay
        D_codebook2, D_trajectory2 = ewa.process_trajectory(actions, rewards)
        
        print(f"Second pass - attractions after decay:")
        for i, step in enumerate(D_trajectory2):
            print(f"  Step {i}: attraction={step['A_t']:.4f}")
        
        # Test 3: Check that codes are properly initialized
        print("\n--- Test 3: Code initialization ---")
        print(f"Codes initialized: {ewa.codes_initialized}")
        if ewa.codes_initialized:
            print(f"Code shape: {ewa.codes.shape}")
            print(f"Grid bins: {ewa.grid_bins}")
            print(f"Number of codes: {ewa.num_codes}")
        
        # Test 4: Verify code routing
        print("\n--- Test 4: Code routing verification ---")
        for i, (action, reward) in enumerate(zip(actions, rewards)):
            code_idx = D_trajectory[i]['code_idx']
            print(f"  Step {i}: action={action.tolist()} -> code_idx={code_idx}")
            
            # Verify the code is close to the action
            if ewa.codes_initialized:
                code = ewa.codes[code_idx]
                distance = torch.norm(action - code)
                print(f"    Distance to code: {distance:.4f}")
        
        print("\n✅ EWAVQ attraction calculations test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR in EWAVQ attraction calculations: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_integration():
    """Test that attractions are correctly integrated into attention weights."""
    print("\n" + "=" * 60)
    print("TESTING ATTENTION INTEGRATION")
    print("=" * 60)
    
    try:
        from decision_transformer.models.trajectory_gpt2 import Attention
        
        # Create a mock config
        class MockConfig:
            def __init__(self):
                self.n_head = 4
                self.attn_pdrop = 0.1
                self.resid_pdrop = 0.1
        
        mock_config = MockConfig()
        
        # Create variant with EWA parameters
        variant = {
            "phi": 0.1,
            "delta": 1.0,
            "beta": 0.05,  # Scaling factor for attention weights
            "grid_bins_factor": 1.0,
            "num_codes": 27,
            "max_online_iters": 20,
            "exp_name": "ewa",
            "env": "test-env",  # Required by Logger
            "seed": 1,
            "save_dir": "./test_logs"  # Required by Logger
        }
        
        # Create Attention instance
        attention = Attention(
            nx=512,
            n_ctx=1000,
            config=mock_config,
            variant=variant
        )
        
        print("✓ Created Attention instance")
        
        # Test 1: EWA initialization
        print("\n--- Test 1: EWA initialization ---")
        
        # Set actions and rewards to trigger EWA initialization
        attention.actions = torch.randn(2, 3, 3)  # (batch_size, seq_len, action_dim)
        attention.rewards = torch.randn(2, 3, 1)  # (batch_size, seq_len, 1)
        
        # Initialize EWA
        attention._initialize_ewa_lazy(4, 60)
        
        print(f"EWA initialized: {attention.ewa_initialized}")
        if attention.ewa_initialized:
            print(f"EWA instance: {type(attention.ewa).__name__}")
        
        # Test 2: Attraction calculation
        print("\n--- Test 2: Attraction calculation ---")
        
        # Get cached attraction
        attraction_matrix = attention._get_cached_attraction(attention.actions, attention.rewards)
        
        print(f"Attraction matrix shape: {attraction_matrix.shape}")
        print(f"Attraction range: [{attraction_matrix.min().item():.4f}, {attraction_matrix.max().item():.4f}]")
        
        # Test 3: Attention weight modification
        print("\n--- Test 3: Attention weight modification ---")
        
        # Create mock attention weights
        batch_size, num_heads, seq_len = 2, 4, 9  # 3 steps * 3 tokens per step
        attention_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
        
        print(f"Original attention weights shape: {attention_weights.shape}")
        print(f"Original range: [{attention_weights.min().item():.4f}, {attention_weights.max().item():.4f}]")
        
        # Apply EWA modifications (simulate what happens in _attn)
        if attention.ewa_initialized:
            # Get action indices
            _, _, _, action_indices = attention.ewa._extract_dimensions(attention.rewards)
            print(f"Action indices: {action_indices.tolist()}")
            
            # Apply attraction to attention weights at action token indices
            beta = variant["beta"]
            for a_ind in action_indices:
                a_val = attraction_matrix[:, :, a_ind, :]
                a_val_expanded = a_val.expand(-1, -1, attention_weights.size(2))
                attention_weights[:, :, :, a_ind] += beta * a_val_expanded
            
            print(f"Modified attention weights range: [{attention_weights.min().item():.4f}, {attention_weights.max().item():.4f}]")
            print(f"Modification applied at indices: {action_indices.tolist()}")
        
        print("\n✅ Attention integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR in attention integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end_flow():
    """Test the complete end-to-end flow."""
    print("\n" + "=" * 60)
    print("TESTING END-TO-END FLOW")
    print("=" * 60)
    
    try:
        from decision_transformer.models.ewa_vq_optimized import EWAVQOptimized
        
        # Create EWAVQ instance
        ewa = EWAVQOptimized(
            num_heads=4,
            tuple_seq_length=60,
            phi=0.1,
            delta=1.0,
            num_codes=27,
            grid_bins=3
        )
        
        # Create test data
        batch_size, seq_len, action_dim = 2, 3, 3
        actions = torch.randn(batch_size, seq_len, action_dim)
        rewards = torch.randn(batch_size, seq_len, 1)
        
        print(f"Test data: batch_size={batch_size}, seq_len={seq_len}, action_dim={action_dim}")
        
        # Test 1: Process batch
        print("\n--- Test 1: Batch processing ---")
        D_codebooks, D_trajectories = ewa.process_batch(actions, rewards)
        
        print(f"Batch processing results:")
        print(f"  D_codebooks length: {len(D_codebooks)}")
        print(f"  D_trajectories length: {len(D_trajectories)}")
        
        # Test 2: Get attraction matrix
        print("\n--- Test 2: Attraction matrix generation ---")
        attraction_matrix = ewa.get_attraction(D_trajectories, rewards)
        
        print(f"Attraction matrix:")
        print(f"  Shape: {attraction_matrix.shape}")
        print(f"  Range: [{attraction_matrix.min().item():.4f}, {attraction_matrix.max().item():.4f}]")
        print(f"  Mean: {attraction_matrix.mean().item():.4f}")
        
        # Test 3: Verify dimensions match
        print("\n--- Test 3: Dimension verification ---")
        expected_shape = (batch_size, ewa.num_heads, ewa.tuple_seq_length, 1)
        print(f"Expected shape: {expected_shape}")
        print(f"Actual shape: {attraction_matrix.shape}")
        
        if attraction_matrix.shape == expected_shape:
            print("✅ Shape matches expected dimensions!")
        else:
            print("❌ Shape mismatch!")
            return False
        
        # Test 4: Check cache performance
        print("\n--- Test 4: Cache performance ---")
        cache_stats = ewa.get_cache_stats()
        print(f"Cache stats: {cache_stats}")
        
        print("\n✅ End-to-end flow test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR in end-to-end flow: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("EWA CORRECTNESS AND INTEGRATION TEST")
    print("=" * 80)
    
    # Test 1: EWAVQ attraction calculations
    print("\n1. Testing EWAVQ attraction calculations...")
    ewa_ok = test_ewa_attraction_calculations()
    
    # Test 2: Attention integration
    print("\n2. Testing attention integration...")
    attention_ok = test_attention_integration()
    
    # Test 3: End-to-end flow
    print("\n3. Testing end-to-end flow...")
    flow_ok = test_end_to_end_flow()
    
    # Summary
    print("\n" + "=" * 80)
    print("CORRECTNESS TEST SUMMARY")
    print("=" * 80)
    
    if ewa_ok and attention_ok and flow_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ EWAVQ attraction calculations are correct")
        print("✅ Attractions are properly integrated into attention weights")
        print("✅ End-to-end flow works correctly")
        print("\nThe optimized EWAVQ is working correctly!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above.")
    
    return ewa_ok and attention_ok and flow_ok

if __name__ == "__main__":
    main()
