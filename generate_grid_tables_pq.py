#!/usr/bin/env python3
"""
Script to pre-generate Product Quantization (PQ) codebooks for common environments.
This solves the curse of dimensionality by splitting high-dimensional action spaces into subspaces.

USAGE:
    python generate_grid_tables_pq.py --env walker2d-medium-replay-v2 --action_dim 6
    python generate_grid_tables_pq.py --env hopper-medium-v2 --action_dim 3
"""

import torch
import os
import pickle
import argparse
from decision_transformer.models.ewa_vq_pq import EWAVQProductQuantization
import numpy as np

def generate_pq_codebooks_offline(action_dim: int, num_subspaces: int = None, 
                                 grid_bins: int = None,
                                 codes_per_subspace: int = None, device: str = "cuda", 
                                 env_name: str = None):
    """
    Generate PQ codebooks offline and cache them for reuse.
    Args:
        action_dim: dimension of action space
        num_subspaces: number of subspaces (auto-determined if None)
        grid_bins: number of bins per dimension (auto-determined if None)
        codes_per_subspace: codes per subspace (auto-determined if None)
        device: device to create tensors on
        env_name: environment name for cache key
    Returns:
        pq_instance: initialized PQ-EWA instance with cached codebooks
    """
    print(f"\n{'='*60}")
    print(f"Generating PQ codebooks for {action_dim}D action space")
    print(f"Environment: {env_name or 'unknown'}")
    print(f"{'='*60}")
    
    # Create PQ-EWA instance (this will auto-determine optimal parameters)
    pq_instance = EWAVQProductQuantization(
        num_heads=4,  # Default, will be overridden during actual use
        tuple_seq_length=60,  # Default, will be overridden during actual use
        phi=0.05,  # Default, will be overridden during actual use
        delta=0.8,  # Default, will be overridden during actual use
        action_dim=action_dim,
        num_subspaces=num_subspaces,
        codes_per_subspace=codes_per_subspace,
        grid_bins=grid_bins,  # Use provided value or None for auto-determination
        max_subspaces=4  # Default max subspaces
    )
    
    # Initialize codebooks (this generates and caches them)
    pq_instance.initialize_codebooks()
    
    # Generate cache key for this configuration
    cache_key = EWAVQProductQuantization._get_cache_key(
        action_dim, 
        pq_instance.num_subspaces, 
        pq_instance.grid_bins,
        pq_instance.codes_per_subspace, 
        env_name
    )
    
    # Save to file cache
    cache_path = EWAVQProductQuantization._get_cache_path(cache_key)
    print(f"\nSaving PQ codebooks to cache: {cache_path}")
    
    # Prepare data for caching
    cache_data = {
        'action_dim': action_dim,
        'num_subspaces': pq_instance.num_subspaces,
        'grid_bins': pq_instance.grid_bins,
        'codes_per_subspace': pq_instance.codes_per_subspace,
        'total_codes': pq_instance.total_codes,
        'subspaces': pq_instance.subspaces,
        'subspace_codebooks': [cb.cpu() for cb in pq_instance.subspace_codebooks],  # Move to CPU for storage
        'env_name': env_name
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    # Store in memory cache
    EWAVQProductQuantization._subspace_cache[cache_key] = cache_data
    
    print(f"✓ PQ codebooks cached successfully!")
    
    return pq_instance

def analyze_pq_coverage(pq_instance: EWAVQProductQuantization):
    """
    Analyze the coverage and efficiency of the PQ codebooks.
    Args:
        pq_instance: initialized PQ-EWA instance
    """
    print(f"\n{'='*60}")
    print("PQ CODEBOOK ANALYSIS")
    print(f"{'='*60}")
    
    action_dim = pq_instance.action_dim
    num_subspaces = pq_instance.num_subspaces
    grid_bins = pq_instance.grid_bins
    codes_per_subspace = pq_instance.codes_per_subspace
    total_codes = pq_instance.total_codes
    
    # Calculate traditional grid approach numbers
    traditional_grid_bins = grid_bins  # Typical value used in original code
    traditional_total_cells = traditional_grid_bins ** action_dim
    
    print(f"Action Space: {action_dim}D")
    print(f"PQ Configuration: {num_subspaces} subspaces × {codes_per_subspace} codes = {total_codes} total codes")
    print(f"Grid Bins: {pq_instance.grid_bins} bins per dimension")
    print(f"Traditional Approach: {traditional_grid_bins}^{action_dim} = {traditional_total_cells:,} cells")
    
    # Calculate efficiency metrics
    efficiency_ratio = traditional_total_cells / total_codes
    memory_savings = (traditional_total_cells - total_codes) / traditional_total_cells * 100
    
    print(f"\nEfficiency Metrics:")
    print(f"  Code reduction: {efficiency_ratio:.1f}x fewer codes")
    print(f"  Memory savings: {memory_savings:.1f}%")
    
    # Analyze each subspace
    print(f"\nSubspace Analysis:")
    for i, (start_idx, end_idx) in enumerate(pq_instance.subspaces):
        subspace_dim = end_idx - start_idx
        subspace_codes = pq_instance.subspace_codebooks[i]
        
        # Calculate coverage for this subspace
        subspace_grid_cells = 6 ** subspace_dim  # Assuming 6 bins per dimension
        subspace_coverage = min(codes_per_subspace, subspace_grid_cells) / subspace_grid_cells * 100
        
        print(f"  Subspace {i}: actions[{start_idx}:{end_idx}] ({subspace_dim}D)")
        print(f"    Codes: {codes_per_subspace} / {subspace_grid_cells:,} cells ({subspace_coverage:.1f}% coverage)")
        print(f"    Code range: [{subspace_codes.min().item():.3f}, {subspace_codes.max().item():.3f}]")
    
    # Overall coverage assessment
    print(f"\nOverall Assessment:")
    if total_codes < 100:
        print(f"  ✓ Excellent: {total_codes} codes is very efficient for {action_dim}D space")
    elif total_codes < 500:
        print(f"  ✓ Good: {total_codes} codes provides good coverage for {action_dim}D space")
    elif total_codes < 1000:
        print(f"  ⚠️  Moderate: {total_codes} codes may be insufficient for {action_dim}D space")
    else:
        print(f"  ❌ Poor: {total_codes} codes is too many for {action_dim}D space")
    
    return {
        'efficiency_ratio': efficiency_ratio,
        'memory_savings': memory_savings,
        'total_codes': total_codes,
        'traditional_cells': traditional_total_cells
    }

def generate_pq_for_variant(variant: dict, action_dim: int):
    """
    Generate PQ codebooks for a specific variant and action dimension.
    Args:
        variant: experiment variant dictionary
        action_dim: dimension of action space
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = variant.get("env", None)
    
    print(f"\nGenerating PQ codebooks for variant:")
    print(f"  Environment: {env_name}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Device: {device}")
    
    try:
        # Generate PQ codebooks
        pq_instance = generate_pq_codebooks_offline(
            action_dim=action_dim,
            num_subspaces=variant.get("num_subspaces"),  # From variant
            grid_bins=variant.get("grid_bins"),  # From variant
            codes_per_subspace=variant.get("codes_per_subspace"),  # From variant
            device=device,
            env_name=env_name
        )
        
        # Analyze the generated codebooks
        analysis = analyze_pq_coverage(pq_instance)
        
        print(f"\n✓ PQ codebooks generated successfully for {env_name}!")
        return pq_instance, analysis
        
    except Exception as e:
        print(f"❌ Error generating PQ codebooks: {e}")
        return None, None

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate PQ codebooks for EWA")
    parser.add_argument("--env", type=str, required=True, help="Environment name")
    parser.add_argument("--action_dim", type=int, required=True, help="Action space dimension")
    parser.add_argument("--num_subspaces", type=int, help="Number of subspaces (auto-determined if not specified)")
    parser.add_argument("--grid_bins", type=int, help="Number of bins per dimension (auto-determined if not specified)")
    parser.add_argument("--codes_per_subspace", type=int, help="Codes per subspace (auto-determined if not specified)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Create a minimal variant for testing
    variant = {
        "env": args.env,
        "phi": 0.05,
        "delta": 0.8,
        "beta": 0.05,
        "num_subspaces": args.num_subspaces,
        "grid_bins": args.grid_bins,
        "codes_per_subspace": args.codes_per_subspace
    }
    
    # Generate PQ codebooks
    pq_instance, analysis = generate_pq_for_variant(variant, args.action_dim)
    
    if pq_instance is not None:
        print(f"\n🎉 Successfully generated PQ codebooks!")
        print(f"   Total codes: {pq_instance.total_codes}")
        print(f"   Subspaces: {pq_instance.num_subspaces}")
        print(f"   Codes per subspace: {pq_instance.codes_per_subspace}")
    else:
        print(f"\n❌ Failed to generate PQ codebooks")
        return 1
    
    return 0

if __name__ == "__main__":
    # Example usage for different environments
    print("PQ Grid Table Generator for EWA")
    print("=" * 50)
    
    # Test with different action dimensions
    test_cases = [
        ("hopper-medium-v2", 3),
        ("walker2d-medium-replay-v2", 6),
        ("halfcheetah-medium-v2", 6),
        ("antmaze-medium-play-v2", 8)
    ]
    
    print("Example configurations:")
    for env, action_dim in test_cases:
        print(f"  {env}: {action_dim}D actions")
    
    print("\nTo generate codebooks for a specific environment:")
    print("  python generate_grid_tables_pq.py --env walker2d-medium-replay-v2 --action_dim 6")
    
    # Run main if called directly
    exit(main())
