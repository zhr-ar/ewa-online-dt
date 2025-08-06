#!/usr/bin/env python3
"""
Script to pre-generate grid tables for common environments.
This will create cached grid tables that can be reused during training.
"""

import torch
import os
import pickle
from decision_transformer.models.ewa_vq import EWAVQ

def generate_grid_table_offline(action_dim, grid_bins, num_codes, device="cuda", env_name=None):
    """
    Generate grid table offline and cache it for reuse.
    Args:
        action_dim: dimension of action space
        grid_bins: number of bins per dimension
        num_codes: number of VQ codes
        device: device to create tensors on
        env_name: environment name for cache key
    Returns:
        grid_table: cached grid table tensor
    """
    cache_key = EWAVQ._get_cache_key(action_dim, grid_bins, num_codes, env_name)
    cache_path = EWAVQ._get_cache_path(cache_key)
    
    # Check if already in memory cache
    if cache_key in EWAVQ._grid_table_cache:
        print(f"✓ Using memory cached grid table for {action_dim}d, {grid_bins} bins, {num_codes} codes")
        return EWAVQ._grid_table_cache[cache_key]
    
    # Check if file cache exists
    if os.path.exists(cache_path):
        print(f"✓ Loading grid table from file cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            grid_table = pickle.load(f)
        # Move to device
        grid_table = grid_table.to(device)
        EWAVQ._grid_table_cache[cache_key] = grid_table
        return grid_table
    
    # Generate new grid table
    print(f"🔄 Generating new grid table for {action_dim}d, {grid_bins} bins, {num_codes} codes...")
    
    # Check if grid size is too large
    grid_size = grid_bins ** action_dim
    max_grid_size = 1000000  # 1M cells max
    
    print(f"Grid table info: action_dim={action_dim}, grid_bins={grid_bins}, grid_size={grid_size}")
    
    if grid_size > max_grid_size:
        print(f"Warning: Grid size {grid_size} is too large for action_dim {action_dim}. Using simplified routing.")
        return None
    
    print(f"Building grid table: {grid_size} cells for action_dim {action_dim}")
    
    # Create temporary codes for grid building (will be replaced during actual initialization)
    temp_codes = torch.randn(num_codes, action_dim, device=device, dtype=torch.float32)
    
    # Create grid table: (grid_bins^action_dim,)
    grid_table = torch.zeros(grid_size, dtype=torch.long, device=device)
    
    # Progress monitoring
    progress_interval = max(1, grid_size // 50)  # Print every 2%
    
    print(f"Starting grid table construction with {grid_size} cells...")
    
    # For each grid cell, find the nearest code
    for i in range(grid_size):
        if i % progress_interval == 0:
            print(f"Grid building progress: {i}/{grid_size} ({i/grid_size*100:.1f}%)")
        
        # Convert linear index to grid coordinates
        coords = EWAVQ._linear_to_grid_coords_static(i, action_dim, grid_bins)
        
        # Convert grid coordinates to action space
        action = EWAVQ._grid_coords_to_action_static(coords, grid_bins, device)
        
        # Find nearest code
        distances = torch.norm(temp_codes - action, dim=1)
        nearest_code = torch.argmin(distances)
        grid_table[i] = nearest_code
    
    print(f"Grid table built successfully!")
    
    # Save to file cache
    print(f"Saving grid table to cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(grid_table.cpu(), f)
    
    # Store in memory cache
    EWAVQ._grid_table_cache[cache_key] = grid_table
    
    return grid_table

def generate_grid_table_for_variant(variant, action_dim):
    """Generate grid table for a specific variant and action dimension."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get parameters from variant
    grid_bins_factor = variant.get("grid_bins_factor", 1.0)
    num_codes = variant.get("num_codes", 16)
    env_name = variant.get("env", None)
    
    # Calculate adaptive grid_bins based on action dimension
    adaptive_grid_bins = max(2, min(8, int(grid_bins_factor * action_dim)))
    
    print(f"Generating grid table for variant:")
    print(f"  - action_dim: {action_dim}")
    print(f"  - grid_bins_factor: {grid_bins_factor}")
    print(f"  - adaptive_grid_bins: {adaptive_grid_bins}")
    print(f"  - num_codes: {num_codes}")
    print(f"  - env_name: {env_name}")
    
    try:
        # This will generate and cache the grid table
        grid_table = generate_grid_table_offline(
            action_dim=action_dim,
            grid_bins=adaptive_grid_bins,
            num_codes=num_codes,
            device=device,
            env_name=env_name
        )
        
        if grid_table is not None:
            print(f"✓ Successfully generated grid table: {grid_table.shape}")
            return grid_table
        else:
            print(f"⚠ Grid table too large, using fallback routing")
            return None
            
    except Exception as e:
        print(f"✗ Error generating grid table: {e}")
        return None

if __name__ == "__main__":
    # This script is now only used for generating grid tables for specific variants
    # The main functionality is in generate_grid_table_for_variant()
    print("This script is used by DecisionTransformer to generate grid tables for specific variants.")
    print("Run main.py to generate grid tables for your specific experiment parameters.") 