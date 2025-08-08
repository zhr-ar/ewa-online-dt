#!/usr/bin/env python3
"""
Script to pre-generate grid tables for common environments.
This will create cached grid tables that can be reused during training.
"""

import torch
import os
import pickle
from decision_transformer.models.ewa_vq import EWAVQ

def _action_to_grid_idx_static(action, grid_bins):
    """
    Static version of action to grid index conversion.
    Args:
        action: tensor of shape (d,)
        grid_bins: number of bins per dimension
    Returns:
        grid_idx: linear index into grid table
    """
    # Map action from [-1, 1] to [0, grid_bins-1]
    grid_coords = ((action + 1.0) / 2.0 * (grid_bins - 1)).clamp(0, grid_bins - 1)
    grid_coords = grid_coords.long()
    
    # Convert to linear index
    grid_idx = 0
    for d in range(len(grid_coords)):
        grid_idx += grid_coords[d].item() * (grid_bins ** d)
    
    return grid_idx

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
        print(f"Using memory cached grid table for {action_dim}d, {grid_bins} bins, {num_codes} codes")
        return EWAVQ._grid_table_cache[cache_key]
    
    # Check if file cache exists
    if os.path.exists(cache_path):
        # print(f"*** Attr min/max/mean: {attr.min().item():.4f}/{attr.max().item():.4f}/{attr.mean().item():.4f}; Attn W  min/max/mean: {w.min().item():.4f}/{w.max().item():.4f}/{w.mean().item():.4f}")

        with open(cache_path, 'rb') as f:
            grid_table = pickle.load(f)
        # Move to device
        grid_table = grid_table.to(device)
        EWAVQ._grid_table_cache[cache_key] = grid_table
        return grid_table
    
    # Generate new grid table
    
    # Check if grid size is too large
    grid_size = grid_bins ** action_dim
    max_grid_size = 1000000  # 1M cells max
    
    if grid_size > max_grid_size:
        return None
    
    # Generate the actual static codes that will be used during training
    print(f"Generating static codes for grid table building...")
    
    # Calculate optimal grid_bins based on num_codes and action_dim
    # We want to ensure that num_codes >= grid_bins^action_dim for full coverage
    optimal_grid_bins = int(num_codes ** (1.0 / action_dim))
    
    # If we don't have enough codes for full coverage, reduce grid_bins
    while optimal_grid_bins ** action_dim > num_codes and optimal_grid_bins > 2:
        optimal_grid_bins -= 1
    
    # For very high-dimensional spaces, we might need to use only 2 bins
    # and accept that we can't cover the full space with the given number of codes
    if optimal_grid_bins ** action_dim > num_codes:
        optimal_grid_bins = 2  # Minimum bins per dimension
    
    # Ensure we don't exceed reasonable limits
    optimal_grid_bins = max(2, min(optimal_grid_bins, 8))
    
    # For high-dimensional environments (6D+), increase num_codes to ensure full coverage
    if action_dim >= 6 and optimal_grid_bins ** action_dim > num_codes:
        # Calculate how many codes we need for full coverage
        required_codes = optimal_grid_bins ** action_dim
        # Cap at a reasonable maximum (e.g., 128 codes)
        max_codes = 128
        if required_codes <= max_codes:
            num_codes = required_codes
        else:
            num_codes = max_codes
    
    static_codes = []
    for i in range(num_codes):
        # Convert linear index to grid coordinates
        coords = EWAVQ._linear_to_grid_coords_static(i, action_dim, optimal_grid_bins)
        
        # Convert grid coordinates to action space
        code = EWAVQ._grid_coords_to_action_static(coords, optimal_grid_bins, device)
        static_codes.append(code)
    
    temp_codes = torch.stack(static_codes)
    print(f"Static codes generated for grid table with full action space coverage: shape={temp_codes.shape}")
    print(f"Code range: [{temp_codes.min().item():.3f}, {temp_codes.max().item():.3f}]")
    
    # Calculate coverage statistics
    total_possible_cells = optimal_grid_bins ** action_dim
    coverage_ratio = min(num_codes, total_possible_cells) / total_possible_cells
    print(f"Action space coverage: {coverage_ratio:.1%} ({min(num_codes, total_possible_cells)}/{total_possible_cells} cells)")
    
    # Update grid_bins to match what we actually used
    grid_bins = optimal_grid_bins
    
    # Create grid table: (grid_bins^action_dim,)
    grid_table = torch.zeros(grid_size, dtype=torch.long, device=device)
    
    # Progress monitoring
    progress_interval = max(1, grid_size // 50)  # Print every 2%
    
    print(f"Starting grid table construction with {grid_size} cells...")
    
    # OPTIMIZATION: Vectorized grid table construction
    # Process in batches for GPU efficiency
    grid_cells_batch_size = 10000
    for batch_start in range(0, grid_size, grid_cells_batch_size):
        batch_end = min(batch_start + grid_cells_batch_size, grid_size)
        
        # Vectorized coordinate conversion
        coords_list = []
        for idx in range(batch_start, batch_end):
            coords = EWAVQ._linear_to_grid_coords_static(idx, action_dim, grid_bins)
            coords_list.append(coords)
        
        # Vectorized action conversion
        actions_batch = []
        for coords in coords_list:
            action = EWAVQ._grid_coords_to_action_static(coords, grid_bins, device)
            actions_batch.append(action)
        
        actions_batch = torch.stack(actions_batch)
        
        # Vectorized nearest neighbor search
        distances = torch.cdist(actions_batch, temp_codes)
        nearest_codes = torch.argmin(distances, dim=1)
        
        grid_table[batch_start:batch_end] = nearest_codes
        
        if batch_start % (grid_cells_batch_size * 10) == 0:
            print(f"Grid building progress: {batch_start}/{grid_size} ({batch_start/grid_size*100:.1f}%)")
    
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
    env_name = variant.get("env", None)
    
    # Calculate adaptive grid_bins based on action dimension
    # For high-dimensional environments, use smaller grid_bins_factor automatically
    if action_dim > 6:
        # Use smaller grid_bins_factor for high-dimensional environments
        adjusted_grid_bins_factor = min(grid_bins_factor, 0.5)
        print(f"⚠️  High-dimensional environment ({action_dim}D), reducing grid_bins_factor from {grid_bins_factor} to {adjusted_grid_bins_factor}")
    else:
        adjusted_grid_bins_factor = grid_bins_factor
    
    adaptive_grid_bins = max(2, min(8, int(adjusted_grid_bins_factor * action_dim)))

    # Automatically determine number of codes based on grid size
    total_grid_cells = adaptive_grid_bins ** action_dim
    
    # Safety check: limit grid size for high-dimensional environments
    max_reasonable_codes = 36  # Fixed maximum for consistent behavior across environments
    if total_grid_cells > max_reasonable_codes:
        print(f"⚠️  Grid size {total_grid_cells} is too large for {action_dim}D actions")
        print(f"   Limiting to {max_reasonable_codes} codes for consistent behavior")
        default_num_codes = max_reasonable_codes
    else:
        default_num_codes = total_grid_cells
    
    # Use calculated default, but allow override via num_codes parameter
    # If num_codes is None or not specified, use the calculated default
    num_codes = variant.get("num_codes")
    if num_codes is None:
        num_codes = default_num_codes
    # Ensure we don't exceed the total number of grid cells
    num_codes = min(num_codes, total_grid_cells)
    
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
            return grid_table
        else:
            return None
            
    except Exception as e:
        return None

if __name__ == "__main__":
    # This script is now only used for generating grid tables for specific variants
    # The main functionality is in generate_grid_table_for_variant()
    print("This script is used by DecisionTransformer to generate grid tables for specific variants.")
    print("Run main.py to generate grid tables for your specific experiment parameters.") 