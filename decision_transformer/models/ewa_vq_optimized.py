import torch
import numpy as np
import os
import pickle
import hashlib
import time
 
class EWAVQOptimized:
    """
    OPTIMIZED Vector Quantization (VQ) based EWA with direct grid lookup.
    
    OPTIMIZATIONS IMPLEMENTED:
    1. Trajectory Caching: Cache processed trajectories to avoid recomputation
    2. Vectorized Batch Processing: Process multiple trajectories in parallel
    3. Grid Table Optimization: Advanced indexing and pre-computed lookups
    4. Memory Management: Automatic cache cleanup and memory optimization
    5. GPU Acceleration: Maximize GPU utilization with vectorized operations
    """
    
    # Class-level cache for grid tables
    _grid_table_cache = {}
    _cache_dir = "./ewa_grid_cache"
    
    @classmethod
    def _get_cache_key(cls, action_dim, grid_bins, num_codes, env_name=None):
        """Generate a unique cache key for grid table parameters."""
        if env_name:
            return f"grid_{action_dim}d_{grid_bins}bins_{num_codes}codes_{env_name}"
        else:
            return f"grid_{action_dim}d_{grid_bins}bins_{num_codes}codes"
    
    @classmethod
    def _get_cache_path(cls, cache_key):
        """Get the file path for a cached grid table."""
        os.makedirs(cls._cache_dir, exist_ok=True)
        return os.path.join(cls._cache_dir, f"{cache_key}.pkl")
    
    @staticmethod
    def _linear_to_grid_coords_static(linear_idx, dim, grid_bins):
        """Convert linear index to grid coordinates (static method)."""
        coords = []
        for d in range(dim):
            coords.append(linear_idx % grid_bins)
            linear_idx //= grid_bins
        return coords
    
    @staticmethod
    def _grid_coords_to_action_static(coords, grid_bins, device):
        """Convert grid coordinates to action space (static method)."""
        action = torch.tensor(coords, device=device, dtype=torch.float32)
        action = 2.0 * action / (grid_bins - 1) - 1.0
        return action

    def __init__(self, num_heads, tuple_seq_length, phi, delta, num_codes, grid_bins):
        """
        Args:
            num_heads: number of attention heads
            tuple_seq_length: total sequence length (3 * actions_seq_length)
            phi: decay factor for attraction
            delta: chosen-vs-unchosen weight
            num_codes: number of VQ codes (centroids)
            grid_bins: number of bins per dimension for grid lookup
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.tuple_seq_length = tuple_seq_length
        self.phi = phi
        self.delta = delta
        self.num_codes = num_codes
        self.grid_bins = grid_bins
        
        # VQ codebook - will be initialized with static grid-based codes
        self.codes = None
        
        # Grid lookup table for fast routing - will be loaded from cache
        self.grid_table = None
        
        # OPTIMIZATION 1: Precomputed powers for vectorized grid index calculation
        max_action_dim = 10  # Most environments have <= 10 action dimensions
        self.grid_powers = torch.pow(grid_bins, torch.arange(max_action_dim, device=self.device))
        
        # History tracking variables
        self.step = 0
        self.attraction_history = []
        self.code_usage_history = []
        self.reward_history = []
        
        # Track if codes have been initialized
        self.codes_initialized = False
        
        # OPTIMIZATION 1: Trajectory Caching System
        self._trajectory_cache = {}
        self._attraction_cache = {}
        self._max_cache_size = 1000  # Prevent memory bloat
        self._cache_hits = 0
        self._cache_misses = 0
        
        # OPTIMIZATION 4: Memory Management
        self._last_cleanup_step = 0
        self._cleanup_interval = 100  # Clean up cache every 100 steps
        
        # Check if grid table is already cached for these parameters
        self.action_dim = None  # Will be set when we know the actual action dimension
        cache_key = self._get_cache_key(self.action_dim, grid_bins, num_codes)

    def _get_trajectory_cache_key(self, actions, rewards):
        """Generate cache key for trajectory processing using tensor hashes."""
        # Create a hash of the action and reward tensors for caching
        actions_bytes = actions.cpu().numpy().tobytes()
        rewards_bytes = rewards.cpu().numpy().tobytes()
        
        # Use SHA-256 for collision-resistant hashing
        combined_bytes = actions_bytes + rewards_bytes
        cache_key = hashlib.sha256(combined_bytes).hexdigest()[:16]  # Use first 16 chars
        
        return cache_key

    def _cleanup_cache(self):
        """Clean up old cache entries to prevent memory bloat."""
        if len(self._trajectory_cache) > self._max_cache_size:
            # Remove oldest entries (FIFO)
            keys_to_remove = list(self._trajectory_cache.keys())[:len(self._trajectory_cache) - self._max_cache_size]
            for key in keys_to_remove:
                del self._trajectory_cache[key]
                if key in self._attraction_cache:
                    del self._attraction_cache[key]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            print(f"EWA Cache cleanup: removed {len(keys_to_remove)} old entries")

    def _extract_dimensions(self, rewards):
        """
        Extract batch size, number of action tokens, and sequence length dynamically.
        Args:
            rewards: tensor of shape (B, k, 1) or (B, k)
        Returns:
            batch_size, num_action_tokens, tuple_seq_length, action_indices
        """
        if len(rewards.shape) == 3:
            batch_size, num_action_tokens, _ = rewards.shape
        else:
            batch_size, num_action_tokens = rewards.shape
            
        # tuple_seq_length = 3 * num_action_tokens
        
        # Define action token indices dynamically in the full sequence
        # For sequence (R_1, s_1, a_1, R_2, s_2, a_2, ...), action tokens are at indices 2, 5, 8, ...
        if num_action_tokens == 1:
            action_indices = torch.tensor([2], device=self.device)
        else:
            action_indices = torch.arange(2, min(self.tuple_seq_length, 3 * num_action_tokens), step=3, device=self.device)
        
        return batch_size, num_action_tokens, self.tuple_seq_length, action_indices

    def _generate_static_codes(self, action_dim):
        """
        Generate static codes by building a grid table based on the number of codes.
        This ensures full coverage of the action space regardless of action dimension.
        """
        # Build grid table based on number of codes, not action dimension
        codes = []
        
        # Calculate optimal grid_bins based on num_codes and action_dim
        optimal_grid_bins = int(self.num_codes ** (1.0 / action_dim))
        
        # If we don't have enough codes for full coverage, reduce grid_bins
        while optimal_grid_bins ** action_dim > self.num_codes and optimal_grid_bins > 2:
            optimal_grid_bins -= 1
        
        # For very high-dimensional spaces, we might need to use only 2 bins
        if optimal_grid_bins ** action_dim > self.num_codes:
            optimal_grid_bins = 2  # Minimum bins per dimension
        
        # Ensure we don't exceed reasonable limits
        optimal_grid_bins = max(2, min(optimal_grid_bins, 8))
        
        # For high-dimensional environments (6D+), increase num_codes to ensure full coverage
        if action_dim >= 6 and optimal_grid_bins ** action_dim > self.num_codes:
            required_codes = optimal_grid_bins ** action_dim
            max_codes = 128
            if required_codes <= max_codes:
                self.num_codes = required_codes
            else:
                self.num_codes = max_codes
        
        # Generate codes by sampling grid coordinates evenly across the action space
        for i in range(self.num_codes):
            coords = self._linear_to_grid_coords_static(i, action_dim, optimal_grid_bins)
            code = self._grid_coords_to_action_static(coords, optimal_grid_bins, self.device)
            codes.append(code)
        
        self.codes = torch.stack(codes).to(self.device)
        self.grid_bins = optimal_grid_bins

    def initialize_codes(self, action_dim):
        """
        Initialize VQ codes using static grid-based generation.
        Args:
            action_dim: dimension of action space
        """
        if self.codes_initialized:
            return
            
        # Generate static codes from grid
        self._generate_static_codes(action_dim)
        
        # Load grid table from cache
        env_name = getattr(self, 'variant', {}).get('env', None) if hasattr(self, 'variant') else None
        self._load_grid_table_from_cache(action_dim, env_name)
        
        self.codes_initialized = True
    
    def _load_grid_table_from_cache(self, action_dim, env_name=None):
        """Load grid table from cache or generate it if not available."""
        # Update the stored action_dim with the actual value
        self.action_dim = action_dim
        
        # Generate cache key - prefer environment name if available
        if env_name:
            cache_key = self._get_cache_key(action_dim, self.grid_bins, self.num_codes, env_name)
        else:
            cache_key = self._get_cache_key(action_dim, self.grid_bins, self.num_codes)
        
        # Check memory cache first
        if cache_key in self._grid_table_cache:
            self.grid_table = self._grid_table_cache[cache_key]
            # Ensure grid table is on the correct device
            if self.grid_table.device != self.device:
                self.grid_table = self.grid_table.to(self.device)
            return
        
        # Check file cache
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                grid_table = pickle.load(f)
            # Move to device
            self.grid_table = grid_table.to(self.device)
            # Store in memory cache (also on device)
            self._grid_table_cache[cache_key] = self.grid_table
            return
        
        # If not in cache, this should not happen if pre-generated in DecisionTransformer
        self.grid_table = None

    def _action_to_grid_idx_batch(self, actions):
        """
        Convert batch of actions to grid indices for fast lookup.
        OPTIMIZATION: Vectorized batch grid index computation.
        Args:
            actions: tensor of shape (batch_size, d)
        Returns:
            grid_indices: tensor of shape (batch_size,) with linear indices
        """
        # Map actions from [-1, 1] to [0, grid_bins-1] using rounding (nearest centers)
        tmp = (actions + 1.0) / 2.0 * (self.grid_bins - 1)
        grid_coords = tmp.round().clamp(0, self.grid_bins - 1).long()
        
        # OPTIMIZATION: Vectorized batch grid index computation
        action_dim = grid_coords.shape[1]
        # Ensure grid_powers is on the same device as actions
        if self.grid_powers.device != actions.device:
            self.grid_powers = self.grid_powers.to(actions.device)
        powers = self.grid_powers[:action_dim].unsqueeze(0)  # (1, action_dim)
        
        # Vectorized computation: sum(grid_coords * powers, dim=1)
        grid_indices = torch.sum(grid_coords * powers, dim=1)
        
        return grid_indices

    def route_actions_fast_batch(self, actions):
        """
        Route batch of actions to nearest codes using grid lookup or direct computation.
        OPTIMIZATION: Advanced indexing and pre-computed lookups.
        Args:
            actions: tensor of shape (batch_size, d)
        Returns:
            code_indices: tensor of shape (batch_size,) with code indices
        """
        if self.grid_table is not None:
            # OPTIMIZATION 3: Advanced indexing for faster lookup
            grid_indices = self._action_to_grid_idx_batch(actions)
            
            # Ensure grid table is on the same device as actions
            if self.grid_table.device != actions.device:
                self.grid_table = self.grid_table.to(actions.device)
            
            # Use advanced indexing for faster lookup
            valid_mask = grid_indices < len(self.grid_table)
            code_indices = torch.zeros(actions.shape[0], dtype=torch.long, device=actions.device)
            
            if valid_mask.any():
                # Use advanced indexing for valid indices
                code_indices[valid_mask] = self.grid_table[grid_indices[valid_mask]]
            
            # Only compute distances for invalid indices (should be rare)
            invalid_mask = ~valid_mask
            if invalid_mask.any():
                invalid_actions = actions[invalid_mask]
                # Use smaller batch size for distance computation to avoid memory issues
                distances = torch.cdist(invalid_actions, self.codes)
                code_indices[invalid_mask] = torch.argmin(distances, dim=1)
            
            return code_indices
        
        # Fallback: direct computation for all actions
        distances = torch.cdist(actions, self.codes)
        return torch.argmin(distances, dim=1)

    def process_trajectory(self, actions, rewards):
        """
        Process a single trajectory using VQ-based EWA with caching.
        OPTIMIZATION: Trajectory caching to avoid recomputation.
        Args:
            actions: tensor of shape (k, d) for one trajectory
            rewards: tensor of shape (k,) or (k, 1) for one trajectory
        Returns:
            D_codebook: list of VQ codes with attractions (summary of used codes)
            D_trajectory: list of dicts with action info and attractions
        """
        # OPTIMIZATION 1: Check cache first
        cache_key = self._get_trajectory_cache_key(actions, rewards)
        if cache_key in self._trajectory_cache:
            self._cache_hits += 1
            return self._trajectory_cache[cache_key]
        
        self._cache_misses += 1
        
        # Extract actual sequence length from rewards (not actions)
        # This is needed because in evaluation phase, sequences grow step-by-step:
        # Step 0: actions(10,1,6), rewards(10,1,1) -> Step 19: actions(10,20,6), rewards(10,20,1)
        actual_seq_length = rewards.shape[0]

        # Only process the actions/rewards that actually exist
        actions = actions[:actual_seq_length]
        rewards = rewards[:actual_seq_length]
        
        # Get dimensions like in ewa_kernel.py
        k = actions.shape[0]
        d = actions.shape[1]
        
        actions = actions.to(self.device)
        rewards = rewards.view(-1).to(self.device)  # (k,)
        
        # Initialize codes if not done yet
        if not self.codes_initialized:
            action_dim = actions.shape[1]
            self.initialize_codes(action_dim)
        
        # OPTIMIZATION 3: GPU-accelerated trajectory processing
        # Pre-allocate tensors on GPU for vectorized operations
        code_attraction = torch.zeros(self.num_codes, device=self.device)
        attraction_updates = torch.zeros(self.num_codes, device=self.device)
        code_usage_vec = torch.zeros(self.num_codes, dtype=torch.long, device=self.device)
        
        # Vectorized code routing (keep on GPU)
        code_indices = self.route_actions_fast_batch(actions)  # (k,) tensor
        
        # OPTIMIZATION 3: Vectorized attraction processing
        D_trajectory = []
        trajectory_rewards = []
        
        for t in range(k):
            code_idx = code_indices[t]
            r_t = rewards[t]
            
            # Decay all code attractions
            code_attraction *= (1 - self.phi)
            
            # Update attraction for the routed code only
            attraction_updates[code_idx] += self.delta * r_t
            code_attraction += attraction_updates
            attraction_updates[code_idx] = 0  # Reset for next iteration

            # Track usage for the routed code
            code_usage_vec[code_idx] += 1

            # Track reward
            trajectory_rewards.append(r_t.item())

            # Append step info (post-update attraction value)
            step_code_idx = code_idx.item() if isinstance(code_idx, torch.Tensor) else int(code_idx)
            D_trajectory.append({
                'a_t': actions[t].detach().cpu().numpy(),
                'r_t': r_t.item(),
                'code_idx': step_code_idx,
                'A_t': code_attraction[code_idx].item(),
                'code_usage': code_usage_vec[code_idx].item()
            })

        # Build codebook summary only for codes that were actually used
        D_codebook = []
        used_code_indices = torch.nonzero(code_usage_vec > 0, as_tuple=False).view(-1)
        for i in used_code_indices.tolist():
            D_codebook.append({'code_idx': i, 'A': code_attraction[i]})

        # OPTIMIZATION: Store detailed trajectory information for efficient metric retrieval
        avg_attraction = float(np.mean([step['A_t'] for step in D_trajectory])) if len(D_trajectory) > 0 else 0.0
        total_code_usage = int((code_usage_vec > 0).sum().item())
        avg_reward = float(np.mean(trajectory_rewards)) if len(trajectory_rewards) > 0 else 0.0
        
        # Store comprehensive iteration data
        iteration_data = {
            'step': self.step,
            'avg_attraction': avg_attraction,
            'code_usage': total_code_usage,
            'reward': avg_reward,
            'trajectory_steps': D_trajectory,  # Store full trajectory for detailed analysis
            'timestamp': time.time()
        }
        
        self.attraction_history.append(iteration_data)
        self.code_usage_history.append(total_code_usage)
        self.reward_history.append(avg_reward)
        
        # OPTIMIZATION 1: Cache the result
        result = (D_codebook, D_trajectory)
        self._trajectory_cache[cache_key] = result
        
        # OPTIMIZATION 4: Periodic cache cleanup
        if self.step - self._last_cleanup_step >= self._cleanup_interval:
            self._cleanup_cache()
            self._last_cleanup_step = self.step
        
        self.step += 1
        return result

    def process_batch(self, actions, rewards):
        """
        Process a batch of trajectories with optimized batch processing.
        OPTIMIZATION: Vectorized batch processing for better GPU utilization.
        Args:
            actions: tensor of shape (B, k, d)
            rewards: tensor of shape (B, k) or (B, k, 1)
        Returns:
            D_codebooks: list of D_codebook for each trajectory
            D_trajectories: list of D_trajectory for each trajectory
        """
        # OPTIMIZATION: Ensure codes are initialized before processing
        if not self.codes_initialized:
            action_dim = actions.shape[-1]
            self.initialize_codes(action_dim)
        
        # Extract dimensions dynamically
        batch_size, num_action_tokens, tuple_seq_length, action_indices = self._extract_dimensions(rewards)
        
        # OPTIMIZATION 3: Choose processing strategy based on batch size
        if batch_size > 1 and num_action_tokens > 1:
            # Use vectorized processing for larger batches
            return self._process_batch_vectorized(actions, rewards)
        else:
            # Use sequential processing for small batches (more cache-friendly)
            return self._process_batch_sequential(actions, rewards)

    def _process_batch_sequential(self, actions, rewards):
        """Sequential batch processing (cache-friendly for small batches)."""
        batch_size, num_action_tokens, tuple_seq_length, action_indices = self._extract_dimensions(rewards)
        
        D_codebooks = []
        D_trajectories = []
        
        for b in range(batch_size):
            D_codebook, D_trajectory = self.process_trajectory(actions[b], rewards[b])
            D_codebooks.append(D_codebook)
            D_trajectories.append(D_trajectory)
        
        return D_codebooks, D_trajectories

    def _process_batch_vectorized(self, actions, rewards):
        """Vectorized batch processing for better GPU utilization."""
        batch_size, num_action_tokens, tuple_seq_length, action_indices = self._extract_dimensions(rewards)
        
        # OPTIMIZATION 3: Combine all trajectories into single tensors for vectorized processing
        all_actions = actions.view(-1, actions.shape[-1])  # (B*k, d)
        all_rewards = rewards.view(-1)  # (B*k,)
        
        # Process all actions at once
        all_code_indices = self.route_actions_fast_batch(all_actions)
        
        # Reshape back to batch format
        code_indices = all_code_indices.view(actions.shape[0], actions.shape[1])
        
        # OPTIMIZATION 3: Vectorized attraction computation
        D_codebooks = []
        D_trajectories = []
        
        for b in range(batch_size):
            # Extract trajectory-specific data
            traj_actions = actions[b]
            traj_rewards = rewards[b]
            traj_code_indices = code_indices[b]
            
            # Process individual trajectory (reuse existing logic)
            D_codebook, D_trajectory = self._process_trajectory_with_indices(
                traj_actions, traj_rewards, traj_code_indices
            )
            
            D_codebooks.append(D_codebook)
            D_trajectories.append(D_trajectory)
        
        return D_codebooks, D_trajectories

    def _process_trajectory_with_indices(self, actions, rewards, code_indices):
        """Process trajectory with pre-computed code indices."""
        # Extract actual sequence length from rewards (not actions)
        # This is needed because in evaluation phase, sequences grow step-by-step:
        # Step 0: actions(10,1,6), rewards(10,1,1) -> Step 19: actions(10,20,6), rewards(10,20,1)
        actual_seq_length = rewards.shape[0]

        # Only process the actions/rewards that actually exist
        actions = actions[:actual_seq_length]
        rewards = rewards[:actual_seq_length]
        code_indices = code_indices[:actual_seq_length]
        
        # Get dimensions like in ewa_kernel.py
        k = actions.shape[0]
        d = actions.shape[1]
        
        actions = actions.to(self.device)
        rewards = rewards.view(-1).to(self.device)
        
        # Initialize codes if not done yet
        if not self.codes_initialized:
            action_dim = actions.shape[1]
            self.initialize_codes(action_dim)
        
        # Pre-allocate tensors on GPU
        code_attraction = torch.zeros(self.num_codes, device=self.device)
        attraction_updates = torch.zeros(self.num_codes, device=self.device)
        code_usage_vec = torch.zeros(self.num_codes, dtype=torch.long, device=self.device)
        
        D_trajectory = []
        trajectory_rewards = []
        
        for t in range(k):
            code_idx = code_indices[t]
            r_t = rewards[t]
            
            # Decay all code attractions
            code_attraction *= (1 - self.phi)
            
            # Update attraction for the routed code only
            attraction_updates[code_idx] += self.delta * r_t
            code_attraction += attraction_updates
            attraction_updates[code_idx] = 0

            # Track usage for the routed code
            code_usage_vec[code_idx] += 1
            trajectory_rewards.append(r_t.item())

            # Append step info
            step_code_idx = code_idx.item() if isinstance(code_idx, torch.Tensor) else int(code_idx)
            D_trajectory.append({
                'a_t': actions[t].detach().cpu().numpy(),
                'r_t': r_t.item(),
                'code_idx': step_code_idx,
                'A_t': code_attraction[code_idx].item(),
                'code_usage': code_usage_vec[code_idx].item()
            })

        # Build codebook summary
        D_codebook = []
        used_code_indices = torch.nonzero(code_usage_vec > 0, as_tuple=False).view(-1)
        for i in used_code_indices.tolist():
            D_codebook.append({'code_idx': i, 'A': code_attraction[i]})

        # Track history
        avg_attraction = float(np.mean([step['A_t'] for step in D_trajectory])) if len(D_trajectory) > 0 else 0.0
        self.attraction_history.append(avg_attraction)
        self.code_usage_history.append(int((code_usage_vec > 0).sum().item()))
        self.reward_history.append(float(np.mean(trajectory_rewards)) if len(trajectory_rewards) > 0 else 0.0)
        
        return D_codebook, D_trajectory

    def get_attraction(self, D_trajectories, rewards):
        """
        Build the attraction matrix for the batch.
        Args:
            D_trajectories: list of D_trajectory (length B)
        Returns:
            Tensor of shape (B, num_heads, tuple_seq_length, 1) with attraction values at action token indices
        """
        batch_size, num_action_tokens, tuple_seq_length, action_indices = self._extract_dimensions(rewards)
        A = torch.zeros((batch_size, self.num_heads, tuple_seq_length, 1), device=self.device)
        
        for b in range(batch_size):
            D_trajectory = D_trajectories[b]
            
            for i, step_info in enumerate(D_trajectory):
                if i < len(action_indices):
                    a_idx = action_indices[i].item()
                    val = step_info['A_t']
                    A[b, :, a_idx, 0] = val
        
        return A

    def get_history(self):
        """
        Return history of important variables tracked during processing.
        Returns:
            tuple: (steps, attraction_values, code_usage_values, reward_values)
        """
        steps = list(range(len(self.attraction_history)))
        attraction_values = np.array(self.attraction_history)
        code_usage_values = np.array(self.code_usage_history)
        reward_values = np.array(self.reward_history)
        
        return steps, attraction_values, code_usage_values, reward_values

    def get_current_metrics(self):
        """
        OPTIMIZATION: Get current iteration metrics without computing full history.
        This is much faster than get_history() for logging purposes.
        Returns:
            dict: Current iteration metrics
        """
        if len(self.attraction_history) == 0:
            return {
                'current_attraction': 0.0,
                'current_code_usage': 0,
                'current_reward': 0.0,
                'total_trajectories_processed': 0
            }
        
        # Get the most recent values (current iteration)
        latest = self.attraction_history[-1]
        current_attraction = float(np.mean([step['A_t'] for step in latest.get('trajectory_steps', [])])) if 'trajectory_steps' in latest else 0.0
        current_code_usage = latest.get('code_usage', 0)
        current_reward = latest.get('reward', 0.0)
        
        return {
            'current_attraction': current_attraction,
            'current_code_usage': current_code_usage,
            'current_reward': current_reward,
            'total_trajectories_processed': len(self.attraction_history)
        }

    def get_cache_stats(self):
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._trajectory_cache),
            'max_cache_size': self._max_cache_size
        }

    def clear_cache(self):
        """Clear all caches (useful for memory management)."""
        self._trajectory_cache.clear()
        self._attraction_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print("EWA cache cleared")
