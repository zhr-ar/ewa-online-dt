import torch
import numpy as np
import os
import pickle

class EWAVQ:
    """
    Vector Quantization (VQ) based EWA with direct grid lookup.
    Uses a fixed codebook of action centroids and direct grid-based routing
    for maximum speed. Much faster than dynamic clustering approaches.
    
    OPTIMIZATIONS IMPLEMENTED:
    1. Precomputed Powers: grid_bins^i precomputed to avoid repeated calculations
    2. Vectorized Grid Index: tensor operations instead of Python loops
    3. Batch Processing: vectorized routing for multiple actions
    4. Grid Table Caching: memory and file-based caching system
    """
    
    # Class-level cache for grid tables
    _grid_table_cache = {}
    _cache_dir = "./ewa_grid_cache"
    
    @classmethod
    def _get_cache_key(cls, action_dim, grid_bins, num_codes, env_name=None):
        """Generate a unique cache key for grid table parameters."""
        
        # Include environment name if provided, otherwise use generic key
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
        # Map grid coordinates [0, grid_bins-1] to action space [-1, 1]
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
        # Precompute grid_bins^i for all possible action dimensions (up to 10)
        max_action_dim = 10  # Most environments have <= 10 action dimensions
        self.grid_powers = torch.pow(grid_bins, torch.arange(max_action_dim, device=self.device))
        
        # History tracking variables
        self.step = 0
        self.attraction_history = []
        self.code_usage_history = []
        self.reward_history = []
        
        # Track if codes have been initialized
        self.codes_initialized = False
        
        # Check if grid table is already cached for these parameters
        # Note: We can't get env_name in __init__, so we'll use a generic key
        # The actual loading will happen in _load_grid_table_from_cache with proper env_name
        # We'll use a placeholder action_dim that will be updated when we know the actual dimension
        self.action_dim = None  # Will be set when we know the actual action dimension
        cache_key = self._get_cache_key(self.action_dim, grid_bins, num_codes)  # Will be None until action_dim is set

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
        # print(f"Generating static codes for action_dim={action_dim}, num_codes={self.num_codes}")
        
        # Build grid table based on number of codes, not action dimension
        # This ensures we always have exactly num_codes cells covering the whole action space
        codes = []
        
        # Calculate how many bins we need per dimension to get close to num_codes
        # For example: if num_codes=36 and action_dim=3, we want 3^3=27, but 4^3=64 is too much
        # So we'll use 3 bins per dimension for 3D, which gives us 27 cells
        # For 6D with 36 codes, we'll use 2 bins per dimension (2^6=64, but we'll only use first 36)
        
        # Calculate optimal grid_bins based on num_codes and action_dim
        # We want to ensure that num_codes >= grid_bins^action_dim for full coverage
        optimal_grid_bins = int(self.num_codes ** (1.0 / action_dim))
        
        # If we don't have enough codes for full coverage, reduce grid_bins
        while optimal_grid_bins ** action_dim > self.num_codes and optimal_grid_bins > 2:
            optimal_grid_bins -= 1
        
        # For very high-dimensional spaces, we might need to use only 2 bins
        # and accept that we can't cover the full space with the given number of codes
        if optimal_grid_bins ** action_dim > self.num_codes:
            optimal_grid_bins = 2  # Minimum bins per dimension
        
        # Ensure we don't exceed reasonable limits
        optimal_grid_bins = max(2, min(optimal_grid_bins, 8))
        
        # For high-dimensional environments (6D+), increase num_codes to ensure full coverage
        if action_dim >= 6 and optimal_grid_bins ** action_dim > self.num_codes:
            # Calculate how many codes we need for full coverage
            required_codes = optimal_grid_bins ** action_dim
            # Cap at a reasonable maximum (e.g., 128 codes)
            max_codes = 128
            if required_codes <= max_codes:
                self.num_codes = required_codes
            else:
                self.num_codes = max_codes
        
        # Generate codes by sampling grid coordinates evenly across the action space
        for i in range(self.num_codes):
            # Convert linear index to grid coordinates
            coords = self._linear_to_grid_coords_static(i, action_dim, optimal_grid_bins)
            
            # Convert grid coordinates to action space
            code = self._grid_coords_to_action_static(coords, optimal_grid_bins, self.device)
            codes.append(code)
        
        self.codes = torch.stack(codes)
        
        # Update grid_bins to match what we actually used
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
            return
        
        # Check file cache
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                grid_table = pickle.load(f)
            # Move to device
            self.grid_table = grid_table.to(self.device)
            # Store in memory cache
            self._grid_table_cache[cache_key] = self.grid_table
            return
        
        # If not in cache, this should not happen if pre-generated in DecisionTransformer
        self.grid_table = None



    def _action_to_grid_idx(self, action):
        """
        Convert action to grid index for fast lookup.
        OPTIMIZATION: Vectorized grid index computation with precomputed powers.
        Args:
            action: tensor of shape (d,)
        Returns:
            grid_idx: linear index into grid table
        """
        # Map action from [-1, 1] to [0, grid_bins-1] using rounding (nearest center)
        tmp = (action + 1.0) / 2.0 * (self.grid_bins - 1)
        grid_coords = tmp.round().clamp(0, self.grid_bins - 1).long()
        
        # OPTIMIZATION 2: Vectorized grid index computation
        # Use precomputed powers instead of Python loop
        action_dim = len(grid_coords)
        powers = self.grid_powers[:action_dim]  # Get powers for this action dimension
        
        # Vectorized computation: sum(grid_coords * powers)
        grid_idx = torch.sum(grid_coords * powers)
        
        return grid_idx
    
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
        powers = self.grid_powers[:action_dim].unsqueeze(0)  # (1, action_dim)
        
        # Vectorized computation: sum(grid_coords * powers, dim=1)
        grid_indices = torch.sum(grid_coords * powers, dim=1)
        
        return grid_indices

    def route_action_fast(self, action): # TODO: remove this function
        """
        Route action to nearest code using grid lookup or direct computation.
        Args:
            action: tensor of shape (d,)
        Returns:
            code_idx: index of nearest code
        """
        if self.grid_table is not None:
            # Use grid lookup for fast routing
            grid_idx = self._action_to_grid_idx(action)
            if grid_idx < len(self.grid_table):
                code_idx = self.grid_table[grid_idx].item()
                return code_idx
        
        # Fallback: direct computation (slower but always works)
        distances = torch.norm(self.codes - action, dim=1)
        return torch.argmin(distances).item()
    
    def route_actions_fast_batch(self, actions):
        """
        Route batch of actions to nearest codes using grid lookup or direct computation.
        OPTIMIZATION: Vectorized batch routing for multiple actions.
        Args:
            actions: tensor of shape (batch_size, d)
        Returns:
            code_indices: tensor of shape (batch_size,) with code indices
        """
        if self.grid_table is not None:
            # Use vectorized grid lookup for fast routing
            grid_indices = self._action_to_grid_idx_batch(actions)
            
            # Check bounds and get code indices
            valid_mask = grid_indices < len(self.grid_table)
            code_indices = torch.zeros(actions.shape[0], dtype=torch.long, device=actions.device)
            
            # Use grid table for valid indices
            if valid_mask.any():
                code_indices[valid_mask] = self.grid_table[grid_indices[valid_mask]]
            
            # Fallback for invalid indices
            invalid_mask = ~valid_mask
            if invalid_mask.any():
                distances = torch.cdist(actions[invalid_mask], self.codes)
                code_indices[invalid_mask] = torch.argmin(distances, dim=1)
            
            return code_indices
        
        # Fallback: direct computation for all actions
        distances = torch.cdist(actions, self.codes)
        return torch.argmin(distances, dim=1)

    def process_trajectory(self, actions, rewards):
        """
        Process a single trajectory using VQ-based EWA.
        Args:
            actions: tensor of shape (k, d) for one trajectory
            rewards: tensor of shape (k,) or (k, 1) for one trajectory
        Returns:
            D_codebook: list of VQ codes with attractions (summary of used codes)
            D_trajectory: list of dicts with action info and attractions
        """
        # Extract actual sequence length from rewards (not actions)
        # This is needed because in evaluation phase, sequences grow step-by-step:
        # Step 0: actions(10,1,6), rewards(10,1,1) -> Step 19: actions(10,20,6), rewards(10,20,1)
        actual_seq_length = rewards.shape[0]

        # Only process the actions/rewards that actually exist
        actions = actions[:actual_seq_length]
        rewards = rewards[:actual_seq_length]
        
        k = actions.shape[0]
        actions = actions.to(self.device)
        rewards = rewards.view(-1).to(self.device)  # (k,)
        
        # Initialize codes if not done yet
        if not self.codes_initialized:
            # Simple initialization based on action dimension
            action_dim = actions.shape[1]
            self.initialize_codes(action_dim)
        
        # OPTIMIZATION: GPU-accelerated trajectory processing
        # Pre-allocate tensors on GPU for vectorized operations
        code_attraction = torch.zeros(self.num_codes, device=self.device)
        attraction_updates = torch.zeros(self.num_codes, device=self.device)
        code_usage_vec = torch.zeros(self.num_codes, dtype=torch.long, device=self.device)
        
        # Vectorized code routing (keep on GPU)
        code_indices = self.route_actions_fast_batch(actions)  # (k,) tensor
        
        # OPTIMIZATION: Vectorized attraction processing
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

            # # DEBUG: Print first few steps for verification (kept as requested)
            # print(f"DEBUG EWA: Step {t}, Action: {actions[t].cpu().numpy()}, Code: {step_code_idx}, Reward: {r_t.item():.4f}, Attraction: {code_attraction[code_idx].item():.4f}")

        # Build codebook summary only for codes that were actually used (includes negative attractions)
        D_codebook = []
        used_code_indices = torch.nonzero(code_usage_vec > 0, as_tuple=False).view(-1)
        for i in used_code_indices.tolist():
            D_codebook.append({'code_idx': i, 'A': code_attraction[i]})

        # Track history using trajectory-based statistics
        avg_attraction = float(np.mean([step['A_t'] for step in D_trajectory])) if len(D_trajectory) > 0 else 0.0
        self.attraction_history.append(avg_attraction)
        self.code_usage_history.append(int((code_usage_vec > 0).sum().item()))
        self.reward_history.append(float(np.mean(trajectory_rewards)) if len(trajectory_rewards) > 0 else 0.0)
        # print(f"\nD_codebook: {D_codebook}")
        # print(f"\nD_trajectory: {D_trajectory}")
        return D_codebook, D_trajectory

    def process_batch(self, actions, rewards):
        """
        Process a batch of trajectories.
        Args:
            actions: tensor of shape (B, k, d)
            rewards: tensor of shape (B, k) or (B, k, 1)
        Returns:
            D_codebooks: list of D_codebook for each trajectory
            D_trajectories: list of D_trajectory for each trajectory
        """
        # Extract dimensions dynamically
        batch_size, num_action_tokens, tuple_seq_length, action_indices = self._extract_dimensions(rewards)
        
        D_codebooks = []
        D_trajectories = []
        
        for b in range(batch_size):
            D_codebook, D_trajectory = self.process_trajectory(actions[b], rewards[b])
            D_codebooks.append(D_codebook)
            D_trajectories.append(D_trajectory)
        return D_codebooks, D_trajectories

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
        

        # print(f"\n--- ATTRACTION MATRIX ---")
        # print(f"Batch size: {batch_size}, Seq length: {tuple_seq_length}")
        # print(f"Action indices: {action_indices.tolist()}")
        
        for b in range(batch_size):
            D_trajectory = D_trajectories[b]

            # print(f"  Trajectory {b}: {len(D_trajectory)} steps")
            for i, step_info in enumerate(D_trajectory):
                if i < len(action_indices):
                    a_idx = action_indices[i].item()
                    val = step_info['A_t']
                    A[b, :, a_idx, 0] = val
                    # print(f"    Step {i}: action_idx={a_idx}, attraction={val:.3f}")
        

        # print(f"  Attraction matrix shape: {A.shape}")
        # print(f"  Attraction range: [{A.min().item():.3f}, {A.max().item():.3f}]")
        # print("-" * 50)
        
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

    # def update_codes_offline(self, all_actions):
    #     """
    #     Update VQ codes using k-means on all available actions.
    #     Call this periodically (e.g., every 10-20 epochs).
    #     Args:
    #         all_actions: tensor of shape (N, d) - all actions from dataset
    #     """
    #     # print(f"Updating VQ codes with {len(all_actions)} actions...")
        
    #     # Convert to numpy for sklearn
    #     actions_np = all_actions.cpu().numpy()
        
    #     # Run k-means clustering
    #     kmeans = KMeans(n_clusters=self.num_codes, random_state=42, n_init=10)
    #     kmeans.fit(actions_np)
        
    #     # Update codes
    #     self.codes = torch.tensor(kmeans.cluster_centers_, device=self.device, dtype=torch.float32)
        
    #     # Grid table is already cached, no need to rebuild
    #     # print(f"VQ codes updated. New code shape: {self.codes.shape}") 