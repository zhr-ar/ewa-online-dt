import torch
import numpy as np
import os
import pickle

class EWAVQ:
    """
    Vector Quantization (VQ) based EWA with direct grid lookup.
    Uses a fixed codebook of action centroids and direct grid-based routing
    for maximum speed. Much faster than dynamic clustering approaches.
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
        print(f"*** EWAVQ: Creating EWAVQ with num_codes={num_codes}, grid_bins={grid_bins} ***")
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
        Generate static codes based on grid coordinates.
        This is much simpler and more reliable than k-means.
        """
        print(f"Generating static codes for action_dim={action_dim}, num_codes={self.num_codes}")
        
        # Generate codes by sampling grid coordinates
        codes = []
        for i in range(self.num_codes):
            # Convert linear index to grid coordinates
            coords = self._linear_to_grid_coords_static(i, action_dim, self.grid_bins)
            # Convert grid coordinates to action space
            code = self._grid_coords_to_action_static(coords, self.grid_bins, self.device)
            codes.append(code)
        
        self.codes = torch.stack(codes)
        print(f"Static codes generated. Shape: {self.codes.shape}")
        print(f"Code range: [{self.codes.min().item():.3f}, {self.codes.max().item():.3f}]")

    def initialize_codes(self, action_dim):
        """
        Initialize VQ codes using static grid-based generation.
        Args:
            action_dim: dimension of action space
        """
        print(f"\n*** EWAVQ: Starting static code initialization ***")
        print(f"Initializing VQ codes for action_dim={action_dim}")
        
        if self.codes_initialized:
            print("VQ codes already initialized, skipping")
            return
            
        # Generate static codes from grid
        self._generate_static_codes(action_dim)
        
        # Load grid table from cache
        env_name = getattr(self, 'variant', {}).get('env', None) if hasattr(self, 'variant') else None
        self._load_grid_table_from_cache(action_dim, env_name)
        
        self.codes_initialized = True
        print(f"VQ codes initialized. Code shape: {self.codes.shape}")
        print(f"Code range: [{self.codes.min().item():.3f}, {self.codes.max().item():.3f}]")
        if self.grid_table is not None:
            print(f"Grid table size: {self.grid_bins}^{self.codes.shape[1]} = {len(self.grid_table)} cells")
        else:
            print("Using fallback routing (no grid table)")
    
    def _load_grid_table_from_cache(self, action_dim, env_name=None):
        """Load grid table from cache or generate it if not available."""
        print(f"Loading grid table from cache for action_dim={action_dim}...")
        
        # Update the stored action_dim with the actual value
        self.action_dim = action_dim
        
        # Generate cache key - prefer environment name if available
        if env_name:
            cache_key = self._get_cache_key(action_dim, self.grid_bins, self.num_codes, env_name)
        else:
            cache_key = self._get_cache_key(action_dim, self.grid_bins, self.num_codes)
        
        # Check memory cache first
        if cache_key in self._grid_table_cache:
            print(f"✓ Using memory cached grid table for {action_dim}d, {self.grid_bins} bins, {self.num_codes} codes")
            self.grid_table = self._grid_table_cache[cache_key]
            return
        
        # Check file cache
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            print(f"✓ Loading grid table from file cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                grid_table = pickle.load(f)
            # Move to device
            self.grid_table = grid_table.to(self.device)
            # Store in memory cache
            self._grid_table_cache[cache_key] = self.grid_table
            return
        
        # If not in cache, this should not happen if pre-generated in DecisionTransformer
        print(f"⚠ Grid table not found in cache! This should not happen if pre-generated in DecisionTransformer.")
        print(f"🔄 Using fallback routing (no grid table)")
        self.grid_table = None



    def _action_to_grid_idx(self, action):
        """
        Convert action to grid index for fast lookup.
        Args:
            action: tensor of shape (d,)
        Returns:
            grid_idx: linear index into grid table
        """
        # Map action from [-1, 1] to [0, grid_bins-1]
        grid_coords = ((action + 1.0) / 2.0 * (self.grid_bins - 1)).clamp(0, self.grid_bins - 1)
        grid_coords = grid_coords.long()
        
        # Convert to linear index
        grid_idx = 0
        for i, coord in enumerate(grid_coords):
            grid_idx += coord * (self.grid_bins ** i)
        
        return grid_idx

    def route_action_fast(self, action):
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
                return self.grid_table[grid_idx].item()
        
        # Fallback: direct computation (slower but always works)
        distances = torch.norm(self.codes - action, dim=1)
        return torch.argmin(distances).item()

    def process_trajectory(self, actions, rewards):
        """
        Process a single trajectory using VQ-based EWA.
        Args:
            actions: tensor of shape (k, d) for one trajectory
            rewards: tensor of shape (k,) or (k, 1) for one trajectory
        Returns:
            D_kernel: list of VQ codes with attractions (like clusters)
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
        
        # Track trajectory info
        D_trajectory = []
        D_kernel = []  # List of VQ codes with attractions: [{'code_idx': idx, 'A': attraction}, ...]
        trajectory_rewards = []
        code_usage = torch.zeros(self.num_codes, device=self.device)
        
        # Initialize attractions per trajectory (not per code) - RESET to zero for each trajectory
        # We compute A_t dynamically from D_kernel, just like in EWAKernel
        
        # print(f"\n=== TRAJECTORY ===")
        # print(f"Total Steps: {k}, Action dim: {actions.shape[1]}")
    
        for t in range(k):
            a_t = actions[t]
            r_t = rewards[t]
            # print("-" * 20)
            # print(f"  \nStep {t}: action={[f'{x:.3f}' for x in a_t.tolist()]}, reward={r_t.item():.3f}")
            
            # Track reward
            trajectory_rewards.append(r_t.item())
            
            # 1. Route action to nearest code (FAST)
            code_idx = self.route_action_fast(a_t)
            code_usage[code_idx] += 1
            # print(f"code_idx of a_t: {code_idx}")
            # print(f"D_kernel: {D_kernel}")
            
            # 2. Decay all VQ code attractions
            for code_info in D_kernel:
                code_info['A'] = (1 - self.phi) * code_info['A']
                # print(f"code_info['A'] of D_kernel: {code_info['A']}")
            
            # 3. Update attraction for the routed code (like cluster attraction)
            attraction_update = self.delta * r_t
            
            # Find if this code already exists in D_kernel
            code_exists = False
            A_t = None  # Will store the attraction value for this step
            for code_info in D_kernel:
                if code_info['code_idx'] == code_idx:  # code_idx is already an integer
                    code_info['A'] += attraction_update
                    A_t = code_info['A']  # Use the updated cluster attraction
                    code_exists = True
                    # print(f"code_info['A'] of D_kernel: {code_info['A']}")
                    break
            
            # If code doesn't exist, add it to D_kernel
            if not code_exists:
                new_code_attraction = attraction_update
                D_kernel.append({'code_idx': code_idx, 'A': new_code_attraction})
                A_t = new_code_attraction  # Use the new code attraction
        
        
            # 4. Log to trajectory
            D_trajectory.append({
                'a_t': a_t.detach().cpu().numpy(),
                'r_t': r_t.item(),
                'code_idx': code_idx,  # code_idx is already an integer
                'A_t': A_t.item() if isinstance(A_t, torch.Tensor) else float(A_t),
                'code_usage': code_usage[code_idx].item()
            })
            # print(f"\nD_trajectory: {D_trajectory}")
            # print(f"\nD_kernel: {D_kernel}")
        
        # Track history
        avg_attraction = np.mean([step['A_t'] for step in D_trajectory])
        self.attraction_history.append(avg_attraction)
        self.code_usage_history.append(torch.sum(code_usage > 0).item())
        self.reward_history.append(np.mean(trajectory_rewards))
        
        
        return D_kernel, D_trajectory

    def process_batch(self, actions, rewards):
        """
        Process a batch of trajectories.
        Args:
            actions: tensor of shape (B, k, d)
            rewards: tensor of shape (B, k) or (B, k, 1)
        Returns:
            D_kernels: list of D_kernel for each trajectory
            D_trajectories: list of D_trajectory for each trajectory
        """
        # Extract dimensions dynamically
        batch_size, num_action_tokens, tuple_seq_length, action_indices = self._extract_dimensions(rewards)
        
        D_kernels = []
        D_trajectories = []
        
        for b in range(batch_size):
            D_kernel, D_trajectory = self.process_trajectory(actions[b], rewards[b])
            D_kernels.append(D_kernel)
            D_trajectories.append(D_trajectory)
        return D_kernels, D_trajectories

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