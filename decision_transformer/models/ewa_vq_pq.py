import torch
import numpy as np
import os
import pickle
import hashlib
import time
from typing import List, Tuple, Dict, Optional

class EWAVQProductQuantization:
    """
    Product Quantization (PQ) based EWA for high-dimensional action spaces.
    
    SOLVES THE CURSE OF DIMENSIONALITY:
    - Instead of 6^6 = 46,656 cells, split into 2×3D subspaces: 3^3 + 3^3 = 54 codes
    - Scales linearly with action dimension instead of exponentially
    
    IMPLEMENTATION:
    1. Split action space into K subspaces (e.g., 6D → 2×3D or 3×2D)
    2. Each subspace gets its own codebook with M codes
    3. Total codes = K × M (much smaller than action_dim^grid_bins)
    4. Actions are quantized per subspace and combined
    """
    
    # Class-level cache for subspace codebooks
    _subspace_cache = {}
    _cache_dir = "./ewa_pq_cache"
    
    @classmethod
    def _get_cache_key(cls, action_dim, num_subspaces, grid_bins, codes_per_subspace, env_name=None):
        """Generate a unique cache key for PQ parameters."""
        if env_name:
            return f"pq_{action_dim}d_{num_subspaces}sub_{grid_bins}b_{codes_per_subspace}c_{env_name}"
        else:
            return f"pq_{action_dim}d_{num_subspaces}sub_{grid_bins}b_{codes_per_subspace}c"
    
    @classmethod
    def _get_cache_path(cls, cache_key):
        """Get the file path for a cached PQ codebook."""
        os.makedirs(cls._cache_dir, exist_ok=True)
        return os.path.join(cls._cache_dir, f"{cache_key}.pkl")
    
    @staticmethod
    def _split_action_space(action_dim: int, num_subspaces: int) -> List[Tuple[int, int]]:
        """
        Split action space into subspaces optimally.
        Args:
            action_dim: total action dimension
            num_subspaces: number of subspaces to create
        Returns:
            List of (start_idx, end_idx) tuples for each subspace
        """
        if num_subspaces > action_dim:
            raise ValueError(f"Cannot create {num_subspaces} subspaces for {action_dim}D actions")
        
        # Optimal split: try to make subspaces as balanced as possible
        base_size = action_dim // num_subspaces
        remainder = action_dim % num_subspaces
        
        subspaces = []
        start_idx = 0
        
        for i in range(num_subspaces):
            # Distribute remainder across first few subspaces
            subspace_size = base_size + (1 if i < remainder else 0)
            end_idx = start_idx + subspace_size
            subspaces.append((start_idx, end_idx))
            start_idx = end_idx
        
        return subspaces 
    
    def _generate_subspace_codes(self, subspace_dim: int, num_codes: int, device: torch.device) -> Tuple[torch.Tensor, int]:
        """
        Generate codes for a single subspace using grid-based approach.
        Args:
            subspace_dim: dimension of this subspace
            num_codes: number of codes to generate
            device: device to create tensors on
        Returns:
            codes: tensor of shape (num_codes, subspace_dim)
            grid_bins: number of bins used for this subspace
        """
        # Use the instance grid_bins for consistent quantization
        grid_bins = self.grid_bins
        
        codes = []
        for i in range(num_codes):
            # Convert linear index to grid coordinates
            coords = self._linear_to_grid_coords(i, subspace_dim, grid_bins)
            # Convert grid coordinates to action space [-1, 1]
            code = self._grid_coords_to_action(coords, grid_bins, device)
            codes.append(code)
        
        return torch.stack(codes), grid_bins
    
    def _linear_to_grid_coords(self, linear_idx: int, dim: int, grid_bins: int) -> List[int]:
        """Convert linear index to grid coordinates."""
        coords = []
        for d in range(dim):
            coords.append(linear_idx % grid_bins)
            linear_idx //= grid_bins
        return coords
    
    def _grid_coords_to_action(self, coords: List[int], grid_bins: int, device: torch.device) -> torch.Tensor:
        """Convert grid coordinates to action space [-1, 1]."""
        action = torch.tensor(coords, device=device, dtype=torch.float32)
        action = 2.0 * action / (grid_bins - 1) - 1.0
        return action

    def __init__(self, num_heads: int, tuple_seq_length: int, phi: float, delta: float, 
                 action_dim: int, num_subspaces: int = None, codes_per_subspace: int = None,
                 grid_bins: int = None, max_subspaces: int = None):
        """
        Args:
            num_heads: number of attention heads
            tuple_seq_length: total sequence length (3 * actions_seq_length)
            phi: decay factor for attraction
            delta: chosen-vs-unchosen weight
            action_dim: dimension of action space
            num_subspaces: number of subspaces (default: auto-determine)
            codes_per_subspace: codes per subspace (default: auto-determine)
            grid_bins: number of bins per dimension for grid quantization (default: from variant)
            max_subspaces: maximum number of subspaces to create (default: from variant)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.tuple_seq_length = tuple_seq_length
        self.phi = phi
        self.delta = delta
        self.action_dim = action_dim
        
        # PQ parameters - use provided values or defaults from variant
        self.grid_bins = grid_bins if grid_bins is not None else 3  # Default fallback
        self.max_subspaces = max_subspaces if max_subspaces is not None else 4  # Default fallback
        
        if num_subspaces is None:
            # Auto-determine optimal number of subspaces
            if action_dim <= 3:
                self.num_subspaces = 1
            elif action_dim <= 6:
                self.num_subspaces = 2  # Split 6D into 2×3D
            elif action_dim <= 9:
                self.num_subspaces = 3  # Split 9D into 3×3D
            else:
                # For very high dimensions, limit to max_subspaces
                self.num_subspaces = min(max_subspaces, max(2, action_dim // 3))
        else:
            self.num_subspaces = num_subspaces
        
        if codes_per_subspace is None:
            # Auto-determine codes per subspace for FULL COVERAGE
            if self.num_subspaces == 1:
                # Single subspace: use grid_bins^action_dim for full coverage
                self.codes_per_subspace = self.grid_bins ** action_dim
            else:
                # Multiple subspaces: each subspace only needs to cover its own dimensions
                # This is the correct Product Quantization approach
                subspace_dim = action_dim // self.num_subspaces
                self.codes_per_subspace = self.grid_bins ** subspace_dim
        else:
            self.codes_per_subspace = codes_per_subspace
        
        # Calculate total codes
        self.total_codes = self.num_subspaces * self.codes_per_subspace
        
        # Track grid bins for caching (already set above)
        
        # Split action space into subspaces
        self.subspaces = self._split_action_space(action_dim, self.num_subspaces)
        
        # Initialize subspace codebooks
        self.subspace_codebooks = []
        self.subspace_attractions = []
        self.subspace_usage = []
        
        # History tracking
        self.step = 0
        self.attraction_history = []
        self.code_usage_history = []
        self.reward_history = []
        
        # Track if codebooks have been initialized
        self.codebooks_initialized = False
        
        # print(f"PQ-EWA initialized: {action_dim}D → {self.num_subspaces} subspaces × {self.codes_per_subspace} codes = {self.total_codes} total codes")
        # for i, (start, end) in enumerate(self.subspaces):
        #     print(f"  Subspace {i}: actions[{start}:{end}] ({end-start}D)")

    def initialize_codebooks(self):
        """Initialize all subspace codebooks."""
        if self.codebooks_initialized:
            return
        
        # print(f"Initializing PQ codebooks for {self.num_subspaces} subspaces...")
        
        # DEBUG: Print detailed setup information
        # print(f"🔧 PQ Codebook Setup:")
        # print(f"   Action dimension: {self.action_dim}")
        # print(f"   Number of subspaces: {self.num_subspaces}")
        # print(f"   Codes per subspace: {self.codes_per_subspace}")
        # print(f"   Grid bins: {self.grid_bins}")
        # print(f"   Total codes: {self.total_codes}")
        # print(f"   Subspace splits: {self.subspaces}")
        
        # Track grid bins across all subspaces
        subspace_grid_bins = []
        
        for i, (start_idx, end_idx) in enumerate(self.subspaces):
            subspace_dim = end_idx - start_idx
            
            # Generate codes for this subspace
            codes, grid_bins = self._generate_subspace_codes(subspace_dim, self.codes_per_subspace, self.device)
            self.subspace_codebooks.append(codes)
            subspace_grid_bins.append(grid_bins)
            
            # Initialize attraction and usage tracking for this subspace
            self.subspace_attractions.append(torch.zeros(self.codes_per_subspace, device=self.device))
            self.subspace_usage.append(torch.zeros(self.codes_per_subspace, dtype=torch.long, device=self.device))
            
            # print(f"  Subspace {i}: {codes.shape} codes, {grid_bins} bins, range [{codes.min().item():.3f}, {codes.max().item():.3f}]")
            
            # DEBUG: Show sample codes for first subspace
            # if i == 0:
            #     print(f"     Sample codes (first 5): {codes[:5].cpu().numpy()}")
        
        # Set overall grid_bins (use the maximum across subspaces for consistency)
        self.grid_bins = max(subspace_grid_bins)
        
        self.codebooks_initialized = True


    def _quantize_action_subspace(self, action_subspace: torch.Tensor, subspace_idx: int) -> int:
        """
        Quantize actions in a single subspace to nearest code.
        Args:
            action_subspace: actions for this subspace (batch_size, subspace_dim)
            subspace_idx: index of the subspace
        Returns:
            code_indices: indices of nearest codes (batch_size,)
        """
        if not self.codebooks_initialized:
            self.initialize_codebooks()
        
        codes = self.subspace_codebooks[subspace_idx]  # (codes_per_subspace, subspace_dim)
        
        # DEBUG: Print quantization details for first batch
        # print(f"🔍 Quantization Debug - Subspace {subspace_idx}:")
        # print(f"   Action subspace shape: {action_subspace.shape}")
        # print(f"   Codes shape: {codes.shape}")
        # print(f"   Action sample: {action_subspace[0].cpu().numpy()}")
        # print(f"   Code sample: {codes[0].cpu().numpy()}")
           
        
        # Compute distances to all codes
        distances = torch.cdist(action_subspace, codes)  # (batch_size, codes_per_subspace)
        
        # DEBUG: Show distance calculations for first action
        # print(f"   Distance matrix shape: {distances.shape}")
        # print(f"   Distances for first action: {distances[0, :10].cpu().numpy()}")
        # print(f"   Min distance: {distances[0].min().item():.4f}")
        
        # Find nearest code for each action
        code_indices = torch.argmin(distances, dim=1)  # (batch_size,)
        
        return code_indices

    def _quantize_actions_pq(self, actions: torch.Tensor) -> List[torch.Tensor]:
        """
        Quantize actions using Product Quantization.
        Args:
            actions: tensor of shape (batch_size, action_dim)
        Returns:
            code_indices_list: list of code indices for each subspace
        """
        if not self.codebooks_initialized:
            self.initialize_codebooks()
        
        code_indices_list = []
        
        for subspace_idx, (start_idx, end_idx) in enumerate(self.subspaces):
            # Extract actions for this subspace
            action_subspace = actions[:, start_idx:end_idx]
            
            # Quantize this subspace
            code_indices = self._quantize_action_subspace(action_subspace, subspace_idx)
            code_indices_list.append(code_indices)
        
        return code_indices_list

    def process_trajectory(self, actions: torch.Tensor, rewards: torch.Tensor):
        """
        Process a single trajectory using PQ-based EWA.
        Args:
            actions: tensor of shape (k, action_dim) for one trajectory
            rewards: tensor of shape (k,) or (k, 1) for one trajectory
        Returns:
            D_codebook: list of codes with attractions (summary of used codes)
            D_trajectory: list of dicts with action info and attractions
        """
        # DEBUG: Print trajectory processing details
        # print(f"\n🔍 EWA VQ PQ DEBUG ")
        # print(f"   Actions shape: {actions.shape}")
        # print(f"   Rewards shape: {rewards.shape}")
        # print(f"   Actions sample: {actions.cpu().numpy()}")
        # print(f"   Rewards sample: {rewards.cpu().numpy()}")
        
        # Initialize codebooks FIRST if not done yet
        if not self.codebooks_initialized:
            self.initialize_codebooks()
        
        # RESET: Clear attractions and usage for each new trajectory
        # print(f"   🔄 Resetting attractions and usage for new trajectory")
        for subspace_idx in range(self.num_subspaces):
            self.subspace_attractions[subspace_idx].zero_()
            self.subspace_usage[subspace_idx].zero_()
        
        # Extract actual sequence length from rewards
        actual_seq_length = rewards.shape[0]
        actions = actions[:actual_seq_length]
        rewards = rewards[:actual_seq_length]
        
        # Get dimensions
        k = actions.shape[0]
        d = actions.shape[1]
        
        actions = actions.to(self.device)
        rewards = rewards.view(-1).to(self.device)
        
        # Quantize actions using PQ
        code_indices_list = self._quantize_actions_pq(actions)  # List of (k,) tensors
        
        # print(f"   📊 Quantization results:")
        # print(f"      Subspaces: {len(code_indices_list)}")
        # for i, codes in enumerate(code_indices_list):
        #     print(f"   Subspace {i}: {codes.cpu().numpy()}")
        
        # Process trajectory step by step
        D_trajectory = []
        trajectory_rewards = []
        
        for t in range(k):
            r_t = rewards[t]
            
            # Decay all subspace attractions
            for subspace_idx in range(self.num_subspaces):
                self.subspace_attractions[subspace_idx] *= (1 - self.phi)
            
            # Update attractions for used codes in each subspace
            for subspace_idx in range(self.num_subspaces):
                code_idx = code_indices_list[subspace_idx][t]
                self.subspace_attractions[subspace_idx][code_idx] += self.delta * r_t
                self.subspace_usage[subspace_idx][code_idx] += 1

                # print(f"      Step {t} out of {k}, Subspace {subspace_idx}:")
                # print(f"        Code index: {code_idx.item()}")
                # print(f"        Reward: {r_t.item():.4f}")
                # print(f"        Attraction before: {(self.subspace_attractions[subspace_idx][code_idx] - self.delta * r_t).item():.4f}")
                # print(f"        Reward: {r_t.item():.4f}")
                # print(f"        Attraction before: {(self.subspace_attractions[subspace_idx][code_idx] - self.delta * r_t).item():.4f}")
                # print(f"        Attraction after: {self.subspace_attractions[subspace_idx][code_idx].item():.4f}")
        
            # Track reward
            trajectory_rewards.append(r_t.item())
            
            # Build trajectory info
            step_info = {
                'a_t': actions[t].detach().cpu().numpy(),
                'r_t': r_t.item(),
                'code_indices': [idx.item() for idx in [code_indices_list[s][t] for s in range(self.num_subspaces)]],
                'subspace_attractions': [self.subspace_attractions[s][code_indices_list[s][t]].item() for s in range(self.num_subspaces)],
                'total_attraction': sum(self.subspace_attractions[s][code_indices_list[s][t]].item() for s in range(self.num_subspaces))
            }
            D_trajectory.append(step_info)
        
        # Build codebook summary for codes that were actually used
        D_codebook = []
        for subspace_idx in range(self.num_subspaces):
            used_codes = torch.nonzero(self.subspace_usage[subspace_idx] > 0, as_tuple=False).view(-1)
            for code_idx in used_codes.tolist():
                D_codebook.append({
                    'subspace_idx': subspace_idx,
                    'code_idx': code_idx,
                    'A': self.subspace_attractions[subspace_idx][code_idx].item(),
                    'usage': self.subspace_usage[subspace_idx][code_idx].item()
                })
        

        # print(f"   📈 Codebook summary:")
        # for code_info in D_codebook:
        #     print(f"      Subspace {code_info['subspace_idx']}, Code {code_info['code_idx']}: "
        #             f"Attraction={code_info['A']:.4f}, Usage={code_info['usage']}")
    
        # Track history
        avg_attraction = float(np.mean([step['total_attraction'] for step in D_trajectory])) if len(D_trajectory) > 0 else 0.0
        total_code_usage = sum(len(torch.nonzero(usage > 0)) for usage in self.subspace_usage)
        avg_reward = float(np.mean(trajectory_rewards)) if len(trajectory_rewards) > 0 else 0.0
        
        iteration_data = {
            'step': self.step,
            'avg_attraction': avg_attraction,
            'code_usage': total_code_usage,
            'reward': avg_reward,
            'trajectory_steps': D_trajectory,
            'timestamp': time.time()
        }
        
        self.attraction_history.append(iteration_data)
        self.code_usage_history.append(total_code_usage)
        self.reward_history.append(avg_reward)

        self.step += 1

        # print(f"\nD_codebook: {D_codebook}")
        # print(f"\nD_trajectory: {D_trajectory}")
        
        return D_codebook, D_trajectory

    def process_batch(self, actions: torch.Tensor, rewards: torch.Tensor):
        """
        Process a batch of trajectories with PQ-based EWA.
        Args:
            actions: tensor of shape (B, k, action_dim)
            rewards: tensor of shape (B, k) or (B, k, 1)
        Returns:
            D_codebooks: list of D_codebook for each trajectory
            D_trajectories: list of D_trajectory for each trajectory
        """
        batch_size = actions.shape[0]
        
        D_codebooks = []
        D_trajectories = []
        
        for b in range(batch_size):
            # print(f"Processing batch {b} out of {batch_size}")
            D_codebook, D_trajectory = self.process_trajectory(actions[b], rewards[b])
            D_codebooks.append(D_codebook)
            D_trajectories.append(D_trajectory)
        
        return D_codebooks, D_trajectories

    def get_attraction(self, D_trajectories: List, rewards: torch.Tensor) -> torch.Tensor:
        """
        Build the attraction matrix for the batch.
        Args:
            D_trajectories: list of D_trajectory (length B)
        Returns:
            Tensor of shape (B, num_heads, tuple_seq_length, 1) with attraction values at action token indices
        """
        batch_size = len(D_trajectories)
        tuple_seq_length = self.tuple_seq_length
        
        # Extract action token indices (assuming they're at positions 2, 5, 8, ...)
        num_action_tokens = rewards.shape[1] if len(rewards.shape) > 1 else 1
        if num_action_tokens == 1:
            action_indices = torch.tensor([2], device=self.device)
        else:
            action_indices = torch.arange(2, min(tuple_seq_length, 3 * num_action_tokens), step=3, device=self.device)
        
        A = torch.zeros((batch_size, self.num_heads, tuple_seq_length, 1), device=self.device)
        
        for b in range(batch_size):
            D_trajectory = D_trajectories[b]
            
            for i, step_info in enumerate(D_trajectory):
                if i < len(action_indices):
                    a_idx = action_indices[i].item()
                    val = step_info['total_attraction']
                    A[b, :, a_idx, 0] = val
        
        return A

    def get_history(self):
        """Return history of important variables tracked during processing."""
        steps = list(range(len(self.attraction_history)))
        attraction_values = np.array(self.attraction_history)
        code_usage_values = np.array(self.code_usage_history)
        reward_values = np.array(self.reward_history)
        
        return steps, attraction_values, code_usage_values, reward_values

    def get_current_metrics(self):
        """Get current iteration metrics efficiently for logging."""
        if len(self.attraction_history) == 0:
            return {
                'current_attraction': 0.0,
                'current_code_usage': 0,
                'current_reward': 0.0,
                'total_trajectories_processed': 0
            }
        
        latest = self.attraction_history[-1]
        current_attraction = latest.get('avg_attraction', 0.0)
        current_code_usage = latest.get('code_usage', 0)
        current_reward = latest.get('reward', 0.0)
        
        return {
            'current_attraction': current_attraction,
            'current_code_usage': current_code_usage,
            'current_reward': current_reward,
            'total_trajectories_processed': len(self.attraction_history)
        }

    def get_cache_stats(self):
        """Get cache performance statistics (kept for compatibility but always returns zeros)."""
        return {
            'cache_hits': 0,
            'cache_misses': 0,
            'hit_rate': 0.0,
            'cache_size': 0,
            'max_cache_size': 0
        }

    def clear_cache(self):
        """Clear all caches for memory management (no-op since cache removed)."""
        # print("PQ-EWA cache cleared (no cache to clear)")

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
