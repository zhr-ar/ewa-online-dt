import torch
import numpy as np

class EWAKernel:
    """
    Kernel-based EWA with clustering for continuous actions (per trajectory).
    Each sequence in Decision Transformer consists of 3 types of tokens (returns, states, actions),
    so if we have k actions, the total sequence length is 3*k.
    This class implements the clustering-based EWA algorithm per trajectory, with batch support.
    Uses dynamic shape extraction for maximum flexibility.
    """
    def __init__(self, num_heads, tuple_seq_length, phi, delta):
        """
        Args:
            num_heads: number of attention heads
            tuple_seq_length: total sequence length (3 * actions_seq_length)
            phi: decay factor for attraction
            delta: chosen-vs-unchosen weight
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.tuple_seq_length = tuple_seq_length
        self.phi = phi
        self.delta = delta
        self.alpha = 0.7  # scale factor for kernel width (sigma)
        self.beta = 0.8   # scale factor for merge radius (for tau)
        
        # History tracking variables
        self.step = 0
        self.attraction_history = []
        self.sigma_history = []
        self.tau_history = []
        self.cluster_count_history = []
        self.reward_history = []
        
        # # Dynamic state (set during processing)
        # self.current_batch_size = None
        # self.current_action_indices = None
        # self.current_tuple_seq_length = None

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

    @staticmethod
    def pairwise_distances(actions):
        """
        Compute all pairwise L2 distances for a set of actions (K, d) -> (K*(K-1)/2,)
        Args:
            actions: (K, d) tensor
        Returns:
            np.ndarray of distances
        """
        K = actions.shape[0]
        dists = []
        for i in range(K):
            for j in range(i+1, K):
                d = torch.norm(actions[i] - actions[j]).item()
                dists.append(d)
        return np.array(dists)

    def pick_sigma_tau(self, actions):
        """
        Pick kernel width sigma and similarity threshold tau per trajectory.
        Args:
            actions: tensor of shape (k, d) for one trajectory
        Returns:
            sigma, tau: kernel width and similarity threshold
        """
        if len(actions) < 2:
            return 1.0, 0.5  # Default values for single action
        
        # Compute pairwise distances
        distances = self.pairwise_distances(actions)
        # Convert numpy array to tensor for torch.median()
        distances_tensor = torch.tensor(distances, device=self.device)
        m_t = torch.median(distances_tensor).item()
        
        # Set kernel width and threshold
        sigma = self.alpha * m_t
        r = self.beta * m_t
        tau = np.exp(-r**2 / (2 * sigma**2)) if sigma > 0 else 0.5
        
        return sigma, tau

    def process_trajectory(self, actions, rewards):
        """
        Process a single trajectory to build clusters and compute attractions.
        OPTIMIZED VERSION: Simplified clustering for speed.
        Args:
            actions: tensor of shape (k, d) for one trajectory
            rewards: tensor of shape (k,) or (k, 1) for one trajectory
        Returns:
            D_kernel: list of clusters with centers and attractions
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
        d = actions.shape[1]
        actions = actions.to(self.device)
        rewards = rewards.view(-1).to(self.device)  # (k,)
        
        # OPTIMIZATION: Use fixed sigma and tau for speed
        sigma = 0.3  # Fixed kernel width
        tau = 0.7    # Fixed similarity threshold
        
        # Track sigma and tau for this trajectory
        self.sigma_history.append(sigma)
        self.tau_history.append(tau)
        
        # Initialize data structures
        D_kernel = []  # List of clusters: [{'mu': center, 'A': attraction}, ...]
        D_trajectory = []  # List of action info: [{'a_t': action, 'r_t': reward, 'cluster_id': id, 'A_t': attraction, 'k_t': similarity}, ...]
        
        # Track rewards for this trajectory
        trajectory_rewards = []
        
        # OPTIMIZATION: Limit number of clusters for speed
        max_clusters = min(10, k)  # Maximum 10 clusters per trajectory
        
        for t in range(k):
            a_t = actions[t]
            r_t = rewards[t]
            
            # Track reward
            trajectory_rewards.append(r_t.item())

            # 1. Compute similarities to all clusters (OPTIMIZED)
            if len(D_kernel) == 0:
                k_js = torch.zeros(0, device=self.device)
            else:
                # OPTIMIZATION: Use vectorized computation
                mus = torch.stack([c['mu'] for c in D_kernel])  # (J, d)
                diff = mus - a_t.unsqueeze(0)  # (J, d)
                k_js = torch.exp(-torch.norm(diff, dim=1) ** 2 / (2 * sigma ** 2))  # (J,)
            
            if len(k_js) > 0:
                j_star = torch.argmax(k_js).item()
                k_star = k_js[j_star].item()
            else:
                j_star = None
                k_star = 0.0

            # 2. Decay all cluster attractions
            for c in D_kernel:
                c['A'] = (1 - self.phi) * c['A']
            
            # 3. Update or create cluster (OPTIMIZED)
            if k_star >= tau and len(D_kernel) > 0 and len(D_kernel) < max_clusters:
                # Similar to existing cluster
                c = D_kernel[j_star]
                c['A'] += self.delta * r_t * k_star
                # Update center with EMA (weighted by k_star)
                c['mu'] = (1 - k_star) * c['mu'] + k_star * a_t
                cluster_id = j_star
                A_t = c['A']
                k_t = k_star
            elif len(D_kernel) < max_clusters:
                # Create new cluster
                new_cluster = {'mu': a_t.clone(), 'A': self.delta * r_t}
                D_kernel.append(new_cluster)
                cluster_id = len(D_kernel) - 1
                A_t = new_cluster['A']
                k_t = k_star if len(D_kernel) > 1 else 1.0
            else:
                # OPTIMIZATION: If too many clusters, just use the most similar existing one
                if len(D_kernel) > 0:
                    j_star = torch.argmax(k_js).item()
                    c = D_kernel[j_star]
                    c['A'] += self.delta * r_t * k_star
                    cluster_id = j_star
                    A_t = c['A']
                    k_t = k_star
                else:
                    cluster_id = 0
                    A_t = self.delta * r_t
                    k_t = 1.0
            
            # Log to D_trajectory
            D_trajectory.append({
                'a_t': a_t.detach().cpu().numpy(),
                'r_t': r_t.item(),
                'cluster_id': cluster_id,
                'A_t': A_t.item() if isinstance(A_t, torch.Tensor) else float(A_t),
                'k_t': k_t
            })
        
        # Track cluster count and average attraction for this trajectory
        self.cluster_count_history.append(len(D_kernel))
        avg_attraction = np.mean([step['A_t'] for step in D_trajectory])
        self.attraction_history.append(avg_attraction)
        self.reward_history.append(np.mean(trajectory_rewards))
        
        return D_kernel, D_trajectory

    def process_batch(self, actions, rewards):
        """
        Process a batch of trajectories.
        OPTIMIZED VERSION: Reduced overhead and early termination.
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
        
        # # OPTIMIZATION: Reduce progress prints for large batches
        # progress_interval = max(1, batch_size // 20)  # Print every 5% instead of 10%
        
        for b in range(batch_size):
            # if b % progress_interval == 0:  # Print progress every 5% of trajectories
            #     print(f"EWA: Processing trajectory {b+1}/{batch_size}")
            D_kernel, D_trajectory = self.process_trajectory(actions[b], rewards[b])
            D_kernels.append(D_kernel)
            D_trajectories.append(D_trajectory)

        return D_kernels, D_trajectories

    def get_attraction(self, D_trajectories, rewards):
        """
        Build the attraction matrix for the batch.
        Args:
            D_trajectories: list of D_trajectory (length B), each is a list of dicts for each action step
        Returns:
            Tensor of shape (B, num_heads, tuple_seq_length, 1) with attraction values at action token indices
        """
        # # Use dynamic batch size from D_trajectories
        # B = len(D_trajectories)  # Use actual batch size from data
        # H = self.num_heads
        # L = self.tuple_seq_length
        # device = self.device
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
            tuple: (steps, attraction_values, sigma_values, tau_values, cluster_counts, reward_values)
        """
        steps = list(range(len(self.attraction_history)))
        attraction_values = np.array(self.attraction_history)
        sigma_values = np.array(self.sigma_history)
        tau_values = np.array(self.tau_history)
        cluster_counts = np.array(self.cluster_count_history)
        reward_values = np.array(self.reward_history)
        
        return steps, attraction_values, sigma_values, tau_values, cluster_counts, reward_values
