import torch
import numpy as np
from collections import deque, OrderedDict

class EWAKernel:
    """
    Kernel-smoothed Experience-Weighted Attraction for continuous actions.
    """

    def __init__(self, num_heads, seq_length, batch_size,
                 phi=0.9, initial_delta=0.7, 
                 delta_decay_rate=0.001,  # Rate at which delta decays
                 sigma_window_size=None,   # Window size for adaptive sigma
                 monitor_interval=100,    # How often to log stats
                 h=0.25):                 # Novelty threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.phi = phi
        self.delta = initial_delta
        self.initial_delta = initial_delta
        self.delta_decay_rate = delta_decay_rate
        self.h = h  # Novelty threshold
        self.sigma = 0.5  # Initial RBF bandwidth

        # --- support dictionary D = {(a_ℓ, u_ℓ)} -----------------
        self.D = OrderedDict()  # Dictionary storing (a_ℓ, u_ℓ) tuples
        # -----------------------------------------------------------
        
        # For adaptive sigma
        self.sigma_window_size = sigma_window_size
        self.recent_actions = deque(maxlen=sigma_window_size)
        
        # For monitoring
        self.monitor_interval = monitor_interval
        self.step = 0
        self.stats = {
            'support_size': [],
            'sigma_values': [],
            'delta_values': [],
            'new_supports': 0,
            'attraction_history': []  # Track attraction values over time
        }

        # Store the last processed actions and rewards for attraction computation
        self.last_actions = None
        self.last_rewards = None

    def _tensor_to_key(self, tensor):
        """Convert tensor to hashable tuple for dictionary key"""
        return tuple(tensor.cpu().numpy().tolist())

    def _key_to_tensor(self, key):
        """Convert tuple key back to tensor"""
        return torch.tensor(key, device=self.device)

    # ---------- kernel function -----------------------------------
    @staticmethod
    def _kernel_rbf(a, c, sigma):
        """RBF kernel k(a,c) with bandwidth sigma, a,c are (d,) tensors"""
        return torch.exp(-torch.norm(a - c) ** 2 / (2 * sigma ** 2))
    # --------------------------------------------------------------

    # ---------- public helper -------------------------------------
    def _compute_A(self, a_batch):
        """
        Compute attraction function A_t(a) using Eq. (1)
        a_batch : (B, d) tensor of actions
        Return  : (B,) tensor A_t(a) using eq (1)
        """
        if len(self.D) == 0:
            return torch.zeros(a_batch.size(0), device=a_batch.device)

        # Get actions and weights from dictionary
        actions = torch.stack([self._key_to_tensor(k) for k in self.D.keys()])  # (M,d)
        weights = torch.tensor(list(self.D.values()), device=a_batch.device)  # (M,)
        
        # pairwise squared distance
        dist2 = ((a_batch[:, None, :] - actions[None, :, :]) ** 2).sum(-1)  # (B,M)
        k_mat = torch.exp(-dist2 / (2 * self.sigma ** 2))             # (B,M)
        return (k_mat * weights).sum(-1)                                    # (B,)
    # --------------------------------------------------------------

    def _update_sigma(self):
        """Update sigma based on median pairwise distance of recent actions"""
        if len(self.recent_actions) < 2:
            return
            
        # Convert recent actions to tensor
        recent = torch.stack(list(self.recent_actions))
        
        # Compute pairwise distances
        dists = torch.cdist(recent, recent)
        
        # Get upper triangular part (excluding diagonal)
        mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
        pairwise_dists = dists[mask]
        
        if len(pairwise_dists) > 0:
            # Update sigma as median distance
            self.sigma = torch.median(pairwise_dists).item()
            
            # Log sigma value
            self.stats['sigma_values'].append(self.sigma)

    def _decay_delta(self):
        """Decay delta value over time"""
        self.delta = self.initial_delta / (1 + self.delta_decay_rate * self.step)
        self.stats['delta_values'].append(self.delta)

    def get_stats(self):
        """Get current statistics about the kernel"""
        return {
            'support_size': len(self.D),
            'current_sigma': self.sigma,
            'current_delta': self.delta,
            'new_supports': self.stats['new_supports'],
            'attraction_history': self.stats['attraction_history']
        }

    def update(self, actions, rewards):
        """
        Update kernel weights based on executed actions and rewards
        actions : (B, k, d)   executed continuous actions at t
        rewards : (B, k, 1)     rewards r_t
        """
        actions = actions.detach().to(self.device)
        rewards = rewards.detach().to(self.device)
        
        # Store actions and rewards for use in get_attraction
        self.last_actions = actions
        self.last_rewards = rewards

        # Update step counter and decay delta
        self.step += 1
        self._decay_delta()

        # Process each sequence in the batch
        for b in range(self.batch_size):
            # Initialize D ← empty list for this sequence
            self.D = OrderedDict()  # Reset dictionary for this sequence
            
            # Process each timestep in the sequence
            for a_t, r_t in zip(actions[b], rewards[b]): # a_t: (d); r_t: (1)
                # Add to recent actions for sigma adaptation
                self.recent_actions.append(a_t)
                
                if len(self.D) == 0:
                    # Initialize dictionary D with first action
                    self.D[self._tensor_to_key(a_t)] = self.delta * r_t
                    self.stats['new_supports'] += 1
                    continue

                # Find nearest support point (a_j, u_j) ∈ D
                actions_D = torch.stack([self._key_to_tensor(k) for k in self.D.keys()])  # (M, d)
                dists = torch.norm(actions_D - a_t, dim=1)     # (M,)  # distances to each point in D
                min_dist, idx = torch.min(dists, dim=0)
                nearest_action_key = list(self.D.keys())[idx]

                # Check novelty criterion: ||a_t - a_j||₂ ≤ h
                if min_dist <= self.h:
                    # Update existing weight: u_j ← φ·u_j + δ_t·r_t
                    self.D[nearest_action_key] = (self.phi * self.D[nearest_action_key] + 
                                                self.delta * r_t)
                else:
                    # Add new support point: (a_t, δ_t·r_t) to D
                    self.D[self._tensor_to_key(a_t)] = self.delta * r_t
                    self.stats['new_supports'] += 1

                # Maintain exactly seq_length support points by keeping only the most recent ones
                if len(self.D) > self.seq_length:
                    # Remove the oldest support point
                    self.D.popitem(last=False)  # Remove first item (oldest)

            # Update sigma periodically
            if self.step % self.monitor_interval == 0:
                self._update_sigma()
                self.stats['support_size'].append(len(self.D))

    def get_attraction(self):
        """
        Get attraction values for action tokens
        Returns: (B, H, L, 1) tensor of attraction values where:
        - B is batch size
        - H is number of attention heads
        - L is sequence length
        - Values are consistent across heads for each sequence
        - Only action tokens have non-zero attraction values
        """
        if len(self.D) == 0 or self.last_actions is None:
            return torch.zeros((self.batch_size, self.num_heads, self.seq_length, 1), device=self.device)
            
        # Get all actions from the batch
        batch_actions = self.last_actions  # (B, L, d)
        
        # Define action token indices in the full sequence
        # For sequence (R_1, s_1, a_1, R_2, s_2, a_2, ...), action tokens are at indices 2, 5, 8, ...
        action_indices = torch.arange(2, self.seq_length, step=3, device=self.device)
        
        # Extract all action tokens from all sequences in the batch
        # Reshape to (B*num_actions, d) to compute attractions for all actions at once
        action_tokens = batch_actions[:, action_indices]  # (B, num_actions, d)
        B, num_actions, d = action_tokens.shape
        action_tokens_flat = action_tokens.reshape(-1, d)  # (B*num_actions, d)
        
        # Compute attraction values for all actions at once
        A_flat = self._compute_A(action_tokens_flat)  # (B*num_actions,)
        
        # Reshape back to (B, num_actions)
        A_reshaped = A_flat.reshape(B, num_actions)
        
        # Initialize full batch attraction tensor with zeros
        A = torch.zeros((self.batch_size, self.num_heads, self.seq_length, 1), device=self.device)
        
        # Place attraction values at the correct indices for all sequences
        A[:, :, action_indices, 0] = A_reshaped.unsqueeze(1).expand(-1, self.num_heads, -1)
        
        # Track attraction values in history
        mean_attraction = A.mean().item()
        self.stats['attraction_history'].append(mean_attraction)
            
        return A
