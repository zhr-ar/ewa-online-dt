import torch
import numpy as np
from collections import deque, OrderedDict

class EWAKernel:
    """
    Kernel-smoothed Experience-Weighted Attraction for continuous actions.
    Each sequence in Decision Transformer consists of 3 types of tokens (returns, states, actions),
    so if we have k actions, the total sequence length is 3*k.
    """

    def __init__(self, num_heads, tuple_seq_length, batch_size,
                 phi=0.9, initial_delta=0.7, 
                 delta_decay_rate=0.001,  # Rate at which delta decays
                 sigma_window_size=None,   # Window size for adaptive sigma
                 monitor_interval=100,    # How often to log stats
                 h=0.25):                 # Novelty threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.tuple_seq_length = tuple_seq_length  # Total sequence length (3 * actions_seq_length)
        self.actions_seq_length = tuple_seq_length // 3  # Number of actions in the sequence
        self.batch_size = batch_size
        self.phi = phi
        self.delta = initial_delta
        self.initial_delta = initial_delta
        self.delta_decay_rate = delta_decay_rate
        self.h = h  # Novelty threshold
        self.sigma = 0.5  # Initial RBF bandwidth

        # --- support dictionaries D = {(a_ℓ, u_ℓ)} for each batch item -----------------
        self.Ds = [OrderedDict() for _ in range(batch_size)]  # List of dictionaries, one per batch item
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

        # Define action token indices in the full sequence
        # For sequence (R_1, s_1, a_1, R_2, s_2, a_2, ...), action tokens are at indices 2, 5, 8, ...
        self.action_indices = torch.arange(2, self.tuple_seq_length, step=3, device=self.device)

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
    def _compute_A(self, act_flat):
        """
        Compute attraction function A_t(a) using Eq. (1)
        act_flat : (B*actions_seq_length, d) tensor of actions
        Return  : (B*actions_seq_length, ) tensor A_t(a) using eq (1)
        """
        # Initialize output tensor for all batch items
        A = torch.zeros(act_flat.size(0), device=act_flat.device)  # (B*actions_seq_length,)
        
        # Process each batch item separately
        for b in range(self.batch_size):
            # Get the dictionary for this batch item
            D = self.Ds[b]
            if len(D) == 0:
                continue
                
            # Get start and end indices for this batch item's actions
            start_idx = b * self.actions_seq_length
            end_idx = (b + 1) * self.actions_seq_length
            batch_actions = act_flat[start_idx:end_idx]  # (actions_seq_length, d)
            
            # Get actions and kernel weights from dictionary
            actions_D = torch.stack([self._key_to_tensor(k) for k in D.keys()])  # (M,d)
            kernel_weights = torch.tensor(list(D.values()), device=act_flat.device)  # (M,)
            
            print(f"\n**********Computing attraction values for batch {b}:**********")
            print(f"actions_D.shape: {actions_D.shape}; Current actions in D: {actions_D}")
            print(f"kernel_weights.shape: {kernel_weights.shape}; Kernel weights in D: {kernel_weights}")
            
            # Compute pairwise squared distances efficiently
            dist2 = ((batch_actions[:, None, :] - actions_D[None, :, :]) ** 2).sum(-1)  # (actions_seq_length,M)
            print(f"batch_actions.shape: {batch_actions.shape}; actions_D.shape: {actions_D.shape}; dist2.shape: {dist2.shape}")
            print(f"dist2 min/max: {dist2.min().item():.4f}/{dist2.max().item():.4f}")
            
            k_mat = torch.exp(-dist2 / (2 * self.sigma ** 2)) # (actions_seq_length,M)
            print(f"k_mat.shape: {k_mat.shape}; k_mat min/max: {k_mat.min().item():.4f}/{k_mat.max().item():.4f}")
            
            # Compute attraction values for this batch item
            A_batch = (k_mat * kernel_weights).sum(-1)  # (actions_seq_length,)
            
            # Store in the correct position in the output tensor
            A[start_idx:end_idx] = A_batch
            
            print(f"A_batch.shape: {A_batch.shape}; A_batch min/max: {A_batch.min().item():.4f}/{A_batch.max().item():.4f}")
        
        print(f"\nFinal concatenated A shape: {A.shape}")
        print(f"A min/max/mean: {A.min().item():.4f}/{A.max().item():.4f}/{A.mean().item():.4f}")
        return A

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
            'support_size': len(self.Ds[0]),
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
        where k is the number of actions in the sequence
        """
        actions = actions.detach().to(self.device)
        rewards = rewards.detach().to(self.device)
        
        print(f"\nEWAKernel Update:")
        print(f"actions shape: {actions.shape}")
        print(f"rewards shape: {rewards.shape}")

        # Update step counter and decay delta
        self.step += 1
        # self._decay_delta()

        # Process each sequence in the batch
        for b in range(self.batch_size):
            print(f"\n**********EWAKernel Update: batch {b}**********")
            
            # Get the dictionary for this batch item
            D = self.Ds[b]
            D.clear()  # Reset dictionary for this sequence
            
            # Process each timestep in the sequence
            for a_t, r_t in zip(actions[b], rewards[b]): # a_t: (d); r_t: (1)
                # Add to recent actions for sigma adaptation
                self.recent_actions.append(a_t)
                
                if len(D) == 0:
                    # Initialize dictionary D with first action and its kernel weight
                    D[self._tensor_to_key(a_t)] = self.delta * r_t
                    self.stats['new_supports'] += 1
                    continue

                # Find nearest support point (a_j, u_j) ∈ D
                actions_D = torch.stack([self._key_to_tensor(k) for k in D.keys()])  # (M, d)
                dists = torch.norm(actions_D - a_t, dim=1)     # (M,)  # distances to each point in D
                min_dist, idx = torch.min(dists, dim=0)
                nearest_action_key = list(D.keys())[idx]

                print(f"min_dist: {min_dist}; idx: {idx}; dists: {dists}; nearest_action_key: {nearest_action_key}")

                # Check novelty criterion: ||a_t - a_j||₂ ≤ h
                if min_dist <= self.h:
                    # Update existing kernel weight: u_j ← φ·u_j + δ_t·r_t
                    D[nearest_action_key] = (self.phi * D[nearest_action_key] + self.delta * r_t)
                else:
                    # Add new support point with kernel weight: (a_t, δ_t·r_t) to D
                    D[self._tensor_to_key(a_t)] = self.delta * r_t
                    self.stats['new_supports'] += 1

                # Maintain exactly actions_seq_length support points by keeping only the most recent ones
                if len(D) > self.actions_seq_length:
                    # Remove the oldest support point
                    D.popitem(last=False)  # Remove first item (oldest)
            
            print(f"Dictionary size after update: {len(D)}")
            print(f"Dictionary: {D}")

            # Update sigma periodically
            if self.step % self.monitor_interval == 0:
                self._update_sigma()
                self.stats['support_size'].append(len(D))
        print(f"\n**********len(Ds): {len(self.Ds)} \nDs: {self.Ds}**********")

    def get_attraction(self, actions=None):
        """
        Get attraction values for action tokens
        Args:
            actions: (B, actions_seq_length, d) tensor of actions to compute attraction for
        Returns: (B, H, L, 1) tensor of attraction values where:
        - B is batch size
        - H is number of attention heads
        - L is total sequence length (3 * actions_seq_length)
        - Values are consistent across heads for each sequence
        - Only action tokens have non-zero attraction values
        """
        if len(self.Ds[0]) == 0:
            return torch.zeros((self.batch_size, self.num_heads, self.tuple_seq_length, 1), device=self.device)
            
        # Get all actions from the batch
        batch_actions = actions  # (B, actions_seq_length, d)
        
        print(f"\nEWAKernel Get Attraction:")
        print(f"batch_actions shape: {batch_actions.shape}")
        
        # Extract all action tokens from all sequences in the batch
        # Reshape to (B*actions_seq_length, d) to compute attractions for all actions at once
        action_tokens = batch_actions  # (B, actions_seq_length, d)
        B, actions_seq_length, d = action_tokens.shape
        action_tokens_flat = action_tokens.reshape(-1, d)  # (B*actions_seq_length, d)
        
        # Compute attraction values for all actions at once using _compute_A
        A_flat = self._compute_A(action_tokens_flat)  # (B*actions_seq_length,)
            
        # Reshape back to (B, actions_seq_length)
        A_reshaped = A_flat.reshape(B, actions_seq_length)
        
        # Initialize full batch attraction tensor with zeros
        attraction_matrix = torch.zeros((self.batch_size, self.num_heads, self.tuple_seq_length, 1), device=self.device)
        
        # attr shape: (batch_size, num_heads, seq_len, 1)
        # Place attraction values at the correct indices for all sequences
        attraction_matrix[:, :, self.action_indices, 0] = A_reshaped.unsqueeze(1).expand(-1, self.num_heads, -1)
        
        print(f"Final attraction_matrix shape: {attraction_matrix.shape}")
        print(f"Non-zero elements in attraction_matrix: {(attraction_matrix != 0).sum().item()}")
        print(f"attraction_matrix min/max/mean: {attraction_matrix.min().item():.4f}/{attraction_matrix.max().item():.4f}/{attraction_matrix.mean().item():.4f}")
        print(f"attraction_matrix: {attraction_matrix}")
        
        # Track attraction values in history
        mean_attraction = attraction_matrix.mean().item()
        self.stats['attraction_history'].append(mean_attraction)
            
        return attraction_matrix
