import torch 
import copy
import numpy as np

class EWADecay:
    def __init__(self, num_heads, seq_length, phi=0.9, initial_delta=0.7, final_delta=0.1, 
                 max_online_iters=100, num_updates_per_online_iter=300, 
                 batch_size=256, replay_size=1000, trajectory_length=1000):
        """
        Initialize EWADecay.
        
        Args:
            num_heads: Number of attention heads
            seq_length: Length of the sequence
            phi: Forgetting factor (memory decay)
            initial_delta: Initial learning rate
            final_delta: Final learning rate
            max_online_iters: Maximum number of online iterations
            num_updates_per_online_iter: Number of updates per online iteration
            batch_size: Batch size
            replay_size: Size of replay buffer
            trajectory_length: Average length of trajectories (defaults to 1000 for MuJoCo environments)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.phi = phi
        self.initial_delta = initial_delta
        self.final_delta = final_delta

        # Estimate how many times a state-action pair is revisited
        # Formula: (total updates × batch size) / (buffer size × trajectory length)
        self.n_visits = (max_online_iters * num_updates_per_online_iter * batch_size) // (replay_size * trajectory_length)

        # Compute decay step: when to decay δ
        self.decay_step = max_online_iters / self.n_visits if self.n_visits > 0 else max_online_iters

        self.step = 0  # Current step counter
        
        # Initialize attraction values as None
        self.attraction_values = None
        self.action_indices = None
        

    def get_delta(self):
        """
        Compute delta based on current step (decays when revisits occur).
        Returns:
            float: Current delta value between initial_delta and final_delta
        """
        decay_ratio = min(1.0, self.step / self.decay_step)
        return self.initial_delta * (1 - decay_ratio) + self.final_delta * decay_ratio

    def update(self, rewards):
        """
        Update attraction values for action tokens.
        
        :param rewards: Tensor of shape [batch_size, num_action_tokens, 1].
        """
        batch_size, num_action_tokens, _ = rewards.shape

        self.attraction_values = torch.ones(
            (batch_size, self.num_heads, self.seq_length, 1), device=self.device
        )

        # Define action token indices
        if num_action_tokens == 1:
            action_indices = torch.tensor([2], device=self.device)
        else:
            action_indices = torch.arange(2, min(self.seq_length, 3 * num_action_tokens), step=3, device=self.device)
        
        # # Normalize rewards to [-1, 1] range within each batch
        # rewards_ewa = copy.deepcopy(rewards)
        # batch_max = rewards_ewa.max(dim=1, keepdim=True)[0]
        # batch_min = rewards_ewa.min(dim=1, keepdim=True)[0]
        # rewards_ewa = 2.0 * (rewards_ewa - batch_min) / (batch_max - batch_min + 1e-6) - 1.0
        
        # Use rewards directly without normalization
        rewards_ewa = rewards 

        # Expand rewards to match attention heads
        rewards_ewa = rewards_ewa.unsqueeze(1)
        rewards_ewa = rewards_ewa.expand(-1, self.num_heads, -1, -1)

        # Get current decayed delta
        current_delta = self.get_delta()

        # Update attraction values at action indices
        for i, a_ind in enumerate(action_indices):
            self.attraction_values[:, :, a_ind, :] = (
                self.phi * self.attraction_values[:, :, a_ind, :] + 
                current_delta * rewards_ewa[:, :, i, :]
            )

        self.action_indices = action_indices
        self.step += 1

    def get_attraction(self):
        """
        Get the attraction values for all tokens in the sequence.
        Returns:
            Tensor of shape [batch_size, num_heads, seq_length, 1] with raw attraction values.
        """
        if self.attraction_values is None:
            raise ValueError("Attraction values not initialized! Call update() first.")

        # Create base matrix of ones
        attraction_matrix = torch.ones_like(self.attraction_values, device=self.device)
        
        # Use raw attraction values without sigmoid transformation
        attraction_matrix[:, :, self.action_indices, :] = self.attraction_values[:, :, self.action_indices, :]
        
        return attraction_matrix
