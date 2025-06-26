import torch 
import copy
import numpy as np
import math
class EWA:
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
        self.delta_value = initial_delta
        self.delta_decay_ratio = 1.0

        self.step = 0  # Current step counter
        
        # Initialize attraction values as None
        self.attraction_values = None
        self.action_indices = None
        
        # Track delta values, decay ratio, and attraction values over time
        self.delta_history = []
        self.delta_decay_ratio_history = []
        self.step_history = []
        self.attraction_history = []


    def get_history(self):
        """
        Get the history of delta values, decay ratios, attraction values, and corresponding steps.
        Returns:
            tuple: (steps, delta_values, decay_ratios, attraction_values) as numpy arrays
        """
        # Convert lists to numpy arrays
        steps = np.array(self.step_history)
        attraction_values = np.array(self.attraction_history)
        
        return steps, attraction_values

    def update(self, actions, rewards):
        """
        Update attraction values for action tokens.
        
        :param rewards: Tensor of shape [batch_size, num_action_tokens, 1].
        """
        batch_size, num_action_tokens, _ = rewards.shape
        actions = actions.detach().to(self.device)
        rewards = rewards.detach().to(self.device)

        # Initialize attraction values if not already done
        if self.attraction_values is None:
            self.attraction_values = torch.zeros(
                (batch_size, self.num_heads, self.seq_length, 1), device=self.device
            )

        # Define action token indices
        if num_action_tokens == 1:
            action_indices = torch.tensor([2], device=self.device)
        else:
            action_indices = torch.arange(2, min(self.seq_length, 3 * num_action_tokens), step=3, device=self.device)
        
        # Use rewards directly without normalization
        rewards_ewa = rewards 

        # Expand rewards to match attention heads
        rewards_ewa = rewards_ewa.unsqueeze(1)
        rewards_ewa = rewards_ewa.expand(-1, self.num_heads, -1, -1)

        # # Get current decayed delta and decay ratio
        # current_delta, current_delta_decay_ratio = self.compute_delta_and_decay_ratio()

        a_ind_prev = 2
        # Only update attraction values at action indices, keeping other indices unchanged
        for i, a_ind in enumerate(action_indices):
            self.attraction_values[:, :, a_ind, :] = (
                self.phi * self.attraction_values[:, :, a_ind_prev, :] + 
                self.initial_delta * rewards_ewa[:, :, i, :]
            )
            # print(f"attraction_values at previous action index {a_ind_prev} \n {self.attraction_values[:, :, a_ind_prev, :]}")
            a_ind_prev = a_ind
            
            # print(f"phi: {self.phi}; initial_delta: {self.initial_delta}")
            # print(f"rewards_ewa at action index {a_ind}: {rewards_ewa[:, :, i, :]}")
            # print(f"attraction_values at action index {a_ind} with shape {self.attraction_values[:, :, a_ind, :].shape} \n {self.attraction_values[:, :, a_ind, :]}")
            # print("=" * 50)
        
        # Record attraction values in history
        # Convert to numpy and take mean across batch and heads for each action token
        attraction_np = self.attraction_values.detach().cpu().numpy()
        # Calculate mean attraction value across all dimensions (batch, heads, action tokens)
        # Extract only the attraction values at action indices and take mean
        action_attractions = attraction_np[:, :, action_indices.cpu().numpy(), :]
        mean_attraction = np.mean(action_attractions)
        self.attraction_history.append(mean_attraction)
        self.action_indices = action_indices
        
        # Increment step counter after all updates are complete
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
        attraction_matrix = torch.zeros_like(self.attraction_values, device=self.device)
        
        # Use raw attraction values without sigmoid transformation
        attraction_matrix[:, :, self.action_indices, :] = self.attraction_values[:, :, self.action_indices, :]
        
        return attraction_matrix
