import torch 
import copy
import numpy as np
import math
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
        self.delta_value = initial_delta
        self.delta_decay_ratio = 1.0

        # Estimate how many times a state-action pair is revisited
        # Formula: (total updates × batch size) / (buffer size × trajectory length)
        self.n_visits = (max_online_iters * num_updates_per_online_iter * batch_size) // (replay_size * trajectory_length)

        # Compute decay step: when delta starts to decay
        self.decay_iter = math.ceil(max_online_iters / self.n_visits) if self.n_visits > 1 else max_online_iters

        self.step = 0  # Current step counter
        
        # Initialize attraction values as None
        self.attraction_values = None
        self.action_indices = None
        
        # Track delta values, decay ratio, and attraction values over time
        self.delta_history = []
        self.delta_decay_ratio_history = []
        self.step_history = []
        self.attraction_history = []


    def compute_delta_and_decay_ratio(self, step=None, record_history=True):
        """
        Compute delta and decay ratio for a given step.
        
        Args:
            step: Step number to compute for. If None, uses current step.
            record_history: Whether to record the values in history.
            
        Returns:
            tuple: (delta_value, decay_ratio)
        """
        if step is None:
            step = self.step
            
        self.delta_decay_ratio = 1 - (step / self.decay_iter)
        self.delta_value = self.initial_delta * (1 - self.delta_decay_ratio) + self.final_delta * self.delta_decay_ratio
        
        if record_history:
            self.delta_history.append(self.delta_value)
            self.delta_decay_ratio_history.append(self.delta_decay_ratio)
            self.step_history.append(step)
            
        return self.delta_value, self.delta_decay_ratio

    def get_history(self):
        """
        Get the history of delta values, decay ratios, attraction values, and corresponding steps.
        Returns:
            tuple: (steps, delta_values, decay_ratios, attraction_values) as numpy arrays
        """
        # Convert lists to numpy arrays
        steps = np.array(self.step_history)
        delta_values = np.array(self.delta_history)
        delta_decay_ratios = np.array(self.delta_decay_ratio_history)
        attraction_values = np.array(self.attraction_history)
        
        return steps, delta_values, delta_decay_ratios, attraction_values

    def update(self, rewards):
        """
        Update attraction values for action tokens.
        
        :param rewards: Tensor of shape [batch_size, num_action_tokens, 1].
        """
        batch_size, num_action_tokens, _ = rewards.shape

        # Initialize attraction values if not already done
        if self.attraction_values is None:
            self.attraction_values = torch.ones(
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

        # Get current decayed delta and decay ratio
        current_delta, current_delta_decay_ratio = self.compute_delta_and_decay_ratio()

        a_ind_prev = 2
        # Update attraction values at action indices
        for i, a_ind in enumerate(action_indices):
            self.attraction_values[:, :, a_ind, :] = (
                self.phi * self.attraction_values[:, :, a_ind_prev, :] + 
                current_delta * rewards_ewa[:, :, i, :]
            )
            a_ind_prev = a_ind
            
        # Record attraction values in history
        # Convert to numpy and take mean across batch and heads for each action token
        attraction_np = self.attraction_values.detach().cpu().numpy()
         attraction_values = np.array(self.attraction_history)

        self.action_indices = action_indices
        
        # Increment step counter after all updates are complete
        self.step += 1
        # print(f"attraction_values: {self.attraction_values}")
        print(f"current_delta: {current_delta}")
        print(f"current_delta_decay_ratio: {current_delta_decay_ratio}")
        # print(f"action_indices: {action_indices}")

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
