import torch 
import copy
# old version
class EWA:
    def __init__(self, num_heads, seq_length, phi=0.9, delta=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.phi = phi  # Forgetting factor (memory decay)
        self.delta = delta  # Learning rate

        # # Attraction values are always fixed size: [256, 4, 60, 1]

    def update(self, rewards):
        """
        Update attraction values for action tokens.
        
        :param rewards: Tensor of shape [batch_size, num_action_tokens, 1].
        """
        batch_size, num_action_tokens, _ = rewards.shape  # Get number of action tokens
    
        self.attraction_values = torch.ones(
            (batch_size, self.num_heads, self.seq_length, 1), device=self.device
        )  # Only num_action_tokens, not full seq_length!


        # Define action token indices
        if num_action_tokens == 1:
            action_indices = torch.tensor([2], device=self.device)  # Only one action token at position 2
        else:
            # Ensure action indices stay within 60
            action_indices = torch.arange(2, min(self.seq_length, 3 * num_action_tokens), step=3, device=self.device)
            # Action tokens at 2, 5, 8, ..., up to num_action_tokens

        # Ensure rewards have the correct shape
        # rewards_ewa = torch.squeeze(rewards, -1)  # Shape: [batch_size, num_action_tokens]
        rewards_ewa = copy.deepcopy(rewards)
        rewards_ewa = rewards_ewa.unsqueeze(1)  # Expand to match attention heads: [batch_size, 1, num_action_tokens]
        rewards_ewa = rewards_ewa.expand(-1, self.num_heads, -1, -1)  # Expand to match heads: [batch_size, 4, num_action_tokens]

        # Only update attraction values **at action indices**, keep the rest as 1
        for i, a_ind in enumerate(action_indices):
            self.attraction_values[:, :, a_ind, :] = (
                self.phi * self.attraction_values[:, :, a_ind, :] + self.delta * rewards_ewa[:, :, i, :]
            )

        # Store action indices for use in attention computation
        self.action_indices = action_indices

    def get_attraction(self):
        """
        Get the attraction values for all tokens in the sequence.
        Returns:
            Tensor of shape [256, num_heads, seq_length, 1].
        """
        if self.attraction_values is None:
            raise ValueError("Attraction values not initialized! Call update() first.")

        # Copy attraction values to avoid modifying original tensor
        attraction_matrix = torch.ones_like(self.attraction_values, device=self.device)
        attraction_matrix[:, :, self.action_indices, :] = self.attraction_values[:, :, self.action_indices, :]
        return attraction_matrix
