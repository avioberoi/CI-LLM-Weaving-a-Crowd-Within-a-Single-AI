import torch
import torch.nn as nn
import torch.nn.functional as F


def _sliced_wasserstein_distance_squared(X: torch.Tensor, Y: torch.Tensor, num_projections: int = 50) -> torch.Tensor:
    """
    Compute **squared** sliced 2-Wasserstein distance between two sets of D-dimensional vectors X and Y.
    X and Y are tensors of shape (batch_size, dim_features).
    The number of samples (batch_size) in X and Y MUST be the same.

    Methodology Reference: The text refers to 2-Wasserstein distance. SWD often approximates W_2.
                         The loss function aims to maximize this distance.
                         The squared distance is often used in loss functions for convenience.

    Parameters:
    - X (torch.Tensor): First set of samples, shape (batch_size, dim_features).
    - Y (torch.Tensor): Second set of samples, shape (batch_size, dim_features).
    - num_projections (int): Number of random 1D projections to use.

    Returns:
    - torch.Tensor: A scalar tensor representing the squared Sliced Wasserstein-2 distance.
    """
    if X.size(0) != Y.size(0):
        raise ValueError(f"X and Y must have the same number of samples (batch_size). "
                         f"Got X: {X.size(0)} and Y: {Y.size(0)}")
    if X.size(1) != Y.size(1):
        raise ValueError(f"X and Y must have the same feature dimension. "
                         f"Got X: {X.size(1)} and Y: {Y.size(1)}")
    if X.size(0) == 0: # Handle empty batch case
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)


    batch_size, dim_features = X.size()
    device = X.device
    dtype = X.dtype

    # Sample random projection directions (L2-normalized)
    # Shape: (dim_features, num_projections) for efficient matmul
    projections = torch.randn(dim_features, num_projections, device=device, dtype=dtype)
    projections = F.normalize(projections, p=2, dim=0) # Normalize each column (direction vector)

    # Project points X and Y onto these random directions
    # X @ projections -> (batch_size, dim_features) @ (dim_features, num_projections) -> (batch_size, num_projections)
    projected_X = X @ projections
    projected_Y = Y @ projections

    # Sort the projected values along the sample dimension (batch_size) for each projection
    # Result shape: (batch_size, num_projections)
    projected_X_sorted, _ = torch.sort(projected_X, dim=0)
    projected_Y_sorted, _ = torch.sort(projected_Y, dim=0)

    # Compute the squared L2 distance between sorted projected samples for each projection
    # (projected_X_sorted - projected_Y_sorted)^2 -> shape (batch_size, num_projections)
    # Then mean over samples (batch_size) to get W_2^2 for each projection
    # This is the squared 1D Wasserstein-2 distance for each projection
    squared_distances_per_projection = torch.sum((projected_X_sorted - projected_Y_sorted)**2, dim=0) / batch_size
    
    # Average over all projections to get the final Sliced Wasserstein-2 distance (squared)
    sw_dist_sq = squared_distances_per_projection.mean()
    
    return sw_dist_sq

def pairwise_sliced_wasserstein_distance_squared(
        agent_representations: list[torch.Tensor], 
        num_projections: int = 50
    ) -> torch.Tensor:
    """
    Compute the average pairwise squared Sliced Wasserstein-2 distance among a list of agent representations.
    This serves as the diversity measure R_SW. Higher means more diverse.

    Parameters:
    - agent_representations (list[torch.Tensor]): List of K tensors [Rep1, Rep2, ..., RepK].
                                                 Each tensor Rep_k has shape (batch_size, dim_features).
                                                 All tensors must have the same batch_size and dim_features.
    - num_projections (int): Number of random 1D projections to use for SWD.

    Returns:
    - torch.Tensor: A scalar tensor, R_SW, representing the average pairwise diversity.
    """
    num_agents = len(agent_representations)
    if num_agents < 2:
        # No diversity to measure if less than 2 agents
        if agent_representations: # if list is not empty but has 1 agent
            return torch.tensor(0.0, device=agent_representations[0].device, dtype=agent_representations[0].dtype)
        else: # if list is empty
            return torch.tensor(0.0)


    total_pairwise_swd_sq = 0.0
    num_pairs = 0

    # Check consistency of shapes and device for all representations
    first_rep = agent_representations[0]
    batch_size, dim_features = first_rep.shape
    device = first_rep.device
    dtype = first_rep.dtype

    for i in range(num_agents):
        if agent_representations[i].shape != (batch_size, dim_features):
            raise ValueError(f"All agent representations must have the same shape. "
                             f"Agent 0 shape: {(batch_size, dim_features)}, "
                             f"Agent {i} shape: {agent_representations[i].shape}")
        if agent_representations[i].device != device or agent_representations[i].dtype != dtype:
             raise ValueError(f"All agent representations must be on the same device and have the same dtype. "
                             f"Agent 0 device: {device}, dtype: {dtype}. "
                             f"Agent {i} device: {agent_representations[i].device}, dtype: {agent_representations[i].dtype}")


    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            swd_sq_ij = _sliced_wasserstein_distance_squared(
                agent_representations[i], 
                agent_representations[j], 
                num_projections=num_projections
            )
            total_pairwise_swd_sq += swd_sq_ij
            num_pairs += 1
    
    if num_pairs == 0: # Should only happen if num_agents < 2, already handled.
        return torch.tensor(0.0, device=device, dtype=dtype)
        
    average_pairwise_swd_sq = total_pairwise_swd_sq / num_pairs
    return average_pairwise_swd_sq

# Define nn.Module wrappers for use in the training loop

class SlicedWassersteinDiversityRegularizer(nn.Module):
    """
    Computes the Sliced Wasserstein Diversity (R_SW) to be maximized.
    This will be used as a regularizer in the loss function (e.g., L_task - lambda * R_SW).
    The input is expected to be a list of K agent representations (e.g., hidden states).
    """
    def __init__(self, num_projections: int = 50):
        super().__init__()
        self.num_projections = num_projections

    def forward(self, agent_representations: list[torch.Tensor]) -> torch.Tensor:
        """
        Parameters:
        - agent_representations (list[torch.Tensor]): List of K tensors [Rep1, ..., RepK],
          where each Rep_k is shape (batch_size, feature_dimension).
          These could be last hidden states, specific layer embeddings, or even logits
          if treated as points in a vector space.
        
        Returns:
        - torch.Tensor: Scalar R_SW, the average pairwise squared Sliced Wasserstein-2 distance.
        """
        return pairwise_sliced_wasserstein_distance_squared(
            agent_representations, 
            num_projections=self.num_projections
        )

class SlicedWassersteinLogitsRegularizer(nn.Module):
    """Sliced Wasserstein regularizer applied on agent output logits (next-token logits)"""
    def __init__(self, num_projections: int = 128):
        super().__init__()
        self.num_projections = num_projections

    def forward(self, agent_logits: list[torch.Tensor]) -> torch.Tensor:
        # Each agent_logits[i] is a tensor of shape (batch_size, vocab_size)
        # (Optionally, one could apply softmax here to compare probability distributions, 
        # but we treat logits as representation vectors in output space.)
        return pairwise_sliced_wasserstein_distance_squared(agent_logits, num_projections=self.num_projections)

class SlicedWassersteinHiddenStateRegularizer(nn.Module):
    """Regularizer on final pre-softmax hidden states of each agent"""
    def __init__(self, num_projections: int = 128):
        super().__init__()
        self.num_projections = num_projections

    def forward(self, agent_hidden_states: list[torch.Tensor]) -> torch.Tensor:
        # agent_hidden_states[i] is shape (batch_size, hidden_dim)
        return pairwise_sliced_wasserstein_distance_squared(agent_hidden_states, num_projections=self.num_projections)

class SlicedWassersteinLayerRegularizer(nn.Module):
    """Regularizer on an intermediate transformer layer representation"""
    def __init__(self, layer_index: int, num_projections: int = 128):
        super().__init__()
        self.layer_index = layer_index  # which layer this regularizer is applied to
        self.num_projections = num_projections

    def forward(self, agent_layer_reps: list[torch.Tensor]) -> torch.Tensor:
        # agent_layer_reps[i] is the representation (batch_size, dim) at the specified layer for agent i
        return pairwise_sliced_wasserstein_distance_squared(agent_layer_reps, num_projections=self.num_projections)