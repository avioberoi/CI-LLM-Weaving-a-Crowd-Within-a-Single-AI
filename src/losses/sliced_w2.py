import torch
import torch.nn as nn
import torch.nn.functional as F


def _sliced_wasserstein_distance_squared(X: torch.Tensor, Y: torch.Tensor, num_projections: int = 50) -> torch.Tensor:
    """
    Compute **squared** sliced 2-Wasserstein distance between two sets of D-dimensional vectors X and Y.
    X and Y are tensors of shape (batch_size, dim_features).
    """
    if X.size(0) != Y.size(0):
        raise ValueError(f"X and Y must have the same number of samples (batch_size). "
                         f"Got X: {X.size(0)} and Y: {Y.size(0)}")
    if X.size(1) != Y.size(1):
        raise ValueError(f"X and Y must have the same feature dimension. "
                         f"Got X: {X.size(1)} and Y: {Y.size(1)}")
    if X.size(0) == 0:  
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)


    batch_size, dim_features = X.size()
    device = X.device
    dtype = X.dtype

   
    projections = torch.randn(dim_features, num_projections, device=device, dtype=dtype)
    projections = F.normalize(projections, p=2, dim=0)  

     
    projected_X = X @ projections
    projected_Y = Y @ projections


    projected_X_sorted, _ = torch.sort(projected_X, dim=0)
    projected_Y_sorted, _ = torch.sort(projected_Y, dim=0)

    squared_distances_per_projection = torch.sum((projected_X_sorted - projected_Y_sorted)**2, dim=0) / batch_size
    
    sw_dist_sq = squared_distances_per_projection.mean()
    
    return sw_dist_sq

def pairwise_sliced_wasserstein_distance_squared(
        agent_representations: list[torch.Tensor], 
        num_projections: int = 50
    ) -> torch.Tensor:
    """
    Compute the average pairwise squared Sliced Wasserstein-2 distance among a list of agent representations.
    This serves as the diversity measure R_SW. Higher means more diverse.
    """
    num_agents = len(agent_representations)
    if num_agents < 2:
        if agent_representations:  
            return torch.tensor(0.0, device=agent_representations[0].device, dtype=agent_representations[0].dtype)
        else:  
            return torch.tensor(0.0)


    total_pairwise_swd_sq = 0.0
    num_pairs = 0

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
    
    if num_pairs == 0:  
        return torch.tensor(0.0, device=device, dtype=dtype)
        
    average_pairwise_swd_sq = total_pairwise_swd_sq / num_pairs
    return average_pairwise_swd_sq


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
        return pairwise_sliced_wasserstein_distance_squared(agent_logits, num_projections=self.num_projections)

class SlicedWassersteinHiddenStateRegularizer(nn.Module):
    """Regularizer on final pre-softmax hidden states of each agent"""
    def __init__(self, num_projections: int = 128):
        super().__init__()
        self.num_projections = num_projections

    def forward(self, agent_hidden_states: list[torch.Tensor]) -> torch.Tensor:
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