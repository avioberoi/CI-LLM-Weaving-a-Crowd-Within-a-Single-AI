import torch
import torch.nn as nn
from torch.distributions import Dirichlet

class DirichletAggregator(nn.Module):
    """
    Aggregator that combines K agent outputs with Dirichlet-weighted averaging.
    Produces a weighted average of agent probability distributions.
    The concentration parameter alpha is fixed (non-learnable in this version).
    """
    def __init__(self, num_agents: int = None, alpha_val: float = None, input_is_logits: bool = True, alpha: float = None):
        super().__init__()
        # Handle alias `alpha`
        if alpha_val is None and alpha is not None:
            alpha_val = alpha
        # Default alpha_val
        alpha_val = alpha_val or 1.0
        if alpha_val <= 0:
            raise ValueError("Dirichlet concentration parameter alpha_val must be positive.")
        # num_agents may be set later
        self.num_agents = num_agents
        self.alpha_val = alpha_val
        self.input_is_logits = input_is_logits

    def forward(self, agent_outputs: list[torch.Tensor], sample_weights: bool = False) -> torch.Tensor:
        """
        Combine agent outputs using Dirichlet-weighted averaging."""
        if not agent_outputs:
            raise ValueError("agent_outputs list cannot be empty.")
        # Infer num_agents if not provided
        if self.num_agents is None:
            self.num_agents = len(agent_outputs)
        if len(agent_outputs) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} agent outputs, but got {len(agent_outputs)}.")


        example_tensor = agent_outputs[0]
        batch_size = example_tensor.size(0)
        device = example_tensor.device
        dtype = example_tensor.dtype

        try:
            outputs = torch.stack(agent_outputs, dim=1) 
        except Exception as e:

            print(f"Error stacking agent_outputs: {e}")
            for i, t in enumerate(agent_outputs):
                print(f"Output {i} shape: {t.shape}, device: {t.device}, dtype: {t.dtype}")
            raise

        if self.input_is_logits:
            outputs = outputs.softmax(dim=2)  
        else:
            with torch.no_grad(): 
                prob_sums = outputs.sum(dim=2)
                if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-3):
                    pass


        if sample_weights:
            concentration_params = torch.full((self.num_agents,), self.alpha_val, device=device, dtype=dtype)
            dirichlet_dist = Dirichlet(concentration_params)
            weight_vectors = dirichlet_dist.rsample((batch_size,))
        else:
            weight_vectors = torch.full((batch_size, self.num_agents), 1.0/self.num_agents,
                                        device=device, dtype=dtype)
        
        weight_vectors = weight_vectors.unsqueeze(-1) 
        
        combined = (outputs * weight_vectors).sum(dim=1) 
        
        return combined