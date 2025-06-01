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
        """
        Parameters:
        - num_agents (int): Number of agent outputs to aggregate.
        - alpha_val (float): The symmetric concentration parameter for the Dirichlet distribution.
                             A value of 1.0 for each component results in a uniform expected weighting.
        - input_is_logits (bool): If True, assumes agent_outputs are logits and applies softmax.
                                  If False, assumes agent_outputs are already probabilities.
        - alpha (float): Alias for alpha_val.
        """
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
        Aggregates agent outputs.

        Parameters:
        - agent_outputs (list[torch.Tensor]): List of K tensors [P1, P2, ..., PK].
                                             Each tensor has shape (batch_size, vocab_size),
                                             representing logits or probabilities.
        - sample_weights (bool): If True, samples weights from the Dirichlet distribution.
                                 If False (default for inference), uses the expected weights
                                 (uniform 1/K for a symmetric Dirichlet with equal alpha_val).
        
        Returns:
        - torch.Tensor: Combined output distribution of shape (batch_size, vocab_size).
        """
        if not agent_outputs:
            raise ValueError("agent_outputs list cannot be empty.")
        # Infer num_agents if not provided
        if self.num_agents is None:
            self.num_agents = len(agent_outputs)
        if len(agent_outputs) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} agent outputs, but got {len(agent_outputs)}.")

        # It's crucial that all tensors in agent_outputs are on the same device.
        # Assuming they are, get device and dtype from the first tensor.
        example_tensor = agent_outputs[0]
        batch_size = example_tensor.size(0)
        device = example_tensor.device
        dtype = example_tensor.dtype

        # Stack outputs for easier computation: shape (batch_size, K, vocab_size)
        try:
            outputs = torch.stack(agent_outputs, dim=1) 
        except Exception as e:
            # This might happen if tensors in the list have inconsistent shapes (other than vocab_size dim)
            # or are on different devices.
            print(f"Error stacking agent_outputs: {e}")
            # You might want to add more detailed shape/device checks here if issues persist
            for i, t in enumerate(agent_outputs):
                print(f"Output {i} shape: {t.shape}, device: {t.device}, dtype: {t.dtype}")
            raise

        if self.input_is_logits:
            outputs = outputs.softmax(dim=2) # Convert logits to probabilities
        else:
            # If inputs are already probabilities, optionally check if they sum to 1 as a sanity check
            with torch.no_grad(): # No need for gradients in this check
                prob_sums = outputs.sum(dim=2)
                if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-3):
                    # print("Warning: input_is_logits is False, but inputs do not appear to be valid probabilities (sum != 1).")
                    # Depending on strictness, could raise error or proceed.
                    # For robustness, if they are not probabilities, applying softmax might be safer anyway,
                    # or ensure the caller always provides correct input type.
                    # If strict, uncomment:
                    # raise ValueError("Inputs specified as probabilities do not sum to 1.")
                    pass


        if sample_weights:
            # Prepare Dirichlet distribution (symmetric with self.alpha_val for each agent)
            concentration_params = torch.full((self.num_agents,), self.alpha_val, device=device, dtype=dtype)
            dirichlet_dist = Dirichlet(concentration_params)
            # Sample K weights for each item in the batch. Shape: [batch_size, K]
            weight_vectors = dirichlet_dist.rsample((batch_size,))
        else:
            # Use expected weights: alpha_i / sum(alphas). For symmetric, this is 1/K.
            weight_vectors = torch.full((batch_size, self.num_agents), 1.0/self.num_agents,
                                        device=device, dtype=dtype)
        
        # Combine agent outputs with these weights
        # outputs shape: (batch_size, K, vocab_size)
        # weight_vectors shape: (batch_size, K) -> unsqueeze to (batch_size, K, 1)
        weight_vectors = weight_vectors.unsqueeze(-1) 
        
        # Perform weighted sum: (P_k * w_k) and sum over K
        combined = (outputs * weight_vectors).sum(dim=1) # Result shape: (batch_size, vocab_size)
        
        return combined