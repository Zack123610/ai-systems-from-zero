import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """Naive multi-head causal self-attention.

    Project 01 intentionally recomputes attention over the full sequence on every
    generation step. Do NOT add a KV cache here.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [batch, seq_len, dim]

        Returns:
            Tensor with shape [batch, seq_len, dim].
        """
        batch, seq_len, _ = x.shape

        # TODO 1: Project x into Q, K, and V.
        # TODO 2: Reshape each tensor into [batch, heads, seq_len, head_dim].
        # TODO 3: Compute scaled attention scores Q @ K^T / sqrt(head_dim).
        # TODO 4: Construct and apply a causal mask so token i cannot see j > i.
        # TODO 5: Apply softmax to obtain attention probabilities.
        # TODO 6: Multiply probabilities by V.
        # TODO 7: Merge heads back to [batch, seq_len, dim] and apply out_proj.
        #
        # Keep this implementation explicit rather than calling PyTorch SDPA;
        # the objective is to see the intermediate tensors yourself.

        raise NotImplementedError("Implement CausalSelfAttention.forward")
