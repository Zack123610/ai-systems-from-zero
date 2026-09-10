import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    TODO:
    1. Compute the mean square over the last dimension.
    2. Normalize x using rsqrt(mean_square + eps).
    3. Multiply by the learnable weight.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement RMSNorm.forward")


class GatedMLP(nn.Module):
    """A small gated feed-forward layer similar to modern decoder-only LLMs."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement a SwiGLU-style gated MLP:
        # activation(gate_proj(x)) * up_proj(x), then down_proj(...)
        raise NotImplementedError("Implement GatedMLP.forward")
