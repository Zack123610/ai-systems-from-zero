from dataclasses import dataclass

import torch
import torch.nn as nn

from attention import CausalSelfAttention
from layers import GatedMLP, RMSNorm


@dataclass
class ModelConfig:
    vocab_size: int = 256
    dim: int = 128
    hidden_dim: int = 384
    num_layers: int = 4
    num_heads: int = 4
    max_seq_len: int = 4096


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        self.attn = CausalSelfAttention(config.dim, config.num_heads)
        self.ffn_norm = RMSNorm(config.dim)
        self.mlp = GatedMLP(config.dim, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Add the two pre-norm residual paths.
        # Hint:
        # x = x + attention(norm(x))
        # x = x + mlp(norm(x))
        raise NotImplementedError("Implement TransformerBlock.forward")


class TinyDecoderLM(nn.Module):
    """A deliberately small decoder-only language model for systems experiments.

    The weights are random in Project 01. That is fine: the goal is inference
    mechanics and performance, not language quality.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Args:
            token_ids: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = token_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("Sequence exceeds max_seq_len")

        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.lm_head(x)
