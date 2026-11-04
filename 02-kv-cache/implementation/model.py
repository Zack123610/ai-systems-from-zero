from dataclasses import dataclass

import torch
import torch.nn as nn

from attention import KVCache, CachedCausalSelfAttention
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
        self.attn = CachedCausalSelfAttention(config.dim, config.num_heads)
        self.ffn_norm = RMSNorm(config.dim)
        self.mlp = GatedMLP(config.dim, config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_output, new_kv = self.attn(
            self.attn_norm(x),
            past_kv=past_kv,
            use_cache=use_cache,
        )
        x = x + attn_output
        x = x + self.mlp(self.ffn_norm(x))
        return x, new_kv


class TinyCachedDecoderLM(nn.Module):
    """Tiny decoder-only model whose layers can return K/V caches."""

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

    def forward(
        self,
        token_ids: torch.Tensor,
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[KVCache] | None]:
        """Process prompt or decode tokens.

        Args:
            token_ids: New token IDs shaped [batch, seq_len].
            past_key_values: One K/V pair per Transformer layer.
            use_cache: Whether to return updated layer caches.

        Returns:
            Logits shaped [batch, seq_len, vocab_size] and updated caches.
        """
        _, seq_len = token_ids.shape

        # TODO 1: Validate that past_key_values has one entry per layer.

        # TODO 2: Find past_len from a cached key tensor. It is zero when no
        # cache is supplied. Reject sequences longer than max_seq_len after
        # including both past and new tokens.

        # TODO 3: Position IDs must begin at past_len, not zero, during decode.
        # Add token and position embeddings to create x.

        # TODO 4: Pass the matching cache entry into each block and collect
        # each updated cache when use_cache is True.

        # TODO 5: Apply final_norm and lm_head, then return logits and either
        # the collected caches or None.
        raise NotImplementedError("Implement cache propagation in model.forward")
