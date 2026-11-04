import math

import torch
import torch.nn as nn

KVCache = tuple[torch.Tensor, torch.Tensor]


class CachedCausalSelfAttention(nn.Module):
    """Multi-head causal attention with an optional per-layer KV cache."""

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

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.reshape(
            batch, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """Run attention for new tokens and optionally extend the cache.

        Args:
            x: New hidden states with shape [batch, query_len, dim].
            past_kv: Cached keys and values, each shaped
                [batch, heads, past_len, head_dim].
            use_cache: Whether to return the full updated K/V tensors.

        Returns:
            Attention output shaped [batch, query_len, dim] and either the
            updated cache or None.
        """
        batch, query_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        new_k = self._split_heads(self.k_proj(x))
        new_v = self._split_heads(self.v_proj(x))

        # TODO 1: If past_kv is present, concatenate past and new K/V along
        # the sequence dimension (dim=2). Otherwise use new_k and new_v.
        # Track past_len; it is zero during prefill and nonzero during decode.

        # TODO 2: Compute scaled scores with shape
        # [batch, heads, query_len, past_len + query_len].
        # Each query must attend to cached keys as well as allowed new keys.

        # TODO 3: Build a causal mask from absolute positions:
        # query positions: past_len ... past_len + query_len - 1
        # key positions:   0 ... past_len + query_len - 1
        # Mask locations where key_position > query_position.

        # TODO 4: Apply softmax, multiply by the full value tensor, merge the
        # heads, and apply out_proj.

        # TODO 5: Return (full_k, full_v) when use_cache is True, otherwise
        # return None as the second tuple element.
        raise NotImplementedError("Implement cached causal self-attention")
