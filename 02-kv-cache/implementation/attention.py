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

        # 1. Extend cached keys and values.
        if past_kv is None:
            past_len = 0
            full_k = new_k
            full_v = new_v
        else:
            past_k, past_v = past_kv
            past_len = past_k.shape[2]
            full_k = torch.cat((past_k, new_k), dim=2)
            full_v = torch.cat((past_v, new_v), dim=2)

        # 2. Scaled dot-product attention scores.
        scores = q @ full_k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # 3. Causal mask using absolute token positions.
        key_len = past_len + query_len

        query_positions = torch.arange(
            past_len,
            key_len,
            device=x.device,
        )
        key_positions = torch.arange(
            key_len,
            device=x.device,
        )

        casual_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        scores = scores.masked_fill(casual_mask, float("-inf"))

        # 4. Attention output.
        attention_probs = torch.softmax(scores, dim=-1)
        attention_output = attention_probs @ full_v

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(batch, query_len, self.dim)
        output = self.out_proj(attention_output)

        # 5. Return the updated cache when requested.
        updated_cache = (full_k, full_v) if use_cache else None
        return output, updated_cache
