"""Residual cascade, per-residue multi-level fusion, learned latent queries."""

import torch
from torch import nn

from .blocks import RaygunBlock, validate_block_config, validate_masked_input, zero_padding


class QueryPool(nn.Module):
    """Pre-norm cross-attention + residual + pre-norm FFN + residual."""

    def __init__(self, dim: int, num_heads: int, ffn_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_ratio * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_ratio * dim, dim), nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                memory_valid_mask: torch.Tensor) -> torch.Tensor:
        # In explicit half/double models these already match; autocast fusion may
        # return BF16 while learned queries remain FP32. Casting preserves grads.
        query = query.to(dtype=memory.dtype)
        normalized_memory = self.memory_norm(memory)
        update, _ = self.attention(
            self.query_norm(query), normalized_memory, normalized_memory,
            key_padding_mask=~memory_valid_mask, need_weights=False,
        )
        query = query + self.attention_dropout(update)
        return query + self.ffn(self.ffn_norm(query))


class CascadeEncoder(nn.Module):
    def __init__(self, dim: int = 1280, num_latents: int = 50, num_blocks: int = 4,
                 num_heads: int = 20, conv_kernel: int = 7,
                 fusion_hidden_dim: int | None = None, pool_ffn_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        validate_block_config(dim, num_heads, conv_kernel)
        for name, value in (("num_latents", num_latents), ("num_blocks", num_blocks),
                            ("pool_ffn_ratio", pool_ffn_ratio)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if fusion_hidden_dim is None:
            fusion_hidden_dim = 2 * dim
        if type(fusion_hidden_dim) is not int or fusion_hidden_dim < 1:
            raise ValueError("fusion_hidden_dim must be a positive integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.dim, self.num_latents = dim, num_latents
        self.blocks = nn.ModuleList([
            RaygunBlock(dim, num_heads, conv_kernel, dropout) for _ in range(num_blocks)
        ])
        self.fusion = nn.Sequential(
            nn.Linear((num_blocks + 1) * dim, fusion_hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(fusion_hidden_dim, dim),
        )
        self.latent_queries = nn.Parameter(torch.empty(num_latents, dim))
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)
        self.pool = QueryPool(dim, num_heads, pool_ffn_ratio, dropout)

    def forward(self, embeddings: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        validate_masked_input(embeddings, input_mask, self.dim)
        h = zero_padding(embeddings, input_mask)
        levels = [h]
        for block in self.blocks:
            h = zero_padding(h + block(h, mask=input_mask), input_mask)
            levels.append(h)
        memory = zero_padding(self.fusion(torch.cat(levels, dim=-1)), input_mask)
        queries = self.latent_queries.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
        return self.pool(queries, memory, memory_valid_mask=input_mask)
