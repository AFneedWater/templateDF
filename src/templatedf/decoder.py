"""Non-autoregressive sequence decoder conditioned only on latent and length."""

import math

import torch
from torch import nn

from .blocks import zero_padding
from .data import AA_ORDER


def validate_output_lengths(output_lengths: torch.Tensor, batch_size: int,
                            device: torch.device, *, minimum: int = 1,
                            maximum: int | None = None) -> torch.Tensor:
    """Reject fractional/bool lengths; explicitly place integer lengths on device."""
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if not isinstance(output_lengths, torch.Tensor):
        raise ValueError("output_lengths must be an integer tensor [B]")
    if output_lengths.ndim != 1 or output_lengths.shape[0] != batch_size:
        raise ValueError("output_lengths must have shape [B], matching latent/input batch")
    if output_lengths.dtype not in integer_dtypes:
        raise ValueError("output_lengths must contain integers, not floats or booleans")
    if (output_lengths < minimum).any() or (
        maximum is not None and (output_lengths > maximum).any()
    ):
        upper = str(maximum) if maximum is not None else "unbounded"
        raise ValueError(f"output_lengths must be in [{minimum}, {upper}]")
    return output_lengths.to(device=device, dtype=torch.long)


def sinusoidal_position_encoding(length: int, dim: int, *, device: torch.device,
                                 dtype: torch.dtype) -> torch.Tensor:
    """Dynamic absolute positions; values never depend on the batch max length."""
    calculation_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    positions = torch.arange(length, device=device, dtype=calculation_dtype)
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=calculation_dtype)
        * (-math.log(10000.0) / dim)
    )
    angles = positions[:, None] * frequencies[None, :]
    return torch.stack((angles.sin(), angles.cos()), dim=-1).reshape(length, dim).to(dtype=dtype)


class TransformerDecoderBlock(nn.Module):
    """Pre-norm self-attention, latent cross-attention, FFN; zero padded queries."""

    def __init__(self, dim: int, num_heads: int, ffn_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_ratio * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_ratio * dim, dim), nn.Dropout(dropout),
        )
        self.self_dropout = nn.Dropout(dropout)
        self.cross_dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                query_valid_mask: torch.Tensor) -> torch.Tensor:
        h = zero_padding(query, query_valid_mask)
        normalized = self.self_norm(h)
        update, _ = self.self_attention(
            normalized, normalized, normalized,
            key_padding_mask=~query_valid_mask, need_weights=False,
        )
        h = zero_padding(h + self.self_dropout(update), query_valid_mask)
        latent_memory = self.memory_norm(memory)
        update, _ = self.cross_attention(
            self.cross_norm(h), latent_memory, latent_memory, need_weights=False,
        )  # All N latent tokens are valid; no causal mask or target tokens.
        h = zero_padding(h + self.cross_dropout(update), query_valid_mask)
        return zero_padding(h + self.ffn(self.ffn_norm(h)), query_valid_mask)


class SequenceDecoder(nn.Module):
    """Basic tensor decoder: positive output lengths; peptide bounds live in AE."""

    aa_order = AA_ORDER

    def __init__(self, dim: int = 1280, num_blocks: int = 4, num_heads: int = 20,
                 ffn_ratio: int = 4, dropout: float = 0.1, length_scale: float = 150):
        super().__init__()
        if type(dim) is not int or dim < 2 or dim % 2:
            raise ValueError("dim must be a positive even integer")
        if type(num_heads) is not int or num_heads < 1 or dim % num_heads:
            raise ValueError("num_heads must divide dim")
        for name, value in (("num_blocks", num_blocks), ("ffn_ratio", ffn_ratio)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if isinstance(length_scale, bool) or not math.isfinite(length_scale) or length_scale <= 0:
            raise ValueError("length_scale must be finite and positive")
        self.dim = dim
        self.length_scale = float(length_scale)
        self.length_encoder = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(dim, num_heads, ffn_ratio, dropout) for _ in range(num_blocks)
        ])
        self.embedding_head = nn.Linear(dim, dim)
        self.amino_acid_head = nn.Linear(dim, len(AA_ORDER))

    def forward(self, latent: torch.Tensor, output_lengths: torch.Tensor) -> dict:
        if (not isinstance(latent, torch.Tensor) or latent.ndim != 3
                or latent.shape[-1] != self.dim or not latent.is_floating_point()):
            raise ValueError(f"latent must be floating [B,N,{self.dim}]")
        if latent.shape[0] < 1 or latent.shape[1] < 1:
            raise ValueError("Latent batch and token dimensions must be nonempty")
        if not torch.isfinite(latent).all():
            raise ValueError("All latent values must be finite")
        lengths = validate_output_lengths(output_lengths, latent.shape[0], latent.device)
        max_length = int(lengths.max().item())
        output_mask = torch.arange(max_length, device=latent.device)[None, :] < lengths[:, None]
        positions = sinusoidal_position_encoding(
            max_length, self.dim, device=latent.device, dtype=latent.dtype,
        )
        calculation_dtype = torch.float64 if latent.dtype == torch.float64 else torch.float32
        normalized_lengths = lengths.to(calculation_dtype).log1p() / math.log1p(self.length_scale)
        length_features = self.length_encoder(normalized_lengths.to(latent.dtype)[:, None])
        h = zero_padding(positions[None, :, :] + length_features[:, None, :], output_mask)
        for block in self.blocks:
            h = block(h, latent, query_valid_mask=output_mask)
        reconstructed = zero_padding(self.embedding_head(h), output_mask)
        logits = zero_padding(self.amino_acid_head(reconstructed), output_mask)
        return {"reconstructed_embeddings": reconstructed, "logits": logits,
                "output_mask": output_mask, "output_lengths": lengths}
