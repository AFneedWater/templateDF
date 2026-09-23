# Adapted from Raygun raygun/modelv2/model_utils.py
# Copyright 2024 Kapil Devkota, Rohit Singh
# Source commit: cd3b3574708a71f6dc134c7719d287b5ab274186
# Original terms: licenses/RAYGUN_LICENSE.txt (CC BY-NC 4.0).
# Changes: explicit validation, dtype-safe right padding, padding zeroing,
# configurable dropout, no length-dependent bias K/V token, and RoPE cache reset.
"""Independent adaptation of the rotary Transformer -> masked convolution Block."""

import torch
from torch import nn
from esm.modules import TransformerLayer


def validate_masked_input(x: torch.Tensor, mask: torch.Tensor, dim: int) -> None:
    if x.ndim != 3 or x.shape[-1] != dim or not x.is_floating_point():
        raise ValueError(f"Expected floating embeddings [B,L,{dim}]")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("Batch and sequence dimensions must be nonempty")
    if mask.shape != x.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Mask must be bool [B,L], True for valid residues")
    if mask.device != x.device:
        raise ValueError("Mask and embeddings must be on the same device")
    if not mask.any(dim=1).all():
        raise ValueError("Each input must have at least one valid residue")
    if (mask[:, 1:] & ~mask[:, :-1]).any():
        raise ValueError("Only right padding is supported")


def zero_padding(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # masked_fill also removes NaN/Inf in invalid positions; multiplication does not.
    return x.masked_fill(~mask.unsqueeze(-1), 0)


def validate_block_config(dim: int, num_heads: int, conv_kernel: int) -> None:
    if type(dim) is not int or dim < 4 or dim % 4:
        raise ValueError("dim must be a positive multiple of 4")
    if type(num_heads) is not int or num_heads < 1 or dim % num_heads:
        raise ValueError("num_heads must divide dim")
    if (dim // num_heads) % 2:
        raise ValueError("Rotary attention requires an even head dimension")
    if type(conv_kernel) is not int or conv_kernel < 2:
        raise ValueError("conv_kernel must be >= 2 (middle kernel is conv_kernel // 2)")


class ConvMasked(nn.Module):
    """Right-padded convolution, preserving Raygun's forward-looking window."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=0)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.masked_fill(~mask[:, None, :], 0)
        padding = x.new_zeros(x.shape[0], x.shape[1], self.kernel_size - 1)
        output = self.conv(torch.cat((x, padding), dim=-1))
        return output.masked_fill(~mask[:, None, :], 0)


class ConvBlock(nn.Module):
    def __init__(self, dim: int, conv_kernel: int):
        super().__init__()
        self.c1 = ConvMasked(dim, dim // 2, conv_kernel)
        self.c2 = ConvMasked(dim // 2, dim // 4, conv_kernel // 2)
        self.c3 = ConvMasked(dim // 4, dim // 2, conv_kernel)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for conv in (self.c1, self.c2, self.c3):
            x = self.activation(conv(x, mask))
        return x.transpose(1, 2)


class RaygunBlock(nn.Module):
    """Returns an update [B,L,D]; the cascade applies its outer residual."""

    def __init__(self, dim: int = 1280, num_heads: int = 20,
                 conv_kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        validate_block_config(dim, num_heads, conv_kernel)
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.dim = dim
        self.encoder = TransformerLayer(
            embed_dim=dim, ffn_embed_dim=2 * dim, attention_heads=num_heads,
            use_rotary_embeddings=True,
            # Raygun uses the upstream default True. Its appended unmasked token
            # rotates at Lmax, so its contribution changes with padded batch size.
            add_bias_kv=False,
        )
        self.convblock = ConvBlock(dim, conv_kernel)
        self.final = nn.Linear(dim // 2, dim)
        self.dropout = nn.Dropout(dropout)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # fair-esm's non-buffer rotary tables are not invalidated by .to(dtype).
        rotary = self.encoder.self_attn.rot_emb
        rotary._seq_len_cached = rotary._cos_cached = rotary._sin_cached = None
        return result

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        validate_masked_input(x, mask, self.dim)
        x = zero_padding(x, mask)
        x, _ = self.encoder(x.transpose(0, 1), self_attn_padding_mask=~mask)
        x = zero_padding(x.transpose(0, 1), mask)
        x = self.convblock(x, mask)
        return zero_padding(self.dropout(self.final(x)), mask)
