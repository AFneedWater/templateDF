"""Peptide autoencoder: the decoder receives only latent Z and output lengths."""

import torch
from torch import nn

from .blocks import validate_masked_input
from .data import AA_ORDER, MIN_LENGTH, MAX_LENGTH
from .decoder import SequenceDecoder, validate_output_lengths
from .encoder import CascadeEncoder


class ProteinAutoencoder(nn.Module):
    aa_order = AA_ORDER

    def __init__(self, dim: int = 1280, num_latents: int = 50,
                 encoder_blocks: int = 4, decoder_blocks: int = 4,
                 num_heads: int = 20, conv_kernel: int = 7,
                 fusion_hidden_dim: int | None = None, pool_ffn_ratio: int = 4,
                 decoder_ffn_ratio: int = 4, dropout: float = 0.1,
                 length_scale: float = 150):
        super().__init__()
        self.dim, self.num_latents = dim, num_latents
        self.encoder = CascadeEncoder(
            dim=dim, num_latents=num_latents, num_blocks=encoder_blocks,
            num_heads=num_heads, conv_kernel=conv_kernel,
            fusion_hidden_dim=fusion_hidden_dim, pool_ffn_ratio=pool_ffn_ratio, dropout=dropout,
        )
        self.decoder = SequenceDecoder(
            dim=dim, num_blocks=decoder_blocks, num_heads=num_heads,
            ffn_ratio=decoder_ffn_ratio, dropout=dropout, length_scale=length_scale,
        )

    @staticmethod
    def _check_condition(condition):
        if condition is not None:
            raise NotImplementedError("Stage 1 does not support nonempty condition (use condition=None)")

    def _validate_input(self, embeddings: torch.Tensor, input_mask: torch.Tensor):
        if not isinstance(embeddings, torch.Tensor) or not isinstance(input_mask, torch.Tensor):
            raise ValueError("embeddings and input_mask must be tensors")
        validate_masked_input(embeddings, input_mask, self.dim)
        lengths = input_mask.sum(dim=1)
        if ((lengths < MIN_LENGTH) | (lengths > MAX_LENGTH)).any():
            raise ValueError(f"Input peptide lengths must be in [{MIN_LENGTH}, {MAX_LENGTH}]")
        if not torch.isfinite(embeddings[input_mask]).all():
            raise ValueError("Valid residue embeddings must be finite")
        return lengths

    def encode(self, embeddings: torch.Tensor, input_mask: torch.Tensor, condition=None) -> torch.Tensor:
        self._check_condition(condition)
        self._validate_input(embeddings, input_mask)
        return self.encoder(embeddings, input_mask)

    def decode(self, latent: torch.Tensor, output_lengths: torch.Tensor, condition=None) -> dict:
        self._check_condition(condition)
        if (not isinstance(latent, torch.Tensor) or latent.ndim != 3
                or latent.shape[1:] != (self.num_latents, self.dim)):
            raise ValueError(f"latent must have shape [B,{self.num_latents},{self.dim}]")
        lengths = validate_output_lengths(
            output_lengths, latent.shape[0], latent.device, minimum=MIN_LENGTH, maximum=MAX_LENGTH,
        )
        return self.decoder(latent, lengths)

    def forward(self, embeddings: torch.Tensor, input_mask: torch.Tensor,
                output_lengths: torch.Tensor | None = None, condition=None) -> dict:
        self._check_condition(condition)
        input_lengths = self._validate_input(embeddings, input_mask)
        lengths = input_lengths if output_lengths is None else validate_output_lengths(
            output_lengths, embeddings.shape[0], embeddings.device,
            minimum=MIN_LENGTH, maximum=MAX_LENGTH,
        )
        latent = self.encoder(embeddings, input_mask)
        result = self.decode(latent, lengths)
        return {**result, "latent": latent}
