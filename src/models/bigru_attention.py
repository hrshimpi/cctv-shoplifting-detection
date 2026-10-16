"""
Temporal attention added on top of the BiGRU sequence autoencoder
(src/models/bigru_autoencoder.py) for Experiment E - not a rewrite.

BiGRUAutoencoder collapses its encoder to a single latent vector using
only the final timestep's forward/backward hidden state. This module
changes exactly that one step: BiGRUAttentionAutoencoder subclasses it,
inherits the same encoder/decoder/projections unchanged, and replaces
the final-hidden-state latent with an attention-weighted pool over
EVERY timestep's encoder output - letting the model learn which parts
of a window matter most for reconstruction, rather than always the
window's end.
"""

import torch
from torch import nn

from src.models.bigru_autoencoder import BiGRUAutoencoder


class TemporalAttention(nn.Module):
    """Additive attention pooling: a learned score per timestep, softmax
    over time, weighted sum - collapses (batch, T, dim) to (batch, dim)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, encoder_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """encoder_outputs: (batch, T, dim).
        Returns (context (batch, dim), attention_weights (batch, T))."""
        scores = self.score(encoder_outputs).squeeze(-1)  # (batch, T)
        weights = torch.softmax(scores, dim=-1)
        context = torch.sum(encoder_outputs * weights.unsqueeze(-1), dim=1)
        return context, weights


class BiGRUAttentionAutoencoder(BiGRUAutoencoder):
    """Same constructor signature and encoder/decoder/output_proj as
    BiGRUAutoencoder (inherited unchanged) - see that module's docstring
    for the base architecture. Adds one new module (temporal attention)
    and overrides only forward() to use it in place of the final hidden
    state. reconstruction_error() is inherited as-is."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(input_dim, hidden_size, num_layers, dropout)
        self.attention = TemporalAttention(hidden_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_length, input_dim) -> reconstruction of the
        same shape."""
        _, window_length, _ = x.shape

        encoder_outputs, _ = self.encoder(x)  # (batch, T, 2*hidden_size)
        context, _ = self.attention(encoder_outputs)
        latent = self.latent_proj(context)  # (batch, hidden_size) - same projection as the parent

        decoder_input = latent.unsqueeze(1).expand(-1, window_length, -1)
        decoder_out, _ = self.decoder(decoder_input)
        return self.output_proj(decoder_out)
