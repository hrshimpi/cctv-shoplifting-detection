"""
Bidirectional-GRU sequence autoencoder for one-class pose anomaly
detection. Trained on normal sequences only; the anomaly score at
inference time is the mean squared reconstruction error (see
src/evaluation/metrics.py).

Architecture: a BiGRU encodes the whole input window into a single
latent vector (the concatenated final forward/backward hidden state of
the last layer, projected down to hidden_size); a unidirectional GRU
decoder reconstructs the window from that latent vector, repeated at
every timestep as the decoder input. This is a standard seq2seq
autoencoder shape, not a from-scratch design.
"""

import torch
from torch import nn


class BiGRUAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Encoder is bidirectional -> final layer's fwd+bwd hidden states
        # concatenate to 2*hidden_size; project back down to hidden_size
        # to use as the (repeated) decoder input.
        self.latent_proj = nn.Linear(hidden_size * 2, hidden_size)

        self.decoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_size, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_length, input_dim) -> reconstruction of the
        same shape."""
        batch_size, window_length, _ = x.shape

        _, h_n = self.encoder(x)
        # h_n: (num_layers * 2, batch, hidden_size); last layer's
        # forward/backward states are the last two entries.
        h_fwd, h_bwd = h_n[-2], h_n[-1]
        latent = self.latent_proj(torch.cat([h_fwd, h_bwd], dim=-1))  # (batch, hidden_size)

        decoder_input = latent.unsqueeze(1).expand(-1, window_length, -1)
        decoder_out, _ = self.decoder(decoder_input)
        return self.output_proj(decoder_out)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared reconstruction error - the anomaly
        score. x: (batch, T, D) -> (batch,)."""
        recon = self.forward(x)
        return torch.mean((recon - x) ** 2, dim=(1, 2))
