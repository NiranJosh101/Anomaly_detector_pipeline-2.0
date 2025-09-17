from typing import Optional, Tuple
import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder using the "repeat-latent" decoder strategy.

    Args:
        seq_len: int
            Length of input windows (L). Used only for sanity docs; not required at runtime.
        n_features: int
            Number of input features / channels (C).
        hidden_size: int = 64
            Hidden dimensionality for encoder/decoder LSTM.
        latent_size: int = 16
            Size of the bottleneck latent vector (z).
        num_layers: int = 1
            Number of stacked LSTM layers for encoder/decoder.
        dropout: float = 0.0
            Dropout between stacked LSTM layers (ignored when num_layers=1).
        bidirectional: bool = False
            If True, encoder LSTM will be bidirectional. (Not recommended for streaming.)
    """

    def __init__(self,
                 seq_len: int,
                 n_features: int,
                 hidden_size: int = 64,
                 latent_size: int = 16,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 ):
        super().__init__()

       
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional


        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        enc_out_dim = hidden_size * (2 if bidirectional else 1)

        self.to_latent = nn.Linear(enc_out_dim, latent_size)

        
        self.decoder = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  
        )

        self.output_layer = nn.Linear(hidden_size, n_features)

        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with a sensible default (Xavier).
        LSTM parameters typically have reasonable defaults; we only initialize
        the linear layers here for clarity."""
        nn.init.xavier_uniform_(self.to_latent.weight)
        nn.init.zeros_(self.to_latent.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        
        # x expected shape: (B, L, C)
        if x.ndim != 3:
            raise ValueError(f"Expected x of shape (B, L, C), got {tuple(x.shape)}")

        enc_out, (h_n, c_n) = self.encoder(x)

        if self.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h_top = torch.cat([h_forward, h_backward], dim=1)  # (B, hidden*2)
        else:
            h_top = h_n[-1]

        # map to latent
        z = self.to_latent(h_top)  # (B, latent_size)
        return z

    def decode(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:

        if seq_len is None:
            seq_len = self.seq_len
        B = z.size(0)

        dec_in = z.unsqueeze(1).repeat(1, seq_len, 1)

        dec_out, (h_n, c_n) = self.decoder(dec_in)

        yhat = self.output_layer(dec_out)
        return yhat

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # encode -> z (B, latent)
        z = self.encode(x)
        # decode -> yhat (B, L, C)
        yhat = self.decode(z, seq_len=x.size(1))
        return yhat

    # Utility helpers
    def reconstruct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience: return reconstruction and latent.

        Returns:
            yhat: (B, L, C)
            z: (B, latent)
        """
        z = self.encode(x)
        yhat = self.decode(z, seq_len=x.size(1))
        return yhat, z

    def save_checkpoint(self, path: str) -> None:
        """Save model state dict and config for later reload."""
        ckpt = {
            'state_dict': self.state_dict(),
            'config': {
                'seq_len': self.seq_len,
                'n_features': self.n_features,
                'hidden_size': self.hidden_size,
                'latent_size': self.latent_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional,
            }
        }
        torch.save(ckpt, path)

    @classmethod
    def load_checkpoint(cls, path: str, map_location: Optional[str] = None) -> 'LSTMAutoencoder':
        """Load from checkpoint saved by `save_checkpoint`.

        Returns an instantiated model with loaded weights. Note: you must pass the
        same device/ dtype mapping via `map_location` if reloading on CPU/GPU.
        """
        ckpt = torch.load(path, map_location=map_location)
        cfg = ckpt['config']
        model = cls(
            seq_len=cfg['seq_len'],
            n_features=cfg['n_features'],
            hidden_size=cfg['hidden_size'],
            latent_size=cfg['latent_size'],
            num_layers=cfg['num_layers'],
            dropout=cfg['dropout'],
            bidirectional=cfg['bidirectional'],
        )
        model.load_state_dict(ckpt['state_dict'])
        return model



def count_params(module: nn.Module) -> int:
    """Return number of trainable parameters (useful for debugging/model cards)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = LSTMAutoencoder(seq_len=64, n_features=1, hidden_size=32, latent_size=8)
    print("params:", count_params(m))
    x = torch.randn(4, 64, 1).to(device)
    m = m.to(device)
    y = m(x)
    print(y.shape)  # expect (4, 64, 1)
