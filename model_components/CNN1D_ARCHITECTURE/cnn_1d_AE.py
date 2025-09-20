from typing import List, Optional, Tuple, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


# Building Blocks 
class ConvBlock1D(nn.Module):
    """Conv1d -> Norm -> Activation"""

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 k: int = 3,
                 s: int = 1,
                 d: int = 1,
                 norm: Literal['batch', 'layer', 'none'] = 'batch',
                 act: Literal['relu', 'gelu', 'elu', 'leaky_relu', 'none'] = 'relu',
                 dropout: float = 0.0,
                 padding: Optional[int] = None):
        super().__init__()
        if padding is None:
            padding = ((k - 1) * d) // 2

        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, dilation=d, padding=padding)

        if norm == 'batch':
            self.norm = nn.BatchNorm1d(out_ch)
        elif norm == 'layer':
            self.norm = nn.GroupNorm(1, out_ch)
        else:
            self.norm = nn.Identity()

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'gelu':
            self.act = nn.GELU()
        elif act == 'elu':
            self.act = nn.ELU(inplace=True)
        elif act == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            self.act = nn.Identity()

        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class ResidualBlock1D(nn.Module):
    """Two ConvBlock1D layers with a residual/skip connection.

    """
    def __init__(self,
                 ch: int,
                 k: int = 3,
                 d: int = 1,
                 norm: Literal['batch', 'layer', 'none'] = 'batch',
                 act: Literal['relu', 'gelu', 'elu', 'leaky_relu', 'none'] = 'relu',
                 dropout: float = 0.0):
        super().__init__()
        self.block1 = ConvBlock1D(ch, ch, k=k, s=1, d=d, norm=norm, act=act, dropout=dropout)
        self.block2 = ConvBlock1D(ch, ch, k=k, s=1, d=d, norm=norm, act=act, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block1(x)
        out = self.block2(out)
        return out + residual


class UpBlock1D(nn.Module):
    """Upsampling block (decoder): Either ConvTranspose1d or Upsample+Conv1d.

    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 mode: Literal['transpose', 'nearest'] = 'transpose',
                 norm: Literal['batch', 'layer', 'none'] = 'batch',
                 act: Literal['relu', 'gelu', 'elu', 'leaky_relu', 'none'] = 'relu',
                 dropout: float = 0.0):
        super().__init__()
        self.mode = mode
        if mode == 'transpose':
            # ConvTranspose1d doubles length when kernel=4, stride=2, padding=1
            self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            self.post = ConvBlock1D(out_ch, out_ch, k=3, s=1, d=1, norm=norm, act=act, dropout=dropout)
        elif mode == 'nearest':
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.post = ConvBlock1D(in_ch, out_ch, k=3, s=1, d=1, norm=norm, act=act, dropout=dropout)
        else:
            raise ValueError("mode must be 'transpose' or 'nearest'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.post(x)
        return x


# CNN Autoencoder
class CNNAutoencoder1D(nn.Module):
    """
    1D CNN Autoencoder with options for residual blocks and dilations.

    """
    def __init__(self,
                 seq_len: int,
                 in_channels: int,
                 channels: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] | int = 3,
                 dilations: List[int] | int = 1,
                 use_residual: bool = True,
                 norm: Literal['batch', 'layer', 'none'] = 'batch',
                 act: Literal['relu', 'gelu', 'elu', 'leaky_relu', 'none'] = 'relu',
                 dropout: float = 0.0,
                 up_mode: Literal['transpose', 'nearest'] = 'transpose',
                 latent_dim: int = 64):
        super().__init__()

        self.seq_len = seq_len
        self.in_channels = in_channels
        self.channels = channels
        self.use_residual = use_residual
        self.norm = norm
        self.act = act
        self.dropout = dropout
        self.up_mode = up_mode
        self.latent_dim = latent_dim

        num_scales = len(channels)
        if seq_len % (2 ** num_scales) != 0:
            raise ValueError(
                f"seq_len={seq_len} must be divisible by 2**num_scales (num_scales={num_scales})."
            )

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_scales
        if isinstance(dilations, int):
            dilations = [dilations] * num_scales
        assert len(kernel_sizes) == num_scales
        assert len(dilations) == num_scales

        # Encoder
        enc_blocks = []
        prev_ch = in_channels
        length = seq_len  # track temporal length across scales
        self.lengths: List[int] = [length]

        for i, (ch, k, d) in enumerate(zip(channels, kernel_sizes, dilations)):
            enc_blocks.append(ConvBlock1D(prev_ch, ch, k=k, s=1, d=d,
                                          norm=norm, act=act, dropout=dropout))
            if use_residual:
                enc_blocks.append(ResidualBlock1D(ch, k=k, d=d, norm=norm, act=act, dropout=dropout))
            enc_blocks.append(ConvBlock1D(ch, ch, k=4, s=2, d=1, norm=norm, act=act, dropout=dropout, padding=1))
            prev_ch = ch
            length = length // 2
            self.lengths.append(length)

        self.encoder = nn.Sequential(*enc_blocks)

        enc_flat_dim = channels[-1] * length
        self.to_latent = nn.Linear(enc_flat_dim, latent_dim)

        # Decoder
        self.from_latent = nn.Linear(latent_dim, enc_flat_dim)

        dec_blocks = []
        curr_ch = channels[-1]
        length_dec = length

        # Mirror the encoder scales in reverse order
        for i, (ch, k, d) in enumerate(reversed(list(zip(channels, kernel_sizes, dilations)))):
            next_ch = channels[-(i+2)] if (i < len(channels) - 1) else in_channels
            dec_blocks.append(UpBlock1D(curr_ch, next_ch, mode=up_mode, norm=norm, act=act, dropout=dropout))
            length_dec = length_dec * 2
            if use_residual and next_ch > 0:  # sanity
                dec_blocks.append(ResidualBlock1D(next_ch, k=k, d=d, norm=norm, act=act, dropout=dropout))
            curr_ch = next_ch

        self.decoder = nn.Sequential(*dec_blocks)
        self.out_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)

        nn.init.xavier_uniform_(self.to_latent.weight)
        nn.init.zeros_(self.to_latent.bias)
        nn.init.xavier_uniform_(self.from_latent.weight)
        nn.init.zeros_(self.from_latent.bias)

    def encode(self, x_blc: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Encode (B, L, C) -> latent (B, latent_dim).

        Returns latent and a tuple with (C_last, L_last) to guide reshaping in decode.
        """
        if x_blc.ndim != 3:
            raise ValueError(f"Expected x of shape (B, L, C), got {tuple(x_blc.shape)}")
        # Rearrange to (B, C, L) for Conv1d
        x = x_blc.permute(0, 2, 1)
        x = self.encoder(x)
        B, C_last, L_last = x.shape
        x = x.reshape(B, C_last * L_last)
        z = self.to_latent(x)
        return z, (C_last, L_last)

    def decode(self, z: torch.Tensor, shape_info: Tuple[int, int]) -> torch.Tensor:
        """Decode latent (B, latent_dim) back to reconstruction (B, L, C)."""
        B = z.size(0)
        C_last, L_last = shape_info
        x = self.from_latent(z)
        x = x.view(B, C_last, L_last)
        x = self.decoder(x)
        x = self.out_conv(x)
        x = x.permute(0, 2, 1).contiguous()
        return x

    def forward(self, x_blc: torch.Tensor) -> torch.Tensor:
        """Full forward: (B, L, C) -> (B, L, C)."""
        if x_blc.size(1) != self.seq_len:
            raise ValueError(
                f"Input seq_len={x_blc.size(1)} differs from model.seq_len={self.seq_len}. "
                "This AE uses stride-2 pyramids; lengths must match the configured seq_len."
            )
        z, shape_info = self.encode(x_blc)
        yhat = self.decode(z, shape_info)
        return yhat

    # Utilities
    def reconstruct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return reconstruction and latent for convenience.
        Args:
            x: (B, L, C)
        Returns:
            yhat: (B, L, C), z: (B, latent_dim)
        """
        z, shape_info = self.encode(x)
        yhat = self.decode(z, shape_info)
        return yhat, z

    def save_checkpoint(self, path: str) -> None:
        ckpt = {
            'state_dict': self.state_dict(),
            'config': {
                'seq_len': self.seq_len,
                'in_channels': self.in_channels,
                'channels': self.channels,
                'use_residual': self.use_residual,
                'norm': self.norm,
                'act': self.act,
                'dropout': self.dropout,
                'up_mode': self.up_mode,
                'latent_dim': self.latent_dim,
            }
        }
        torch.save(ckpt, path)

    @classmethod
    def load_checkpoint(cls, path: str, map_location: Optional[str] = None) -> 'CNNAutoencoder1D':
        ckpt = torch.load(path, map_location=map_location)
        cfg = ckpt['config']
        model = cls(**cfg)
        model.load_state_dict(ckpt['state_dict'])
        return model


# Helpers 
def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


