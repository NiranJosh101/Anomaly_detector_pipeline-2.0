import torch
import torch.nn as nn

class GRUAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1, dropout=0.0, bidirectional=False):
        """
        GRU Autoencoder for time-series anomaly detection.

        Args:
            input_dim (int): Number of features per timestep (C).
            hidden_dim (int): Number of hidden units in GRU layers.
            latent_dim (int): Size of the latent bottleneck vector.
            num_layers (int): Number of stacked GRU layers.
            dropout (float): Dropout rate (applied between stacked layers).
            bidirectional (bool): If True, encoder GRU is bidirectional.
        """
        super(GRUAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        
        # GRU expects input: (batch, seq_len, input_dim) if batch_first=True
        self.encoder_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  
            bidirectional=bidirectional
        )

        enc_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.enc_to_latent = nn.Linear(enc_out_dim, latent_dim)

        # Decoder GRU
        self.decoder_gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        
        self.output_layer = nn.Linear(hidden_dim, input_dim)


    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        # Encode 
        enc_out, h_n = self.encoder_gru(x)

        if self.bidirectional:
            h_forward = h_n[-2, :, :]   
            h_backward = h_n[-1, :, :]  
            h_top = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_top = h_n[-1, :, :]

        z = self.enc_to_latent(h_top)  

        # Decode 
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)

        dec_out, _ = self.decoder_gru(z_repeated) 

        
        recon = self.output_layer(dec_out)  # (batch, seq_len, input_dim)

        return recon
