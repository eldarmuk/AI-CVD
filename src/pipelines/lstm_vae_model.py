import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_VAE(nn.Module):
    """
    LSTM-based Variational Autoencoder for sequence reconstruction.
    
    Architecture:
      - Encoder: LSTM processing (B, seq_len, feat) -> (B, hidden)
      - Bottleneck: Maps hidden state to latent distribution parameters (mu, logvar)
      - Decoder: LSTM initialized with latent vector z, reconstructing the sequence.
    
    Note: z is fed to the decoder at every timestep to maintain global context.
    """
    
    def __init__(self, input_dim: int = 25, sequence_length: int = 96, 
                 embedding_dim: int = 32, hidden_dim: int = 64):
        super(LSTM_VAE, self).__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Variational Heads
        self.fc_mu = nn.Linear(hidden_dim, embedding_dim)
        self.fc_logvar = nn.Linear(hidden_dim, embedding_dim)
        
        # Decoder
        self.fc_decoder_input = nn.Linear(embedding_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.fc_output = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x: torch.Tensor):
        # x: (B, seq_len, input_dim)
        _, (hidden, _) = self.encoder_lstm(x)
        
        # Flatten hidden state: (1, B, hidden) -> (B, hidden)
        hidden = hidden.squeeze(0)
        
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        device = z.device
        
        # Project z to hidden dim
        hidden_z = self.fc_decoder_input(z) # (B, hidden)
        
        # Init decoder state with latent vector
        h0 = hidden_z.unsqueeze(0) # (1, B, hidden)
        c0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        
        # Latent conditioning: Feed z at every timestep
        # (B, hidden) -> (B, seq_len, hidden)
        decoder_input = hidden_z.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Decode
        out_lstm, _ = self.decoder_lstm(decoder_input, (h0, c0))
        
        # Map back to feature space: (B, seq_len, input_dim)
        return self.fc_output(out_lstm)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z


def test_run():
    """Simple sanity check for dimensions."""
    B, L, F_in = 64, 96, 25
    H, Z_dim = 64, 32
    
    print(f"Initializing model with [B={B}, L={L}, F={F_in}]...")
    
    model = LSTM_VAE(
        input_dim=F_in,
        sequence_length=L,
        embedding_dim=Z_dim,
        hidden_dim=H
    )
    
    dummy_input = torch.randn(B, L, F_in)
    recon_x, mu, logvar, z = model(dummy_input)
    
    # Logging shapes
    print(f"  + Input shape:   {dummy_input.shape}")
    print(f"  + Recon shape:   {recon_x.shape}")
    print(f"  + Latent z:      {z.shape}")
    
    # Assertions
    try:
        assert recon_x.shape == dummy_input.shape
        assert z.shape == (B, Z_dim)
        print("\n+ Integrity check passed. Dimensions align.")
    except AssertionError as e:
        print(f"\n! Dimension mismatch: {e}")

if __name__ == "__main__":
    test_run()