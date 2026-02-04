import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
from pathlib import Path

# --- CONFIG ---
BASE_DIR = Path(".")
PATHS = {
    "parquet": BASE_DIR / "data/processed/multimodal_features.parquet",
    "static_csv": BASE_DIR / "data/processed/static_features.csv",
    "stats": BASE_DIR / "data/processed/anomaly_detection/normalization_stats.json",
    "model_save": BASE_DIR / "models/lstm_cvae/best_checkpoint.pt"
}
(BASE_DIR / "models/lstm_cvae").mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 96
DYN_DIM = 12   # Dynamic Features
STAT_DIM = 14  # Static Features (Age, Gender, Comorbidities...)
HIDDEN_DIM = 64
EMBEDDING_DIM = 32
BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-3

# --- 1. THE CVAE MODEL ---
class LSTM_CVAE(nn.Module):
    def __init__(self, dyn_dim, stat_dim, seq_len, embed_dim, hidden_dim):
        super(LSTM_CVAE, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Encoder: Takes [Dynamic + Static_Repeated]
        # We repeat static features for every timestep to feed into LSTM
        self.encoder_lstm = nn.LSTM(dyn_dim + stat_dim, hidden_dim, batch_first=True)
        
        # Latent Projections
        self.fc_mu = nn.Linear(hidden_dim, embed_dim)
        self.fc_logvar = nn.Linear(hidden_dim, embed_dim)
        
        # Decoder: Takes [Z + Static] -> Hidden
        # We project Z+Static to hidden state to initialize decoder
        self.fc_decoder_init = nn.Linear(embed_dim + stat_dim, hidden_dim)
        
        # Decoder Input: We feed [Zero + Static] or [Recon_prev + Static]?
        # Standard Seq2Seq: We feed the static features at every step
        self.decoder_lstm = nn.LSTM(stat_dim + embed_dim, hidden_dim, batch_first=True)
        
        # Output Projection
        self.fc_output = nn.Linear(hidden_dim, dyn_dim)

    def encode(self, dyn, stat):
        # Expand static to (B, Seq, Stat)
        stat_expanded = stat.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Concat: (B, Seq, Dyn+Stat)
        combined = torch.cat([dyn, stat_expanded], dim=2)
        
        _, (hidden, _) = self.encoder_lstm(combined)
        hidden = hidden.squeeze(0) # (B, Hidden)
        
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, stat):
        _ = z.size(0)
        
        # Prepare Decoder Input
        # We will feed [Z, Static] repeated at every timestep
        # This is a "Teacher Forcing = 0" approach (pure generative)
        
        # Z: (B, Embed) -> (B, Seq, Embed)
        z_expanded = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Stat: (B, Stat) -> (B, Seq, Stat)
        stat_expanded = stat.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Input: (B, Seq, Embed + Stat)
        decoder_input = torch.cat([z_expanded, stat_expanded], dim=2)
        
        # Init Hidden State
        # h0 = self.fc_decoder_init(torch.cat([z, stat], dim=1)).unsqueeze(0)
        # c0 = torch.zeros(1, batch_size, self.hidden_dim).to(z.device)
        
        # Run Decoder
        out, _ = self.decoder_lstm(decoder_input)
        
        return self.fc_output(out)

    def forward(self, dyn, stat):
        mu, logvar = self.encode(dyn, stat)
        z = self.reparameterize(mu, logvar)
        recon_dyn = self.decode(z, stat)
        return recon_dyn, mu, logvar

# --- 2. DATASET ---
class HybridDataset(Dataset):
    def __init__(self, parquet_path, static_csv_path, dyn_cols, seq_len=96):
        print("Loading data...")
        self.df = pd.read_parquet(parquet_path)
        self.static_df = pd.read_csv(static_csv_path).set_index('senior_id')
        
        # Filter only Healthy (Label 3 == 0) for training
        print("Filtering for Healthy Training Data...")
        self.df = self.df[self.df['label_3'] == 0].reset_index(drop=True)
        
        self.dyn_cols = dyn_cols
        self.seq_len = seq_len
        self.seniors = self.df['senior_id'].unique()
        
        # Pre-compute normalization stats (Dynamic only)
        print("Computing Normalization Stats...")
        dyn_data = self.df[dyn_cols].values
        self.mean = torch.tensor(np.nanmean(dyn_data, axis=0), dtype=torch.float32)
        self.std = torch.tensor(np.nanstd(dyn_data, axis=0), dtype=torch.float32)
        
        # Save stats
        stats = {'mean': self.mean.tolist(), 'std': self.std.tolist()}
        with open(PATHS['stats'], 'w') as f:
            json.dump(stats, f)

        # Index windows
        self.indices = []
        print("Indexing windows...")
        for sid in self.seniors:
            # Check if we have static data for this senior
            if sid not in self.static_df.index:
                continue
            
            sub_indices = self.df[self.df['senior_id'] == sid].index.values
            if len(sub_indices) < seq_len:
                continue
            
            # Create valid start indices
            # Stride = 96 (No overlap for training speed, or overlap=1 for dense?)
            # Let's do Stride=48 to get more data
            starts = sub_indices[:-seq_len:48] 
            for s in starts:
                self.indices.append((sid, s))
                
        print(f"Total Training Windows: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sid, start_idx = self.indices[idx]
        
        # Get Dynamic Window
        # Note: In a real heavy loop, iloc is slow. 
        # Ideally we convert DF to Tensor in __init__, but RAM might limit us.
        # Fast path: Use pre-extracted numpy array if possible.
        # For now, standard pandas access.
        
        window = self.df.iloc[start_idx : start_idx + self.seq_len][self.dyn_cols].values
        window = torch.tensor(window, dtype=torch.float32)
        
        # Normalize
        window = torch.where(torch.isnan(window), self.mean, window)
        window = (window - self.mean) / self.std
        
        # Get Static Vector
        # We assume static_df is numeric (One-Hot Encoded if needed)
        # You might need to add pd.get_dummies in __init__
        static_vec = self.static_df.loc[sid].values
        static_vec = torch.tensor(static_vec, dtype=torch.float32)
        
        return window, static_vec

# --- 3. LOSS FUNCTION ---
def cvae_loss(recon, x, mu, logvar, beta=0.5):
    mse = nn.functional.mse_loss(recon, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kld, mse, kld

# --- 4. TRAIN LOOP ---
def main():
    # Define Columns
    dyn_cols = [
        'temperature', 'heartrate', 'sbp', 'dbp', 'pulse_pressure', 
        'steps', 'saturation', 'hr_volatility', 'bp_trend', 
        'steps_rolling_sum_6h', 'shock_index', 'recent_event_burden'
    ]
    
    # Init Dataset
    dataset = HybridDataset(PATHS['parquet'], PATHS['static_csv'], dyn_cols)
    
    # Handle Static Features Encoding (if they are strings like 'Male')
    # If dataset.static_df has strings, this will crash.
    # Check and fix:
    if dataset.static_df.select_dtypes(include=['object']).shape[1] > 0:
        print("Encoding categorical static features...")
        dataset.static_df = pd.get_dummies(dataset.static_df, drop_first=True)
        # Update stat_dim
    
    stat_dim = dataset.static_df.shape[1]
    print(f"Dynamic Dim: {len(dyn_cols)}, Static Dim: {stat_dim}")
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # workers=0 for Windows safety
    
    # Init Model
    model = LSTM_CVAE(len(dyn_cols), stat_dim, SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("Starting CVAE Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (dyn, stat) in enumerate(loader):
            dyn, stat = dyn.to(DEVICE), stat.to(DEVICE)
            
            optimizer.zero_grad()
            
            recon, mu, logvar = model(dyn, stat)
            loss, mse, kld = cvae_loss(recon, dyn, mu, logvar)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f} (MSE: {mse.item():.4f} KL: {kld.item():.4f})")
                
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(loader):.4f}")
        torch.save(model.state_dict(), PATHS['model_save'])

if __name__ == "__main__":
    main()