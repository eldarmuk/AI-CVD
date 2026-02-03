import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import time
from pathlib import Path
from typing import Tuple, Optional

from .lstm_vae_model import LSTM_VAE

class Config:
    X_TRAIN = Path("data/processed/sequences/X_train_anomaly.npy")
    STATS_PATH = Path("data/processed/sequences/normalization_stats.json")
    MODEL_SAVE = Path("models/lstm_vae_best.pt")
    
    SEQ_LEN = 96
    N_FEATURES = 25
    BATCH_SIZE = 512
    LR = 1e-3
    EPOCHS = 10
    BETA = 0.001  # KL weight (annealing suggested for complex datasets)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- UTILS ---
def get_normalization_stats(npy_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes global Mean/Std from .npy file, ignoring NaNs.
    Uses chunked processing to handle large datasets within RAM limits.
    """
    if Config.STATS_PATH.exists():
        print(f"+ Loading cached stats from {Config.STATS_PATH}")
        with open(Config.STATS_PATH, 'r') as f:
            stats = json.load(f)
        return np.array(stats['mean'], dtype=np.float32), np.array(stats['std'], dtype=np.float32)

    print(f"+ Computing robust stats from {npy_path} (this may take a moment)...")
    
    # Memmap for zero-copy access
    X = np.memmap(npy_path, dtype='float32', mode='r')
    n_samples = X.shape[0] // (Config.SEQ_LEN * Config.N_FEATURES)
    X = X.reshape(n_samples, Config.SEQ_LEN, Config.N_FEATURES)
    
    # Accumulators
    feat_sum = np.zeros(Config.N_FEATURES)
    feat_sq_sum = np.zeros(Config.N_FEATURES)
    feat_count = np.zeros(Config.N_FEATURES)
    
    chunk_size = 50_000
    for i in range(0, n_samples, chunk_size):
        chunk = np.array(X[i:i+chunk_size]).reshape(-1, Config.N_FEATURES)
        
        # Mask NaNs
        mask = ~np.isnan(chunk)
        valid_data = np.nan_to_num(chunk, nan=0.0)
        
        feat_sum += valid_data.sum(axis=0)
        feat_sq_sum += (valid_data ** 2).sum(axis=0)
        feat_count += mask.sum(axis=0)
        
        if i % (chunk_size * 5) == 0:
            print(f"  > Processed {i}/{n_samples} samples")

    # Finalize stats (Std = sqrt(E[x^2] - E[x]^2))
    mean = feat_sum / np.maximum(feat_count, 1)
    mean_sq = feat_sq_sum / np.maximum(feat_count, 1)
    var = np.maximum(mean_sq - mean**2, 0) # Clip negative variance
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0 # Prevent div/0

    Config.STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(Config.STATS_PATH, 'w') as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
        
    return mean.astype(np.float32), std.astype(np.float32)

# --- DATASET ---
class MmapDataset(Dataset):
    def __init__(self, npy_path: Path, mean: np.ndarray, std: np.ndarray, label_path: Optional[Path] = None):
        self.data = np.memmap(npy_path, dtype='float32', mode='r')
        n_samples = self.data.shape[0] // (Config.SEQ_LEN * Config.N_FEATURES)
        self.data = self.data.reshape(n_samples, Config.SEQ_LEN, Config.N_FEATURES)
        
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)
        self.labels = np.load(label_path) if label_path else None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.data[idx])) # (96, 25)
        
        # Robust Normalization:
        # 1. Identify NaNs
        nan_mask = torch.isnan(x)
        
        # 2. Fill NaNs with global mean
        # (Since we subtract mean later, these positions effectively become 0)
        x = torch.where(nan_mask, self.mean, x)
        
        # 3. Normalize
        x = (x - self.mean) / self.std
        
        # 4. Cleanup any lingering artifacts
        x = torch.nan_to_num(x, nan=0.0)

        if self.labels is not None:
            return x, torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, x # Target == Input for AE

# --- LOSS ---
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """Computes VAE loss: MSE (Reconstruction) + Beta * KL Divergence"""
    
    # Reconstruction term
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    
    # KL Divergence term
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld / x.size(0) # Normalize by batch size
    
    return recon_loss + (beta * kld), recon_loss, kld

# --- TRAINING ---
def train():
    print(f"Starting training on {Config.DEVICE}")
    
    # Setup Data
    mean, std = get_normalization_stats(Config.X_TRAIN)
    dataset = MmapDataset(Config.X_TRAIN, mean, std)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                        num_workers=4, pin_memory=True, drop_last=True)
    
    print(f"+ Dataset loaded: {len(dataset):,} samples")

    # Setup Model
    model = LSTM_VAE(
        input_dim=Config.N_FEATURES, 
        hidden_dim=64, 
        embedding_dim=32
    ).to(Config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    
    Config.MODEL_SAVE.parent.mkdir(parents=True, exist_ok=True)
    

    # Training Loop
    for epoch in range(Config.EPOCHS):
        model.train()
        start_time = time.time()
        metrics = {'loss': 0.0, 'recon': 0.0, 'kl': 0.0}
        
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(Config.DEVICE)
            
            # Optimization
            optimizer.zero_grad()
            recon_x, mu, logvar, _ = model(x)
            loss, recon, kl = vae_loss(recon_x, x, mu, logvar, beta=Config.BETA)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            # Tracking
            metrics['loss'] += loss.item()
            metrics['recon'] += recon.item()
            metrics['kl'] += kl.item()
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"  Batch {batch_idx}/{len(loader)} | "
                      f"Loss: {loss.item():.4f} (Recon: {recon.item():.4f} KL: {kl.item():.4f})")

        # Epoch Summary
        avg_loss = metrics['loss'] / len(loader)
        duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | {duration:.1f}s | Avg Loss: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), Config.MODEL_SAVE)

if __name__ == "__main__":
    train()