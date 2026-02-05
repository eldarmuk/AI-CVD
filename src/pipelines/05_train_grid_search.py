import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score

# --- CONFIG ---
BASE_DIR = Path(".")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024
EPOCHS = 100 

# --- MODEL CLASS (Standard) ---
class LSTM_VAE(nn.Module):
    def __init__(self, input_dim, seq_len, embed_dim, hidden_dim):
        super(LSTM_VAE, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, embed_dim)
        self.fc_logvar = nn.Linear(hidden_dim, embed_dim)
        
        # Decoder
        self.fc_decoder_input = nn.Linear(embed_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        hidden = hidden.squeeze(0)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        batch_size = z.size(0)
        device = z.device
        
        hidden_proj = self.fc_decoder_input(z)
        h0 = hidden_proj.unsqueeze(0)
        c0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        
        # Decoder Input: Repeat Z (Global Context)
        decoder_input = hidden_proj.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        out, _ = self.decoder_lstm(decoder_input, (h0, c0))
        return self.fc_output(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

# --- DATASET CLASS ---
class MmapDataset(Dataset):
    def __init__(self, npy_path, seq_len, n_features, mean, std):
        self.data = np.memmap(npy_path, dtype='float32', mode='r')
        n_samples = self.data.shape[0] // (seq_len * n_features)
        self.data = self.data.reshape(n_samples, seq_len, n_features)
        
        self.mean = mean.cpu().numpy()
        self.std = std.cpu().numpy() + 1e-6 
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        x = np.array(self.data[idx])
        # Normalize on the fly
        x = np.nan_to_num(x, nan=0.0)
        x = (x - self.mean) / self.std
        return torch.tensor(x, dtype=torch.float32)

def loss_function(recon_x, x, mu, logvar):
    mse = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + 0.001 * kld 

# --- TRAINER ---
def train_config(name, hidden_dim, embed_dim, seq_len):
    print(f"\n>>> STARTING RUN: {name} (H={hidden_dim}, E={embed_dim})")
    
    # Paths
    X_train_path = BASE_DIR / "data/processed/anomaly_detection/X_train.npy"
    X_test_path = BASE_DIR / "data/processed/anomaly_detection/X_test.npy"
    y_test_path = BASE_DIR / "data/processed/anomaly_detection/y_test.npy"
    stats_path = BASE_DIR / "data/processed/anomaly_detection/normalization_stats.json"
    
    save_dir = BASE_DIR / "models" / "grid_search" / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load Stats
    with open(stats_path, 'r') as f:
        stats = json.load(f)
        mean = torch.tensor(stats['mean']).to(DEVICE)
        std = torch.tensor(stats['std']).to(DEVICE)
    
    n_features = len(stats['mean'])
    print(f"  Features: {n_features}")

    # Load Data
    train_dataset = MmapDataset(X_train_path, seq_len, n_features, mean, std)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # Preload Test Data
    X_test_raw = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    # Normalize Test
    mean_np = mean.cpu().numpy()
    std_np = std.cpu().numpy() + 1e-6
    X_test_norm = (np.nan_to_num(X_test_raw, nan=0.0) - mean_np) / std_np
    X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32).to(DEVICE)

    # Model
    model = LSTM_VAE(n_features, seq_len, embed_dim, hidden_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, verbose=True)
    
    best_auc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(DEVICE)
            if batch.shape[1] != seq_len:
                batch = batch[:, :seq_len, :]

            optimizer.zero_grad()
            recon, mu, logvar, _ = model(batch)
            loss = loss_function(recon, batch, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            if X_test_tensor.shape[1] != seq_len:
                X_eval = X_test_tensor[:, :seq_len, :]
            else:
                X_eval = X_test_tensor
            
            recon, _, _, _ = model(X_eval)
            errors = torch.mean((X_eval - recon) ** 2, dim=[1, 2]).cpu().numpy()
            current_auc = roc_auc_score(y_test, errors)

        print(f"[{name}] Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f} | AUC={current_auc:.4f}")
        
        scheduler.step(current_auc)

        if current_auc > best_auc:
            best_auc = current_auc
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            with open(save_dir / "metrics.txt", "w") as f:
                f.write(f"Best AUC: {best_auc:.5f}\nEpoch: {epoch}\nConfig: H{hidden_dim}_E{embed_dim}")
    
    print(f"Finished {name}. Best: {best_auc:.4f}")
    return best_auc

if __name__ == "__main__":
    # Baseline
    train_config("run_standard", hidden_dim=64, embed_dim=32, seq_len=96)
    # Larger
    train_config("run_large", hidden_dim=128, embed_dim=64, seq_len=96)
    # Bottleneck
    train_config("run_bottleneck", hidden_dim=64, embed_dim=16, seq_len=96)