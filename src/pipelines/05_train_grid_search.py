import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score

# --- CONFIG ---
BASE_DIR = Path(".")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024
EPOCHS = 30
PATIENCE = 8
KL_BETA_MAX = 0.001

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

def loss_function(recon_x, x, mu, logvar, beta=0.001):
    mse = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kld


def get_beta(epoch, max_epochs=100, beta_max=0.001):
    """Gradually increase KL weight from 0 to beta_max."""
    return min(beta_max, beta_max * (epoch / (max_epochs * 0.5)))


def validate(model, val_loader, device, beta):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    all_errors = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            recon, mu, logvar, _ = model(x)
            loss = loss_function(recon, x, mu, logvar, beta=beta)

            recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kld.item()

            errors = torch.mean((x - recon) ** 2, dim=[1, 2])
            all_errors.extend(errors.cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())

    n_batches = len(val_loader)
    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kl': total_kl / n_batches,
        'errors': np.array(all_errors),
        'labels': np.array(all_labels)
    }


def find_optimal_threshold(errors, labels):
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, errors)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)

    return {
        'threshold': float(thresholds[ix]),
        'gmean': float(gmeans[ix]),
        'tpr': float(tpr[ix]),
        'fpr': float(fpr[ix])
    }

# --- TRAINER ---
def train_config(name, hidden_dim, embed_dim, seq_len):
    print(f"\n>>> STARTING RUN: {name} (H={hidden_dim}, E={embed_dim})")
    
    # Paths
    X_train_path = BASE_DIR / "data/processed/anomaly_detection/X_train.npy"
    X_val_path = BASE_DIR / "data/processed/anomaly_detection/X_val.npy"
    y_val_path = BASE_DIR / "data/processed/anomaly_detection/y_val.npy"
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
    
    # Preload Val Data
    X_val_raw = np.load(X_val_path)
    y_val = np.load(y_val_path)

    # Normalize Val
    mean_np = mean.cpu().numpy()
    std_np = std.cpu().numpy() + 1e-6
    X_val_norm = (np.nan_to_num(X_val_raw, nan=0.0) - mean_np) / std_np
    X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda"
    )

    # Model
    model = LSTM_VAE(n_features, seq_len, embed_dim, hidden_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    best_threshold = None
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        current_beta = get_beta(epoch, EPOCHS, KL_BETA_MAX)
        
        for batch in train_loader:
            batch = batch.to(DEVICE)
            if batch.shape[1] != seq_len:
                batch = batch[:, :seq_len, :]

            optimizer.zero_grad()
            recon, mu, logvar, _ = model(batch)
            loss = loss_function(recon, batch, mu, logvar, beta=current_beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        val_metrics = validate(model, val_loader, DEVICE, current_beta)
        threshold_info = find_optimal_threshold(val_metrics['errors'], val_metrics['labels'])
        current_auc = roc_auc_score(val_metrics['labels'], val_metrics['errors'])

        print(
            f"[{name}] Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f} "
            f"| Val Loss={val_metrics['loss']:.4f} | Val AUROC={current_auc:.4f} | Beta={current_beta:.6f}"
        )

        scheduler.step(current_auc)

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_threshold = threshold_info
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_auc': current_auc,
                'threshold': threshold_info['threshold'],
            }, save_dir / "best_model.pt")
            with open(save_dir / "metrics.txt", "w") as f:
                f.write(
                    f"Best Val Loss: {best_val_loss:.5f}\n"
                    f"Val AUROC: {current_auc:.5f}\n"
                    f"Best Threshold: {threshold_info['threshold']:.5f}\n"
                    f"Epoch: {epoch}\nConfig: H{hidden_dim}_E{embed_dim}"
                )
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Finished {name}. Best Val Loss: {best_val_loss:.4f}")
    return best_val_loss

if __name__ == "__main__":
    # Baseline
    train_config("run_standard", hidden_dim=64, embed_dim=32, seq_len=96)
    # Larger
    train_config("run_large", hidden_dim=128, embed_dim=64, seq_len=96)
    # Bottleneck
    train_config("run_bottleneck", hidden_dim=64, embed_dim=16, seq_len=96)