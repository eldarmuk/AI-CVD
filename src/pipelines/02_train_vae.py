import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Tuple, Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..components.lstm_vae_model import LSTM_VAE
from ..components.utils import get_feature_columns

class Config:
    X_TRAIN = Path("data/processed/anomaly_detection/X_train.npy")
    X_VAL = Path("data/processed/anomaly_detection/X_val.npy")
    Y_VAL = Path("data/processed/anomaly_detection/y_val.npy")
    STATS_PATH = Path("data/processed/anomaly_detection/normalization_stats.json")
    MODEL_SAVE_DIR = Path("models/lstm_vae")
    BEST_MODEL_PATH = MODEL_SAVE_DIR / "best_checkpoint.pt"
    VAL_METRICS_PATH = MODEL_SAVE_DIR / "validation_metrics.json"
    PARQUET_PATH = Path("data/processed/multimodal_features.parquet")
    
    SEQ_LEN = 96
    FEATURE_NAMES = get_feature_columns(pd.read_parquet(PARQUET_PATH))
    N_FEATURES = len(FEATURE_NAMES)
    BATCH_SIZE = 512
    LR = 1e-3
    EPOCHS = 20
    BETA = 0.001  # KL weight (annealing suggested for complex datasets)
    KL_CYCLES = 4
    KL_WARMUP_RATIO = 0.5
    PATIENCE = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0 if os.name == "nt" else 4

# --- UTILS ---
def get_normalization_stats(npy_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Mean/Std from the training .npy file only, ignoring NaNs.
    Uses chunked processing to handle large datasets within RAM limits.
    """
    expected_source = str(npy_path.resolve())
    if Config.STATS_PATH.exists():
        with open(Config.STATS_PATH, 'r') as f:
            stats = json.load(f)
        if (
            stats.get('fit_split') == 'train'
            and stats.get('source_path') == expected_source
            and stats.get('seq_len') == Config.SEQ_LEN
            and stats.get('n_features') == Config.N_FEATURES
        ):
            print(f"+ Loading cached train-only stats from {Config.STATS_PATH}")
            return np.array(stats['mean'], dtype=np.float32), np.array(stats['std'], dtype=np.float32)
        print("+ Ignoring cached normalization stats without matching train-only provenance")

    print(f"+ Fitting normalization stats on TRAIN data only: {npy_path}")
    
    # Memmap for zero-copy access while preserving the .npy header shape.
    X = np.load(npy_path, mmap_mode='r')
    if X.ndim != 3 or X.shape[1:] != (Config.SEQ_LEN, Config.N_FEATURES):
        raise ValueError(
            f"Expected training array shape (n, {Config.SEQ_LEN}, {Config.N_FEATURES}), "
            f"got {X.shape}"
        )
    n_samples = X.shape[0]
    
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
        json.dump({
            "mean": mean.tolist(),
            "std": std.tolist(),
            "fit_split": "train",
            "source_path": expected_source,
            "source_shape": list(X.shape),
            "seq_len": Config.SEQ_LEN,
            "n_features": Config.N_FEATURES,
            "nan_policy": "ignore_when_fitting_fill_with_train_mean_when_transforming",
        }, f, indent=2)
        
    return mean.astype(np.float32), std.astype(np.float32)

# --- DATASET ---
class NumpyDataset(Dataset):
    def __init__(self, data, mean: np.ndarray, std: np.ndarray, labels: Optional[np.ndarray] = None):
        if isinstance(data, Path):
            self.data = np.load(data, mmap_mode='r')
        else:
            self.data = data
        if self.data.ndim != 3 or self.data.shape[1:] != (Config.SEQ_LEN, Config.N_FEATURES):
            raise ValueError(
                f"Expected sequence array shape (n, {Config.SEQ_LEN}, {Config.N_FEATURES}), "
                f"got {self.data.shape}"
            )
        if labels is not None and len(labels) != len(self.data):
            raise ValueError(f"Label length mismatch: X={len(self.data)}, y={len(labels)}")
        
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)
        self.labels = labels

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

            recon_x, mu, logvar, _ = model(x)
            loss, recon, kl = vae_loss(recon_x, x, mu, logvar, beta)

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

            errors = torch.mean((x - recon_x) ** 2, dim=[1, 2])
            all_errors.extend(errors.cpu().numpy())
            all_labels.extend(y.numpy())

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


def get_beta(epoch, max_epochs=20, beta_max=0.001):
    """Cyclical KL annealing from 0 to beta_max within each cycle."""
    cycle_length = max(max_epochs / Config.KL_CYCLES, 1)
    cycle_progress = (epoch % cycle_length) / cycle_length
    warmup_progress = min(cycle_progress / Config.KL_WARMUP_RATIO, 1.0)
    return beta_max * warmup_progress

# --- TRAINING ---
def train():
    print(f"Starting training on {Config.DEVICE}")
    print(f"+ Sequence contract: {Config.SEQ_LEN} time steps x {Config.N_FEATURES} features")
    
    # Setup Data
    mean, std = get_normalization_stats(Config.X_TRAIN)
    train_dataset = NumpyDataset(Config.X_TRAIN, mean, std)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.DEVICE.type == "cuda",
        drop_last=True,
    )
    
    X_val = np.load(Config.X_VAL)
    y_val = np.load(Config.Y_VAL)
    val_dataset = NumpyDataset(X_val, mean, std, labels=y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.DEVICE.type == "cuda",
    )
    
    print(f"+ Train dataset: {len(train_dataset):,} samples")
    print(f"+ Val dataset: {len(val_dataset):,} samples")
    print(f"  Anomalies: {np.sum(y_val):,}")
    print(f"  Normal: {len(y_val) - np.sum(y_val):,}")

    # Setup Model
    model = LSTM_VAE(
        input_dim=Config.N_FEATURES, 
        hidden_dim=64, 
        embedding_dim=32
    ).to(Config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    Config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    

    # Training Loop
    best_val_loss = float('inf')
    best_val_auc = 0.0
    patience_counter = 0
    best_threshold = None

    for epoch in range(Config.EPOCHS):
        model.train()
        start_time = time.time()
        metrics = {'loss': 0.0, 'recon': 0.0, 'kl': 0.0}

        current_beta = get_beta(epoch, Config.EPOCHS, Config.BETA)

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(Config.DEVICE)
            
            # Optimization
            optimizer.zero_grad()
            recon_x, mu, logvar, _ = model(x)
            loss, recon, kl = vae_loss(recon_x, x, mu, logvar, beta=current_beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            # Tracking
            metrics['loss'] += loss.item()
            metrics['recon'] += recon.item()
            metrics['kl'] += kl.item()
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} (Recon: {recon.item():.4f} KL: {kl.item():.4f})")

        # Validation
        val_metrics = validate(model, val_loader, Config.DEVICE, current_beta)
        threshold_info = find_optimal_threshold(val_metrics['errors'], val_metrics['labels'])

        val_auc = roc_auc_score(val_metrics['labels'], val_metrics['errors'])

        # Epoch Summary
        avg_loss = metrics['loss'] / len(train_loader)
        duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | {duration:.1f}s | Beta: {current_beta:.6f}")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f} (Recon: {val_metrics['recon']:.4f}, KL: {val_metrics['kl']:.4f})")
        print(f"  Val AUROC: {val_auc:.4f}")
        print(f"  Val Threshold: {threshold_info['threshold']:.4f} (G-Mean: {threshold_info['gmean']:.4f})")

        scheduler.step(val_auc)

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_threshold = threshold_info
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_auc': val_auc,
                'threshold': threshold_info['threshold'],
            }, Config.BEST_MODEL_PATH)

            print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        print()

    val_metrics_data = {
        'best_val_loss': float(best_val_loss),
        'best_threshold': best_threshold,
        'note': 'Threshold selected on VALIDATION set, will be applied to TEST set'
    }

    with open(Config.VAL_METRICS_PATH, 'w') as f:
        json.dump(val_metrics_data, f, indent=2)

if __name__ == "__main__":
    train()
