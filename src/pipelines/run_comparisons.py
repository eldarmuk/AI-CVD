"""
FIXED Comparison Script for Paper (No Pickle Errors)

Changes:
1. All np.load() calls use allow_pickle=True to avoid Windows pickle errors
2. Removed LaTeX table printing (keeping only console output and PDFs)
3. All models use proper train/val/test splits
4. Thresholds selected on validation, applied to test
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
from pathlib import Path
import json
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(".")
X_TRAIN_PATH = BASE_DIR / "data/processed/anomaly_detection/X_train.npy"
X_VAL_PATH = BASE_DIR / "data/processed/anomaly_detection/X_val.npy"
Y_VAL_PATH = BASE_DIR / "data/processed/anomaly_detection/y_val.npy"
X_TEST_PATH = BASE_DIR / "data/processed/anomaly_detection/X_test.npy"
Y_TEST_PATH = BASE_DIR / "data/processed/anomaly_detection/y_test.npy"
STATS_PATH = BASE_DIR / "data/processed/anomaly_detection/normalization_stats.json"
LSTM_VAE_CHECKPOINT = BASE_DIR / "models/lstm_vae/best_checkpoint.pt"
VAL_METRICS_PATH = BASE_DIR / "models/lstm_vae/validation_metrics.json"
RESULTS_DIR = BASE_DIR / "models/lstm_vae/paper_figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- MODEL DEFINITIONS ---
class LSTM_VAE(nn.Module):
    def __init__(self, input_dim=11, seq_len=96, embed_dim=32, hidden_dim=64):
        super(LSTM_VAE, self).__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, embed_dim)
        self.fc_logvar = nn.Linear(hidden_dim, embed_dim)
        self.fc_decoder_input = nn.Linear(embed_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_output = nn.Linear(hidden_dim, input_dim)
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def encode(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        return self.fc_mu(hidden.squeeze(0)), self.fc_logvar(hidden.squeeze(0))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        batch_size, device = z.size(0), z.device
        hidden_proj = self.fc_decoder_input(z)
        h0 = hidden_proj.unsqueeze(0)
        c0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        decoder_input = hidden_proj.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder_lstm(decoder_input, (h0, c0))
        return self.fc_output(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


class StandardAE(nn.Module):
    """Non-temporal baseline: Standard dense autoencoder."""
    def __init__(self, seq_len=96, input_dim=11):
        super(StandardAE, self).__init__()
        flat_dim = seq_len * input_dim
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, flat_dim)
        )
        self.seq_len = seq_len
        self.input_dim = input_dim
    
    def forward(self, x):
        batch_size = x.size(0)
        flat = x.view(batch_size, -1)
        encoded = self.encoder(flat)
        decoded = self.decoder(encoded)
        return decoded.view(batch_size, self.seq_len, self.input_dim)

# --- UTILS ---
def load_and_normalize_data(data_path, labels_path=None):
    """
    Load and normalize data using EXACT SAME normalization as training/evaluation.
    
    CRITICAL: Must match the normalization in training script exactly!
    """
    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)
        mean = torch.tensor(stats['mean'], dtype=torch.float32).to(DEVICE)
        std = torch.tensor(stats['std'], dtype=torch.float32).to(DEVICE)  # NO +1e-6!

    # Detect file type
    with open(data_path, "rb") as f:
        magic = f.read(6)

    if magic.startswith(b"\x93NUMPY"):
        X = np.load(data_path, allow_pickle=True)
    else:
        # Raw memmap (train)
        seq_len = 96
        input_dim = len(stats['mean'])
        raw = np.memmap(data_path, dtype=np.float32, mode='r')
        n_samples = raw.size // (seq_len * input_dim)
        X = raw.reshape(n_samples, seq_len, input_dim)

    # CRITICAL: Use SAME normalization as training
    X_tensor = torch.from_numpy(X).float().to(DEVICE)
    
    # Fill NaNs with mean (SAME as training)
    nan_mask = torch.isnan(X_tensor)
    mean_expanded = mean.view(1, 1, -1).expand_as(X_tensor)
    X_tensor = torch.where(nan_mask, mean_expanded, X_tensor)
    
    # Normalize
    std_expanded = std.view(1, 1, -1).expand_as(X_tensor)
    X_tensor = (X_tensor - mean_expanded) / std_expanded
    
    # Clean up
    X_tensor = torch.nan_to_num(X_tensor, nan=0.0)

    if labels_path:
        y = np.load(labels_path, allow_pickle=True)
        return X_tensor, y

    return X_tensor

def bootstrap_auc_ci(y_true, y_scores, n_bootstraps=1000, seed=42):
    """Compute bootstrap confidence interval for AUROC."""
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    n = len(y_true)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(score)

    return np.percentile(bootstrapped_scores, 2.5), np.percentile(bootstrapped_scores, 97.5)


def calculate_ppv_at_prevalence(sensitivity, specificity, prevalence):
    """Calculate Positive Predictive Value at true prevalence."""
    tp = sensitivity * prevalence
    fp = (1 - specificity) * (1 - prevalence)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def get_metrics_comprehensive(y_true, y_scores, threshold, model_name, true_prevalence=0.0001):
    """
    Compute comprehensive metrics including realistic prevalence estimates.
    
    Args:
        threshold: Detection threshold (from validation set)
        true_prevalence: Realistic event prevalence (default 0.01% = 0.0001)
    """
    # ROC metrics
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    ci_lower, ci_upper = bootstrap_auc_ci(y_true, y_scores)
    
    # Threshold-based metrics (at balanced test set)
    y_pred = (y_scores >= threshold).astype(int)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision_balanced = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Realistic prevalence metrics
    ppv_realistic = calculate_ppv_at_prevalence(sensitivity, specificity, true_prevalence)
    
    # AUPRC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall_curve, precision_curve)
    
    return {
        "Model": model_name,
        "AUROC": roc_auc,
        "CI_Lower": ci_lower,
        "CI_Upper": ci_upper,
        "AUPRC": auprc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision_Balanced": precision_balanced,
        "PPV_Realistic": ppv_realistic,
        "FPR": fpr,
        "TPR": tpr,
        "Scores": y_scores,
        "Threshold": threshold
    }


def train_unsupervised_baseline(X_train, X_val, y_val, model_class, epochs=20, lr=1e-3):
    """Train unsupervised model (AE) with validation monitoring."""
    model = model_class().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_train_subset = X_train 
    
    train_loader = DataLoader(X_train_subset, batch_size=256, shuffle=True)
    
    best_val_auc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        for x_batch in train_loader:
            optimizer.zero_grad()
            recon = model(x_batch)
            loss = nn.functional.mse_loss(recon, x_batch)
            loss.backward()
            optimizer.step()
        
        # Validation (compute reconstruction error)
        model.eval()
        with torch.no_grad():
            val_recon = model(X_val)
            val_errors = torch.mean((X_val - val_recon) ** 2, dim=[1, 2]).cpu().numpy()
            val_auc = roc_auc_score(y_val, val_errors)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model


def main():
    print("="*80)
    print("COMPARISON STUDY: LSTM-VAE vs Baselines")
    print("="*80)
    
    # Load data
    print("\n>>> Loading data...")
    X_train = load_and_normalize_data(X_TRAIN_PATH)
    X_val, y_val = load_and_normalize_data(X_VAL_PATH, Y_VAL_PATH)
    X_test, y_test = load_and_normalize_data(X_TEST_PATH, Y_TEST_PATH)
    
    print(f"Train: {X_train.shape}")
    print(f"Val: {X_val.shape} (Anomalies: {np.sum(y_val)})")
    print(f"Test: {X_test.shape} (Anomalies: {np.sum(y_test)})")
    
    # Load validation threshold (selected on validation set)
    with open(VAL_METRICS_PATH, 'r') as f:
        val_metrics = json.load(f)
        threshold_lstm_vae = val_metrics['best_threshold']['threshold']
    
    print(f"\nLSTM-VAE threshold (from validation): {threshold_lstm_vae:.4f}")
    
    results = []
    
    # --- 1. LSTM-VAE (Unsupervised) ---
    print("\n[1/2] Evaluating LSTM-VAE...")
    try:
        model = LSTM_VAE(input_dim=X_test.shape[-1], seq_len=96, embed_dim=32, hidden_dim=64).to(DEVICE)
        checkpoint = torch.load(LSTM_VAE_CHECKPOINT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        with torch.no_grad():
            recon, _, _, _ = model(X_test)
            scores = torch.mean((X_test - recon) ** 2, dim=[1, 2]).cpu().numpy()
        
        results.append(get_metrics_comprehensive(
            y_test, scores, threshold_lstm_vae, "LSTM-VAE", true_prevalence=0.0001
        ))
        print("✓ LSTM-VAE evaluated")
    except Exception as e:
        print(f"✗ LSTM-VAE failed: {e}")
    
    # --- 2. Standard AE (Unsupervised, Non-Temporal) ---
    print("\n[2/2] Training Standard AE...")
    try:
        model = train_unsupervised_baseline(
            X_train, X_val, y_val, 
            lambda: StandardAE(seq_len=96, input_dim=X_test.shape[-1]),
            epochs=20
        )
        
        model.eval()
        with torch.no_grad():
            recon = model(X_test)
            scores = torch.mean((X_test - recon) ** 2, dim=[1, 2]).cpu().numpy()
        
        # Find optimal threshold on validation
        with torch.no_grad():
            val_recon = model(X_val)
            val_errors = torch.mean((X_val - val_recon) ** 2, dim=[1, 2]).cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_val, val_errors)
        gmeans = np.sqrt(tpr * (1 - fpr))
        threshold_ae = thresholds[np.argmax(gmeans)]
        
        results.append(get_metrics_comprehensive(
            y_test, scores, threshold_ae, "Standard AE", true_prevalence=0.0001
        ))
        print("✓ Standard AE trained and evaluated")
    except Exception as e:
        print(f"✗ Standard AE failed: {e}")
    
    # --- REPORT RESULTS ---
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    print(f"{'Model':<25} | {'AUROC (95% CI)':<25} | {'Sens':<6} | {'Spec':<6} | {'Prec@Bal':<9} | {'PPV@0.01%':<10}")
    print("-"*100)
    
    for res in results:
        print(
            f"{res['Model']:<25} | "
            f"{res['AUROC']:.3f} ({res['CI_Lower']:.3f}-{res['CI_Upper']:.3f}) | "
            f"{res['Sensitivity']:.3f}  | "
            f"{res['Specificity']:.3f}  | "
            f"{res['Precision_Balanced']:.3f}     | "
            f"{res['PPV_Realistic']:.4f}"
        )
    
    print("\nKEY:")
    print("  Sens = Sensitivity (Recall)")
    print("  Spec = Specificity")
    print("  Prec@Bal = Precision on balanced test set (50/50)")
    print("  PPV@0.01% = Positive Predictive Value at realistic prevalence (0.01%)")
    
    # --- PLOTS ---
    print("\n>>> Generating figures...")
    
    # 1. ROC Curve
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#2ca02c']
    
    for i, res in enumerate(results):
        lw = 2.5 if "LSTM" in res['Model'] else 1.5
        plt.plot(res['FPR'], res['TPR'], 
                label=f"{res['Model']} (AUC = {res['AUROC']:.2f})",
                color=colors[i], linewidth=lw)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('Receiver Operating Characteristic', fontsize=22, fontweight='bold')
    plt.legend(loc='lower right', frameon=True, fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_roc.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(RESULTS_DIR / "comparison_roc.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {RESULTS_DIR / 'comparison_roc.pdf'}")
    plt.close()
    
    # 2. Error Distribution Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    def plot_separation(ax, scores, y_true, title, threshold):
        normal_scores = scores[y_true == 0]
        anomaly_scores = scores[y_true == 1]
        
        # Use percentile for intelligent zooming
        p05 = np.percentile(normal_scores, 5)
        p95_normal = np.percentile(normal_scores, 95)
        p95_anomaly = np.percentile(anomaly_scores, 95)
        upper = max(p95_normal, p95_anomaly) * 1.2
        
        normal_clip = np.clip(normal_scores, p05, upper)
        anomaly_clip = np.clip(anomaly_scores, p05, upper)
        
        # Plot with enhanced visual separation
        sns.kdeplot(normal_clip, color='#2E86AB', shade=True, alpha=0.6, 
                   linewidth=2.5, label='Normal', ax=ax)
        sns.kdeplot(anomaly_clip, color='#A23B72', shade=True, alpha=0.6,
                   linewidth=2.5, label='Anomaly', ax=ax)
        
        # Add rugplot for actual data points (sampled for clarity)
        sample_idx_normal = np.random.choice(len(normal_clip), min(500, len(normal_clip)), replace=False)
        sample_idx_anomaly = np.random.choice(len(anomaly_clip), min(500, len(anomaly_clip)), replace=False)
        
        ax.scatter(normal_clip[sample_idx_normal], np.zeros(len(sample_idx_normal)) - 0.02,
                  alpha=0.3, s=10, color='#2E86AB', marker='|')
        ax.scatter(anomaly_clip[sample_idx_anomaly], np.zeros(len(sample_idx_anomaly)) - 0.02,
                  alpha=0.3, s=10, color='#A23B72', marker='|')
        
        # Threshold line with fill
        if p05 <= threshold <= upper:
            ax.axvline(threshold, color='#F18F01', linestyle='--', linewidth=3,
                      label=f'Threshold', zorder=10)
            
            # Add shaded regions for decision boundaries
            ax.axvspan(p05, threshold, alpha=0.1, color='#2E86AB', zorder=0)
            ax.axvspan(threshold, upper, alpha=0.1, color='#A23B72', zorder=0)
        
        ax.set_title(title, fontsize=24, fontweight='bold', pad=12)
        ax.set_xlabel("Reconstruction Error", fontsize=20)
        ax.set_ylabel("Density", fontsize=20)
        ax.legend(loc='upper right', fontsize=16, framealpha=0.95)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
        ax.set_xlim(p05, upper)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Plot LSTM-VAE and Standard AE
    if len(results) >= 2:
        plot_separation(axes[0], results[0]['Scores'], y_test,
                       "LSTM-VAE",
                       results[0]['Threshold'])
        plot_separation(axes[1], results[1]['Scores'], y_test,
                       "Standard AE",
                       results[1]['Threshold'])
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_distributions.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(RESULTS_DIR / "comparison_distributions.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {RESULTS_DIR / 'comparison_distributions.pdf'}")
    plt.close()
    
    # Save metrics to CSV for paper
    metrics_df = pd.DataFrame([{
        'Model': res['Model'],
        'AUROC': f"{res['AUROC']:.3f}",
        'CI_Lower': f"{res['CI_Lower']:.3f}",
        'CI_Upper': f"{res['CI_Upper']:.3f}",
        'Sensitivity': f"{res['Sensitivity']:.3f}",
        'Specificity': f"{res['Specificity']:.3f}",
        'Precision_Balanced': f"{res['Precision_Balanced']:.3f}",
        'PPV_Realistic': f"{res['PPV_Realistic']:.4f}",
        'AUPRC': f"{res['AUPRC']:.3f}"
    } for res in results])
    
    metrics_df.to_csv(RESULTS_DIR / "comparison_metrics.csv", index=False)
    print(f"✓ Saved: {RESULTS_DIR / 'comparison_metrics.csv'}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {RESULTS_DIR}/")
    print("  - comparison_roc.pdf")
    print("  - comparison_distributions.pdf")
    print("  - comparison_metrics.csv")


if __name__ == "__main__":
    main()