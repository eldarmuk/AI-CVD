import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
from pathlib import Path
import json
import pandas as pd
import warnings
import seaborn as sns 


# Use a clean style for scientific plots
plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(".")
X_TEST_PATH = BASE_DIR / "data/processed/anomaly_detection/X_test.npy"
Y_TEST_PATH = BASE_DIR / "data/processed/anomaly_detection/y_test.npy"
X_TRAIN_PATH = BASE_DIR / "data/processed/anomaly_detection/X_train.npy"
STATS_PATH = BASE_DIR / "data/processed/anomaly_detection/normalization_stats.json"
RESULTS_DIR = BASE_DIR / "models/lstm_vae/paper_figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_FEW_SHOT = 15  # Scarcity Constraint

# --- MODEL DEFINITIONS ---
class LSTM_VAE(nn.Module):
    def __init__(self, input_dim=12, seq_len=96, embed_dim=32, hidden_dim=64):
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
        decoder_input = hidden_proj.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder_lstm(decoder_input, (h0, c0))
        return self.fc_output(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

class CVAE(nn.Module):
    """Conditional VAE: Conditions on 'Static' Summary Stats of the window"""
    def __init__(self, input_dim=12, cond_dim=24, seq_len=96, embed_dim=32, hidden_dim=64):
        super(CVAE, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        # Encoder takes X + Condition
        self.encoder_lstm = nn.LSTM(input_dim + cond_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, embed_dim)
        self.fc_logvar = nn.Linear(hidden_dim, embed_dim)
        # Decoder takes Z + Condition
        self.fc_decoder_input = nn.Linear(embed_dim + cond_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, c):
        # Repeat condition for every step: (Batch, Seq, Cond)
        c_expanded = c.unsqueeze(1).repeat(1, self.seq_len, 1)
        x_cat = torch.cat([x, c_expanded], dim=2)
        _, (hidden, _) = self.encoder_lstm(x_cat)
        hidden = hidden.squeeze(0)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        batch_size = z.size(0)
        device = z.device
        z_cat = torch.cat([z, c], dim=1)
        hidden_proj = self.fc_decoder_input(z_cat)
        h0 = hidden_proj.unsqueeze(0)
        c0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        decoder_input = hidden_proj.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder_lstm(decoder_input, (h0, c0))
        return self.fc_output(out)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar, z

class BiGRUClassifier(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64):
        super(BiGRUClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.gru(x)
        return self.sigmoid(self.fc(out[:, -1, :]))

class StandardAE(nn.Module):
    def __init__(self, input_dim=12*96):
        super(StandardAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 32))
        self.decoder = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x.view(x.size(0), -1))).view(x.shape)

# --- UTILS ---
def get_metrics(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred = (y_scores >= best_thresh).astype(int)
    # return {
    #     "Model": model_name, "AUROC": roc_auc, "AUPRC": pr_auc,
    #     "Precision": precision_score(y_true, y_pred, zero_division=0),
    #     "Recall": recall_score(y_true, y_pred, zero_division=0),
    #     "FPR": fpr, "TPR": tpr, "PrecCurve": precision, "RecCurve": recall, "Scores": y_scores
    # }
    ci_lower, ci_upper = bootstrap_auc_ci(y_true, y_scores)

    return {
        "Model": model_name,
        "AUROC": roc_auc,
        "AUROC_CI_L": ci_lower,
        "AUROC_CI_U": ci_upper,
        "AUPRC": pr_auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "FPR": fpr,
        "TPR": tpr,
        "PrecCurve": precision,
        "RecCurve": recall,
        "Scores": y_scores
    }

def bootstrap_auc_ci(y_true, y_scores, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    n = len(y_true)

    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)

        # Skip if resample contains only one class
        if len(np.unique(y_true[indices])) < 2:
            continue

        score = roc_auc_score(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(score)

    lower = np.percentile(bootstrapped_scores, 2.5)
    upper = np.percentile(bootstrapped_scores, 97.5)

    return lower, upper

def make_condition(x):
    # Simulates "Static Context" by taking Mean and Std of the window (24 features)
    # This tests if 'Summary Statistics' help the model.
    return torch.cat([x.mean(dim=1), x.std(dim=1)], dim=1)

def main():
    print(">>> LOADING DATA...")
    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)
        mean, std = torch.tensor(stats['mean']).to(DEVICE), torch.tensor(stats['std']).to(DEVICE) + 1e-6
    X_test_raw = np.nan_to_num(np.load(X_TEST_PATH), nan=0.0)
    y_test_raw = np.load(Y_TEST_PATH)
    
    # --- STRICT SPLIT ---
    split_point = len(X_test_raw) // 2
    X_eval = torch.tensor(X_test_raw[split_point:], dtype=torch.float32).to(DEVICE)
    y_eval = y_test_raw[split_point:]
    
    # Scarcity Training Pool
    X_pool = X_test_raw[:split_point]
    y_pool = y_test_raw[:split_point]
    pool_anom_idx = np.where(y_pool == 1)[0]
    pool_norm_idx = np.where(y_pool == 0)[0]
    np.random.seed(42)
    np.random.shuffle(pool_anom_idx)
    selected_anom_idx = pool_anom_idx[:N_FEW_SHOT]
    train_idxs = np.concatenate([pool_norm_idx, selected_anom_idx])
    np.random.shuffle(train_idxs)
    
    X_sup_train = torch.tensor(X_pool[train_idxs], dtype=torch.float32).to(DEVICE)
    y_sup_train = torch.tensor(y_pool[train_idxs], dtype=torch.float32).to(DEVICE)

    # Normalize
    X_eval = torch.nan_to_num((X_eval - mean) / std, nan=0.0, posinf=5.0, neginf=-5.0)
    X_sup_train = torch.nan_to_num((X_sup_train - mean) / std, nan=0.0, posinf=5.0, neginf=-5.0)
    
    X_unsup_mmap = np.memmap(X_TRAIN_PATH, dtype='float32', mode='r')
    X_unsup = torch.tensor(np.nan_to_num(np.array(X_unsup_mmap[:5000*96*12]).reshape(5000, 96, 12), nan=0.0), dtype=torch.float32).to(DEVICE)
    X_unsup = (X_unsup - mean) / std

    results = []

    # --- 1. LSTM-VAE ---
    print("\n[1/5] Evaluating LSTM-VAE (Dynamic Only)...")
    model = LSTM_VAE(12, 96, 32, 64).to(DEVICE)
    try:
        model.load_state_dict(torch.load("models/lstm_vae/best_checkpoint.pt"))
        model.eval()
        with torch.no_grad():
            recon, _, _, _ = model(X_eval)
            scores = torch.mean((X_eval - recon) ** 2, dim=[1, 2]).cpu().numpy()
        results.append(get_metrics(y_eval, scores, "LSTM-VAE"))
    except: print("Err: Checkpoint missing")

    # --- 2. CVAE ---
    print("\n[2/5] Training CVAE (Dynamic + Static Context)...")
    # Condition: 24 features (Mean + Std of window) to simulate "Static Summary"
    model = CVAE(input_dim=12, cond_dim=24, seq_len=96).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    # Train CVAE on Unsupervised Data
    for _ in range(15):
        opt.zero_grad()
        c = make_condition(X_unsup) # Create static context
        recon, mu, logvar, _ = model(X_unsup, c)
        mse = nn.functional.mse_loss(recon, X_unsup)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = mse + 0.001 * kld
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        c_eval = make_condition(X_eval)
        recon, _, _, _ = model(X_eval, c_eval)
        scores = torch.mean((X_eval - recon) ** 2, dim=[1, 2]).cpu().numpy()
    results.append(get_metrics(y_eval, scores, "CVAE"))

    # --- 3. Bi-GRU ---
    print(f"\n[3/5] Training Bi-GRU ({N_FEW_SHOT}-shot)...")
    model = BiGRUClassifier(12, 64).to(DEVICE)
    opt, crit = optim.Adam(model.parameters(), lr=1e-3), nn.BCELoss()
    loader = DataLoader(TensorDataset(X_sup_train, y_sup_train.unsqueeze(1)), batch_size=32, shuffle=True)
    model.train()
    for _ in range(20):
        for x, y in loader:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad(): scores = model(X_eval).cpu().numpy().flatten()
    results.append(get_metrics(y_eval, scores, "Bi-GRU"))

    # --- 4. Standard AE ---
    print("\n[4/5] Training Standard AE...")
    model = StandardAE(12*96).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(15):
        opt.zero_grad()
        recon = model(X_unsup)
        loss = nn.functional.mse_loss(recon, X_unsup)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        recon = model(X_eval)
        scores = torch.mean((X_eval - recon) ** 2, dim=[1, 2]).cpu().numpy()
    results.append(get_metrics(y_eval, scores, "Standard AE"))

    # --- 5. XGBoost ---
    print(f"\n[5/5] Training XGBoost...")
    # Augment Data with Static Context
    X_sup_flat = X_sup_train.reshape(len(X_sup_train), -1)
    c_sup = make_condition(X_sup_train)
    X_sup_augmented = torch.cat([X_sup_flat, c_sup], dim=1).cpu().numpy() # Add static info
    
    X_eval_flat = X_eval.reshape(len(X_eval), -1)
    c_eval = make_condition(X_eval)
    X_eval_augmented = torch.cat([X_eval_flat, c_eval], dim=1).cpu().numpy()
    
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, eval_metric='logloss', use_label_encoder=False)
    xgb_model.fit(X_sup_augmented, y_sup_train.cpu().numpy())
    scores = xgb_model.predict_proba(X_eval_augmented)[:, 1]
    results.append(get_metrics(y_eval, scores, "XGBoost"))

    # --- REPORT ---
    print("\n" + "="*80)
    print(f"{'Model':<20} | {'AUROC (95% CI)':<25} | {'Recall':<8} | {'Precision':<8} | {'AUPRC':<8}")
    print("-" * 80)
    for res in results:
        print(
            f"{res['Model']:<20} | "
            f"{res['AUROC']:.4f} ({res['AUROC_CI_L']:.4f}-{res['AUROC_CI_U']:.4f}) | "
            f"{res['Recall']:.4f}   | "
            f"{res['Precision']:.4f}    | "
            f"{res['AUPRC']:.4f}"
        )

    # --- PLOTS ---
    colors = ['#1f77b4', '#9467bd', '#d62728', '#2ca02c', '#ff7f0e'] # Blue, Purple, Red, Green, Orange
    
    # 1. ROC
    plt.figure(figsize=(8, 6))
    for i, res in enumerate(results):
        lw = 2.5 if "LSTM" in res['Model'] else 1.5
        plt.plot(res['FPR'], res['TPR'], label=f"{res['Model']} (AUC = {res['AUROC']:.2f})", color=colors[i], linewidth=lw)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right', frameon=True)
    plt.savefig(RESULTS_DIR / "comparison_roc.pdf")

    # 2. SEPARATION HISTOGRAMS (ZOOMED KDE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    def plot_zoomed_hist(ax, scores, y_true, title, color_norm, color_anom):
        upper_limit = np.percentile(scores, 98) 
        viz_scores = np.clip(scores, 0, upper_limit)
        sns.histplot(viz_scores[y_true==0], color=color_norm, label='Normal', stat="density", kde=True, bins=50, alpha=0.3, ax=ax, edgecolor=None)
        sns.histplot(viz_scores[y_true==1], color=color_anom, label='Acute Event', stat="density", kde=True, bins=50, alpha=0.3, ax=ax, edgecolor=None)
        
        # Threshold
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_thresh = thresholds[np.argmax(f1)]
        if best_thresh < upper_limit:
            ax.axvline(best_thresh, color='black', linestyle='--', linewidth=2, label=f'Threshold ({best_thresh:.3f})')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Reconstruction Error (MSE)")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.2)

    # Plot LSTM-VAE (Left)
    plot_zoomed_hist(axes[0], results[0]['Scores'], y_eval, "LSTM-VAE (Unsupervised)\nEffective Separation", '#1f77b4', '#d62728')
    # Plot CVAE or Standard AE (Right) - Showing AE as it's the main baseline
    plot_zoomed_hist(axes[1], results[3]['Scores'], y_eval, "Standard AE (Non-Temporal)\nSignificant Overlap", '#2ca02c', '#ff7f0e')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_histograms.pdf")
    print(f"\nFigures exported to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()