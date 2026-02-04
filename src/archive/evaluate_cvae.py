import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from pathlib import Path

# --- CONFIG ---
BASE_DIR = Path(".")
PATHS = {
    "parquet": BASE_DIR / "data/processed/multimodal_features.parquet",
    "static_csv": BASE_DIR / "data/processed/static_features.csv",
    "stats": BASE_DIR / "data/processed/anomaly_detection/normalization_stats.json",
    "model": BASE_DIR / "models/lstm_cvae/best_checkpoint.pt"
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 96
HIDDEN_DIM = 64
EMBEDDING_DIM = 32

# --- MODEL CLASS (Fixed to match Training Script) ---
class LSTM_CVAE(nn.Module):
    def __init__(self, dyn_dim, stat_dim, seq_len, embed_dim, hidden_dim):
        super(LSTM_CVAE, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder_lstm = nn.LSTM(dyn_dim + stat_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, embed_dim)
        self.fc_logvar = nn.Linear(hidden_dim, embed_dim)
        
        # Decoder
        # NOTE: This name must match train_lstm_cvae.py exactly!
        self.fc_decoder_init = nn.Linear(embed_dim + stat_dim, hidden_dim) 
        
        self.decoder_lstm = nn.LSTM(stat_dim + embed_dim, hidden_dim, batch_first=True)
        self.fc_output = nn.Linear(hidden_dim, dyn_dim)

    def encode(self, dyn, stat):
        stat_expanded = stat.unsqueeze(1).repeat(1, self.seq_len, 1)
        combined = torch.cat([dyn, stat_expanded], dim=2)
        _, (hidden, _) = self.encoder_lstm(combined)
        hidden = hidden.squeeze(0)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, stat):
        # Prepare Decoder Input
        z_expanded = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        stat_expanded = stat.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoder_input = torch.cat([z_expanded, stat_expanded], dim=2)
        
        out, _ = self.decoder_lstm(decoder_input)
        return self.fc_output(out)

    def forward(self, dyn, stat):
        mu, logvar = self.encode(dyn, stat)
        z = self.reparameterize(mu, logvar)
        recon_dyn = self.decode(z, stat)
        return recon_dyn, mu, logvar

def evaluate():
    print("Loading Data...")
    df = pd.read_parquet(PATHS['parquet'])
    static_df = pd.read_csv(PATHS['static_csv']).set_index('senior_id')
    
    # Handle Categorical Encoding in Static DF (Must match training logic)
    if static_df.select_dtypes(include=['object']).shape[1] > 0:
        static_df = pd.get_dummies(static_df, drop_first=True)

    # Load Stats
    with open(PATHS['stats'], 'r') as f:
        stats = json.load(f)
        mean = torch.tensor(stats['mean']).to(DEVICE)
        std = torch.tensor(stats['std']).to(DEVICE)

    # Dynamic Columns (Must match training list)
    dyn_cols = [
        'temperature', 'heartrate', 'sbp', 'dbp', 'pulse_pressure', 
        'steps', 'saturation', 'hr_volatility', 'bp_trend', 
        'steps_rolling_sum_6h', 'shock_index', 'recent_event_burden'
    ]

    # Initialize Model
    stat_dim = static_df.shape[1]
    dyn_dim = len(dyn_cols)
    print(f"Model Dimensions -> Dyn: {dyn_dim}, Stat: {stat_dim}")
    
    model = LSTM_CVAE(dyn_dim, stat_dim, SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load(PATHS['model'], map_location=DEVICE, weights_only=False))
    model.eval()

    # --- PREPARE TEST DATA ---
    print("Preparing Test Samples...")
    
    # Filter for relevant seniors only
    valid_seniors = [sid for sid in df['senior_id'].unique() if sid in static_df.index]
    df = df[df['senior_id'].isin(valid_seniors)]
    
    # Identify indices
    valid_ends = np.arange(SEQ_LEN-1, len(df))
    labels = df['label_3'].values[valid_ends]
    
    # Indices of Acute Events
    acute_indices = valid_ends[labels == 1]
    
    # Indices of Normal Events (Sampled to match Acute count)
    normal_indices_pool = valid_ends[labels == 0]
    
    if len(normal_indices_pool) > len(acute_indices):
        normal_indices = np.random.choice(normal_indices_pool, len(acute_indices), replace=False)
    else:
        normal_indices = normal_indices_pool
        
    test_indices = np.concatenate([acute_indices, normal_indices])
    print(f"Test Set: {len(acute_indices)} Acute vs {len(normal_indices)} Normal")

    # --- INFERENCE ---
    y_true = []
    y_scores = []
    
    print(f"Running Inference on {len(test_indices)} windows...")
    with torch.no_grad():
        for i, end_idx in enumerate(test_indices):
            start_idx = end_idx - SEQ_LEN + 1
            
            # 1. Get Dynamic Window
            dyn_win = df.iloc[start_idx : end_idx+1][dyn_cols].values
            dyn_t = torch.tensor(dyn_win, dtype=torch.float32).to(DEVICE)
            
            # Normalize
            dyn_t = torch.where(torch.isnan(dyn_t), mean, dyn_t)
            dyn_t = (dyn_t - mean) / std
            dyn_t = dyn_t.unsqueeze(0) # (1, Seq, Feat)
            
            # 2. Get Static Vector
            sid = df.iloc[end_idx]['senior_id']
            stat_vec = static_df.loc[sid].values
            stat_t = torch.tensor(stat_vec, dtype=torch.float32).to(DEVICE).unsqueeze(0) # (1, Stat)
            
            # 3. Forward Pass
            recon, _, _ = model(dyn_t, stat_t)
            
            # 4. Error (MSE)
            mse = torch.mean((dyn_t - recon) ** 2).item()
            
            y_scores.append(mse)
            y_true.append(df.iloc[end_idx]['label_3'])
            
            if i % 1000 == 0:
                print(f"Tested {i}/{len(test_indices)}...")

    # --- METRICS ---
    auc = roc_auc_score(y_true, y_scores)
    print("\n" + "="*30)
    print(f"CVAE AUROC: {auc:.4f}")
    print("="*30 + "\n")
    
    # Threshold Report
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    best_thresh = thresholds[ix]
    
    y_pred = [1 if s > best_thresh else 0 for s in y_scores]
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    
    # Save ROC Plot
    plt.figure()
    plt.plot(fpr, tpr, label=f'CVAE (AUC={auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.title("ROC Curve: Conditional VAE")
    plt.legend()
    plt.savefig("models/lstm_cvae/roc_curve.png")
    print("Saved models/lstm_cvae/roc_curve.png")

if __name__ == "__main__":
    evaluate()