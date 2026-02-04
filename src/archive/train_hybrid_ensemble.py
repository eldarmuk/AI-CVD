import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- CONFIG ---
BASE_DIR = Path(".")
PATHS = {
    "parquet": BASE_DIR / "data/processed/multimodal_features.parquet",
    "static_csv": BASE_DIR / "data/processed/static_features.csv",
    "stats": BASE_DIR / "data/processed/anomaly_detection/normalization_stats.json",
    "model": BASE_DIR / "models/lstm_vae/best_checkpoint.pt"
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 96
N_FEATURES = 12 
HIDDEN_DIM = 64
EMBEDDING_DIM = 32

# --- MODEL DEFINITION ---
class LSTM_VAE(nn.Module):
    def __init__(self, input_dim=12, sequence_length=96, embedding_dim=32, hidden_dim=64):
        super(LSTM_VAE, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, embedding_dim)
        self.fc_logvar = nn.Linear(hidden_dim, embedding_dim)
        self.fc_decoder_input = nn.Linear(embedding_dim, hidden_dim)
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
        decoder_input = hidden_proj.unsqueeze(1).repeat(1, self.sequence_length, 1)
        out, _ = self.decoder_lstm(decoder_input, (h0, c0))
        return self.fc_output(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar # Standard 3-tuple return

def generate_hybrid_dataset(df, static_df, model, mean, std, feature_cols):
    print("Generating Hybrid Dataset (FULL POPULATION)...")
    
    # 1. Identify Population
    acute_seniors = df[df['label_3'] == 1]['senior_id'].unique()
    normal_seniors = df[~df['senior_id'].isin(acute_seniors)]['senior_id'].unique()
    
    # USE EVERYONE (No Downsampling)
    target_seniors = np.concatenate([acute_seniors, normal_seniors])
    print(f"Processing {len(target_seniors)} seniors ({len(acute_seniors)} Acute, {len(normal_seniors)} Normal)")
    
    # Index Static DF
    static_df = static_df.set_index('senior_id')
    
    # Context Cols
    context_cols = ['hour', 'is_night', 'day_of_week', 'recent_event_burden', 'shock_index']
    context_cols = [c for c in context_cols if c in df.columns]

    dataset_records = []
    model.eval()
    
    with torch.no_grad():
        for i, sid in enumerate(target_seniors):
            sub = df[df['senior_id'] == sid].reset_index(drop=True)
            if len(sub) < SEQ_LEN: 
                continue
            
            # --- VAE Inference ---
            dyn_data = sub[feature_cols].values
            dyn_t = torch.tensor(dyn_data, dtype=torch.float32).to(DEVICE)
            dyn_t = torch.where(torch.isnan(dyn_t), mean, dyn_t)
            dyn_t = (dyn_t - mean) / std
            
            # Windows
            valid_end_indices = np.arange(SEQ_LEN-1, len(sub))
            labels = sub['label_3'].values[valid_end_indices]
            
            # SAMPLING STRATEGY (Updated)
            # 1. Take ALL Acute windows
            # 2. Take a limited sample of Normal windows per senior (e.g., 50) 
            #    Reason: We have 8000 seniors. If we take all 1000 windows per senior, 
            #    we will have 8 million rows. That's too slow for today. 
            #    50 windows * 8000 seniors = 400,000 rows (Manageable & Diverse).
            
            anom_idxs = valid_end_indices[labels == 1]
            norm_idxs = valid_end_indices[labels == 0]
            
            # Take ALL anomalies for this senior
            indices_to_take = list(anom_idxs)
            
            # Take random sample of normals (e.g. 50 per senior)
            if len(norm_idxs) > 50:
                chosen_norm = np.random.choice(norm_idxs, 50, replace=False)
                indices_to_take.extend(chosen_norm)
            else:
                indices_to_take.extend(norm_idxs)
                
            final_indices = np.array(indices_to_take)
            final_indices.sort()
            
            if len(final_indices) == 0:
                continue
            
            # Batch Inference
            batch_wins = []
            for end_idx in final_indices:
                start = end_idx - SEQ_LEN + 1
                batch_wins.append(dyn_t[start : end_idx+1])
            
            batch_wins = torch.stack(batch_wins)
            
            recon, _, _ = model(batch_wins)
            errors = torch.mean((batch_wins - recon) ** 2, dim=[1, 2]).cpu().numpy()
            
            # --- Merge Data ---
            try:
                static_vals = static_df.loc[sid].to_dict()
            except KeyError:
                continue
            
            for k, end_idx in enumerate(final_indices):
                context_vals = sub.loc[end_idx, context_cols].to_dict()
                
                record = {
                    'senior_id': sid,
                    'vae_error': float(errors[k]),
                    'label': int(sub.loc[end_idx, 'label_3']),
                    **static_vals,
                    **context_vals
                }
                dataset_records.append(record)
                
            if i % 500 == 0:
                print(f"Processed {i}/{len(target_seniors)} seniors...")

    return pd.DataFrame(dataset_records)
    
def main():
    # Load Resources
    with open(PATHS['stats'], 'r') as f:
        stats = json.load(f)
        mean = torch.tensor(stats['mean']).to(DEVICE)
        std = torch.tensor(stats['std']).to(DEVICE)
        
    model = LSTM_VAE(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    model.load_state_dict(torch.load(PATHS['model'], map_location=DEVICE, weights_only=False))
    
    df = pd.read_parquet(PATHS['parquet'])
    static_df = pd.read_csv(PATHS['static_csv'])
    
    dyn_cols = [
        'temperature', 'heartrate', 'sbp', 'dbp', 'pulse_pressure', 
        'steps', 'saturation', 'hr_volatility', 'bp_trend', 
        'steps_rolling_sum_6h', 'shock_index', 'recent_event_burden'
    ]
    
# 1. Generate Dataset (FULL)
    hybrid_df = generate_hybrid_dataset(df, static_df, model, mean, std, dyn_cols)
    print(f"Hybrid Shape: {hybrid_df.shape}")
    
    hybrid_df = hybrid_df.dropna()
    hybrid_df = pd.get_dummies(hybrid_df, columns=['gender'], drop_first=True)
    
    # 2. PATIENT-LEVEL SPLIT
    print("Splitting by Senior ID...")
    all_seniors = hybrid_df['senior_id'].unique()
    train_sids, test_sids = train_test_split(all_seniors, test_size=0.2, random_state=42)
    
    train_df = hybrid_df[hybrid_df['senior_id'].isin(train_sids)]
    test_df = hybrid_df[hybrid_df['senior_id'].isin(test_sids)]
    
    X_train = train_df.drop(['label', 'senior_id'], axis=1)
    y_train = train_df['label']
    X_test = test_df.drop(['label', 'senior_id'], axis=1)
    y_test = test_df['label']
    
    # 3. Calculate Scale Weight (Vital for Imbalanced Data)
    # scale_pos_weight = count(negative) / count(positive)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_weight = n_neg / n_pos
    print(f"Training with scale_pos_weight = {scale_weight:.2f}")
    
    # 4. Train XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.05, 
        scale_pos_weight=scale_weight, # <--- CRITICAL FIX
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    
    # 5. Evaluate
    probs = xgb.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"\nFINAL HYBRID AUROC: {auc:.4f}")
    
    print(classification_report(y_test, (probs > 0.5).astype(int)))
        
    # Plot Importance
    plt.figure(figsize=(10, 8))
    imps = xgb.feature_importances_
    idxs = np.argsort(imps)[::-1][:20]
    plt.barh(range(len(idxs)), imps[idxs])
    plt.yticks(range(len(idxs)), X_train.columns[idxs])
    plt.title("Feature Importance (Hybrid - Valid Split)")
    plt.tight_layout()
    plt.savefig("models/hybrid_importance.png")
    print("Saved importance plot.")

if __name__ == "__main__":
    main()