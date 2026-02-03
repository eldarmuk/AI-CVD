
# TODO: Optimize this code (output paths, plotting, etc.)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
from pathlib import Path
from sklearn.metrics import roc_curve, classification_report

from ..components.lstm_vae_model import LSTM_VAE
from ..components.utils import get_feature_columns

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data/processed/anomaly_detection"
PARQUET_PATH = BASE_DIR / "data/processed/multimodal_features.parquet"
MODEL_DIR = BASE_DIR / "models/lstm_vae"
OUTPUT_DIR = MODEL_DIR / "analysis"

PATHS = {
    "X_test": DATA_DIR / "X_test.npy",
    "y_test": DATA_DIR / "y_test.npy",
    "parquet": PARQUET_PATH,
    "stats": DATA_DIR / "normalization_stats.json",
    "model": MODEL_DIR / "best_checkpoint.pt"
}

for subfolder in ["metrics", "windows", "patients", "features"]:
    (OUTPUT_DIR / subfolder).mkdir(parents=True, exist_ok=True)

SEQ_LEN = 96
HIDDEN_DIM = 64
EMBEDDING_DIM = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_NAMES = get_feature_columns(pd.read_parquet(PATHS['parquet']))
N_FEATURES = len(FEATURE_NAMES)

def load_resources():
    print(f"Loading resources from {DATA_DIR}...")
    
    with open(PATHS['stats'], 'r') as f:
        stats = json.load(f)
        mean = torch.tensor(stats['mean']).to(DEVICE)
        std = torch.tensor(stats['std']).to(DEVICE)
        
    print(f"Loading model from {PATHS['model']}...")
    model = LSTM_VAE(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    try:
        model.load_state_dict(torch.load(PATHS['model'], map_location=DEVICE, weights_only=False))
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
        
    model.eval()
    return model, mean, std

def calculate_threshold(y_true, y_scores):
    """Finds best threshold using G-Mean."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    return thresholds[ix]

def analyze_feature_contribution(model, X_test, y_test, mean, std):
    """Calculates which features contribute most to the reconstruction error."""
    print("\n[Analysis 1/3] Calculating Feature Importance...")
    
    X_tensor = torch.from_numpy(X_test).float().to(DEVICE)
    
    nan_mask = torch.isnan(X_tensor)
    mean_exp = mean.unsqueeze(0).unsqueeze(0).expand_as(X_tensor)
    std_exp = std.unsqueeze(0).unsqueeze(0).expand_as(X_tensor)
    X_tensor = torch.where(nan_mask, mean_exp, X_tensor)
    X_tensor = (X_tensor - mean_exp) / std_exp
    
    feature_errors = np.zeros(N_FEATURES)
    count = 0
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            recon, _, _, _ = model(batch)
            error = torch.mean((batch - recon) ** 2, dim=[0, 1])
            feature_errors += error.cpu().numpy() * len(batch)
            count += len(batch)
            
    avg_feat_error = feature_errors / count
    
    plt.figure(figsize=(12, 6))
    indices = np.argsort(avg_feat_error)[::-1]
    sorted_errors = avg_feat_error[indices]
    sorted_names = [FEATURE_NAMES[i] for i in indices]
    
    sns.barplot(x=sorted_errors, y=sorted_names, palette="viridis", hue=sorted_names, legend=False)
    plt.title("Reconstruction Error by Feature (Higher = Harder to Model)")
    plt.xlabel("Mean Squared Error")
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "features" / "feature_importance.png"
    plt.savefig(save_path)
    print(f"Saved {save_path}")

def plot_reconstruction(model, x_sample, title, filename):
    """Plots Input vs Reconstruction for 3 key features."""
    model.eval()
    with torch.no_grad():
        x_in = x_sample.unsqueeze(0).to(DEVICE)
        recon, _, _, _ = model(x_in)
        
    x_in = x_in.cpu().numpy()[0]
    recon = recon.cpu().numpy()[0]
    
    # Plot indices (HeartRate=1, Steps=5, SBP=2) - Verify these match your columns!
    feat_indices = [1, 5, 2] 
    feat_labels = ['Heart Rate', 'Steps', 'SBP']
    
    plt.figure(figsize=(12, 8))
    for i, (idx, label) in enumerate(zip(feat_indices, feat_labels)):
        plt.subplot(3, 1, i+1)
        plt.plot(x_in[:, idx], label='Input', color='black', alpha=0.7)
        plt.plot(recon[:, idx], label='Reconstruction', color='red', linestyle='--')
        plt.title(f"{label} - {title}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    save_path = OUTPUT_DIR / "windows" / filename
    plt.savefig(save_path)
    print(f"Saved {save_path}")

def analyze_windows(model, mean, std):
    """Diagnoses specific windows (TP/FN/FP)."""
    print("\n[Analysis 2/3] Diagnosing Windows...")
    X_test = np.load(PATHS['X_test'])
    y_test = np.load(PATHS['y_test'])
    
    X_tensor = torch.from_numpy(X_test).float().to(DEVICE)
    mean_exp = mean.unsqueeze(0).unsqueeze(0).expand_as(X_tensor)
    std_exp = std.unsqueeze(0).unsqueeze(0).expand_as(X_tensor)
    X_tensor = torch.where(torch.isnan(X_tensor), mean_exp, X_tensor)
    X_tensor = (X_tensor - mean_exp) / std_exp
    
    errors = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), 256):
            batch = X_tensor[i:i+256]
            recon, _, _, _ = model(batch)
            loss = torch.mean((batch - recon) ** 2, dim=[1, 2])
            errors.extend(loss.cpu().numpy())
    errors = np.array(errors)
    
    threshold = calculate_threshold(y_test, errors)
    print(f"Optimal Threshold (G-Mean): {threshold:.4f}")
    
    y_pred = (errors > threshold).astype(int)
    
    # TP: High Error on Anomaly
    tp_idxs = np.where((y_test == 1) & (y_pred == 1))[0]
    if len(tp_idxs) > 0:
        best_tp = tp_idxs[np.argmax(errors[tp_idxs])]
        plot_reconstruction(model, X_tensor[best_tp], "True Positive (Detected)", "window_TP.png")

    # FN: Low Error on Anomaly (Missed)
    fn_idxs = np.where((y_test == 1) & (y_pred == 0))[0]
    if len(fn_idxs) > 0:
        worst_fn = fn_idxs[np.argmin(errors[fn_idxs])]
        plot_reconstruction(model, X_tensor[worst_fn], "False Negative (Missed)", "window_FN.png")
        
    # FP: High Error on Normal (False Alarm)
    fp_idxs = np.where((y_test == 0) & (y_pred == 1))[0]
    if len(fp_idxs) > 0:
        worst_fp = fp_idxs[np.argmax(errors[fp_idxs])]
        plot_reconstruction(model, X_tensor[worst_fp], "False Positive (False Alarm)", "window_FP.png")

    # Save Classification Report
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'])
    print("\nClassification Report:")
    print(report)
    with open(OUTPUT_DIR / "metrics" / "classification_report.txt", "w") as f:
        f.write(report)
        
    return threshold

def analyze_patients(model, mean, std, threshold):
    """Finds TP, TN, FP, FN examples in patient timelines."""
    print("\n[Analysis 3/3] Analyzing Patient Timelines...")
    df = pd.read_parquet(PATHS['parquet'])
    
    feat_cols = [c for c in df.columns if c not in {'senior_id', 'timestamp', 'label_1', 'label_2', 'label_3', 'hour', 'is_night', 'day_of_week'}]
    
    def get_senior_score(sid):
        sub = df[df['senior_id'] == sid].sort_values('timestamp').reset_index(drop=True)
        if len(sub) < SEQ_LEN:
            return None
        
        feats = sub[feat_cols].values.astype(np.float32)
        ft = torch.from_numpy(feats).to(DEVICE)
        
        nan_mask = torch.isnan(ft)
        mean_exp = mean.unsqueeze(0).expand_as(ft)
        ft = torch.where(nan_mask, mean_exp, ft)
        ft = (ft - mean.unsqueeze(0)) / std.unsqueeze(0)
        
        windows = []
        times = []
        labels = []
        for i in range(len(sub) - SEQ_LEN + 1):
            windows.append(ft[i:i+SEQ_LEN])
            times.append(sub['timestamp'].iloc[i+SEQ_LEN-1])
            labels.append(sub['label_3'].iloc[i+SEQ_LEN-1])
        windows = torch.stack(windows)
        
        scores = []
        with torch.no_grad():
            for i in range(0, len(windows), 256):
                batch = windows[i:i+256]
                recon, _, _, _ = model(batch)
                loss = torch.mean((batch - recon) ** 2, dim=[1, 2])
                scores.extend(loss.cpu().numpy())
                
        res = pd.DataFrame({'timestamp': times, 'raw': scores, 'label': labels})
        res['smooth'] = res['raw'].rolling(24, min_periods=1).mean()
        return res

    categories = {'TP': None, 'FN': None, 'FP': None}
    seniors = df['senior_id'].unique()
    random.shuffle(seniors)
    
    for sid in seniors:
        if all(categories.values()):
            break
        res = get_senior_score(sid)
        if res is None:
            continue
        
        # TP: Has label 1 AND smoothed score > threshold
        if not categories['TP'] and not res[(res['label']==1) & (res['smooth'] > threshold)].empty:
            categories['TP'] = (sid, res)
            
        # FN: Has label 1 BUT smoothed score NEVER crosses threshold
        acute_events = res[res['label']==1]
        if not categories['FN'] and not acute_events.empty and (acute_events['smooth'] < threshold).all():
            categories['FN'] = (sid, res)
            
        # FP: Has NO label 1, but smoothed score > threshold * 1.5
        if not categories['FP'] and (res['label']==0).all() and (res['smooth'] > threshold * 1.5).any():
            categories['FP'] = (sid, res)

    # Plot
    for cat, data in categories.items():
        if not data: 
            continue
        sid, res = data
        
        plt.figure(figsize=(10, 5))
        plt.plot(res['timestamp'], res['raw'], color='lightblue', alpha=0.5, label='Raw Noise')
        plt.plot(res['timestamp'], res['smooth'], color='black', label='Smoothed Risk')
        plt.axhline(threshold, color='orange', linestyle='--', label='Threshold')
        
        events = res[res['label']==1]
        if not events.empty:
            plt.scatter(events['timestamp'], events['smooth'], color='red', s=50, label='Acute Event', zorder=5)
            
        plt.title(f"Example {cat} - Senior {sid}")
        plt.legend()
        plt.tight_layout()
        save_path = OUTPUT_DIR / "patients" / f"patient_{cat}.png"
        plt.savefig(save_path)
        print(f"Saved {save_path}")

def main():
    model, mean, std = load_resources()
    
    X_test = np.load(PATHS['X_test'])
    y_test = np.load(PATHS['y_test'])
    
    analyze_feature_contribution(model, X_test, y_test, mean, std)
    threshold = analyze_windows(model, mean, std)
    analyze_patients(model, mean, std, threshold)
    
    print(f"\nAnalysis Complete. Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()