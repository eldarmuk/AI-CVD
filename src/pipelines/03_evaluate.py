import argparse
from sklearn import metrics
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve, confusion_matrix
import json
from pathlib import Path
from typing import Tuple, Dict

from src.components.utils import get_feature_columns

from ..components.lstm_vae_model import LSTM_VAE

class Config:
    BASE_DIR = Path("data/processed/anomaly_detection")
    PARQUET_PATH = Path("data/processed/multimodal_features.parquet")
    MODEL_DIR = Path("models/lstm_vae")
    
    PATHS = {
        "X_val": BASE_DIR / "X_val.npy",
        "y_val": BASE_DIR / "y_val.npy",
        "X_test": BASE_DIR / "X_test.npy",
        "y_test": BASE_DIR / "y_test.npy",
        "val_active_mask": BASE_DIR / "val_active_window_mask.npy",
        "test_active_mask": BASE_DIR / "test_active_window_mask.npy",
        "stats": BASE_DIR / "normalization_stats.json",
        "model": MODEL_DIR / "best_checkpoint.pt",
        "val_metrics": MODEL_DIR / "validation_metrics.json",
        "plots": MODEL_DIR / "metrics"
    }
    REAL_WORLD_DIR = Path("models/real_world_auprc")
    
    SEQ_LEN = 96
    FEATURE_NAMES = get_feature_columns(pd.read_parquet(PARQUET_PATH))
    N_FEATURES = len(FEATURE_NAMES)
    HIDDEN_DIM = 64
    EMBEDDING_DIM = 32
    BATCH_SIZE = 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Config.PATHS['plots'].mkdir(parents=True, exist_ok=True)

def load_and_preprocess(split: str, apply_active_mask: bool = True) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Loads split data and applies the robust normalization logic used in training.
    """
    x_key = f"X_{split}"
    y_key = f"y_{split}"
    print(f"+ Loading data from {Config.PATHS[x_key]}...")
    
    with open(Config.PATHS['stats'], 'r') as f:
        stats = json.load(f)
        mean = torch.tensor(stats['mean']).to(Config.DEVICE)
        std = torch.tensor(stats['std']).to(Config.DEVICE)
    
    X_data = np.load(Config.PATHS[x_key])
    y_data = np.load(Config.PATHS[y_key])
    mask_key = f"{split}_active_mask"
    active_mask_applied = False
    if apply_active_mask and Config.PATHS[mask_key].exists():
        active_mask = np.load(Config.PATHS[mask_key]).astype(bool)
        if len(active_mask) != len(y_data):
            raise ValueError(
                f"{split} active-window mask length mismatch: mask={len(active_mask)}, y={len(y_data)}"
            )
        X_data = X_data[active_mask]
        y_data = y_data[active_mask]
        active_mask_applied = True
    
    X = torch.from_numpy(X_data).float().to(Config.DEVICE)
    
    nan_mask = torch.isnan(X)
    mean_expanded = mean.view(1, 1, -1).expand_as(X)
    X = torch.where(nan_mask, mean_expanded, X)
    
    std_expanded = std.view(1, 1, -1).expand_as(X)
    X = (X - mean_expanded) / std_expanded
    
    print(f"  > {split.capitalize()} Shape: {X.shape}")
    print(f"  > Class Balance: {np.unique(y_data, return_counts=True)}")
    print(f"  > Active-window mask applied: {active_mask_applied}")
    
    return X, y_data

def compute_reconstruction_error(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """
    Runs batched inference and calculates Mean Squared Error (MSE) per sample.
    """
    print("+ Running inference...")
    model.eval()
    errors = []
    
    with torch.no_grad():
        for i in range(0, len(X), Config.BATCH_SIZE):
            batch = X[i : i + Config.BATCH_SIZE]
            recon, _, _, _ = model(batch)
            
            loss = torch.mean((batch - recon) ** 2, dim=[1, 2])
            errors.extend(loss.cpu().numpy())
            
    return np.array(errors)

def bootstrap_auc_ci(y_true: np.ndarray, scores: np.ndarray,
                     n_bootstraps: int = 1000,
                     seed: int = 42):
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []

    n = len(y_true)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], scores[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(bootstrapped_scores)
    lower = sorted_scores[int(0.025 * len(sorted_scores))]
    upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return lower, upper

def analyze_thresholds(y_true: np.ndarray, errors: np.ndarray) -> Dict:
    """
    Computes AUROC and finds the optimal threshold using G-Mean.
    """
    auc = roc_auc_score(y_true, errors)
    auprc = average_precision_score(y_true, errors)
    fpr, tpr, thresholds = roc_curve(y_true, errors)
    
    # G-Mean = sqrt(TPR * (1 - FPR))
    # Balances sensitivity and specificity
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    best_thresh = thresholds[ix]
    
    return {
        "auc": auc,
        "auprc": auprc,
        "best_threshold": best_thresh,
        "gmean": gmeans[ix],
        "fpr": fpr,
        "tpr": tpr
    }


def write_sequence_metrics(y_true: np.ndarray, errors: np.ndarray, metrics_data: Dict, apply_active_mask: bool):
    Config.REAL_WORLD_DIR.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, errors)
    np.savez_compressed(
        Config.REAL_WORLD_DIR / "imbalanced_pr_vae.npz",
        y_true=y_true,
        scores=errors,
        precision=precision,
        recall=recall,
    )
    output = {
        "model": "CCA-TAVAE",
        "mode": "vae",
        "auroc": float(metrics_data["auc"]),
        "auprc": float(metrics_data["auprc"]),
        "auroc_ci_low": float(metrics_data["ci_lower"]),
        "auroc_ci_high": float(metrics_data["ci_upper"]),
        "threshold": float(metrics_data["best_threshold"]),
        "threshold_source": "validation G-Mean from models/lstm_vae/validation_metrics.json",
        "n_windows": int(len(y_true)),
        "positives": int(np.sum(y_true == 1)),
        "negatives": int(np.sum(y_true == 0)),
        "prevalence": float(np.mean(y_true == 1)),
        "active_window_mask_applied": bool(apply_active_mask),
        "prediction_window": "4 hours",
        "cohort": "elite",
    }
    with open(Config.REAL_WORLD_DIR / "metrics_vae.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

def save_plots(y_true, errors, metrics: Dict, split_label: str = "test"):
    """
    Generates and saves publication-quality evaluation figures.
    """
    sns.set_style("whitegrid")
    best_thresh = metrics['best_threshold']
    
    # 1. ROC Curve
    plt.figure(figsize=(8, 8))
    plt.plot(metrics['fpr'], metrics['tpr'], label=f'LSTM-VAE (AUC = {metrics["auc"]:.3f})', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(Config.PATHS['plots'] / f"roc_curve_{split_label}.png", dpi=300)
    plt.close()

    # 2. Error Distribution (Log Scale)
    plt.figure(figsize=(12, 6))
    errors_log = np.log1p(errors)  # log(1+x)
    
    sns.histplot(errors_log[y_true==0], color='green', label='Normal', stat="density", element="step", fill=True, alpha=0.3)
    sns.histplot(errors_log[y_true==1], color='red', label='Anomaly', stat="density", element="step", fill=True, alpha=0.3)
    plt.axvline(np.log1p(best_thresh), color='k', linestyle='--', label='Threshold')
    
    plt.title(f"Reconstruction Error (Log Scale) | AUC={metrics['auc']:.3f}")
    plt.xlabel("Log(MSE Loss)")
    plt.legend()
    plt.savefig(Config.PATHS['plots'] / f"dist_log_{split_label}.png", dpi=300)
    plt.close()

    # 3. Zoomed Error Distribution (Linear Scale)
    plt.figure(figsize=(10, 6))
    viz_errors = np.clip(errors, 0, np.percentile(errors, 99))
    
    sns.histplot(viz_errors[y_true==0], color='green', label='Normal', stat="density", kde=True, bins=50, alpha=0.3)
    sns.histplot(viz_errors[y_true==1], color='red', label='Anomaly', stat="density", kde=True, bins=50, alpha=0.3)
    
    plt.axvline(best_thresh, color='blue', linestyle='--', linewidth=2, label=f'Threshold ({best_thresh:.3f})')
    plt.title("Reconstruction Error Separation (Linear Scale)")
    plt.xlabel("MSE Loss")
    plt.legend()
    plt.savefig(Config.PATHS['plots'] / f"dist_zoomed_{split_label}.png", dpi=300)
    plt.close()

    # 3. Confusion Matrix
    y_pred = (errors > best_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Normal', 'Pred Anomaly'],
                yticklabels=['True Normal', 'True Anomaly'])
    plt.title(f"Confusion Matrix @ Threshold {best_thresh:.4f}")
    plt.savefig(Config.PATHS['plots'] / f"confusion_matrix_{split_label}.png", dpi=300)
    plt.close()
    
    print(f"+ Plots saved to {Config.PATHS['plots']}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate unsupervised VAE reconstruction metrics.")
    parser.add_argument("--mode", choices=["vae"], default="vae")
    parser.add_argument(
        "--no-active-window-mask",
        action="store_true",
        help="Evaluate all standard sequences instead of the local density filtered subset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    apply_active_mask = not args.no_active_window_mask
    print(f"Device: {Config.DEVICE}")
    X_val_tensor, y_val = load_and_preprocess("val", apply_active_mask=apply_active_mask)
    X_test_tensor, y_test = load_and_preprocess("test", apply_active_mask=apply_active_mask)
    
    model = LSTM_VAE(
        input_dim=Config.N_FEATURES, 
        hidden_dim=Config.HIDDEN_DIM, 
        embedding_dim=Config.EMBEDDING_DIM
    ).to(Config.DEVICE)
    
    print(f"+ Loading weights from {Config.PATHS['model']}...")
    checkpoint = torch.load(Config.PATHS['model'], map_location=Config.DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    val_errors = compute_reconstruction_error(model, X_val_tensor)
    test_errors = compute_reconstruction_error(model, X_test_tensor)

    if Config.PATHS['val_metrics'].exists():
        with open(Config.PATHS['val_metrics'], 'r') as f:
            val_metrics = json.load(f)
        threshold = float(val_metrics.get("best_threshold", {}).get("threshold", val_metrics.get("best_threshold", 0.0)))
    else:
        threshold_info = analyze_thresholds(y_val, val_errors)
        threshold = float(threshold_info["best_threshold"])

    test_metrics = analyze_thresholds(y_test, test_errors)
    test_metrics["best_threshold"] = threshold

    ci_lower, ci_upper = bootstrap_auc_ci(y_test, test_errors)
    test_metrics["ci_lower"] = ci_lower
    test_metrics["ci_upper"] = ci_upper

    print("\n" + "="*30)
    print(" FINAL RESULTS (TEST)")
    print(f" AUROC: {test_metrics['auc']:.4f} "
          f"(95% CI: {test_metrics['ci_lower']:.4f}-{test_metrics['ci_upper']:.4f})")
    print(f" AUPRC: {test_metrics['auprc']:.4f}")
    print(f" Threshold (from VAL): {threshold:.4f}")
    print("="*30 + "\n")

    write_sequence_metrics(y_test, test_errors, test_metrics, apply_active_mask)
    save_plots(y_test, test_errors, test_metrics, split_label="test")

if __name__ == "__main__":
    main()
