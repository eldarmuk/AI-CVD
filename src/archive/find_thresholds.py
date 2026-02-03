import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, classification_report
from torch.utils.data import DataLoader
from src.archive.train_model import NPYDataset, BiGRUAttention, FEATURE_COUNT

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/bi_gru_best.pt"
TEST_X = "data/processed/sequences/X_test.npy"
TEST_Y = "data/processed/sequences/y_test.npy"
BATCH_SIZE = 512

def find_best_thresholds():
    print(f"Loading model from {MODEL_PATH}...")
    model = BiGRUAttention(input_dim=FEATURE_COUNT, num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Loading Test Data...")
    ds = NPYDataset(TEST_X, TEST_Y)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    
    all_probs = []
    all_targets = []
    
    print("Running Inference (Collecting Raw Probabilities)...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits, _ = model(x)
            # Softmax to get probabilities (0.0 to 1.0)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.numpy())
            
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    # We focus on Class 3 (Acute) first as it's the most critical
    print("\n--- DIAGNOSING ACUTE CLASS (3) ---")
    y_true_acute = (all_targets == 3).astype(int)
    y_probs_acute = all_probs[:, 3] # Column 3 is Acute probability
    
    print(f"Max Prob assigned to Acute: {y_probs_acute.max():.4f}")
    print(f"Avg Prob assigned to Acute patients: {y_probs_acute[y_true_acute==1].mean():.4f}")
    print(f"Avg Prob assigned to Healthy patients: {y_probs_acute[y_true_acute==0].mean():.4f}")
    
    # Precision-Recall Search
    precisions, recalls, thresholds = precision_recall_curve(y_true_acute, y_probs_acute)
    
    # Find threshold that gives at least 40% Recall
    optimal_idx = np.argmax(recalls < 0.40) # First index where recall drops below 40%
    if optimal_idx == 0: 
        optimal_idx = len(thresholds) - 1 # Fallback
    
    best_thresh = thresholds[optimal_idx]
    best_recall = recalls[optimal_idx]
    best_prec = precisions[optimal_idx]
    
    print(f"\nOPTIMAL THRESHOLD for Acute: {best_thresh:.4f}")
    print(f"At this threshold -> Recall: {best_recall:.2%} | Precision: {best_prec:.2%}")
    
    
    plt.figure()
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Acute Class PR Curve (Best Thresh={best_thresh:.3f})')
    plt.grid()
    plt.savefig("models/acute_pr_curve.png")
    print("Saved models/acute_pr_curve.png")

    # Generate Final Report using Custom Thresholds
    print("\n--- CUSTOM REPORT (Acute Thresh > 0.05?) ---")
    # Rule based prediction:
    # 1. If Acute_Prob > 0.05 -> Predict 3
    # 2. Else If Potential_Prob > 0.10 -> Predict 2
    # 3. Else If Low_Prob > 0.15 -> Predict 1
    # 4. Else Predict 0
    
    y_final_pred = np.zeros_like(all_targets)
    
    # Vectorized logic (Priority: 3 > 2 > 1)
    # Note: These thresholds (0.05, 0.10) are hypotheses. The script outputs the real optimal ones above.
    # Let's use the one we found for Acute.
    t3 = best_thresh
    t2 = t3 * 2 # Heuristic
    t1 = t2 * 1.5
    
    # Apply priority masks
    mask3 = all_probs[:, 3] > t3
    mask2 = (all_probs[:, 2] > t2) & (~mask3)
    mask1 = (all_probs[:, 1] > t1) & (~mask3) & (~mask2)
    
    y_final_pred[mask1] = 1
    y_final_pred[mask2] = 2
    y_final_pred[mask3] = 3
    
    print(classification_report(all_targets, y_final_pred, target_names=['Healthy', 'Low', 'Potential', 'Acute']))

if __name__ == "__main__":
    find_best_thresholds()