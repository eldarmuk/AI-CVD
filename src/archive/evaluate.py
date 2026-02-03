import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, classification_report
from torch.utils.data import DataLoader
from src.archive.train_model import NPYDataset, BiGRUAttention, FEATURE_COUNT

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/bi_gru_best.pt"
TEST_X = "data/processed/archive_early_warning_system/sequences/X_test.npy"
TEST_Y = "data/processed/archive_early_warning_system/sequences/y_test.npy"
BATCH_SIZE = 512 # Larger batch for inference is faster

def plot_pr_curve(y_true, y_probs, score):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'Bi-GRU (AUPRC = {score:.3f})')
    plt.plot([0, 1], [y_true.mean(), y_true.mean()], linestyle='--', label='Random Guess')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/pr_curve.png')
    print("Saved models/pr_curve.png")

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Alert', 'Alert'], yticklabels=['No Alert', 'Alert'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Threshold = 0.5)')
    plt.savefig('models/confusion_matrix.png')
    print("Saved models/confusion_matrix.png")

def evaluate():
    print(f"Loading best model from {MODEL_PATH}...")
    model = BiGRUAttention(input_dim=FEATURE_COUNT).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Loading Test Data...")
    # Note: We use the Test set now, which is "unseen" data
    test_ds = NPYDataset(TEST_X, TEST_Y)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    
    y_true = []
    y_probs = []
    
    print("Running Inference...")
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits, _ = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            
            y_probs.extend(probs)
            y_true.extend(y.numpy().ravel())
            
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    # --- METRICS ---
    auprc = average_precision_score(y_true, y_probs)
    print("\n--- TEST RESULTS ---")
    print(f"Test AUPRC: {auprc:.4f}")
    
    # Optimal Threshold Search (Maximize F1)
    # Since 0.5 might be too high for a "shy" model
    _ = 0
    _ = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (y_probs >= thresh).astype(int)
        _ = average_precision_score(y_true, preds) # Using AP as proxy or standard F1
        # Let's use sklearn classification report for full detail
        pass
        
    print("\nReport at Threshold = 0.5:")
    print(classification_report(y_true, (y_probs >= 0.5).astype(int)))
    
    # Plotting
    
    plot_pr_curve(y_true, y_probs, auprc)
    
    
    plot_confusion_matrix(y_true, (y_probs >= 0.5).astype(int))

if __name__ == "__main__":
    evaluate()