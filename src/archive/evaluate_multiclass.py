import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from src.archive.train_model import NPYDataset, BiGRUAttention, FEATURE_COUNT

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/bi_gru_best.pt"
TEST_X = "data/processed/archive_early_warning_system/sequences/X_test.npy"
TEST_Y = "data/processed/archive_early_warning_system/sequences/y_test.npy"
BATCH_SIZE = 512
class_names = ['Healthy (0)', 'Low (1)', 'Potential (2)', 'Acute (3)']

def evaluate():
    print(f"Loading best model from {MODEL_PATH}...")
    # NOTE: Ensure num_classes=4 matches your training script
    model = BiGRUAttention(input_dim=FEATURE_COUNT, num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Loading Test Data...")
    test_ds = NPYDataset(TEST_X, TEST_Y)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    
    y_true = []
    y_pred = []
    y_probs = []
    
    print("Running Inference...")
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits, _ = model(x)
            
            # Get Probabilities and Hard Predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            y_probs.extend(probs.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # --- METRICS ---
    print("\n" + "="*40)
    print("       FINAL MULTI-CLASS REPORT")
    print("="*40)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    # Normalize by row (True Label) to see "Recall" per class
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Normalized Confusion Matrix (Recall)')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix_multiclass.png')
    print("Saved models/confusion_matrix_multiclass.png")

if __name__ == "__main__":
    evaluate()