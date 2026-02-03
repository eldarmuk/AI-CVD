import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os

def load_data(split):
    print(f"Loading {split}...")
    X = np.load(f"data/processed/archive_early_warning_system/flat/{split}_flat_X.npy")
    y = np.load(f"data/processed/archive_early_warning_system/flat/{split}_flat_y.npy")
    return X, y

def main():
    # 1. Load Data
    X_train, y_train = load_data("train")
    X_val, y_val = load_data("val")
    X_test, y_test = load_data("test")
    
    print(f"Train Shape: {X_train.shape}, Positive Rate: {np.mean(y_train > 0):.2%}")
    
    # Load Feature Names
    feature_names = None
    names_path = "data/processed/archive_early_warning_system/flat/flat_feature_names.json"
    if os.path.exists(names_path):
        with open(names_path, "r") as f:
            feature_names = json.load(f)
        print(f"Loaded {len(feature_names)} feature names.")
    else:
        print("Warning: feature_names.json not found. Using generic names.")
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    # 2. Setup XGBoost
    # Calculate weights manually sample-wise
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=4,
        n_jobs=-1,
        tree_method='hist', # Faster
        early_stopping_rounds=10
    )
    
    # 3. Train
    print("Training XGBoost...")
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # 4. Evaluate
    print("\n--- EVALUATING ON TEST SET ---")
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    
    print(classification_report(y_test, preds, target_names=['Healthy', 'Low', 'Potential', 'Acute']))
    
    # Check AUPRC for Acute Class (Class 3)
    y_test_acute = (y_test == 3).astype(int)
    prob_acute = probs[:, 3]
    auprc = average_precision_score(y_test_acute, prob_acute)
    print(f"Acute Class AUPRC: {auprc:.4f}")
    
    # Feature Importance Plot
    plt.figure(figsize=(12, 8)) # Made figure larger for readability
    importances = model.feature_importances_
    
    # Get Top 20 indices
    indices = np.argsort(importances)[::-1][:20] 
    
    top_feature_names = [feature_names[i] for i in indices]
    
    plt.title("Top 20 Feature Importances")
    plt.bar(range(20), importances[indices])
    plt.xticks(range(20), top_feature_names, rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig("models/archive_bi_gru_early_warning/old_plots/xgb_importance.png")
    print("Saved feature importance plot to models/archive_bi_gru_early_warning/old_plots/xgb_importance.png")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, preds, normalize='true')
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=['Healthy','Low','Potential','Acute'], 
                yticklabels=['Healthy','Low','Potential','Acute'])
    plt.title("XGBoost Confusion Matrix (Recall)")
    plt.savefig("models/archive_bi_gru_early_warning/old_plots/xgb_confusion.png")
    
    # Save Model
    with open("models/archive_bi_gru_early_warning/xgb_best.pkl", "wb") as f:
        pickle.dump(model, f)
        print("Saved models/archive_bi_gru_early_warning/xgb_best.pkl")

if __name__ == "__main__":
    main()