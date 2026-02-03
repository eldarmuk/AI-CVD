"""
generate_sequences.py

Corrected pipeline for Senior Early Warning System.
- Implements "Any Event in Horizon" logic (Max severity in future window).
- Handles NaNs via Forward Fill.
- Performs Downsampling BEFORE writing to disk (saves IO/Space).
- Outputs separate .npy files for efficiency (Standard for DL).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- CONFIGURATION ---
SEQ_LEN = 96         # 24 hours history (assuming 15m intervals)
HORIZON = 96         # Predict events within the NEXT 24 hours
STRIDE = 4           # Slide by 1 hour
SEED = 42
DOWNSAMPLE_RATIO = 20 # 1 Positive : 20 Negatives

def get_feature_cols(df):
    exclude = ["timestamp", "senior_id", "label_1", "label_2", "label_3"]
    return [c for c in df.columns if c not in exclude]

def impute_and_scale(df, feature_cols, train_seniors):
    """
    1. Fills NaNs (Forward fill per senior).
    2. Fits scaler on TRAIN seniors only.
    3. Transforms the whole DF.
    """
    print("Imputing missing values (ffill, bfill, fillna)...")
    # Step A: Forward fill per senior
    df[feature_cols] = df.groupby("senior_id")[feature_cols].ffill()
    # Step B: Backward fill per senior
    df[feature_cols] = df.groupby("senior_id")[feature_cols].bfill()
    # Step C: Fill any remaining NaNs with 0 (or global mean if available)
    df[feature_cols] = df[feature_cols].fillna(0)

    print("Fitting Scaler on Training data...")
    scaler = StandardScaler()
    train_mask = df["senior_id"].isin(train_seniors)
    scaler.fit(df.loc[train_mask, feature_cols])

    print("Transforming data...")
    df[feature_cols] = scaler.transform(df[feature_cols]).astype(np.float32)
    return df

def collect_indices(df, seniors, seq_len, horizon, stride, downsample=False):
    """
    Pass 1: Don't save data yet. Just find the valid START INDICES and their LABELS.
    This allows us to downsample *indices* before allocating massive memory.
    """
    indices = []  # Stores (senior_id, start_row_idx, max_severity_label)

    for sid in tqdm(seniors, desc="Indexing Windows"):
        group = df[df["senior_id"] == sid]
        if len(group) < seq_len + horizon:
            continue

        # Get raw arrays for speed
        g_label_1 = group["label_1"].values
        g_label_2 = group["label_2"].values
        g_label_3 = group["label_3"].values
        g_indices = group.index.values  # The global row indices in the main DF

        # Valid starts: from 0 up to length - seq - horizon
        max_start = len(group) - seq_len - horizon
        starts = np.arange(0, max_start + 1, stride)

        for start in starts:
            end = start + seq_len
            # Look at the window [end : end + horizon]
            fw_1 = g_label_1[end : end + horizon]
            fw_2 = g_label_2[end : end + horizon]
            fw_3 = g_label_3[end : end + horizon]

            # Skip windows where future is unknown
            if len(fw_1) == 0 or len(fw_2) == 0 or len(fw_3) == 0:
                continue

            # Multi-class max severity logic
            if np.any(fw_3 > 0):
                label = 3
            elif np.any(fw_2 > 0):
                label = 2
            elif np.any(fw_1 > 0):
                label = 1
            else:
                label = 0

            indices.append((g_indices[start], label))

    return indices

def balance_indices(indices, ratio):
    """
    Filters the list of indices to balance classes.
    """
    print(f"Balancing {len(indices)} windows...")
    arr = np.array(indices, dtype=object) # (global_idx, label)
    
    # Separate by class
    # Assuming "Positive" is label >= 1 (or just label 3 depending on your goal)
    # Let's assume User wants to predict Level 3 specifically vs others, 
    # OR multi-class. If Multi-class, we usually downsample only class 0.
    
    labels = arr[:, 1].astype(int)
    
    # Indices where Label is 0 (No Alert)
    neg_mask = (labels == 0)
    # Indices where Label is > 0 (Alert 1, 2, or 3)
    pos_mask = ~neg_mask
    
    neg_indices = arr[neg_mask]
    pos_indices = arr[pos_mask]
    
    n_pos = len(pos_indices)
    n_neg_keep = n_pos * ratio
    
    if len(neg_indices) > n_neg_keep:
        # Random choice without replacement
        keep_idx = np.random.choice(len(neg_indices), size=n_neg_keep, replace=False)
        neg_indices = neg_indices[keep_idx]
        
    print(f"Stats: Positives={len(pos_indices)}, Negatives={len(neg_indices)} (Ratio 1:{ratio})")
    
    # Combine and Shuffle
    balanced = np.concatenate([pos_indices, neg_indices])
    np.random.shuffle(balanced)
    return balanced

def write_memmap(df, balanced_indices, feature_cols, out_x, out_y):
    """
    Pass 2: Write the actual data using the selected indices.
    """
    n_samples = len(balanced_indices)
    n_feats = len(feature_cols)
    
    print(f"Writing {n_samples} windows to {out_x}...")
    
    fp_X = np.memmap(out_x, dtype='float32', mode='w+', shape=(n_samples, SEQ_LEN, n_feats))
    fp_y = np.memmap(out_y, dtype='int8', mode='w+', shape=(n_samples,))
    
    # Load all data into RAM for fast random access? 
    # 20M rows * 33 cols * 4 bytes = ~2.6GB. 
    # YES, we can hold the whole DF in RAM. It's faster than random seek on disk.
    data_matrix = df[feature_cols].values
    
    for i, (start_idx, label) in enumerate(tqdm(balanced_indices, desc="Writing Memmap")):
        # start_idx is the global row index
        seq = data_matrix[start_idx : start_idx + SEQ_LEN]
        
        fp_X[i] = seq
        fp_y[i] = label
        
        if i % 10000 == 0:
            fp_X.flush()
            fp_y.flush()
            
    fp_X.flush()
    fp_y.flush()

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--input", default="data/processed/multimodal_features.parquet")
    args.add_argument("--out_dir", default="data/processed/archive_early_warning_system/sequences")
    parsed = args.parse_args()
    
    out_dir = Path(parsed.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_parquet(parsed.input)
    df = df.sort_values(["senior_id", "timestamp"]).reset_index(drop=True)
    
    # Split Seniors
    seniors = df["senior_id"].unique()
    np.random.seed(SEED)
    np.random.shuffle(seniors)
    n_train = int(len(seniors) * 0.8)
    n_val = int(len(seniors) * 0.1)
    
    train_s = seniors[:n_train]
    val_s = seniors[n_train : n_train+n_val]
    test_s = seniors[n_train+n_val:]
    
    # Features & Scaling
    feats = get_feature_cols(df)
    df = impute_and_scale(df, feats, train_s)
    
    # --- PROCESS TRAIN (With Downsampling) ---
    print("\n--- Processing TRAIN ---")
    train_indices = collect_indices(df, train_s, SEQ_LEN, HORIZON, STRIDE)
    train_balanced = balance_indices(train_indices, ratio=DOWNSAMPLE_RATIO)
    write_memmap(df, train_balanced, feats, out_dir/"X_train.npy", out_dir/"y_train.npy")
    
    # --- PROCESS VAL (No Downsampling usually, or less aggressive) ---
    print("\n--- Processing VAL ---")
    val_indices = collect_indices(df, val_s, SEQ_LEN, HORIZON, STRIDE)
    # Convert to array format expected by writer
    val_final = np.array(val_indices, dtype=object) 
    write_memmap(df, val_final, feats, out_dir/"X_val.npy", out_dir/"y_val.npy")
    
    # --- PROCESS TEST ---
    print("\n--- Processing TEST ---")
    test_indices = collect_indices(df, test_s, SEQ_LEN, HORIZON, STRIDE)
    test_final = np.array(test_indices, dtype=object)
    write_memmap(df, test_final, feats, out_dir/"X_test.npy", out_dir/"y_test.npy")

    print(f"\nDone. Files saved to {out_dir}")

if __name__ == "__main__":
    main()