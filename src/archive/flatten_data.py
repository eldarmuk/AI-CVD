import numpy as np
import os
import json
from tqdm import tqdm

# CONFIG
PATHS = {
    "X_train": "data/processed/archive_early_warning_system/sequences/X_train.npy",
    "y_train": "data/processed/archive_early_warning_system/sequences/y_train.npy",
    "X_val": "data/processed/archive_early_warning_system/sequences/X_val.npy",
    "y_val": "data/processed/archive_early_warning_system/sequences/y_val.npy",
    "X_test": "data/processed/archive_early_warning_system/sequences/X_test.npy",
    "y_test": "data/processed/archive_early_warning_system/sequences/y_test.npy"
}
SEQ_LEN = 96
N_FEATS = 33

# --- CRITICAL: UPDATE THIS LIST TO MATCH YOUR EXACT FEATURE ORDER ---
# These must match the 33 columns in your X_train.npy
BASE_FEATURES = [
    "temperature", "heartrate", "sbp", "dbp", "pulse_pressure", 
    "steps", "saturation", "hr_volatility", "bp_trend", "shock_index", 
    "recent_event_burden",
    # Add your static features (diseases, age, etc.) here to reach 33
    "age", "gender", "feature_14", "feature_15", "feature_16",
    "feature_17", "feature_18", "feature_19", "feature_20",
    "feature_21", "feature_22", "feature_23", "feature_24",
    "feature_25", "feature_26", "feature_27", "feature_28",
    "feature_29", "feature_30", "feature_31", "feature_32"
]

def generate_flat_names():
    stats = ['mean', 'std', 'min', 'max', 'last']
    flat_names = []
    
    if len(BASE_FEATURES) != N_FEATS:
        print(f"WARNING: BASE_FEATURES has {len(BASE_FEATURES)} items, but N_FEATS is {N_FEATS}.")
    
    # The flattening loop does: for each chunk -> calculate stats -> hstack
    # Order in hstack: [mean, std, min, max, last]
    # This means all means come first, then all stds, etc.
    # OR does it do feature by feature? 
    # Let's check the code below:
    # flat = np.hstack([f_mean, f_std, f_min, f_max, f_last])
    # f_mean shape is (B, 33). hstack puts them side by side.
    # So indices 0-32 are means, 33-65 are stds, etc.
    
    for stat in stats:
        for feat in BASE_FEATURES:
            flat_names.append(f"{feat}_{stat}")
            
    return flat_names

def flatten_and_save(x_path, y_path, prefix):
    print(f"Processing {prefix}...")
    
    # 1. Load Data
    file_size = os.path.getsize(x_path)
    bytes_per_sample = SEQ_LEN * N_FEATS * 4
    
    # Handle Header Offset if present
    offset_x = 0
    if file_size % bytes_per_sample != 0:
        offset_x = 128
        
    n_samples = (file_size - offset_x) // bytes_per_sample
    
    print(f"  Loading {n_samples} samples from {x_path}...")
    X = np.memmap(x_path, dtype='float32', mode='r', offset=offset_x, shape=(n_samples, SEQ_LEN, N_FEATS))
    
    # Load Y
    y_size = os.path.getsize(y_path)
    offset_y = 128 if y_size % 1 != 0 else 0
    y = np.memmap(y_path, dtype='int8', mode='r', offset=offset_y, shape=(n_samples,))
    
    # 2. Flatten Loop (Chunked to save RAM)
    chunk_size = 10000
    X_flat_list = []
    
    for i in tqdm(range(0, n_samples, chunk_size)):
        # Load chunk to RAM
        X_chunk = np.array(X[i:i+chunk_size]) # (B, 96, 33)
        
        # Calculate Stats
        # axis 1 is the time dimension
        f_mean = np.mean(X_chunk, axis=1)
        f_std  = np.std(X_chunk, axis=1)
        f_min  = np.min(X_chunk, axis=1)
        f_max  = np.max(X_chunk, axis=1)
        f_last = X_chunk[:, -1, :] # The most recent value
        
        # Concatenate: (B, 33*5)
        # Result columns: [mean_feat0, mean_feat1..., std_feat0, std_feat1...]
        flat = np.hstack([f_mean, f_std, f_min, f_max, f_last])
        X_flat_list.append(flat)
        
    # 3. Stack and Save
    X_final = np.vstack(X_flat_list)
    y_final = np.array(y) # Load y fully to RAM
    
    print(f"  Saving {prefix} (Shape {X_final.shape})...")
    np.save(f"data/processed/archive_early_warning_system/flat/{prefix}_flat_X.npy", X_final)
    np.save(f"data/processed/archive_early_warning_system/flat/{prefix}_flat_y.npy", y_final)

if __name__ == "__main__":
    # Save Feature Names First
    flat_names = generate_flat_names()
    with open("data/processed/archive_early_warning_system/flat/flat_feature_names.json", "w") as f:
        json.dump(flat_names, f)
    print("Saved feature names to data/processed/archive_early_warning_system/flat/flat_feature_names.json")

    flatten_and_save(PATHS["X_train"], PATHS["y_train"], "train")
    flatten_and_save(PATHS["X_val"],   PATHS["y_val"],   "val")
    flatten_and_save(PATHS["X_test"],  PATHS["y_test"],  "test")