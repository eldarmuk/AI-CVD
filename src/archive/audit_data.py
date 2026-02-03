import numpy as np
import os

# Paths (adjust if yours are different)
PATHS = {
    "y_train": "data/processed/archive_early_warning_system/sequences/y_train.npy",
    "y_val": "data/processed/archive_early_warning_system/sequences/y_val.npy"
}

def audit():
    print("--- DATA AUDIT ---")
    for name, path in PATHS.items():
        if not os.path.exists(path):
            print(f"[{name}] File not found: {path}")
            continue
            
        # 1. Determine size and shape
        file_size = os.path.getsize(path)
        # Check for header (assuming saved via the generation script which might have offsets)
        # We'll try to guess based on standard int8 (1 byte per sample)
        
        # If it's a raw int8 file (memmap from generation script)
        n_samples = file_size  
        offset = 0
        
        # Heuristic: Check if standard .npy header exists
        if file_size > 128:
            header_magic = b'\x93NUMPY'
            with open(path, 'rb') as f:
                header = f.read(128)
                if header.startswith(header_magic):
                    offset = 128
                    n_samples = file_size - 128
        
        print(f"\nScanning {name}...")
        print(f"  > File Size: {file_size / 1024 / 1024:.2f} MB")
        print(f"  > Est. Samples: {n_samples}")
        
        # 2. Load and Count
        # We load in chunks to avoid RAM explosion if files are huge
        y = np.memmap(path, dtype='int8', mode='r', offset=offset, shape=(n_samples,))
        
        # Count positives (Value > 0)
        # We iterate in chunks of 1 million for safety
        total_pos = 0
        chunk_size = 1_000_000
        
        for i in range(0, n_samples, chunk_size):
            chunk = y[i : i + chunk_size]
            total_pos += np.sum(chunk > 0)
            
        pos_rate = (total_pos / n_samples) * 100
        print(f"  > Total Positives: {total_pos}")
        print(f"  > Positive Rate:   {pos_rate:.4f}%")
        
        if name == "y_train":
            if pos_rate < 1.0:
                print("  [CRITICAL WARNING] Training data has <1% positives. Model cannot learn.")
            elif pos_rate < 5.0:
                print("  [WARNING] Training positive rate is low (<5%). Consider higher POS_WEIGHT.")
            else:
                print("  [OK] Training balance looks healthy.")

if __name__ == "__main__":
    audit()