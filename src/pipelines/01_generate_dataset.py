"""
Phase 3 (Revision) - Anomaly Detection Dataset Generation

Generates sequences for unsupervised anomaly detection from multimodal_features.parquet.

Key Features:
- Splits seniors 80/20 (Train/Test)
- Training data: Only "Normal" windows (label_1=0, label_2=0, label_3=0)
- Test data: 50% Anomaly windows (label_3=1) + 50% Normal windows
- Preserves senior integrity (no mixing train/test seniors)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import random
import logging
from typing import Tuple, List
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent / 'data' / 'processed'
PARQUET_PATH = DATA_DIR / 'multimodal_features.parquet'
OUTPUT_DIR = DATA_DIR / 'anomaly_detection'
SEQ_LEN = 96
STEP_CAP = 2000
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42
TRAIN_STRIDE = 12

X_TRAIN_PATH = OUTPUT_DIR / 'X_train.npy'
X_TEST_PATH = OUTPUT_DIR / 'X_test.npy'
Y_TEST_PATH = OUTPUT_DIR / 'y_test.npy'
METADATA_PATH = OUTPUT_DIR / 'anomaly_dataset_metadata.txt'


def load_and_clean_data(parquet_path: Path) -> pd.DataFrame:
    """Load parquet file and apply basic cleaning."""
    logger.info(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    logger.info(f"Raw shape: {df.shape}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if 'steps' in df.columns:
        logger.info(f"Steps before capping - Min: {df['steps'].min()}, Max: {df['steps'].max()}")
        df['steps'] = df['steps'].clip(upper=STEP_CAP)
        logger.info(f"Steps after capping - Max: {df['steps'].max()}")
    
    df = df.sort_values(['senior_id', 'timestamp']).reset_index(drop=True)
    
    logger.info(f"Cleaned shape: {df.shape}")
    return df

def split_seniors(df: pd.DataFrame, train_split: float = 0.8) -> Tuple[List, List]:
    """
    Split unique senior_ids into Train and Test sets.
    Ensures no senior appears in both sets.
    """
    unique_seniors = df['senior_id'].unique()
    n_seniors = len(unique_seniors)
    
    logger.info(f"Total unique seniors: {n_seniors:,}")
    
    random.seed(RANDOM_SEED)
    shuffled_seniors = list(unique_seniors)
    random.shuffle(shuffled_seniors)
    
    split_idx = int(n_seniors * train_split)
    train_seniors = shuffled_seniors[:split_idx]
    test_seniors = shuffled_seniors[split_idx:]
    
    logger.info(f"Train seniors: {len(train_seniors):,} ({100*len(train_seniors)/n_seniors:.1f}%)")
    logger.info(f"Test seniors: {len(test_seniors):,} ({100*len(test_seniors)/n_seniors:.1f}%)")
    
    return train_seniors, test_seniors


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract feature columns (exclude metadata)."""
    exclude_cols = {'senior_id', 'timestamp', 'label_1', 'label_2', 'label_3', 
                    'hour', 'is_night', 'day_of_week'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")
    return feature_cols


def extract_sequences_from_senior(
    df_senior: pd.DataFrame,
    seq_len: int,
    stride: int = 1,
    require_all_normal: bool = False,
    normal_mask_series: pd.Series = None
) -> List[np.ndarray]:
    sequences = []
    n_rows = len(df_senior)
    
    if n_rows < seq_len:
        return sequences
    
    for start_idx in range(0, n_rows - seq_len + 1, stride):
        end_idx = start_idx + seq_len
        window = df_senior.iloc[start_idx:end_idx]
        
        if require_all_normal:
            window_normal_mask = normal_mask_series.iloc[start_idx:end_idx]
            if not window_normal_mask.all():
                continue
        
        sequences.append(window.values)
    
    return sequences


def generate_training_data(df, train_seniors, seq_len, feature_cols):
    logger.info(f"Generating TRAINING data (Stride={TRAIN_STRIDE})...")
    
    CHUNK_SIZE = 20000 
    chunk_dir = OUTPUT_DIR / 'temp_chunks'
    if chunk_dir.exists(): 
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    batch_buffer = []
    saved_chunks = []
    total_windows_count = 0
    
    normal_mask = df['senior_id'].isin(train_seniors) & \
                  (df['label_1'] == 0) & (df['label_2'] == 0) & (df['label_3'] == 0)
    normal_mask_series = pd.Series(normal_mask, index=df.index)

    for idx, senior_id in enumerate(train_seniors):
        senior_indices = df[df['senior_id'] == senior_id].index
        df_senior = df.loc[senior_indices].reset_index(drop=True)
        
        if len(df_senior) < seq_len:
            continue
            
        mask_senior = normal_mask_series.loc[senior_indices].reset_index(drop=True)
        senior_features = df_senior[feature_cols].values.astype(np.float32)
        sequences = []
        n_rows = len(df_senior)
        
        for start in range(0, n_rows - seq_len + 1, TRAIN_STRIDE):
            end = start + seq_len
            
            if not mask_senior.iloc[start:end].all():
                continue
                
            sequences.append(senior_features[start:end])

        if sequences:
            batch_buffer.extend(sequences)
            total_windows_count += len(sequences)
        
        # Dump chunk
        if len(batch_buffer) >= CHUNK_SIZE:
            chunk_path = chunk_dir / f"chunk_{len(saved_chunks)}.npy"
            np.save(chunk_path, np.array(batch_buffer, dtype=np.float32))
            saved_chunks.append(chunk_path)
            batch_buffer = []
            logger.info(f"   > Saved chunk {len(saved_chunks)}. Total: {total_windows_count:,}")

        if (idx + 1) % 500 == 0:
            logger.info(f"   Processed {idx + 1:,} train seniors...")

    if batch_buffer:
        chunk_path = chunk_dir / f"chunk_{len(saved_chunks)}.npy"
        np.save(chunk_path, np.array(batch_buffer, dtype=np.float32))
        saved_chunks.append(chunk_path)

    logger.info(f"Merging {len(saved_chunks)} chunks ({total_windows_count} windows)...")
    fp = np.memmap(X_TRAIN_PATH, dtype='float32', mode='w+', shape=(total_windows_count, seq_len, len(feature_cols)))
    
    idx = 0
    for chunk_path in saved_chunks:
        data = np.load(chunk_path)
        fp[idx : idx + len(data)] = data
        idx += len(data)
        fp.flush()
        os.remove(chunk_path)
    
    chunk_dir.rmdir()
    logger.info(f"Saved X_train.npy ({fp.nbytes / 1e9:.2f} GB)")
    return fp

def generate_testing_data(df, test_seniors, seq_len, feature_cols):
    logger.info("Generating TESTING data...")
    
    test_df = df[df['senior_id'].isin(test_seniors)].copy().reset_index(drop=True)
    pos_windows = []
    neg_candidates = []
    
    for senior_id in test_seniors:
        senior_indices = test_df[test_df['senior_id'] == senior_id].index
        df_senior = test_df.loc[senior_indices]
        
        if len(df_senior) < seq_len:
            continue
        
        feats = df_senior[feature_cols].values.astype(np.float32)
        
        l1 = df_senior['label_1'].values
        l2 = df_senior['label_2'].values
        l3 = df_senior['label_3'].values
        n_rows = len(df_senior)
        
        for i in range(seq_len, n_rows):
            if l3[i-1] == 1:
                pos_windows.append(feats[i-seq_len : i])
                
        for i in range(0, n_rows - seq_len + 1, seq_len):
            if (l1[i:i+seq_len].sum() == 0) and \
               (l2[i:i+seq_len].sum() == 0) and \
               (l3[i:i+seq_len].sum() == 0):
                neg_candidates.append(feats[i:i+seq_len])
                
    n_pos = len(pos_windows)
    n_neg = len(neg_candidates)
    
    if n_pos == 0:
        logger.warning("No anomaly windows found in Test set!")
        return np.array([]), np.array([])
        
    n_sample = min(n_pos, n_neg)
    logger.info(f"Found {n_pos} Anomaly windows vs {n_neg} Normal candidates.")
    logger.info(f"Sampling {n_sample} balanced windows each (Total {n_sample*2}).")
    
    random.seed(42)
    neg_windows = random.sample(neg_candidates, n_sample)
    
    X = np.array(pos_windows[:n_sample] + neg_windows, dtype=np.float32)
    y = np.array([1] * n_sample + [0] * n_sample, dtype=np.int32)
    
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    
    return X[idx], y[idx]


def save_sequences(X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
    logger.info(f"\nSaving sequences to {OUTPUT_DIR}...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if isinstance(X_train, np.memmap):
        X_train.flush()
        logger.info(f"  + Flushed X_train.npy (Already saved via memmap): {X_train.shape}")
    else:
        np.save(X_TRAIN_PATH, X_train)
        logger.info(f"  + Saved X_train.npy: {X_train.shape}")
    
    np.save(X_TEST_PATH, X_test)
    logger.info(f"  + Saved X_test.npy: {X_test.shape}")
    
    np.save(Y_TEST_PATH, y_test)
    logger.info(f"  + Saved y_test.npy: {y_test.shape}")

def save_metadata(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_train_seniors: int,
    n_test_seniors: int,
    feature_cols: List[str]
) -> None:
    n_anomalies = np.sum(y_test)
    n_normal = len(y_test) - n_anomalies
    
    metadata = f"""Anomaly Detection Dataset

Configuration:
  Sequence length: {SEQ_LEN}
  Step cap: {STEP_CAP}
  Train/Test split: {100*TRAIN_SPLIT:.0f}/{100*(1-TRAIN_SPLIT):.0f}
  Train seniors: {n_train_seniors:,}
  Test seniors: {n_test_seniors:,}

Training Data:
  Shape: {X_train.shape}
  Samples: {len(X_train):,}
  (All normal windows)

Testing Data:
  Shape: {X_test.shape}
  Anomalies: {n_anomalies:,}
  Normal: {n_normal:,}

Features: {len(feature_cols)}
"""
    
    with open(METADATA_PATH, 'w') as f:
        f.write(metadata)
    
    logger.info("  + Saved anomaly_dataset_metadata.txt")


def main():
    logger.info("=" * 100)
    logger.info("PHASE 3 (REVISION) - ANOMALY DETECTION DATASET GENERATION")
    logger.info("=" * 100)
    
    df = load_and_clean_data(PARQUET_PATH)
    train_seniors, test_seniors = split_seniors(df, TRAIN_SPLIT)
    feature_cols = get_feature_columns(df)
    X_train = generate_training_data(df, train_seniors, SEQ_LEN, feature_cols)
    X_test, y_test = generate_testing_data(df, test_seniors, SEQ_LEN, feature_cols)
    
    if len(X_train) == 0:
        logger.error("Training data is empty! Aborting.")
        return
    
    if len(X_test) == 0:
        logger.error("Test data is empty! Aborting.")
        return
    
    save_sequences(X_train, X_test, y_test)
    save_metadata(X_train, X_test, y_test, len(train_seniors), len(test_seniors), feature_cols)
    
    logger.info("\n" + "=" * 100)
    logger.info("DATASET GENERATION COMPLETE")
    logger.info("=" * 100)


if __name__ == '__main__':
    main()
