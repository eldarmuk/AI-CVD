"""
Phase 3 (Revision) - Anomaly Detection Dataset Generation

Generates sequences for unsupervised anomaly detection from multimodal_features.parquet.

Key Features:
- Splits seniors 70/15/15 (Train/Val/Test)
- Training data: Only "Normal" windows (label_1=0, label_2=0, label_3=0)
- Val/Test data: 50% Anomaly windows (label_3=1) + 50% Normal windows
- Preserves senior integrity (no mixing train/val/test seniors)
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import random
import logging
from typing import Tuple, List, Dict
import shutil

from src.components.utils import get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'processed'
PARQUET_PATH = DATA_DIR / 'multimodal_features.parquet'
PROCESSED_DB_PATH = Path(__file__).parent.parent.parent / 'db' / 'hrp_processed.db'
OUTPUT_DIR = DATA_DIR / 'anomaly_detection'
SEQ_LEN = int(os.getenv("SEQ_LEN", "96"))
STEP_CAP = 2000
RANDOM_SEED = 42
TRAIN_STRIDE = 11
TRAIN_MAX_WINDOWS = int(os.getenv("TRAIN_MAX_WINDOWS", "500000"))

X_TRAIN_PATH = OUTPUT_DIR / 'X_train.npy'
X_VAL_PATH = OUTPUT_DIR / 'X_val.npy'
X_TEST_PATH = OUTPUT_DIR / 'X_test.npy'
Y_VAL_PATH = OUTPUT_DIR / 'y_val.npy'
Y_TEST_PATH = OUTPUT_DIR / 'y_test.npy'
METADATA_PATH = OUTPUT_DIR / 'anomaly_dataset_metadata.txt'
STATS_PATH = OUTPUT_DIR / 'normalization_stats.json'
SPLIT_MANIFEST_PATH = OUTPUT_DIR / 'split_manifest.json'
TRAIN_ROWS_PATH = OUTPUT_DIR / 'train_rows.parquet'
VAL_ROWS_PATH = OUTPUT_DIR / 'val_rows.parquet'
TEST_ROWS_PATH = OUTPUT_DIR / 'test_rows.parquet'


def get_parquet_scan_path(parquet_path: Path) -> str:
    """Return a DuckDB read_parquet path for a file or Parquet dataset directory."""
    if parquet_path.is_dir():
        return (parquet_path / "*.parquet").as_posix()
    return parquet_path.as_posix()


def discover_dataset_metadata(parquet_path: Path) -> Tuple[List[str], List, set]:
    """Read only metadata/senior ids via DuckDB; never materialize all rows."""
    import duckdb

    scan_path = get_parquet_scan_path(parquet_path)
    con = duckdb.connect()
    columns = [
        row[0] for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{scan_path}')"
        ).fetchall()
    ]
    seniors = [
        row[0] for row in con.execute(
            f"""
            SELECT DISTINCT senior_id
            FROM read_parquet('{scan_path}')
            ORDER BY senior_id
            """
        ).fetchall()
    ]
    l3_seniors = {
        row[0] for row in con.execute(
            f"""
            SELECT senior_id
            FROM read_parquet('{scan_path}')
            GROUP BY senior_id
            HAVING max(CAST(label_3 AS INTEGER)) = 1
            """
        ).fetchall()
    }
    con.close()
    return columns, seniors, l3_seniors


def discover_l3_alert_seniors_from_db(db_path: Path) -> set:
    """Read true Level 3 alert history from the processed database when available."""
    if not db_path.exists():
        logger.warning(f"Processed DB not found for Level 3 senior audit: {db_path}")
        return set()

    conn = sqlite3.connect(db_path)
    try:
        alert_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(alerts)").fetchall()
        }
        if 'severity' not in alert_cols:
            logger.warning(
                "Skipping DB Level 3 senior audit because alerts.severity is absent; "
                "using Parquet label_3 history only."
            )
            return set()
        rows = conn.execute(
            "SELECT DISTINCT senior_id FROM alerts WHERE severity = 3"
        ).fetchall()
    finally:
        conn.close()
    return {row[0] for row in rows}


def split_senior_ids(unique_seniors: List) -> Tuple[List, List, List]:
    """Split already-discovered senior ids 70/15/15."""
    n_seniors = len(unique_seniors)
    random.seed(RANDOM_SEED)
    shuffled = list(unique_seniors)
    random.shuffle(shuffled)

    train_idx = int(n_seniors * 0.70)
    val_idx = int(n_seniors * 0.85)

    train_seniors = shuffled[:train_idx]
    val_seniors = shuffled[train_idx:val_idx]
    test_seniors = shuffled[val_idx:]

    logger.info(f"Train: {len(train_seniors)}, Val: {len(val_seniors)}, Test: {len(test_seniors)}")
    assert_disjoint_splits({
        'train': train_seniors,
        'val': val_seniors,
        'test': test_seniors,
    })
    return train_seniors, val_seniors, test_seniors


def write_split_manifest(
    train_seniors: List,
    train_pure_healthy_seniors: List,
    train_l3_excluded_seniors: List,
    val_seniors: List,
    test_seniors: List,
) -> None:
    """Persist subject-wise split ids without duplicating the full Parquet dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_map = {
        'train': list(train_seniors),
        'val': list(val_seniors),
        'test': list(test_seniors),
    }
    assert_disjoint_splits(split_map)
    manifest = {
        'random_seed': RANDOM_SEED,
        'split_unit': 'senior_id',
        'train_seniors': list(map(str, train_seniors)),
        'train_pure_healthy_seniors': list(map(str, train_pure_healthy_seniors)),
        'train_l3_excluded_seniors': list(map(str, train_l3_excluded_seniors)),
        'val_seniors': list(map(str, val_seniors)),
        'test_seniors': list(map(str, test_seniors)),
        'train_policy': 'X_train uses only train_pure_healthy_seniors: senior_id with zero label_3 rows in entire history.',
        'note': 'Rows are streamed from multimodal_features.parquet; full split parquet copies are not materialized.',
    }
    with open(SPLIT_MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  + Saved split manifest -> {SPLIT_MANIFEST_PATH}")


def iter_parquet_files(parquet_path: Path) -> List[Path]:
    if parquet_path.is_dir():
        return sorted(parquet_path.glob("*.parquet"))
    return [parquet_path]


def clean_streamed_part(df: pd.DataFrame) -> pd.DataFrame:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'steps' in df.columns:
        df['steps'] = df['steps'].clip(upper=STEP_CAP)
    if 'bp_trend' in df.columns:
        df['bp_trend'] = df['bp_trend'].fillna(0.0)
    return df.sort_values(['senior_id', 'timestamp']).reset_index(drop=True)


def iter_senior_frames(
    parquet_path: Path,
    seniors: set,
    columns: List[str],
) -> pd.DataFrame:
    for part_path in iter_parquet_files(parquet_path):
        df_part = pd.read_parquet(part_path, columns=columns)
        df_part = df_part[df_part['senior_id'].isin(seniors)]
        if df_part.empty:
            continue
        df_part = clean_streamed_part(df_part)
        for _, df_senior in df_part.groupby('senior_id', sort=False):
            yield df_senior.reset_index(drop=True)


def normal_window_starts(df_senior: pd.DataFrame, seq_len: int, stride: int) -> np.ndarray:
    n_rows = len(df_senior)
    if n_rows < seq_len:
        return np.array([], dtype=np.int64)

    normal = (
        (df_senior['label_1'].to_numpy() == 0) &
        (df_senior['label_2'].to_numpy() == 0) &
        (df_senior['label_3'].to_numpy() == 0)
    ).astype(np.int32)
    prefix = np.concatenate([[0], np.cumsum(normal)])
    starts = np.arange(0, n_rows - seq_len + 1, stride, dtype=np.int64)
    valid = (prefix[starts + seq_len] - prefix[starts]) == seq_len
    return starts[valid]


def count_training_windows_streaming(
    parquet_path: Path,
    train_seniors: List,
    read_columns: List[str],
    seq_len: int,
) -> int:
    total = 0
    for idx, df_senior in enumerate(iter_senior_frames(parquet_path, set(train_seniors), read_columns)):
        total += len(normal_window_starts(df_senior, seq_len, TRAIN_STRIDE))
        if (idx + 1) % 500 == 0:
            logger.info(f"   Counted {idx + 1:,} train seniors; candidate windows: {total:,}")
    return total


def generate_training_data_streaming(
    parquet_path: Path,
    train_seniors: List,
    seq_len: int,
    feature_cols: List[str],
    read_columns: List[str],
):
    logger.info(f"Generating TRAINING data stream (Stride={TRAIN_STRIDE})...")
    total_candidates = count_training_windows_streaming(parquet_path, train_seniors, read_columns, seq_len)
    if total_candidates == 0:
        return np.array([])

    n_selected = min(total_candidates, TRAIN_MAX_WINDOWS) if TRAIN_MAX_WINDOWS > 0 else total_candidates
    rng = np.random.default_rng(RANDOM_SEED)
    selected_global = np.sort(rng.choice(total_candidates, size=n_selected, replace=False))

    logger.info(f"Training normal windows: {total_candidates:,}; selected: {n_selected:,}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train = np.lib.format.open_memmap(
        X_TRAIN_PATH,
        dtype='float32',
        mode='w+',
        shape=(n_selected, seq_len, len(feature_cols)),
    )

    global_idx = 0
    write_idx = 0
    selected_ptr = 0
    selected_count = len(selected_global)

    for senior_idx, df_senior in enumerate(iter_senior_frames(parquet_path, set(train_seniors), read_columns)):
        starts = normal_window_starts(df_senior, seq_len, TRAIN_STRIDE)
        if len(starts) == 0:
            continue

        feats = df_senior[feature_cols].to_numpy(dtype=np.float32, copy=True)
        for start in starts:
            if selected_ptr >= selected_count:
                break
            if global_idx == selected_global[selected_ptr]:
                X_train[write_idx] = feats[start:start + seq_len]
                write_idx += 1
                selected_ptr += 1
                if write_idx % 25000 == 0:
                    X_train.flush()
                    logger.info(f"   Wrote {write_idx:,}/{n_selected:,} train windows")
            global_idx += 1

        if (senior_idx + 1) % 500 == 0:
            logger.info(f"   Processed {senior_idx + 1:,} train seniors")

    X_train.flush()
    logger.info(f"Saved X_train.npy ({X_train.nbytes / 1e9:.2f} GB)")
    return X_train


def collect_eval_candidates_streaming(
    parquet_path: Path,
    seniors: List,
    seq_len: int,
    feature_cols: List[str],
    read_columns: List[str],
    split_label: str,
) -> Tuple[List[np.ndarray], int]:
    pos_windows = []
    neg_count = 0

    for df_senior in iter_senior_frames(parquet_path, set(seniors), read_columns):
        if len(df_senior) <= seq_len:
            continue
        feats = df_senior[feature_cols].to_numpy(dtype=np.float32, copy=True)
        l1 = df_senior['label_1'].to_numpy()
        l2 = df_senior['label_2'].to_numpy()
        l3 = df_senior['label_3'].to_numpy()
        n_rows = len(df_senior)

        normal = ((l1 == 0) & (l2 == 0) & (l3 == 0)).astype(np.int32)
        prefix = np.concatenate([[0], np.cumsum(normal)])

        for target_idx in range(seq_len, n_rows):
            window_start = target_idx - seq_len
            window_is_pure_normal = (prefix[target_idx] - prefix[window_start]) == seq_len
            if window_is_pure_normal and l3[target_idx] == 1:
                pos_windows.append(feats[window_start:target_idx])

        neg_targets = np.arange(seq_len, n_rows, seq_len, dtype=np.int64)
        if len(neg_targets):
            neg_valid = (prefix[neg_targets + 1] - prefix[neg_targets - seq_len]) == (seq_len + 1)
            neg_count += int(neg_valid.sum())

    logger.info(f"{split_label}: positive windows={len(pos_windows):,}; negative candidates={neg_count:,}")
    return pos_windows, neg_count


def sample_negative_windows_streaming(
    parquet_path: Path,
    seniors: List,
    seq_len: int,
    feature_cols: List[str],
    read_columns: List[str],
    selected_neg_indices: np.ndarray,
) -> List[np.ndarray]:
    neg_windows = []
    global_neg_idx = 0
    selected_ptr = 0
    selected_count = len(selected_neg_indices)

    for df_senior in iter_senior_frames(parquet_path, set(seniors), read_columns):
        if len(df_senior) <= seq_len or selected_ptr >= selected_count:
            continue
        feats = df_senior[feature_cols].to_numpy(dtype=np.float32, copy=True)
        l1 = df_senior['label_1'].to_numpy()
        l2 = df_senior['label_2'].to_numpy()
        l3 = df_senior['label_3'].to_numpy()
        n_rows = len(df_senior)

        normal = ((l1 == 0) & (l2 == 0) & (l3 == 0)).astype(np.int32)
        prefix = np.concatenate([[0], np.cumsum(normal)])

        for target_idx in range(seq_len, n_rows, seq_len):
            if selected_ptr >= selected_count:
                break
            window_start = target_idx - seq_len
            is_negative = (prefix[target_idx + 1] - prefix[window_start]) == (seq_len + 1)
            if not is_negative:
                continue
            if global_neg_idx == selected_neg_indices[selected_ptr]:
                neg_windows.append(feats[window_start:target_idx])
                selected_ptr += 1
            global_neg_idx += 1

    return neg_windows


def generate_testing_data_streaming(
    parquet_path: Path,
    seniors: List,
    seq_len: int,
    feature_cols: List[str],
    read_columns: List[str],
    split_label: str,
):
    logger.info(f"Generating {split_label} data stream...")
    pos_windows, neg_count = collect_eval_candidates_streaming(
        parquet_path, seniors, seq_len, feature_cols, read_columns, split_label
    )
    n_pos = len(pos_windows)
    if n_pos == 0:
        logger.warning(f"No anomaly windows found in {split_label} set!")
        return np.array([]), np.array([])

    n_sample = min(n_pos, neg_count)
    rng = np.random.default_rng(RANDOM_SEED)
    selected_neg_indices = np.sort(rng.choice(neg_count, size=n_sample, replace=False))
    neg_windows = sample_negative_windows_streaming(
        parquet_path, seniors, seq_len, feature_cols, read_columns, selected_neg_indices
    )

    X = np.array(pos_windows[:n_sample] + neg_windows, dtype=np.float32)
    y = np.array([1] * n_sample + [0] * n_sample, dtype=np.int32)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def write_normalization_stats(
    X_train_path: Path,
    feature_cols: List[str],
    train_pure_healthy_seniors: List,
) -> None:
    """Fit normalization stats on X_train only and persist reproducibility metadata."""
    X = np.load(X_train_path, mmap_mode='r')
    if X.ndim != 3:
        raise ValueError(f"Expected X_train to have shape (n, seq_len, n_features), got {X.shape}")

    n_features = X.shape[-1]
    feat_sum = np.zeros(n_features, dtype=np.float64)
    feat_sq_sum = np.zeros(n_features, dtype=np.float64)
    feat_count = np.zeros(n_features, dtype=np.float64)

    chunk_size = 25_000
    for start in range(0, X.shape[0], chunk_size):
        chunk = np.asarray(X[start:start + chunk_size]).reshape(-1, n_features)
        mask = ~np.isnan(chunk)
        valid = np.nan_to_num(chunk, nan=0.0)
        feat_sum += valid.sum(axis=0)
        feat_sq_sum += (valid ** 2).sum(axis=0)
        feat_count += mask.sum(axis=0)

    mean = feat_sum / np.maximum(feat_count, 1)
    mean_sq = feat_sq_sum / np.maximum(feat_count, 1)
    var = np.maximum(mean_sq - mean ** 2, 0)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0

    stats = {
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "feature_cols": feature_cols,
        "fit_split": "train",
        "fit_cohort": "pure_healthy_train_seniors_zero_l3_history",
        "train_pure_healthy_seniors": list(map(str, train_pure_healthy_seniors)),
        "source_path": str(X_train_path.resolve()),
        "source_shape": list(X.shape),
        "seq_len": int(X.shape[1]),
        "n_features": int(X.shape[2]),
        "nan_policy": "ignore_when_fitting_fill_with_train_mean_when_transforming",
    }
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  + Saved train-only normalization stats -> {STATS_PATH}")


def assert_disjoint_splits(split_map: Dict[str, List]) -> None:
    """Fail fast if a senior appears in more than one split."""
    split_sets = {name: set(seniors) for name, seniors in split_map.items()}
    pairs = [('train', 'val'), ('train', 'test'), ('val', 'test')]
    overlaps = {
        f"{left}_{right}": sorted(split_sets[left] & split_sets[right])
        for left, right in pairs
        if split_sets[left] & split_sets[right]
    }

    if overlaps:
        preview = {name: values[:10] for name, values in overlaps.items()}
        raise ValueError(f"Senior-wise split leakage detected: {preview}")


def write_subjectwise_split_artifacts(
    df: pd.DataFrame,
    train_seniors: List,
    val_seniors: List,
    test_seniors: List
) -> None:
    """
    Persist row-level split parquet files before rolling windows are generated.
    These files are useful for auditing that no senior_id crosses train/val/test.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_map = {
        'train': list(train_seniors),
        'val': list(val_seniors),
        'test': list(test_seniors),
    }
    assert_disjoint_splits(split_map)

    split_paths = {
        'train': TRAIN_ROWS_PATH,
        'val': VAL_ROWS_PATH,
        'test': TEST_ROWS_PATH,
    }

    for split_name, seniors in split_map.items():
        split_df = df[df['senior_id'].isin(seniors)].copy()
        observed_seniors = set(split_df['senior_id'].unique())
        missing_seniors = set(seniors) - observed_seniors
        if missing_seniors:
            raise ValueError(
                f"{split_name} split is missing seniors after filtering: "
                f"{sorted(missing_seniors)[:10]}"
            )
        split_df.to_parquet(split_paths[split_name], index=False)
        logger.info(
            f"  + Saved {split_name} split rows: {split_df.shape} -> "
            f"{split_paths[split_name]}"
        )

    manifest = {
        'random_seed': RANDOM_SEED,
        'split_unit': 'senior_id',
        'train_seniors': list(map(str, train_seniors)),
        'val_seniors': list(map(str, val_seniors)),
        'test_seniors': list(map(str, test_seniors)),
        'paths': {name: str(path) for name, path in split_paths.items()},
    }
    with open(SPLIT_MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"  + Saved split manifest -> {SPLIT_MANIFEST_PATH}")


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

def split_seniors(df: pd.DataFrame) -> Tuple[List, List, List]:
    """Split 70/15/15 for train/val/test."""
    unique_seniors = df['senior_id'].unique()
    n_seniors = len(unique_seniors)
    
    random.seed(RANDOM_SEED)
    shuffled = list(unique_seniors)
    random.shuffle(shuffled)
    
    train_idx = int(n_seniors * 0.70)
    val_idx = int(n_seniors * 0.85)
    
    train_seniors = shuffled[:train_idx]
    val_seniors = shuffled[train_idx:val_idx]
    test_seniors = shuffled[val_idx:]
    
    logger.info(f"Train: {len(train_seniors)}, Val: {len(val_seniors)}, Test: {len(test_seniors)}")
    assert_disjoint_splits({
        'train': train_seniors,
        'val': val_seniors,
        'test': test_seniors,
    })
    
    return train_seniors, val_seniors, test_seniors

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

    train_df = df[df['senior_id'].isin(train_seniors)].copy()

    for idx, (senior_id, df_senior_raw) in enumerate(train_df.groupby('senior_id', sort=False)):
        df_senior = df_senior_raw.reset_index(drop=True)
        
        if len(df_senior) < seq_len:
            continue
            
        mask_senior = normal_mask_series.loc[df_senior_raw.index].reset_index(drop=True)
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

def generate_testing_data(df, test_seniors, seq_len, feature_cols, split_label: str = "TEST"):
    logger.info(f"Generating {split_label} data...")
    
    test_df = df[df['senior_id'].isin(test_seniors)].copy().reset_index(drop=True)
    pos_windows = []
    neg_candidates = []
    
    for senior_id, df_senior in test_df.groupby('senior_id', sort=False):
        df_senior = df_senior.sort_values('timestamp').reset_index(drop=True)
        
        if len(df_senior) <= seq_len:
            continue
        
        feats = df_senior[feature_cols].values.astype(np.float32)
        
        l1 = df_senior['label_1'].values
        l2 = df_senior['label_2'].values
        l3 = df_senior['label_3'].values
        n_rows = len(df_senior)
        
        # The target row is deliberately outside the lookback window. Requiring
        # normal labels inside the input prevents the target from being derived
        # from any timestamp contained in X[t-window_size:t].
        for target_idx in range(seq_len, n_rows):
            window_start = target_idx - seq_len
            window_end = target_idx
            window_is_pure_normal = (
                l1[window_start:window_end].sum() == 0 and
                l2[window_start:window_end].sum() == 0 and
                l3[window_start:window_end].sum() == 0
            )
            if window_is_pure_normal and l3[target_idx] == 1:
                pos_windows.append(feats[window_start:window_end])
                
        for target_idx in range(seq_len, n_rows, seq_len):
            window_start = target_idx - seq_len
            window_end = target_idx
            if (l1[window_start:window_end + 1].sum() == 0) and \
               (l2[window_start:window_end + 1].sum() == 0) and \
               (l3[window_start:window_end + 1].sum() == 0):
                neg_candidates.append(feats[window_start:window_end])
                
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


def save_sequences(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> None:
    logger.info(f"\nSaving sequences to {OUTPUT_DIR}...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if isinstance(X_train, np.memmap):
        X_train.flush()
        logger.info(f"  + Flushed X_train.npy (Already saved via memmap): {X_train.shape}")
    else:
        np.save(X_TRAIN_PATH, X_train)
        logger.info(f"  + Saved X_train.npy: {X_train.shape}")
    
    np.save(X_VAL_PATH, X_val)
    logger.info(f"  + Saved X_val.npy: {X_val.shape}")
    np.save(X_TEST_PATH, X_test)
    logger.info(f"  + Saved X_test.npy: {X_test.shape}")
    
    np.save(Y_VAL_PATH, y_val)
    logger.info(f"  + Saved y_val.npy: {y_val.shape}")
    np.save(Y_TEST_PATH, y_test)
    logger.info(f"  + Saved y_test.npy: {y_test.shape}")

def save_metadata(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_seniors: List,
    train_pure_healthy_seniors: List,
    train_l3_excluded_seniors: List,
    val_seniors: List,
    test_seniors: List,
    feature_cols: List[str]
) -> None:
    n_val_anomalies = np.sum(y_val)
    n_val_normal = len(y_val) - n_val_anomalies
    n_test_anomalies = np.sum(y_test)
    n_test_normal = len(y_test) - n_test_anomalies
    
    metadata = f"""Anomaly Detection Dataset

Configuration:
  Sequence length: {SEQ_LEN}
  Step cap: {STEP_CAP}
  Train split seniors: {len(train_seniors):,}
  Pure healthy train seniors used for X_train: {len(train_pure_healthy_seniors):,}
  Train seniors excluded due to any Level 3 history: {len(train_l3_excluded_seniors):,}
  Val seniors: {len(val_seniors):,}
  Test seniors: {len(test_seniors):,}
  Random seed: {RANDOM_SEED}
  Train max windows: {TRAIN_MAX_WINDOWS}

Training Data:
  Shape: {X_train.shape}
  Samples: {len(X_train):,}
  Cohort: senior_id with zero label_3 rows in entire history
  Windows: all labels normal within lookback

Testing Data:
  Shape: {X_test.shape}
    Anomalies: {n_test_anomalies:,}
    Normal: {n_test_normal:,}

Validation Data:
    Shape: {X_val.shape}
    Anomalies: {n_val_anomalies:,}
    Normal: {n_val_normal:,}

Features: {len(feature_cols)}
Feature Columns:
{json.dumps(feature_cols)}

Senior ID Distributions:
  train_seniors: {json.dumps(list(map(str, train_seniors)))}
  train_pure_healthy_seniors: {json.dumps(list(map(str, train_pure_healthy_seniors)))}
  train_l3_excluded_seniors: {json.dumps(list(map(str, train_l3_excluded_seniors)))}
  val_seniors: {json.dumps(list(map(str, val_seniors)))}
  test_seniors: {json.dumps(list(map(str, test_seniors)))}
"""
    
    with open(METADATA_PATH, 'w') as f:
        f.write(metadata)
    
    logger.info("  + Saved anomaly_dataset_metadata.txt")


def main():
    logger.info("=" * 100)
    logger.info("PHASE 3 (REVISION) - ANOMALY DETECTION DATASET GENERATION")
    logger.info("=" * 100)

    logger.info(f"Discovering Parquet metadata from {PARQUET_PATH}...")
    columns, unique_seniors, parquet_l3_seniors = discover_dataset_metadata(PARQUET_PATH)
    db_l3_seniors = discover_l3_alert_seniors_from_db(PROCESSED_DB_PATH)
    l3_seniors = parquet_l3_seniors | db_l3_seniors
    logger.info(
        f"Columns: {len(columns)}; seniors: {len(unique_seniors):,}; "
        f"seniors with Level 3 history: {len(l3_seniors):,} "
        f"(parquet labels={len(parquet_l3_seniors):,}, db alerts={len(db_l3_seniors):,})"
    )

    train_seniors, val_seniors, test_seniors = split_senior_ids(unique_seniors)
    train_pure_healthy_seniors = [sid for sid in train_seniors if sid not in l3_seniors]
    train_l3_excluded_seniors = [sid for sid in train_seniors if sid in l3_seniors]
    logger.info(
        f"Pure healthy train seniors: {len(train_pure_healthy_seniors):,}; "
        f"excluded train seniors with Level 3 history: {len(train_l3_excluded_seniors):,}"
    )
    write_split_manifest(
        train_seniors,
        train_pure_healthy_seniors,
        train_l3_excluded_seniors,
        val_seniors,
        test_seniors,
    )

    feature_cols = get_feature_columns(pd.DataFrame(columns=columns))
    read_columns = list(dict.fromkeys(['senior_id', 'timestamp', 'label_1', 'label_2', 'label_3'] + feature_cols))
    logger.info(f"Model features: {len(feature_cols)}")
    logger.info(f"TRAIN_MAX_WINDOWS: {TRAIN_MAX_WINDOWS:,} (set env TRAIN_MAX_WINDOWS=0 for all windows)")

    X_train = generate_training_data_streaming(
        PARQUET_PATH, train_pure_healthy_seniors, SEQ_LEN, feature_cols, read_columns
    )
    X_val, y_val = generate_testing_data_streaming(
        PARQUET_PATH, val_seniors, SEQ_LEN, feature_cols, read_columns, split_label="VALIDATION"
    )
    X_test, y_test = generate_testing_data_streaming(
        PARQUET_PATH, test_seniors, SEQ_LEN, feature_cols, read_columns, split_label="TEST"
    )
    
    if len(X_train) == 0:
        logger.error("Training data is empty! Aborting.")
        return
    
    if len(X_val) == 0:
        logger.error("Validation data is empty! Aborting.")
        return

    if len(X_test) == 0:
        logger.error("Test data is empty! Aborting.")
        return
    
    save_sequences(X_train, X_val, y_val, X_test, y_test)
    write_normalization_stats(X_TRAIN_PATH, feature_cols, train_pure_healthy_seniors)
    save_metadata(
        X_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_seniors,
        train_pure_healthy_seniors,
        train_l3_excluded_seniors,
        val_seniors,
        test_seniors,
        feature_cols
    )
    
    logger.info("\n" + "=" * 100)
    logger.info("DATASET GENERATION COMPLETE")
    logger.info("=" * 100)


if __name__ == '__main__':
    main()
