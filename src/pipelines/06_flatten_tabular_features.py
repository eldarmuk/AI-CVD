"""
Vectorized flattening of existing sequence datasets for tabular baselines.

This script deliberately reuses the curated 3D anomaly-detection arrays instead
of generating new sliding windows from Parquet. It converts each
`[n_samples, 96, n_features]` split into one tabular row per existing sample.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "anomaly_detection"
FEATURE_NAMES_PATH = PROJECT_ROOT / "data" / "processed" / "feature_names.txt"
STATS_PATH = DATA_DIR / "normalization_stats.json"

STATIC_FEATURES = {
    "age",
    "gender",
    "cardiovascular",
    "metabolic_endocrine",
    "neurological",
    "psychiatric_cognitive",
    "musculoskeletal",
    "respiratory",
    "gastro_renal_urologic",
    "oncological",
    "sensory",
    "other_functional_risk",
    "other",
}

SUMMARY_STATS = {
    "mean": np.nanmean,
    "std": np.nanstd,
    "max": np.nanmax,
    "min": np.nanmin,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def reduce_nan_stat(values: np.ndarray, reducer) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.", category=RuntimeWarning)
        return reducer(values, axis=1)


def load_feature_names(feature_names_path: Path, stats_path: Path) -> list[str]:
    if feature_names_path.exists():
        text = feature_names_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Feature names file is empty: {feature_names_path}")
        if text.startswith("["):
            names = json.loads(text)
        else:
            names = [line.strip() for line in text.splitlines() if line.strip()]
        return list(map(str, names))

    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        names = stats.get("feature_cols")
        if names:
            logger.warning(
                "%s not found; using feature_cols from %s.",
                feature_names_path,
                stats_path,
            )
            return list(map(str, names))

    raise FileNotFoundError(
        f"Could not resolve feature names. Expected {feature_names_path} or "
        f"feature_cols in {stats_path}."
    )


def load_targets(data_dir: Path, split: str, n_samples: int) -> np.ndarray:
    direct_path = data_dir / f"y_{split}.npy"
    l3_path = data_dir / f"y_{split}_l3.npy"

    if direct_path.exists():
        y = np.load(direct_path)
    elif split == "train":
        logger.warning("No train target file found; generating all-zero y_train_l3.npy.")
        y = np.zeros(n_samples, dtype=np.int8)
    elif l3_path.exists():
        y = np.load(l3_path)
    else:
        raise FileNotFoundError(f"Missing target array for split '{split}': {direct_path}")

    y = np.asarray(y).astype(np.int8)
    if y.shape[0] != n_samples:
        raise ValueError(f"Target length mismatch for {split}: X={n_samples}, y={y.shape[0]}")
    return y


def flatten_sequences(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    if X.ndim != 3:
        raise ValueError(f"Expected a 3D array [samples, time, features], got shape {X.shape}")
    if X.shape[2] != len(feature_names):
        raise ValueError(
            f"Feature name count mismatch: X has {X.shape[2]} features, "
            f"but {len(feature_names)} names were provided."
        )

    dynamic_indices = [idx for idx, name in enumerate(feature_names) if name not in STATIC_FEATURES]
    static_indices = [idx for idx, name in enumerate(feature_names) if name in STATIC_FEATURES]

    parts: list[np.ndarray] = []
    columns: list[str] = []

    if dynamic_indices:
        dynamic = X[:, :, dynamic_indices]
        for stat_name, reducer in SUMMARY_STATS.items():
            stat_values = reduce_nan_stat(dynamic, reducer)
            parts.append(stat_values)
            columns.extend(f"{feature_names[idx]}_{stat_name}" for idx in dynamic_indices)

    if static_indices:
        static_values = X[:, 0, static_indices]
        parts.append(static_values)
        columns.extend(feature_names[idx] for idx in static_indices)

    X_flat = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    return pd.DataFrame(X_flat, columns=columns)


def process_split(data_dir: Path, split: str, feature_names: list[str]) -> dict:
    x_path = data_dir / f"X_{split}.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing sequence array for split '{split}': {x_path}")

    logger.info("Loading %s", x_path)
    X = np.load(x_path)
    logger.info("%s shape: %s", split, X.shape)

    X_flat = flatten_sequences(X, feature_names)
    y = load_targets(data_dir, split, X.shape[0])

    x_out = data_dir / f"X_{split}_flat.parquet"
    y_out = data_dir / f"y_{split}_l3.npy"
    mask_out = data_dir / f"{split}_pure_healthy_mask.npy"

    X_flat.to_parquet(x_out, index=False)
    np.save(y_out, y)
    np.save(mask_out, y == 0)

    logger.info(
        "Saved %s: X_flat=%s, y=%s, positives=%s",
        split,
        X_flat.shape,
        y.shape,
        int(y.sum()),
    )
    return {
        "X_source": str(x_path),
        "X_flat_path": str(x_out),
        "y_path": str(y_out),
        "pure_healthy_mask_path": str(mask_out),
        "X_shape": list(X.shape),
        "X_flat_shape": list(X_flat.shape),
        "y_shape": list(y.shape),
        "positives": int(y.sum()),
        "columns": list(X_flat.columns),
    }


def write_metadata(data_dir: Path, feature_names: list[str], split_metadata: dict[str, dict]) -> None:
    dynamic_features = [name for name in feature_names if name not in STATIC_FEATURES]
    static_features = [name for name in feature_names if name in STATIC_FEATURES]
    metadata = {
        "source": "Existing curated anomaly_detection X_*.npy sequence arrays",
        "flattening": {
            "dynamic_temporal_features": dynamic_features,
            "static_features": static_features,
            "dynamic_temporal_stats": list(SUMMARY_STATS.keys()),
            "static_policy": "first time step value, X[:, 0, feature_index]",
            "sample_iteration": False,
        },
        "feature_names": feature_names,
        "splits": split_metadata,
    }
    with open(data_dir / "flat_feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectorized flattening of existing 3D sequence arrays for tabular baselines."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--feature-names", type=Path, default=FEATURE_NAMES_PATH)
    parser.add_argument("--stats-path", type=Path, default=STATS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_names = load_feature_names(args.feature_names, args.stats_path)
    split_metadata = {
        split: process_split(args.data_dir, split, feature_names)
        for split in ["train", "val", "test"]
    }
    write_metadata(args.data_dir, feature_names, split_metadata)
    logger.info("Vectorized tabular flattening complete: %s", args.data_dir)


if __name__ == "__main__":
    main()
