"""
Extract a mixed supervised Level 3 training cache from the current Parquet.

This script rebuilds data/processed/anomaly_detection/few_shot_train_level3_balanced.npz
from the active multimodal_features.parquet and the current train split manifest.
It is intended to keep supervised classical baselines temporally aligned with
the current lookahead label horizon.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.components.utils import get_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PARQUET_PATH = DATA_DIR / "multimodal_features.parquet"
ANOMALY_DIR = DATA_DIR / "anomaly_detection"
SPLIT_MANIFEST_PATH = ANOMALY_DIR / "split_manifest.json"
OUTPUT_PATH = ANOMALY_DIR / "few_shot_train_level3_balanced.npz"

SEQ_LEN = int(os.getenv("SEQ_LEN", "96"))
STEP_CAP = 2000
RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parquet_files(parquet_path: Path) -> list[Path]:
    return sorted(parquet_path.glob("*.parquet")) if parquet_path.is_dir() else [parquet_path]


def load_feature_columns(parquet_path: Path) -> list[str]:
    schema_df = pd.read_parquet(parquet_files(parquet_path)[0]).head(0)
    return get_feature_columns(schema_df)


def load_train_l3_seniors(split_manifest_path: Path) -> list[str]:
    if not split_manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {split_manifest_path}. Run 01_generate_dataset.py first."
        )
    with open(split_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    candidates = manifest.get("train_l3_excluded_seniors") or manifest.get("train_seniors")
    if not candidates:
        raise ValueError("No train seniors found in split manifest.")
    return list(map(str, candidates))


def clean_part(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "steps" in df.columns:
        df["steps"] = df["steps"].clip(upper=STEP_CAP)
    if "bp_trend" in df.columns:
        df["bp_trend"] = df["bp_trend"].fillna(0.0)
    return df.sort_values(["senior_id", "timestamp"]).reset_index(drop=True)


def iter_senior_frames(
    parquet_path: Path,
    seniors: Iterable[str],
    columns: list[str],
) -> Iterable[pd.DataFrame]:
    senior_set = set(map(str, seniors))
    for part_path in parquet_files(parquet_path):
        part = pd.read_parquet(part_path, columns=columns)
        part = part[part["senior_id"].astype(str).isin(senior_set)]
        if part.empty:
            continue
        part = clean_part(part)
        for _, df_senior in part.groupby("senior_id", sort=False):
            yield df_senior.reset_index(drop=True)


def collect_supervised_windows(
    parquet_path: Path,
    train_l3_seniors: list[str],
    feature_cols: list[str],
    seq_len: int,
) -> tuple[list[np.ndarray], list[np.ndarray], set[str], int]:
    read_columns = list(dict.fromkeys(["senior_id", "timestamp", "label_1", "label_2", "label_3"] + feature_cols))
    positive_windows: list[np.ndarray] = []
    negative_candidates: list[np.ndarray] = []
    positive_seniors: set[str] = set()
    scanned_seniors = 0

    for df_senior in iter_senior_frames(parquet_path, train_l3_seniors, read_columns):
        scanned_seniors += 1
        if len(df_senior) <= seq_len:
            continue
        feats = df_senior[feature_cols].to_numpy(dtype=np.float32, copy=True)
        l1 = df_senior["label_1"].to_numpy()
        l2 = df_senior["label_2"].to_numpy()
        l3 = df_senior["label_3"].to_numpy()
        normal = ((l1 == 0) & (l2 == 0) & (l3 == 0)).astype(np.int32)
        prefix = np.concatenate([[0], np.cumsum(normal)])
        senior_id = str(df_senior["senior_id"].iloc[0])
        senior_pos_starts: list[int] = []
        senior_neg_starts: list[int] = []

        for target_idx in range(seq_len, len(df_senior)):
            window_start = target_idx - seq_len
            lookback_is_normal = (prefix[target_idx] - prefix[window_start]) == seq_len
            if lookback_is_normal and l3[target_idx] == 1:
                senior_pos_starts.append(window_start)

        for target_idx in range(seq_len, len(df_senior), seq_len):
            window_start = target_idx - seq_len
            if (prefix[target_idx + 1] - prefix[window_start]) == (seq_len + 1):
                senior_neg_starts.append(window_start)

        if senior_pos_starts:
            positive_seniors.add(senior_id)
            positive_windows.extend(feats[start:start + seq_len] for start in senior_pos_starts)
            negative_candidates.extend(feats[start:start + seq_len] for start in senior_neg_starts)

        if scanned_seniors % 10 == 0:
            logger.info(
                "Scanned %s train Level 3 seniors; positives=%s; negative candidates=%s",
                scanned_seniors,
                len(positive_windows),
                len(negative_candidates),
            )

    return positive_windows, negative_candidates, positive_seniors, scanned_seniors


def save_balanced_cache(
    positive_windows: list[np.ndarray],
    negative_candidates: list[np.ndarray],
    positive_seniors: set[str],
    output_path: Path,
    feature_cols: list[str],
    seq_len: int,
) -> None:
    if not positive_windows:
        raise RuntimeError("No Level 3 positive training windows found in the train split.")
    if not negative_candidates:
        raise RuntimeError("No normal negative training windows found for positive train seniors.")

    n_sample = min(len(positive_windows), len(negative_candidates))
    rng = np.random.default_rng(RANDOM_SEED)
    pos_idx = rng.choice(len(positive_windows), size=n_sample, replace=False)
    neg_idx = rng.choice(len(negative_candidates), size=n_sample, replace=False)

    X_pos = np.asarray([positive_windows[idx] for idx in pos_idx], dtype=np.float32)
    X_neg = np.asarray([negative_candidates[idx] for idx in neg_idx], dtype=np.float32)
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.asarray([1] * n_sample + [0] * n_sample, dtype=np.int8)

    order = rng.permutation(len(y))
    X = X[order]
    y = y[order]

    metadata = {
        "source_parquet": str(PARQUET_PATH),
        "split_manifest": str(SPLIT_MANIFEST_PATH),
        "seq_len": seq_len,
        "target": "label_3 with target row immediately after the 96-step lookback",
        "lookback_policy": "positive and negative windows require all labels normal inside the lookback",
        "negative_policy": "sampled from the same train seniors that contributed positive windows",
        "positive_windows_available": len(positive_windows),
        "negative_candidates_available": len(negative_candidates),
        "sampled_per_class": int(n_sample),
        "positive_seniors": sorted(positive_seniors),
        "feature_cols": feature_cols,
        "random_seed": RANDOM_SEED,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y, metadata=json.dumps(metadata))
    logger.info(
        "Saved mixed supervised train cache: X=%s, y=%s, positives=%s -> %s",
        X.shape,
        y.shape,
        int(y.sum()),
        output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract balanced mixed Level 3 train cache.")
    parser.add_argument("--parquet", type=Path, default=PARQUET_PATH)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_cols = load_feature_columns(args.parquet)
    train_l3_seniors = load_train_l3_seniors(args.split_manifest)
    logger.info("Using %s train Level 3 candidate seniors from split manifest.", len(train_l3_seniors))
    logger.info("Feature columns: %s", len(feature_cols))

    positives, negatives, positive_seniors, scanned_seniors = collect_supervised_windows(
        args.parquet,
        train_l3_seniors,
        feature_cols,
        args.seq_len,
    )
    logger.info(
        "Scanned %s seniors; positive windows=%s; negative candidates=%s; positive seniors=%s",
        scanned_seniors,
        len(positives),
        len(negatives),
        len(positive_seniors),
    )
    save_balanced_cache(positives, negatives, positive_seniors, args.output, feature_cols, args.seq_len)


if __name__ == "__main__":
    main()
