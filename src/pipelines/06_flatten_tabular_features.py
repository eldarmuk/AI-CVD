"""
Window-level tabular feature extraction for horizontal baselines.

This pipeline converts each 96-step, 24-hour lookback window from
multimodal_features.parquet into a single row of summary statistics. It reuses
the senior-wise split manifest produced by 01_generate_dataset.py so classical
baselines are evaluated on the same subject-isolated partitions as the sequence
models.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from src.components.utils import get_feature_columns
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.components.utils import get_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "anomaly_detection"
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "multimodal_features.parquet"
SPLIT_MANIFEST_PATH = DATA_DIR / "split_manifest.json"

SEQ_LEN = int(os.getenv("SEQ_LEN", "96"))
STEP_CAP = 2000

PHYSIOLOGY_FEATURES = [
    "temperature",
    "heartrate",
    "sbp",
    "dbp",
    "saturation",
    "steps",
    "pulse_pressure",
    "shock_index",
    "hr_volatility",
    "bp_trend",
]

SPARSITY_FEATURES = [
    "time_since_last_temperature",
    "time_since_last_heartrate",
    "time_since_last_sbp",
    "time_since_last_dbp",
    "time_since_last_pulse_pressure",
    "time_since_last_steps",
    "time_since_last_saturation",
]

STATIC_FEATURES = [
    "age",
    "gender",
    "recent_event_burden",
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
]

TEMPORAL_SNAPSHOT_FEATURES = [
    "hour_sin",
    "hour_cos",
    "steps_rolling_sum_6h",
]

LABEL_COLUMNS = ["label_1", "label_2", "label_3"]
ID_COLUMNS = ["senior_id", "timestamp"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parquet_files(parquet_path: Path) -> list[Path]:
    return sorted(parquet_path.glob("*.parquet")) if parquet_path.is_dir() else [parquet_path]


def clean_part(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "steps" in df.columns:
        df["steps"] = df["steps"].clip(upper=STEP_CAP)
    if "bp_trend" in df.columns:
        df["bp_trend"] = df["bp_trend"].fillna(0.0)
    return df.sort_values(["senior_id", "timestamp"]).reset_index(drop=True)


def load_split_manifest(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {path}. Run src/pipelines/01_generate_dataset.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return {
        "train": list(map(str, manifest["train_seniors"])),
        "val": list(map(str, manifest["val_seniors"])),
        "test": list(map(str, manifest["test_seniors"])),
    }


def resolve_columns(parquet_path: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    first_part = parquet_files(parquet_path)[0]
    schema_df = pd.read_parquet(first_part).head(0)
    model_features = get_feature_columns(schema_df)
    physiology = [col for col in PHYSIOLOGY_FEATURES if col in model_features]
    sparsity = [col for col in SPARSITY_FEATURES if col in model_features]
    static = [col for col in STATIC_FEATURES if col in model_features]
    temporal = [col for col in TEMPORAL_SNAPSHOT_FEATURES if col in model_features]
    missing = sorted(
        (set(PHYSIOLOGY_FEATURES) | set(SPARSITY_FEATURES) | set(STATIC_FEATURES))
        - set(model_features)
    )
    if missing:
        logger.warning("Expected feature columns absent from parquet schema: %s", missing)
    return physiology, sparsity, static, temporal


def iter_senior_frames(
    parquet_path: Path,
    seniors: Iterable[str],
    read_columns: list[str],
) -> Iterable[pd.DataFrame]:
    senior_set = set(map(str, seniors))
    for part_path in parquet_files(parquet_path):
        part = pd.read_parquet(part_path, columns=read_columns)
        part = part[part["senior_id"].astype(str).isin(senior_set)]
        if part.empty:
            continue
        part = clean_part(part)
        for _, df_senior in part.groupby("senior_id", sort=False):
            yield df_senior.reset_index(drop=True)


def nan_skew(window: np.ndarray) -> np.ndarray:
    mean = np.nanmean(window, axis=0)
    centered = window - mean
    second = np.nanmean(centered ** 2, axis=0)
    third = np.nanmean(centered ** 3, axis=0)
    denom = np.power(second, 1.5)
    return np.divide(third, denom, out=np.zeros_like(third), where=denom > 1e-12)


def summarize_window(
    window: pd.DataFrame,
    target_row: pd.Series,
    physiology_cols: list[str],
    sparsity_cols: list[str],
    static_cols: list[str],
    temporal_cols: list[str],
) -> dict[str, float | int | str | pd.Timestamp]:
    row: dict[str, float | int | str | pd.Timestamp] = {
        "senior_id": target_row["senior_id"],
        "target_timestamp": target_row["timestamp"],
    }

    if physiology_cols:
        values = window[physiology_cols].to_numpy(dtype=np.float32, copy=True)
        stats = {
            "mean": np.nanmean(values, axis=0),
            "std": np.nanstd(values, axis=0),
            "min": np.nanmin(values, axis=0),
            "max": np.nanmax(values, axis=0),
            "median": np.nanmedian(values, axis=0),
            "skew": nan_skew(values),
            "last_value": values[-1],
            "delta": values[-1] - values[0],
        }
        for stat_name, stat_values in stats.items():
            for col, value in zip(physiology_cols, stat_values):
                row[f"{col}_{stat_name}"] = float(value) if np.isfinite(value) else np.nan

    if sparsity_cols:
        gaps = window[sparsity_cols].to_numpy(dtype=np.float32, copy=True)
        stats = {
            "max_gap": np.nanmax(gaps, axis=0),
            "mean_gap": np.nanmean(gaps, axis=0),
            "final_staleness": gaps[-1],
        }
        for stat_name, stat_values in stats.items():
            for col, value in zip(sparsity_cols, stat_values):
                row[f"{col}_{stat_name}"] = float(value) if np.isfinite(value) else np.nan

    for col in static_cols + temporal_cols:
        value = window[col].iloc[-1]
        row[col] = value

    return row


def extract_split(
    split_name: str,
    seniors: list[str],
    parquet_path: Path,
    output_dir: Path,
    seq_len: int,
    physiology_cols: list[str],
    sparsity_cols: list[str],
    static_cols: list[str],
    temporal_cols: list[str],
    max_windows: int | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    read_columns = list(
        dict.fromkeys(ID_COLUMNS + LABEL_COLUMNS + physiology_cols + sparsity_cols + static_cols + temporal_cols)
    )
    rows: list[dict] = []
    labels: list[int] = []
    pure_healthy: list[bool] = []

    for senior_idx, df_senior in enumerate(iter_senior_frames(parquet_path, seniors, read_columns), start=1):
        if len(df_senior) <= seq_len:
            continue
        label_matrix = df_senior[LABEL_COLUMNS].to_numpy(dtype=np.int8, copy=True)
        all_normal = (label_matrix.sum(axis=1) == 0).astype(np.int32)
        normal_prefix = np.concatenate([[0], np.cumsum(all_normal)])

        for target_idx in range(seq_len, len(df_senior)):
            window_start = target_idx - seq_len
            window = df_senior.iloc[window_start:target_idx]
            target_row = df_senior.iloc[target_idx]
            rows.append(
                summarize_window(
                    window,
                    target_row,
                    physiology_cols,
                    sparsity_cols,
                    static_cols,
                    temporal_cols,
                )
            )
            labels.append(int(target_row["label_3"]))
            input_is_normal = (normal_prefix[target_idx] - normal_prefix[window_start]) == seq_len
            pure_healthy.append(bool(input_is_normal and int(target_row["label_3"]) == 0))

            if max_windows is not None and len(rows) >= max_windows:
                logger.info("%s reached max_windows=%s", split_name, max_windows)
                break
        if max_windows is not None and len(rows) >= max_windows:
            break
        if senior_idx % 25 == 0:
            logger.info("%s seniors processed=%s; windows=%s", split_name, senior_idx, len(rows))

    if not rows:
        raise RuntimeError(f"No tabular windows generated for split '{split_name}'.")

    X = pd.DataFrame(rows)
    y = np.asarray(labels, dtype=np.int8)
    pure_mask = np.asarray(pure_healthy, dtype=bool)

    output_dir.mkdir(parents=True, exist_ok=True)
    X_path = output_dir / f"X_{split_name}_flat.parquet"
    y_path = output_dir / f"y_{split_name}_l3.npy"
    mask_path = output_dir / f"{split_name}_pure_healthy_mask.npy"

    X.to_parquet(X_path, index=False)
    np.save(y_path, y)
    np.save(mask_path, pure_mask)
    logger.info(
        "Saved %s: X=%s y=%s positives=%s pure_healthy=%s",
        split_name,
        X.shape,
        y.shape,
        int(y.sum()),
        int(pure_mask.sum()),
    )
    return X, y, pure_mask


def write_metadata(
    output_dir: Path,
    seq_len: int,
    physiology_cols: list[str],
    sparsity_cols: list[str],
    static_cols: list[str],
    temporal_cols: list[str],
    split_shapes: dict[str, dict],
) -> None:
    feature_names = [
        col
        for col in split_shapes["train"]["columns"]
        if col not in {"senior_id", "target_timestamp"}
    ]
    metadata = {
        "source_parquet": str(PARQUET_PATH),
        "split_manifest": str(SPLIT_MANIFEST_PATH),
        "seq_len": seq_len,
        "lookback_hours": seq_len * 5 / 60,
        "target": "label_3 at the row immediately after the lookback window",
        "physiology_columns": physiology_cols,
        "sparsity_columns": sparsity_cols,
        "static_columns": static_cols,
        "temporal_snapshot_columns": temporal_cols,
        "feature_names": feature_names,
        "split_shapes": split_shapes,
        "pure_healthy_mask_policy": (
            "True when all label_1/label_2/label_3 values are zero inside the "
            "lookback window and target label_3 is zero."
        ),
    }
    with open(output_dir / "flat_feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract tabular statistics from 96-step windows.")
    parser.add_argument("--parquet", type=Path, default=PARQUET_PATH)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument(
        "--max-windows-per-split",
        type=int,
        default=int(os.getenv("MAX_WINDOWS_PER_SPLIT", "0")),
        help="Optional smoke-test cap. Use 0 for all windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_windows = args.max_windows_per_split or None
    splits = load_split_manifest(args.split_manifest)
    physiology_cols, sparsity_cols, static_cols, temporal_cols = resolve_columns(args.parquet)

    split_shapes: dict[str, dict] = {}
    for split_name in ["train", "val", "test"]:
        X, y, pure_mask = extract_split(
            split_name,
            splits[split_name],
            args.parquet,
            args.output_dir,
            args.seq_len,
            physiology_cols,
            sparsity_cols,
            static_cols,
            temporal_cols,
            max_windows,
        )
        split_shapes[split_name] = {
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "positives": int(y.sum()),
            "pure_healthy": int(pure_mask.sum()),
            "columns": list(X.columns),
        }

    write_metadata(
        args.output_dir,
        args.seq_len,
        physiology_cols,
        sparsity_cols,
        static_cols,
        temporal_cols,
        split_shapes,
    )
    logger.info("Tabular feature extraction complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
