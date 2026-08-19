"""Train Hybrid-CGTA Net on aligned sequence and tabular inputs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.models.hybrid_cgta_net import HybridCGTANet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "anomaly_detection"
MODEL_DIR = PROJECT_ROOT / "models" / "hybrid_cgta"
REPORT_DIR = PROJECT_ROOT / "reports"
HYBRID_REPORT_DIR = REPORT_DIR / "hybrid_cgta"

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
CIRCADIAN_FEATURES = {"hour_sin", "hour_cos", "is_night"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HybridDataset(Dataset):
    def __init__(
        self,
        X_dynamic: np.ndarray,
        X_static: np.ndarray,
        X_circadian: np.ndarray,
        X_tabular: np.ndarray,
        y: np.ndarray,
    ) -> None:
        self.X_dynamic = torch.from_numpy(X_dynamic.astype(np.float32, copy=False))
        self.X_static = torch.from_numpy(X_static.astype(np.float32, copy=False))
        self.X_circadian = torch.from_numpy(X_circadian.astype(np.float32, copy=False))
        self.X_tabular = torch.from_numpy(X_tabular.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            self.X_dynamic[idx],
            self.X_static[idx],
            self.X_circadian[idx],
            self.X_tabular[idx],
            self.y[idx],
        )


def read_stats(stats_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return (
        np.asarray(stats["mean"], dtype=np.float32),
        np.asarray(stats["std"], dtype=np.float32),
        list(map(str, stats["feature_cols"])),
    )


def normalize_sequences(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    normalized = np.where(np.isnan(X), mean.reshape(1, 1, -1), X)
    normalized = (normalized - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def split_sequence_inputs(
    X: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    normalized = normalize_sequences(X, mean, std)
    static_indices = [idx for idx, name in enumerate(feature_names) if name in STATIC_FEATURES]
    circadian_indices = [idx for idx, name in enumerate(feature_names) if name in CIRCADIAN_FEATURES]
    dynamic_indices = [
        idx for idx, name in enumerate(feature_names)
        if name not in STATIC_FEATURES and name not in CIRCADIAN_FEATURES
    ]
    if {"hour_sin", "hour_cos"} - set(feature_names):
        raise ValueError("Hybrid-CGTA requires hour_sin and hour_cos in sequence features.")
    if not static_indices or not circadian_indices or not dynamic_indices:
        raise ValueError("Hybrid-CGTA requires dynamic, static, and circadian feature groups.")
    groups = {
        "dynamic": [feature_names[idx] for idx in dynamic_indices],
        "static": [feature_names[idx] for idx in static_indices],
        "circadian": [feature_names[idx] for idx in circadian_indices],
    }
    return (
        normalized[:, :, dynamic_indices],
        normalized[:, 0, static_indices],
        normalized[:, :, circadian_indices],
        y.astype(np.float32),
        groups,
    )


def numeric_tabular_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    drop_cols = [col for col in ["senior_id", "target_timestamp"] if col in df.columns]
    df = df.drop(columns=drop_cols)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_tabular(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, list[float]]]:
    columns = list(train_df.columns)
    val_df = val_df.reindex(columns=columns)
    test_df = test_df.reindex(columns=columns)
    train_values = train_df.to_numpy(dtype=np.float32)
    val_values = val_df.to_numpy(dtype=np.float32)
    test_values = test_df.to_numpy(dtype=np.float32)

    fill = np.nanmedian(train_values, axis=0)
    fill = np.nan_to_num(fill, nan=0.0, posinf=0.0, neginf=0.0)
    train_values = np.where(np.isnan(train_values), fill.reshape(1, -1), train_values)
    val_values = np.where(np.isnan(val_values), fill.reshape(1, -1), val_values)
    test_values = np.where(np.isnan(test_values), fill.reshape(1, -1), test_values)

    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((train_values - mean) / std).astype(np.float32),
        ((val_values - mean) / std).astype(np.float32),
        ((test_values - mean) / std).astype(np.float32),
        columns,
        {"median": fill.tolist(), "mean": mean.tolist(), "std": std.tolist()},
    )


def load_supervised_sequences(data_dir: Path, archive_path: Path) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(archive_path, allow_pickle=True)
    X = archive["X"]
    y = np.asarray(archive["y"], dtype=np.int8)
    mask = np.load(data_dir / "train_supervised_active_window_mask.npy").astype(bool)
    if len(mask) != len(y):
        raise ValueError(f"Train supervised mask mismatch: mask={len(mask)}, y={len(y)}")
    return X[mask], y[mask]


def load_eval_sequences(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(data_dir / f"X_{split}.npy")
    y = np.load(data_dir / f"y_{split}.npy").astype(np.int8)
    mask = np.load(data_dir / f"{split}_active_window_mask.npy").astype(bool)
    if len(mask) != len(y):
        raise ValueError(f"{split} mask mismatch: mask={len(mask)}, y={len(y)}")
    return X[mask], y[mask]


def assert_alignment(
    split: str,
    X_seq: np.ndarray,
    X_tab: np.ndarray,
    y_seq: np.ndarray,
    y_tab: np.ndarray,
) -> None:
    if len(X_seq) != len(X_tab):
        raise ValueError(f"{split} sequence/tabular length mismatch: seq={len(X_seq)}, tab={len(X_tab)}")
    if len(y_seq) != len(y_tab):
        raise ValueError(f"{split} target length mismatch: seq={len(y_seq)}, tab={len(y_tab)}")
    if not np.array_equal(y_seq.astype(np.int8), y_tab.astype(np.int8)):
        raise ValueError(f"{split} target arrays are not exactly aligned.")


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for X_dynamic, X_static, X_circadian, X_tabular, y in loader:
            probs = model(
                X_dynamic.to(device),
                X_static.to(device),
                X_circadian.to(device),
                X_tabular.to(device),
            )
            labels.append(y.numpy())
            scores.append(probs.detach().cpu().numpy())
    return np.concatenate(labels), np.concatenate(scores)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    y_true, scores = predict(model, loader, device)
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    losses: list[float] = []
    for X_dynamic, X_static, X_circadian, X_tabular, y in loader:
        optimizer.zero_grad()
        probs = model(
            X_dynamic.to(device),
            X_static.to(device),
            X_circadian.to(device),
            X_tabular.to(device),
        )
        loss = criterion(probs, y.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def append_benchmark_row(metrics: dict[str, Any], report_dir: Path) -> None:
    summary_path = report_dir / "benchmark_summary.csv"
    row = {
        "model": metrics["model"],
        "model_family": metrics["model_family"],
        "status": metrics["status"],
        "val_balanced_auroc": metrics["val_auroc"],
        "val_balanced_auprc": metrics["val_auprc"],
        "test_auroc": metrics["test_auroc"],
        "test_auprc": metrics["test_auprc"],
        "test_threshold_source": "direct probability ranking",
        "train_windows": metrics["train_windows"],
        "test_windows": metrics["test_windows"],
        "test_positives": metrics["test_positives"],
        "test_negatives": metrics["test_negatives"],
        "active_window_mask_applied": metrics["active_window_mask_applied"],
    }
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = summary[summary["model"] != metrics["model"]]
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True, sort=False)
        summary = summary.sort_values(["test_auprc", "test_auroc"], ascending=False, na_position="last")
    else:
        summary = pd.DataFrame([row])
    summary.to_csv(summary_path, index=False)
    logger.info("Appended Hybrid-CGTA metrics to %s", summary_path)


def build_loaders(args: argparse.Namespace):
    mean, std, feature_names = read_stats(args.data_dir / "normalization_stats.json")
    X_train_seq, y_train_seq = load_supervised_sequences(args.data_dir, args.supervised_train_archive)
    X_val_seq, y_val_seq = load_eval_sequences(args.data_dir, "val")
    X_test_seq, y_test_seq = load_eval_sequences(args.data_dir, "test")

    y_train_tab = np.load(args.data_dir / "y_train_supervised.npy").astype(np.int8)
    y_val_tab = np.load(args.data_dir / "y_val_l3.npy").astype(np.int8)
    y_test_tab = np.load(args.data_dir / "y_test_l3.npy").astype(np.int8)

    train_tab_df = numeric_tabular_frame(args.data_dir / "X_train_supervised_flat.parquet")
    val_tab_df = numeric_tabular_frame(args.data_dir / "X_val_flat.parquet")
    test_tab_df = numeric_tabular_frame(args.data_dir / "X_test_flat.parquet")

    assert_alignment("train_supervised", X_train_seq, train_tab_df.to_numpy(), y_train_seq, y_train_tab)
    assert_alignment("val", X_val_seq, val_tab_df.to_numpy(), y_val_seq, y_val_tab)
    assert_alignment("test", X_test_seq, test_tab_df.to_numpy(), y_test_seq, y_test_tab)

    X_train_tab, X_val_tab, X_test_tab, tabular_columns, tabular_stats = normalize_tabular(train_tab_df, val_tab_df, test_tab_df)
    X_train_dyn, X_train_static, X_train_circ, y_train, groups = split_sequence_inputs(
        X_train_seq, y_train_seq, mean, std, feature_names
    )
    X_val_dyn, X_val_static, X_val_circ, y_val, _ = split_sequence_inputs(X_val_seq, y_val_seq, mean, std, feature_names)
    X_test_dyn, X_test_static, X_test_circ, y_test, _ = split_sequence_inputs(X_test_seq, y_test_seq, mean, std, feature_names)

    train_loader = DataLoader(
        HybridDataset(X_train_dyn, X_train_static, X_train_circ, X_train_tab, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(HybridDataset(X_val_dyn, X_val_static, X_val_circ, X_val_tab, y_val), batch_size=args.batch_size)
    test_loader = DataLoader(HybridDataset(X_test_dyn, X_test_static, X_test_circ, X_test_tab, y_test), batch_size=args.batch_size)
    metadata = {
        "feature_names": feature_names,
        "feature_groups": groups,
        "tabular_columns": tabular_columns,
        "tabular_stats": tabular_stats,
        "train_windows": int(len(y_train)),
        "val_windows": int(len(y_val)),
        "test_windows": int(len(y_test)),
        "test_positives": int(np.sum(y_test == 1)),
        "test_negatives": int(np.sum(y_test == 0)),
        "input_dims": {
            "dynamic_dim": int(X_train_dyn.shape[2]),
            "static_dim": int(X_train_static.shape[1]),
            "circadian_dim": int(X_train_circ.shape[2]),
            "tabular_dim": int(X_train_tab.shape[1]),
        },
    }
    return train_loader, val_loader, test_loader, metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.hybrid_report_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, metadata = build_loaders(args)
    dims = metadata["input_dims"]
    model = HybridCGTANet(
        dynamic_dim=dims["dynamic_dim"],
        static_dim=dims["static_dim"],
        circadian_dim=dims["circadian_dim"],
        tabular_dim=dims["tabular_dim"],
        sequence_hidden_dim=args.sequence_hidden_dim,
        sequence_embedding_dim=args.sequence_embedding_dim,
        tabular_embedding_dim=args.tabular_embedding_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()
    best_path = args.model_dir / "best_model.pt"
    best_val_auprc = -np.inf
    best_epoch = -1
    patience = 0
    history: list[dict[str, Any]] = []

    logger.info(
        "Training Hybrid-CGTA on %s aligned windows; val=%s, test=%s, dims=%s, device=%s",
        metadata["train_windows"],
        metadata["val_windows"],
        metadata["test_windows"],
        dims,
        device,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip)
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
        }
        history.append(row)
        logger.info(
            "Epoch %03d | loss=%.4f | val_auroc=%.4f | val_auprc=%.4f",
            epoch,
            train_loss,
            row["val_auroc"],
            row["val_auprc"],
        )
        if row["val_auprc"] > best_val_auprc:
            best_val_auprc = row["val_auprc"]
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_auprc": best_val_auprc,
                    "metadata": metadata,
                    "hyperparameters": vars(args),
                },
                best_path,
            )
        else:
            patience += 1
            if patience >= args.patience:
                logger.info("Early stopping at epoch %s.", epoch)
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    y_test_true, test_scores = predict(model, test_loader, device)
    np.savez_compressed(args.hybrid_report_dir / "predictions_hybrid_cgta_net.npz", y_test=y_test_true, test_scores=test_scores)

    output = {
        "model": "Hybrid-CGTA Net",
        "model_family": "Hybrid Circadian-Gated Deep Sequence",
        "status": "fit",
        "best_epoch": int(best_epoch),
        "best_val_auprc": float(best_val_auprc),
        "val_auroc": val_metrics["auroc"],
        "val_auprc": val_metrics["auprc"],
        "test_auroc": test_metrics["auroc"],
        "test_auprc": test_metrics["auprc"],
        "train_windows": metadata["train_windows"],
        "val_windows": metadata["val_windows"],
        "test_windows": metadata["test_windows"],
        "test_positives": metadata["test_positives"],
        "test_negatives": metadata["test_negatives"],
        "active_window_mask_applied": True,
        "prediction_window": "4 hours",
        "cohort": "elite",
        "target": "level_2_or_level_3",
        "input_dims": dims,
        "feature_groups": metadata["feature_groups"],
        "tabular_columns": metadata["tabular_columns"],
    }
    with open(args.model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    with open(args.model_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    append_benchmark_row(output, args.report_dir)

    logger.info("Saved best Hybrid-CGTA model to %s", best_path)
    logger.info("Hybrid-CGTA Test AUROC=%.4f | Test AUPRC=%.4f", output["test_auroc"], output["test_auprc"])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hybrid-CGTA Net with aligned sequence and tabular features.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--supervised-train-archive", type=Path, default=DATA_DIR / "few_shot_train_intervention_balanced.npz")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--hybrid-report-dir", type=Path, default=HYBRID_REPORT_DIR)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-hidden-dim", type=int, default=96)
    parser.add_argument("--sequence-embedding-dim", type=int, default=64)
    parser.add_argument("--tabular-embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
