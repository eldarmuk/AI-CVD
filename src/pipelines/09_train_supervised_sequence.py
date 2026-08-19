"""Train and evaluate an end-to-end supervised sequence classifier."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.models.supervised_sequence_net import SupervisedSequenceNet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "anomaly_detection"
MODEL_DIR = PROJECT_ROOT / "models" / "supervised_seq"
REPORT_DIR = PROJECT_ROOT / "reports" / "supervised_seq"

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

ACTIVE_CORE_FEATURES = {"heartrate", "sbp", "dbp", "saturation"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    def __init__(self, X_dynamic: np.ndarray, X_static: np.ndarray, y: np.ndarray):
        self.X_dynamic = torch.from_numpy(X_dynamic.astype(np.float32, copy=False))
        self.X_static = torch.from_numpy(X_static.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X_dynamic[idx], self.X_static[idx], self.y[idx]


def load_stats(stats_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    feature_names = list(map(str, stats["feature_cols"]))
    return mean, std, feature_names


def active_window_mask(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    core_indices = [idx for idx, name in enumerate(feature_names) if name in ACTIVE_CORE_FEATURES]
    if not core_indices:
        raise ValueError(f"None of the active core features were found: {sorted(ACTIVE_CORE_FEATURES)}")
    core = X[:, :, core_indices]
    valid_nonzero = np.isfinite(core) & (np.abs(core) > 1e-6)
    return valid_nonzero.any(axis=(1, 2))


def apply_optional_mask(X: np.ndarray, y: np.ndarray, mask_path: Path | None, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if mask_path is not None and mask_path.exists():
        mask = np.load(mask_path).astype(bool)
        if len(mask) != len(y):
            raise ValueError(f"Mask length mismatch for {mask_path}: mask={len(mask)}, y={len(y)}")
    else:
        mask = active_window_mask(X, feature_names)
    logger.info("Active-window filtering retained %s/%s windows (%.2f%%).", int(mask.sum()), len(mask), mask.mean() * 100)
    return X[mask], y[mask]


def prepare_arrays(
    X: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X.ndim != 3:
        raise ValueError(f"Expected 3D sequence data, got {X.shape}")
    if X.shape[2] != len(feature_names):
        raise ValueError(f"Feature mismatch: X has {X.shape[2]} features, stats define {len(feature_names)}")

    normalized = np.where(np.isnan(X), mean.reshape(1, 1, -1), X)
    normalized = (normalized - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    static_indices = [idx for idx, name in enumerate(feature_names) if name in STATIC_FEATURES]
    dynamic_indices = [idx for idx, name in enumerate(feature_names) if name not in STATIC_FEATURES]
    if not static_indices or not dynamic_indices:
        raise ValueError("Both dynamic and static features are required.")

    return normalized[:, :, dynamic_indices], normalized[:, 0, static_indices], y.astype(np.float32)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for X_dynamic, X_static, y in loader:
            probs = model(X_dynamic.to(device), X_static.to(device))
            scores.append(probs.detach().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(scores)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    y_true, scores = predict(model, loader, device)
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
    }


def load_supervised_train(path: Path) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    X = archive["X"]
    y = np.asarray(archive["y"]).astype(np.int8)
    if len(X) != len(y):
        raise ValueError(f"Supervised train length mismatch: X={len(X)}, y={len(y)}")
    return X, y


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(data_dir / f"X_{split}.npy")
    y = np.load(data_dir / f"y_{split}.npy").astype(np.int8)
    if len(X) != len(y):
        raise ValueError(f"{split} length mismatch: X={len(X)}, y={len(y)}")
    return X, y


def train(args: argparse.Namespace) -> dict[str, Any]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    mean, std, feature_names = load_stats(args.data_dir / "normalization_stats.json")
    X_train_raw, y_train_raw = load_supervised_train(args.supervised_train_archive)
    X_train_raw, y_train_raw = apply_optional_mask(X_train_raw, y_train_raw, None, feature_names)
    X_val_raw, y_val_raw = load_split(args.data_dir, "val")
    X_val_raw, y_val_raw = apply_optional_mask(
        X_val_raw,
        y_val_raw,
        args.data_dir / "val_active_window_mask.npy",
        feature_names,
    )
    X_test_raw, y_test_raw = load_split(args.data_dir, "test")
    X_test_raw, y_test_raw = apply_optional_mask(
        X_test_raw,
        y_test_raw,
        args.data_dir / "test_active_window_mask.npy",
        feature_names,
    )

    X_train_dyn, X_train_static, y_train = prepare_arrays(X_train_raw, y_train_raw, mean, std, feature_names)
    X_val_dyn, X_val_static, y_val = prepare_arrays(X_val_raw, y_val_raw, mean, std, feature_names)
    X_test_dyn, X_test_static, y_test = prepare_arrays(X_test_raw, y_test_raw, mean, std, feature_names)

    train_loader = DataLoader(
        SequenceDataset(X_train_dyn, X_train_static, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(SequenceDataset(X_val_dyn, X_val_static, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SequenceDataset(X_test_dyn, X_test_static, y_test), batch_size=args.batch_size, shuffle=False)

    model = SupervisedSequenceNet(
        dynamic_dim=X_train_dyn.shape[2],
        static_dim=X_train_static.shape[1],
        sequence_length=X_train_dyn.shape[1],
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    best_val_auprc = -np.inf
    best_epoch = -1
    patience = 0
    history: list[dict[str, Any]] = []
    best_path = args.model_dir / "best_model.pt"

    logger.info(
        "Training SupervisedSequenceNet on %s windows; val=%s, test=%s, device=%s",
        len(y_train),
        len(y_val),
        len(y_test),
        device,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for X_dynamic, X_static, y in train_loader:
            optimizer.zero_grad()
            probs = model(X_dynamic.to(device), X_static.to(device))
            loss = criterion(probs, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))

        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
        }
        history.append(row)
        logger.info(
            "Epoch %03d | loss=%.4f | val_auroc=%.4f | val_auprc=%.4f",
            epoch,
            row["train_loss"],
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
                    "feature_names": feature_names,
                    "dynamic_dim": X_train_dyn.shape[2],
                    "static_dim": X_train_static.shape[1],
                    "sequence_length": X_train_dyn.shape[1],
                    "hidden_dim": args.hidden_dim,
                    "embedding_dim": args.embedding_dim,
                    "dropout": args.dropout,
                    "best_epoch": best_epoch,
                    "best_val_auprc": best_val_auprc,
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
    np.savez_compressed(args.report_dir / "predictions_supervised_sequence_net.npz", y_test=y_test_true, test_scores=test_scores)

    output = {
        "model": "Supervised Sequence Net",
        "model_family": "Supervised Deep Sequence",
        "status": "fit",
        "best_epoch": int(best_epoch),
        "best_val_auprc": float(best_val_auprc),
        "val_auroc": val_metrics["auroc"],
        "val_auprc": val_metrics["auprc"],
        "test_auroc": test_metrics["auroc"],
        "test_auprc": test_metrics["auprc"],
        "train_windows": int(len(y_train)),
        "val_windows": int(len(y_val)),
        "test_windows": int(len(y_test)),
        "test_positives": int(np.sum(y_test == 1)),
        "test_negatives": int(np.sum(y_test == 0)),
        "active_window_mask_applied": True,
        "prediction_window": "4 hours",
        "cohort": "elite",
    }
    with open(args.model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    with open(args.model_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved best model to %s", best_path)
    logger.info("Test AUROC=%.4f | Test AUPRC=%.4f", output["test_auroc"], output["test_auprc"])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the supervised sequence crisis classifier.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--supervised-train-archive", type=Path, default=DATA_DIR / "few_shot_train_intervention_balanced.npz")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
