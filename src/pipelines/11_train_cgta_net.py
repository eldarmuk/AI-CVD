"""Train Circadian-Gated Temporal Attention Network on Level 2/3 targets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.models.cgta_net import CGTANet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "anomaly_detection"
MODEL_DIR = PROJECT_ROOT / "models" / "cgta_net"
REPORT_DIR = PROJECT_ROOT / "reports"
CGTA_REPORT_DIR = REPORT_DIR / "cgta_net"

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
ACTIVE_CORE_FEATURES = {"heartrate", "sbp", "dbp", "saturation"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CGTADataset(Dataset):
    def __init__(self, X_dynamic: np.ndarray, X_static: np.ndarray, X_circadian: np.ndarray, y: np.ndarray):
        self.X_dynamic = torch.from_numpy(X_dynamic.astype(np.float32, copy=False))
        self.X_static = torch.from_numpy(X_static.astype(np.float32, copy=False))
        self.X_circadian = torch.from_numpy(X_circadian.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X_dynamic[idx], self.X_static[idx], self.X_circadian[idx], self.y[idx]


def read_stats(stats_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return (
        np.asarray(stats["mean"], dtype=np.float32),
        np.asarray(stats["std"], dtype=np.float32),
        list(map(str, stats["feature_cols"])),
    )


def load_supervised_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    X = archive["X"]
    y = np.asarray(archive["y"], dtype=np.int8)
    if len(X) != len(y):
        raise ValueError(f"Supervised archive length mismatch: X={len(X)}, y={len(y)}")
    return X, y


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(data_dir / f"X_{split}.npy")
    y = np.load(data_dir / f"y_{split}.npy").astype(np.int8)
    if len(X) != len(y):
        raise ValueError(f"{split} length mismatch: X={len(X)}, y={len(y)}")
    return X, y


def active_window_mask(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    core_indices = [idx for idx, name in enumerate(feature_names) if name in ACTIVE_CORE_FEATURES]
    if not core_indices:
        raise ValueError(f"Missing active-window core features: {sorted(ACTIVE_CORE_FEATURES)}")
    core = X[:, :, core_indices]
    return (np.isfinite(core) & (np.abs(core) > 1e-6)).any(axis=(1, 2))


def apply_active_mask(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    mask_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if mask_path is not None and mask_path.exists():
        mask = np.load(mask_path).astype(bool)
        if len(mask) != len(y):
            raise ValueError(f"Mask length mismatch for {mask_path}: mask={len(mask)}, y={len(y)}")
    else:
        mask = active_window_mask(X, feature_names)
    logger.info("Active-window filtering retained %s/%s windows (%.2f%%).", int(mask.sum()), len(mask), mask.mean() * 100)
    return X[mask], y[mask]


def prepare_inputs(
    X: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    normalized = np.where(np.isnan(X), mean.reshape(1, 1, -1), X)
    normalized = (normalized - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    static_indices = [idx for idx, name in enumerate(feature_names) if name in STATIC_FEATURES]
    circadian_indices = [idx for idx, name in enumerate(feature_names) if name in CIRCADIAN_FEATURES]
    dynamic_indices = [
        idx
        for idx, name in enumerate(feature_names)
        if name not in STATIC_FEATURES and name not in CIRCADIAN_FEATURES
    ]
    if not static_indices:
        raise ValueError("No static clinical features found.")
    if {"hour_sin", "hour_cos"} - set(feature_names):
        raise ValueError("CGTA-Net requires hour_sin and hour_cos in feature_names.")
    if not circadian_indices:
        raise ValueError("No circadian features found.")
    if not dynamic_indices:
        raise ValueError("No dynamic physiological features found.")

    feature_groups = {
        "dynamic": [feature_names[idx] for idx in dynamic_indices],
        "static": [feature_names[idx] for idx in static_indices],
        "circadian": [feature_names[idx] for idx in circadian_indices],
    }
    return (
        normalized[:, :, dynamic_indices],
        normalized[:, 0, static_indices],
        normalized[:, :, circadian_indices],
        y.astype(np.float32),
        feature_groups,
    )


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for X_dynamic, X_static, X_circadian, y in loader:
            probs = model(X_dynamic.to(device), X_static.to(device), X_circadian.to(device))
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
    for X_dynamic, X_static, X_circadian, y in loader:
        optimizer.zero_grad()
        probs = model(X_dynamic.to(device), X_static.to(device), X_circadian.to(device))
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
    report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    logger.info("Appended CGTA-Net metrics to %s", summary_path)


def integrated_gradients(
    model: CGTANet,
    X_dynamic: torch.Tensor,
    X_static: torch.Tensor,
    X_circadian: torch.Tensor,
    device: torch.device,
    steps: int,
) -> np.ndarray:
    model.eval()
    baseline = torch.zeros_like(X_dynamic, device=device)
    X_dynamic = X_dynamic.to(device)
    X_static = X_static.to(device)
    X_circadian = X_circadian.to(device)
    total_gradients = torch.zeros_like(X_dynamic, device=device)

    with torch.backends.cudnn.flags(enabled=False):
        for alpha in torch.linspace(0.0, 1.0, steps, device=device):
            interpolated = baseline + alpha * (X_dynamic - baseline)
            interpolated.requires_grad_(True)
            output = model(interpolated, X_static, X_circadian).sum()
            gradients = torch.autograd.grad(output, interpolated)[0]
            total_gradients += gradients.detach()

    return ((X_dynamic - baseline) * total_gradients / steps).detach().cpu().numpy()


def write_explainability_plot(
    model: CGTANet,
    loader: DataLoader,
    dynamic_features: list[str],
    circadian_features: list[str],
    output_path: Path,
    device: torch.device,
    sample_size: int,
    ig_steps: int,
) -> None:
    X_dynamic, X_static, X_circadian, _ = next(iter(loader))
    n = min(sample_size, len(X_dynamic))
    attributions = integrated_gradients(
        model,
        X_dynamic[:n],
        X_static[:n],
        X_circadian[:n],
        device,
        ig_steps,
    )
    importance = np.mean(np.abs(attributions), axis=0).T
    with torch.no_grad():
        gates = model.gate_values(X_circadian[:n].to(device)).detach().cpu().numpy()
    gate_curve = gates.mean(axis=(0, 2))

    fig, (ax_heatmap, ax_gate) = plt.subplots(
        2,
        1,
        figsize=(13, max(8, 0.28 * len(dynamic_features))),
        gridspec_kw={"height_ratios": [5, 1]},
        sharex=True,
    )
    image = ax_heatmap.imshow(importance, aspect="auto", cmap="magma")
    ax_heatmap.set_yticks(np.arange(len(dynamic_features)))
    ax_heatmap.set_yticklabels(dynamic_features, fontsize=7)
    ax_heatmap.set_ylabel("Dynamic Feature")
    ax_heatmap.set_title("CGTA-Net Integrated Gradients Across 96-Step Lookback")
    fig.colorbar(image, ax=ax_heatmap, fraction=0.018, pad=0.01, label="Mean |Attribution|")

    ax_gate.plot(gate_curve, color="#0f766e", linewidth=2)
    ax_gate.set_ylabel("Mean Gate")
    ax_gate.set_xlabel("5-Minute Time Step")
    ax_gate.set_title(f"Circadian Gate Activation ({', '.join(circadian_features)})")
    ax_gate.grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved CGTA explainability plot to %s", output_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.cgta_report_dir.mkdir(parents=True, exist_ok=True)

    mean, std, feature_names = read_stats(args.data_dir / "normalization_stats.json")
    X_train_raw, y_train_raw = load_supervised_archive(args.supervised_train_archive)
    X_train_raw, y_train_raw = apply_active_mask(X_train_raw, y_train_raw, feature_names)
    X_val_raw, y_val_raw = load_split(args.data_dir, "val")
    X_val_raw, y_val_raw = apply_active_mask(X_val_raw, y_val_raw, feature_names, args.data_dir / "val_active_window_mask.npy")
    X_test_raw, y_test_raw = load_split(args.data_dir, "test")
    X_test_raw, y_test_raw = apply_active_mask(X_test_raw, y_test_raw, feature_names, args.data_dir / "test_active_window_mask.npy")

    X_train_dyn, X_train_static, X_train_circ, y_train, groups = prepare_inputs(X_train_raw, y_train_raw, mean, std, feature_names)
    X_val_dyn, X_val_static, X_val_circ, y_val, _ = prepare_inputs(X_val_raw, y_val_raw, mean, std, feature_names)
    X_test_dyn, X_test_static, X_test_circ, y_test, _ = prepare_inputs(X_test_raw, y_test_raw, mean, std, feature_names)

    train_loader = DataLoader(
        CGTADataset(X_train_dyn, X_train_static, X_train_circ, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(CGTADataset(X_val_dyn, X_val_static, X_val_circ, y_val), batch_size=args.batch_size)
    test_loader = DataLoader(CGTADataset(X_test_dyn, X_test_static, X_test_circ, y_test), batch_size=args.batch_size)

    model = CGTANet(
        dynamic_dim=X_train_dyn.shape[2],
        static_dim=X_train_static.shape[1],
        circadian_dim=X_train_circ.shape[2],
        hidden_dim=args.hidden_dim,
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
        "Training CGTA-Net on %s L2+L3 windows; val=%s, test=%s, device=%s",
        len(y_train),
        len(y_val),
        len(y_test),
        device,
    )
    logger.info("Feature groups: dynamic=%s, static=%s, circadian=%s", len(groups["dynamic"]), len(groups["static"]), groups["circadian"])

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
                    "feature_groups": groups,
                    "feature_names": feature_names,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
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
    np.savez_compressed(args.cgta_report_dir / "predictions_cgta_net.npz", y_test=y_test_true, test_scores=test_scores)

    output = {
        "model": "CGTA-Net",
        "model_family": "Circadian-Gated Deep Sequence",
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
        "target": "level_2_or_level_3",
        "circadian_features": groups["circadian"],
        "dynamic_features": groups["dynamic"],
        "static_features": groups["static"],
    }
    with open(args.model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    with open(args.model_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    write_explainability_plot(
        model,
        test_loader,
        groups["dynamic"],
        groups["circadian"],
        args.report_dir / "cgta_explainability.png",
        device,
        args.explain_samples,
        args.ig_steps,
    )
    append_benchmark_row(output, args.report_dir)

    logger.info("Saved best CGTA-Net model to %s", best_path)
    logger.info("CGTA-Net Test AUROC=%.4f | Test AUPRC=%.4f", output["test_auroc"], output["test_auprc"])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CGTA-Net for Level 2/3 deterioration prediction.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--supervised-train-archive", type=Path, default=DATA_DIR / "few_shot_train_intervention_balanced.npz")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--cgta-report-dir", type=Path, default=CGTA_REPORT_DIR)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--explain-samples", type=int, default=64)
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
