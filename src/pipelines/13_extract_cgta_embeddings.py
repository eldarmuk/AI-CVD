"""Extract active-window CGTA-Net embeddings for deep-tree fusion."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.cgta_net import CGTANet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "anomaly_detection"
CGTA_CHECKPOINT = PROJECT_ROOT / "models" / "cgta_net" / "best_model.pt"

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


class CGTAEmbeddingDataset(Dataset):
    def __init__(self, X_dynamic: np.ndarray, X_static: np.ndarray, X_circadian: np.ndarray):
        self.X_dynamic = torch.from_numpy(X_dynamic.astype(np.float32, copy=False))
        self.X_static = torch.from_numpy(X_static.astype(np.float32, copy=False))
        self.X_circadian = torch.from_numpy(X_circadian.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.X_dynamic)

    def __getitem__(self, idx: int):
        return self.X_dynamic[idx], self.X_static[idx], self.X_circadian[idx]


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


def split_inputs(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    normalized = normalize_sequences(X, mean, std)
    static_indices = [idx for idx, name in enumerate(feature_names) if name in STATIC_FEATURES]
    circadian_indices = [idx for idx, name in enumerate(feature_names) if name in CIRCADIAN_FEATURES]
    dynamic_indices = [
        idx for idx, name in enumerate(feature_names)
        if name not in STATIC_FEATURES and name not in CIRCADIAN_FEATURES
    ]
    if not static_indices or not circadian_indices or not dynamic_indices:
        raise ValueError("CGTA embedding extraction requires dynamic, static, and circadian feature groups.")
    groups = {
        "dynamic": [feature_names[idx] for idx in dynamic_indices],
        "static": [feature_names[idx] for idx in static_indices],
        "circadian": [feature_names[idx] for idx in circadian_indices],
    }
    return normalized[:, :, dynamic_indices], normalized[:, 0, static_indices], normalized[:, :, circadian_indices], groups


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


def load_cgta_model(checkpoint_path: Path, device: torch.device) -> tuple[CGTANet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    groups = checkpoint["feature_groups"]
    model = CGTANet(
        dynamic_dim=len(groups["dynamic"]),
        static_dim=len(groups["static"]),
        circadian_dim=len(groups["circadian"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 96)),
        num_layers=int(checkpoint.get("num_layers", 1)),
        dropout=float(checkpoint.get("dropout", 0.25)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def extract_embeddings(
    model: CGTANet,
    X_dynamic: np.ndarray,
    X_static: np.ndarray,
    X_circadian: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(CGTAEmbeddingDataset(X_dynamic, X_static, X_circadian), batch_size=batch_size, shuffle=False)
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for X_dyn, X_static_batch, X_circ in loader:
            batch_embed = model.classifier_embedding(
                X_dyn.to(device),
                X_static_batch.to(device),
                X_circ.to(device),
            )
            embeddings.append(batch_embed.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def process_split(
    split: str,
    X: np.ndarray,
    y: np.ndarray,
    data_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
    feature_names: list[str],
    model: CGTANet,
    batch_size: int,
    device: torch.device,
) -> dict:
    X_dyn, X_static, X_circ, groups = split_inputs(X, mean, std, feature_names)
    embeddings = extract_embeddings(model, X_dyn, X_static, X_circ, batch_size, device)
    out_path = data_dir / f"X_{split}_cgta_embed.npy"
    np.save(out_path, embeddings)
    logger.info("Saved %s embeddings: %s -> %s", split, embeddings.shape, out_path)
    return {
        "split": split,
        "path": str(out_path),
        "shape": list(embeddings.shape),
        "positives": int(y.sum()),
        "feature_groups": groups,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CGTA penultimate embeddings for active windows.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CGTA_CHECKPOINT)
    parser.add_argument("--supervised-train-archive", type=Path, default=DATA_DIR / "few_shot_train_intervention_balanced.npz")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    mean, std, feature_names = read_stats(args.data_dir / "normalization_stats.json")
    model, checkpoint = load_cgta_model(args.checkpoint, device)

    split_inputs_map = {
        "train": load_supervised_sequences(args.data_dir, args.supervised_train_archive),
        "val": load_eval_sequences(args.data_dir, "val"),
        "test": load_eval_sequences(args.data_dir, "test"),
    }
    metadata = {
        "source_checkpoint": str(args.checkpoint),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "embedding_layer": "CGTANet.classifier[:-2]",
        "splits": {},
    }
    for split, (X, y) in split_inputs_map.items():
        metadata["splits"][split] = process_split(
            split,
            X,
            y,
            args.data_dir,
            mean,
            std,
            feature_names,
            model,
            args.batch_size,
            device,
        )

    with open(args.data_dir / "cgta_embedding_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("CGTA embedding extraction complete.")


if __name__ == "__main__":
    main()
