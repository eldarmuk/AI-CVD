"""Train CGTA-RF fusion over tabular summaries and deep sequence embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "anomaly_detection"
MODEL_DIR = PROJECT_ROOT / "models" / "cgta_rf_fusion"
REPORT_DIR = PROJECT_ROOT / "reports"
FUSION_REPORT_DIR = REPORT_DIR / "cgta_rf_fusion"
RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def numeric_tabular_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    drop_cols = [col for col in ["senior_id", "target_timestamp"] if col in df.columns]
    df = df.drop(columns=drop_cols)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_fusion_matrices(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    X_train_tab = numeric_tabular_frame(data_dir / "X_train_supervised_flat.parquet")
    X_val_tab = numeric_tabular_frame(data_dir / "X_val_flat.parquet").reindex(columns=X_train_tab.columns)
    X_test_tab = numeric_tabular_frame(data_dir / "X_test_flat.parquet").reindex(columns=X_train_tab.columns)

    X_train_embed = np.load(data_dir / "X_train_cgta_embed.npy")
    X_val_embed = np.load(data_dir / "X_val_cgta_embed.npy")
    X_test_embed = np.load(data_dir / "X_test_cgta_embed.npy")

    y_train = np.load(data_dir / "y_train_supervised.npy").astype(np.int8)
    y_val = np.load(data_dir / "y_val_l3.npy").astype(np.int8)
    y_test = np.load(data_dir / "y_test_l3.npy").astype(np.int8)

    for split, X_tab, X_embed, y in [
        ("train", X_train_tab, X_train_embed, y_train),
        ("val", X_val_tab, X_val_embed, y_val),
        ("test", X_test_tab, X_test_embed, y_test),
    ]:
        if len(X_tab) != len(X_embed) or len(X_tab) != len(y):
            raise ValueError(
                f"{split} fusion alignment mismatch: tab={len(X_tab)}, embed={len(X_embed)}, y={len(y)}"
            )

    embed_cols = [f"cgta_embed_{idx:03d}" for idx in range(X_train_embed.shape[1])]
    X_train = pd.concat([X_train_tab.reset_index(drop=True), pd.DataFrame(X_train_embed, columns=embed_cols)], axis=1)
    X_val = pd.concat([X_val_tab.reset_index(drop=True), pd.DataFrame(X_val_embed, columns=embed_cols)], axis=1)
    X_test = pd.concat([X_test_tab.reset_index(drop=True), pd.DataFrame(X_test_embed, columns=embed_cols)], axis=1)
    return X_train, X_val, X_test, y_train, y_val, y_test


def gmean_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    gmeans = np.sqrt(tpr * (1.0 - fpr))
    idx = int(np.argmax(gmeans))
    return float(thresholds[idx]), float(gmeans[idx])


def optimal_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    idx = int(np.argmax(f1))
    threshold = thresholds[min(idx, len(thresholds) - 1)] if len(thresholds) else 0.5
    return float(f1[idx]), float(threshold)


def precision_at_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float = 0.80) -> float:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    valid = precision[recall >= target_recall]
    if len(valid) == 0:
        return 0.0
    return float(np.max(valid))


def bootstrap_auc_ci(y_true: np.ndarray, scores: np.ndarray, n_bootstraps: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    aucs: list[float] = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(float(roc_auc_score(y_true[idx], scores[idx])))
    if not aucs:
        return float("nan"), float("nan")
    return tuple(np.percentile(aucs, [2.5, 97.5]).tolist())


def evaluate_scores(
    y_val: np.ndarray,
    val_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    model_name: str,
    model_family: str,
) -> dict[str, Any]:
    threshold, val_gmean = gmean_threshold(y_val, val_scores)
    y_pred = (test_scores >= threshold).astype(np.int8)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    f1, f1_threshold = optimal_f1(y_test, test_scores)
    ci_low, ci_high = bootstrap_auc_ci(y_test, test_scores)
    return {
        "model": model_name,
        "model_family": model_family,
        "status": "fit",
        "val_balanced_auroc": float(roc_auc_score(y_val, val_scores)),
        "val_balanced_auprc": float(average_precision_score(y_val, val_scores)),
        "val_gmean_threshold": threshold,
        "val_gmean": val_gmean,
        "test_auroc": float(roc_auc_score(y_test, test_scores)),
        "test_auroc_ci_low": ci_low,
        "test_auroc_ci_high": ci_high,
        "test_auprc": float(average_precision_score(y_test, test_scores)),
        "test_optimal_f1": f1,
        "test_optimal_f1_threshold": f1_threshold,
        "test_precision_at_80_recall": precision_at_recall(y_test, test_scores, 0.80),
        "test_confusion_tn": int(cm[0, 0]),
        "test_confusion_fp": int(cm[0, 1]),
        "test_confusion_fn": int(cm[1, 0]),
        "test_confusion_tp": int(cm[1, 1]),
        "test_threshold_source": "balanced validation G-Mean",
    }


def append_benchmark_row(metrics: dict[str, Any], report_dir: Path) -> None:
    summary_path = report_dir / "benchmark_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = summary[summary["model"] != metrics["model"]]
        summary = pd.concat([summary, pd.DataFrame([metrics])], ignore_index=True, sort=False)
        summary = summary.sort_values(["test_auprc", "test_auroc"], ascending=False, na_position="last")
    else:
        summary = pd.DataFrame([metrics])
    summary.to_csv(summary_path, index=False)
    logger.info("Appended %s metrics to %s", metrics["model"], summary_path)


def make_random_forest_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def embedding_only_frame(X: pd.DataFrame) -> pd.DataFrame:
    embed_cols = [col for col in X.columns if col.startswith("cgta_embed_")]
    if not embed_cols:
        raise ValueError("No CGTA embedding columns found for embeddings-only ablation.")
    return X.loc[:, embed_cols].copy()


def write_fusion_shap(model: Pipeline, X_test: pd.DataFrame, output_path: Path, sample_size: int) -> None:
    import shap

    if len(X_test) > sample_size:
        X_sample = X_test.sample(sample_size, random_state=RANDOM_SEED)
    else:
        X_sample = X_test.copy()
    imputer = model.named_steps["imputer"]
    rf = model.named_steps["model"]
    X_imputed = pd.DataFrame(imputer.transform(X_sample), columns=X_sample.columns)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_imputed)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    plt.figure(figsize=(11, 8))
    shap.summary_plot(shap_values, X_imputed, show=False, max_display=30)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved fusion SHAP beeswarm to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CGTA-RF deep-tree fusion classifier.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--fusion-report-dir", type=Path, default=FUSION_REPORT_DIR)
    parser.add_argument("--shap-sample-size", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.fusion_report_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_fusion_matrices(args.data_dir)
    model = make_random_forest_pipeline()
    logger.info("Training CGTA-RF Fusion on X=%s with positives=%s.", X_train.shape, int(y_train.sum()))
    model.fit(X_train, y_train)
    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_scores(
        y_val,
        val_scores,
        y_test,
        test_scores,
        model_name="CGTA-RF Fusion",
        model_family="Deep-Tree Fusion",
    )
    metrics.update(
        {
            "train_windows": int(len(y_train)),
            "test_windows": int(len(y_test)),
            "test_positives": int(y_test.sum()),
            "test_negatives": int((y_test == 0).sum()),
            "active_window_mask_applied": True,
            "fusion_features": int(X_train.shape[1]),
            "tabular_features": int(pd.read_parquet(args.data_dir / "X_train_supervised_flat.parquet").shape[1]),
            "cgta_embedding_features": int(np.load(args.data_dir / "X_train_cgta_embed.npy").shape[1]),
        }
    )

    with open(args.model_dir / "cgta_rf_fusion.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(args.model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(args.fusion_report_dir / "predictions_cgta_rf_fusion.npz", y_test=y_test, test_scores=test_scores)
    pd.DataFrame({"feature": X_train.columns}).to_csv(args.model_dir / "feature_names.csv", index=False)

    append_benchmark_row(metrics, args.report_dir)
    write_fusion_shap(model, X_test, args.report_dir / "fusion_shap_beeswarm.png", args.shap_sample_size)
    logger.info("CGTA-RF Fusion Test AUROC=%.4f | Test AUPRC=%.4f", metrics["test_auroc"], metrics["test_auprc"])

    X_train_embed = embedding_only_frame(X_train)
    X_val_embed = embedding_only_frame(X_val)
    X_test_embed = embedding_only_frame(X_test)
    embedding_model = make_random_forest_pipeline()
    logger.info(
        "Training Embeddings-Only RF ablation on X=%s with positives=%s.",
        X_train_embed.shape,
        int(y_train.sum()),
    )
    embedding_model.fit(X_train_embed, y_train)
    embed_val_scores = embedding_model.predict_proba(X_val_embed)[:, 1]
    embed_test_scores = embedding_model.predict_proba(X_test_embed)[:, 1]
    embed_metrics = evaluate_scores(
        y_val,
        embed_val_scores,
        y_test,
        embed_test_scores,
        model_name="Embeddings-Only RF",
        model_family="Deep Embedding Ablation",
    )
    embed_metrics.update(
        {
            "train_windows": int(len(y_train)),
            "test_windows": int(len(y_test)),
            "test_positives": int(y_test.sum()),
            "test_negatives": int((y_test == 0).sum()),
            "active_window_mask_applied": True,
            "fusion_features": int(X_train_embed.shape[1]),
            "tabular_features": 0,
            "cgta_embedding_features": int(X_train_embed.shape[1]),
        }
    )
    with open(args.model_dir / "cgta_rf_embeddings_only.pkl", "wb") as f:
        pickle.dump(embedding_model, f)
    with open(args.model_dir / "embeddings_only_metrics.json", "w", encoding="utf-8") as f:
        json.dump(embed_metrics, f, indent=2)
    np.savez_compressed(
        args.fusion_report_dir / "predictions_embeddings_only_rf.npz",
        y_test=y_test,
        test_scores=embed_test_scores,
    )
    append_benchmark_row(embed_metrics, args.report_dir)
    logger.info(
        "Embeddings-Only RF Test AUROC=%.4f | Test AUPRC=%.4f",
        embed_metrics["test_auroc"],
        embed_metrics["test_auprc"],
    )


if __name__ == "__main__":
    main()
