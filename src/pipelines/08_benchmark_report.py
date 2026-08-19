"""
Unified benchmark report and interpretability outputs.

Combines classical baseline metrics from 07_run_classical_baselines.py with
existing CCA-TAVAE / few-shot evaluation artifacts when present. Also produces
ROC/PR comparison plots and optional SHAP summaries for the XGBoost baseline.
"""

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
from sklearn.metrics import auc, precision_recall_curve, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSICAL_REPORT_DIR = PROJECT_ROOT / "reports" / "classical_baselines"
CLASSICAL_MODEL_DIR = PROJECT_ROOT / "models" / "classical_baselines"
TABULAR_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "anomaly_detection"
REAL_WORLD_DIR = PROJECT_ROOT / "models" / "real_world_auprc"
SUPERVISED_SEQ_DIR = PROJECT_ROOT / "models" / "supervised_seq"
SUPERVISED_SEQ_REPORT_DIR = PROJECT_ROOT / "reports" / "supervised_seq"
CGTA_MODEL_DIR = PROJECT_ROOT / "models" / "cgta_net"
CGTA_REPORT_DIR = PROJECT_ROOT / "reports" / "cgta_net"
HYBRID_CGTA_MODEL_DIR = PROJECT_ROOT / "models" / "hybrid_cgta"
HYBRID_CGTA_REPORT_DIR = PROJECT_ROOT / "reports" / "hybrid_cgta"
FUSION_MODEL_DIR = PROJECT_ROOT / "models" / "cgta_rf_fusion"
FUSION_REPORT_DIR = PROJECT_ROOT / "reports" / "cgta_rf_fusion"
REPORT_DIR = PROJECT_ROOT / "reports"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_classical_summary(classical_report_dir: Path) -> pd.DataFrame:
    path = classical_report_dir / "classical_baseline_metrics.csv"
    if not path.exists():
        logger.warning("Classical metrics not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["model_family"] = np.where(
        df["model"].isin(["isolation_forest", "one_class_svm"]),
        "Classical Anomaly Detector",
        np.where(df["model"].str.contains("forest|xgboost", regex=True), "Tree Ensemble", "Regularized Linear"),
    )
    return df


def current_active_test_windows(tabular_data_dir: Path) -> int | None:
    mask_path = tabular_data_dir / "test_active_window_mask.npy"
    y_path = tabular_data_dir / "y_test.npy"
    if mask_path.exists():
        return int(np.load(mask_path).astype(bool).sum())
    if y_path.exists():
        return int(len(np.load(y_path)))
    return None


def metric_window_count_matches(metrics: dict[str, Any], expected_windows: int | None, key: str) -> bool:
    if expected_windows is None:
        return True
    recorded = metrics.get(key)
    if recorded is None:
        return True
    try:
        return int(recorded) == expected_windows
    except (TypeError, ValueError):
        return True


def load_existing_sequence_metrics(real_world_dir: Path, expected_test_windows: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    mappings = {
        "metrics_vae.json": ("CCA-TAVAE", "Deep Generative"),
        "metrics_head.json": ("CCA-TAVAE Few-Shot Head", "Deep/Few-Shot Hybrid"),
    }
    for filename, (model_name, family) in mappings.items():
        path = real_world_dir / filename
        if not path.exists():
            continue
        metrics = read_json(path)
        if not metric_window_count_matches(metrics, expected_test_windows, "n_windows"):
            logger.warning("Skipping stale %s because its test window count does not match the current split.", path)
            continue
        rows.append(
            {
                "model": model_name,
                "model_family": family,
                "val_balanced_auroc": np.nan,
                "val_balanced_auprc": np.nan,
                "val_gmean_threshold": np.nan,
                "val_gmean": np.nan,
                "test_auroc": metrics.get("auroc", np.nan),
                "test_auroc_ci_low": np.nan,
                "test_auroc_ci_high": np.nan,
                "test_auprc": metrics.get("auprc", np.nan),
                "test_optimal_f1": np.nan,
                "test_optimal_f1_threshold": np.nan,
                "test_precision_at_80_recall": np.nan,
                "test_confusion_tn": np.nan,
                "test_confusion_fp": np.nan,
                "test_confusion_fn": np.nan,
                "test_confusion_tp": np.nan,
                "test_threshold_source": "external real-world artifact",
                "test_windows": metrics.get("n_windows", np.nan),
                "test_positives": metrics.get("positives", np.nan),
                "test_negatives": metrics.get("negatives", np.nan),
                "test_prevalence": metrics.get("prevalence", np.nan),
            }
        )
    return pd.DataFrame(rows)


def load_supervised_sequence_metrics(supervised_seq_dir: Path, expected_test_windows: int | None = None) -> pd.DataFrame:
    path = supervised_seq_dir / "metrics.json"
    if not path.exists():
        return pd.DataFrame()
    metrics = read_json(path)
    if not metric_window_count_matches(metrics, expected_test_windows, "test_windows"):
        logger.warning("Skipping stale %s because its test window count does not match the current split.", path)
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "model": metrics.get("model", "Supervised Sequence Net"),
                "model_family": metrics.get("model_family", "Supervised Deep Sequence"),
                "status": metrics.get("status", "fit"),
                "val_balanced_auroc": metrics.get("val_auroc", np.nan),
                "val_balanced_auprc": metrics.get("val_auprc", np.nan),
                "val_gmean_threshold": np.nan,
                "val_gmean": np.nan,
                "test_auroc": metrics.get("test_auroc", np.nan),
                "test_auroc_ci_low": np.nan,
                "test_auroc_ci_high": np.nan,
                "test_auprc": metrics.get("test_auprc", np.nan),
                "test_optimal_f1": np.nan,
                "test_optimal_f1_threshold": np.nan,
                "test_precision_at_80_recall": np.nan,
                "test_confusion_tn": np.nan,
                "test_confusion_fp": np.nan,
                "test_confusion_fn": np.nan,
                "test_confusion_tp": np.nan,
                "test_threshold_source": "direct probability ranking",
                "train_windows": metrics.get("train_windows", np.nan),
                "test_windows": metrics.get("test_windows", np.nan),
                "test_positives": metrics.get("test_positives", np.nan),
                "test_negatives": metrics.get("test_negatives", np.nan),
                "active_window_mask_applied": metrics.get("active_window_mask_applied", np.nan),
            }
        ]
    )


def load_cgta_metrics(cgta_model_dir: Path, expected_test_windows: int | None = None) -> pd.DataFrame:
    path = cgta_model_dir / "metrics.json"
    if not path.exists():
        return pd.DataFrame()
    metrics = read_json(path)
    if not metric_window_count_matches(metrics, expected_test_windows, "test_windows"):
        logger.warning("Skipping stale %s because its test window count does not match the current split.", path)
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "model": metrics.get("model", "CGTA-Net"),
                "model_family": metrics.get("model_family", "Circadian-Gated Deep Sequence"),
                "status": metrics.get("status", "fit"),
                "val_balanced_auroc": metrics.get("val_auroc", np.nan),
                "val_balanced_auprc": metrics.get("val_auprc", np.nan),
                "val_gmean_threshold": np.nan,
                "val_gmean": np.nan,
                "test_auroc": metrics.get("test_auroc", np.nan),
                "test_auroc_ci_low": np.nan,
                "test_auroc_ci_high": np.nan,
                "test_auprc": metrics.get("test_auprc", np.nan),
                "test_optimal_f1": np.nan,
                "test_optimal_f1_threshold": np.nan,
                "test_precision_at_80_recall": np.nan,
                "test_confusion_tn": np.nan,
                "test_confusion_fp": np.nan,
                "test_confusion_fn": np.nan,
                "test_confusion_tp": np.nan,
                "test_threshold_source": "direct probability ranking",
                "train_windows": metrics.get("train_windows", np.nan),
                "test_windows": metrics.get("test_windows", np.nan),
                "test_positives": metrics.get("test_positives", np.nan),
                "test_negatives": metrics.get("test_negatives", np.nan),
                "active_window_mask_applied": metrics.get("active_window_mask_applied", np.nan),
            }
        ]
    )


def load_hybrid_cgta_metrics(hybrid_model_dir: Path, expected_test_windows: int | None = None) -> pd.DataFrame:
    path = hybrid_model_dir / "metrics.json"
    if not path.exists():
        return pd.DataFrame()
    metrics = read_json(path)
    if not metric_window_count_matches(metrics, expected_test_windows, "test_windows"):
        logger.warning("Skipping stale %s because its test window count does not match the current split.", path)
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "model": metrics.get("model", "Hybrid-CGTA Net"),
                "model_family": metrics.get("model_family", "Hybrid Circadian-Gated Deep Sequence"),
                "status": metrics.get("status", "fit"),
                "val_balanced_auroc": metrics.get("val_auroc", np.nan),
                "val_balanced_auprc": metrics.get("val_auprc", np.nan),
                "val_gmean_threshold": np.nan,
                "val_gmean": np.nan,
                "test_auroc": metrics.get("test_auroc", np.nan),
                "test_auroc_ci_low": np.nan,
                "test_auroc_ci_high": np.nan,
                "test_auprc": metrics.get("test_auprc", np.nan),
                "test_optimal_f1": np.nan,
                "test_optimal_f1_threshold": np.nan,
                "test_precision_at_80_recall": np.nan,
                "test_confusion_tn": np.nan,
                "test_confusion_fp": np.nan,
                "test_confusion_fn": np.nan,
                "test_confusion_tp": np.nan,
                "test_threshold_source": "direct probability ranking",
                "train_windows": metrics.get("train_windows", np.nan),
                "test_windows": metrics.get("test_windows", np.nan),
                "test_positives": metrics.get("test_positives", np.nan),
                "test_negatives": metrics.get("test_negatives", np.nan),
                "active_window_mask_applied": metrics.get("active_window_mask_applied", np.nan),
            }
        ]
    )


def load_fusion_metrics(fusion_model_dir: Path, expected_test_windows: int | None = None) -> pd.DataFrame:
    path = fusion_model_dir / "metrics.json"
    if not path.exists():
        return pd.DataFrame()
    metrics = read_json(path)
    if not metric_window_count_matches(metrics, expected_test_windows, "test_windows"):
        logger.warning("Skipping stale %s because its test window count does not match the current split.", path)
        return pd.DataFrame()
    return pd.DataFrame([metrics])


def write_benchmark_summary(args: argparse.Namespace) -> pd.DataFrame:
    expected_test_windows = current_active_test_windows(args.tabular_data_dir)
    frames = [
        load_classical_summary(args.classical_report_dir),
        load_existing_sequence_metrics(args.real_world_dir, expected_test_windows),
        load_supervised_sequence_metrics(args.supervised_seq_dir, expected_test_windows),
        load_cgta_metrics(args.cgta_model_dir, expected_test_windows),
        load_hybrid_cgta_metrics(args.hybrid_cgta_model_dir, expected_test_windows),
        load_fusion_metrics(args.fusion_model_dir, expected_test_windows),
    ]
    frames = [df for df in frames if not df.empty]
    if not frames:
        raise FileNotFoundError("No benchmark metric artifacts found to summarize.")
    summary = pd.concat(frames, ignore_index=True, sort=False)
    summary = summary.sort_values(["test_auprc", "test_auroc"], ascending=False, na_position="last")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.report_dir / "benchmark_summary.csv", index=False)
    logger.info("Saved %s", args.report_dir / "benchmark_summary.csv")
    return summary


def plot_classical_prediction_file(ax_roc: plt.Axes, ax_pr: plt.Axes, path: Path, label: str) -> None:
    data = np.load(path)
    y = data["y_test"]
    scores = data["test_scores"]
    if len(np.unique(y)) < 2:
        logger.warning("Skipping %s curves because only one class is present.", label)
        return
    fpr, tpr, _ = roc_curve(y, scores)
    precision, recall, _ = precision_recall_curve(y, scores)
    ax_roc.plot(fpr, tpr, lw=1.8, label=f"{label} AUROC={auc(fpr, tpr):.3f}")
    ax_pr.plot(recall, precision, lw=1.8, label=f"{label} AUPRC={auc(recall, precision):.3f}")


def plot_sequence_prediction_file(ax_pr: plt.Axes, path: Path, label: str) -> None:
    data = np.load(path)
    if {"recall", "precision"}.issubset(data.files):
        recall = data["recall"]
        precision = data["precision"]
    else:
        y = data["y_true"]
        scores = data["scores"]
        precision, recall, _ = precision_recall_curve(y, scores)
    ax_pr.plot(recall, precision, lw=1.8, linestyle="--", label=label)


def write_comparison_plot(args: argparse.Namespace) -> None:
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    classical_files = sorted(args.classical_report_dir.glob("predictions_*.npz"))
    for path in classical_files:
        label = path.stem.replace("predictions_", "").replace("_", " ")
        plot_classical_prediction_file(ax_roc, ax_pr, path, label)

    sequence_files = {
        "CCA-TAVAE real-world PR": args.real_world_dir / "imbalanced_pr_vae.npz",
        "CCA-TAVAE few-shot real-world PR": args.real_world_dir / "imbalanced_pr_head.npz",
    }
    for label, path in sequence_files.items():
        if path.exists():
            plot_sequence_prediction_file(ax_pr, path, label)

    supervised_seq_path = args.supervised_seq_report_dir / "predictions_supervised_sequence_net.npz"
    if supervised_seq_path.exists():
        metrics_path = args.supervised_seq_dir / "metrics.json"
        supervised_label = "Supervised Sequence Net"
        if metrics_path.exists():
            supervised_label = read_json(metrics_path).get("model", supervised_label)
        plot_classical_prediction_file(ax_roc, ax_pr, supervised_seq_path, supervised_label)

    cgta_path = args.cgta_report_dir / "predictions_cgta_net.npz"
    if cgta_path.exists():
        plot_classical_prediction_file(ax_roc, ax_pr, cgta_path, "CGTA-Net")

    hybrid_path = args.hybrid_cgta_report_dir / "predictions_hybrid_cgta_net.npz"
    if hybrid_path.exists():
        plot_classical_prediction_file(ax_roc, ax_pr, hybrid_path, "Hybrid-CGTA Net")

    fusion_path = args.fusion_report_dir / "predictions_cgta_rf_fusion.npz"
    if fusion_path.exists():
        plot_classical_prediction_file(ax_roc, ax_pr, fusion_path, "CGTA-RF Fusion")

    ax_roc.plot([0, 1], [0, 1], color="black", linestyle=":", lw=1)
    ax_roc.set_title("ROC Comparison")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(fontsize=8)
    ax_roc.grid(alpha=0.25)

    ax_pr.set_title("Precision-Recall Comparison")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(fontsize=8)
    ax_pr.grid(alpha=0.25)

    fig.tight_layout()
    output_path = args.report_dir / "benchmark_roc_pr_comparison.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[c for c in ["senior_id", "target_timestamp"] if c in df.columns]).copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def write_shap_or_importance(args: argparse.Namespace) -> None:
    model_path = args.classical_model_dir / "xgboost.pkl"
    X_path = args.tabular_data_dir / "X_test_flat.parquet"
    if not model_path.exists() or not X_path.exists():
        logger.warning("Skipping SHAP: missing %s or %s", model_path, X_path)
        return

    model = load_pickle(model_path)
    X_raw = feature_matrix(pd.read_parquet(X_path))
    if len(X_raw) > args.shap_sample_size:
        X_raw = X_raw.sample(args.shap_sample_size, random_state=42)
    imputer = model.named_steps["imputer"]
    xgb_model = model.named_steps["model"]
    fitted_features = getattr(imputer, "feature_names_in_", None)
    if fitted_features is not None:
        missing = [feature for feature in fitted_features if feature not in X_raw.columns]
        if missing:
            logger.warning("Skipping SHAP: missing fitted features in X_test_flat.parquet: %s", missing)
            return
        X_raw = X_raw.loc[:, list(fitted_features)]
    X = imputer.transform(X_raw)

    try:
        import shap

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({"feature": X_raw.columns, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False
        )
        shap_df.to_csv(args.report_dir / "xgboost_shap_feature_importance.csv", index=False)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_raw, show=False, max_display=25)
        plt.tight_layout()
        plt.savefig(args.report_dir / "xgboost_shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved XGBoost SHAP outputs.")
    except ImportError:
        logger.warning("SHAP is not installed; writing native XGBoost importance fallback.")
        importance = getattr(xgb_model, "feature_importances_", None)
        if importance is None:
            return
        imp_df = pd.DataFrame({"feature": X_raw.columns, "importance": importance}).sort_values(
            "importance", ascending=False
        )
        imp_df.to_csv(args.report_dir / "xgboost_feature_importance_fallback.csv", index=False)
        top = imp_df.head(25).sort_values("importance")
        plt.figure(figsize=(9, 7))
        plt.barh(top["feature"], top["importance"])
        plt.xlabel("Native XGBoost importance")
        plt.title("XGBoost Feature Importance")
        plt.tight_layout()
        plt.savefig(args.report_dir / "xgboost_feature_importance_fallback.png", dpi=300)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate unified benchmark summary and figures.")
    parser.add_argument("--classical-report-dir", type=Path, default=CLASSICAL_REPORT_DIR)
    parser.add_argument("--classical-model-dir", type=Path, default=CLASSICAL_MODEL_DIR)
    parser.add_argument("--tabular-data-dir", type=Path, default=TABULAR_DATA_DIR)
    parser.add_argument("--real-world-dir", type=Path, default=REAL_WORLD_DIR)
    parser.add_argument("--supervised-seq-dir", type=Path, default=SUPERVISED_SEQ_DIR)
    parser.add_argument("--supervised-seq-report-dir", type=Path, default=SUPERVISED_SEQ_REPORT_DIR)
    parser.add_argument("--cgta-model-dir", type=Path, default=CGTA_MODEL_DIR)
    parser.add_argument("--cgta-report-dir", type=Path, default=CGTA_REPORT_DIR)
    parser.add_argument("--hybrid-cgta-model-dir", type=Path, default=HYBRID_CGTA_MODEL_DIR)
    parser.add_argument("--hybrid-cgta-report-dir", type=Path, default=HYBRID_CGTA_REPORT_DIR)
    parser.add_argument("--fusion-model-dir", type=Path, default=FUSION_MODEL_DIR)
    parser.add_argument("--fusion-report-dir", type=Path, default=FUSION_REPORT_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--shap-sample-size", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_benchmark_summary(args)
    write_comparison_plot(args)
    write_shap_or_importance(args)
    logger.info("Benchmark report generation complete.")


if __name__ == "__main__":
    main()
