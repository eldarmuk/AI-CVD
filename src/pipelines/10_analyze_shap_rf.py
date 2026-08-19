"""Random Forest SHAP explainability and clinical calibration analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "anomaly_detection"
REPORT_DIR = PROJECT_ROOT / "reports"
RANDOM_SEED = 42
TEMPORAL_ABLATION_PATTERNS = ("hour", "day_of_week", "is_night", "timestamp")


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = {"senior_id", "target_timestamp"}
    X = df.drop(columns=[col for col in drop_cols if col in df.columns]).copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def drop_temporal_features(X: pd.DataFrame) -> pd.DataFrame:
    temporal_cols = [
        col for col in X.columns if any(pattern in col.lower() for pattern in TEMPORAL_ABLATION_PATTERNS)
    ]
    if temporal_cols:
        print(f"Circadian ablation dropping {len(temporal_cols)} temporal columns: {', '.join(temporal_cols)}")
    return X.drop(columns=temporal_cols)


def load_supervised_train(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    X_path = data_dir / "X_train_supervised_flat.parquet"
    y_path = data_dir / "y_train_supervised.npy"
    X = feature_matrix(pd.read_parquet(X_path))
    y = np.load(y_path).astype(np.int8)
    if len(X) != len(y):
        raise ValueError(f"Supervised train length mismatch: X={len(X)}, y={len(y)}")
    return drop_temporal_features(X), y


def load_test(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    X_path = data_dir / "X_test_flat.parquet"
    y_path = data_dir / "y_test_l3.npy"
    if not y_path.exists():
        y_path = data_dir / "y_test.npy"
    X = feature_matrix(pd.read_parquet(X_path))
    y = np.load(y_path).astype(np.int8)
    if len(X) != len(y):
        raise ValueError(f"Test length mismatch: X={len(X)}, y={len(y)}")
    return drop_temporal_features(X), y


def train_random_forest(X_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    model = Pipeline(
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
    model.fit(X_train, y_train)
    return model


def calculate_operating_point(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    youden_j = tpr - fpr
    idx = int(np.argmax(youden_j))
    threshold = float(thresholds[idx])
    y_pred = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "threshold": threshold,
        "youden_j": float(youden_j[idx]),
        "precision_ppv": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def write_calibration_report(metrics: dict[str, float | int], output_path: Path) -> None:
    text = f"""Random Forest Clinical Calibration - Circadian Ablation

Operating threshold: {metrics["threshold"]:.6f}
Threshold criterion: Youden's J statistic

Precision (Positive Predictive Value): {metrics["precision_ppv"]:.4f}
Recall (Sensitivity): {metrics["recall_sensitivity"]:.4f}
Specificity: {metrics["specificity"]:.4f}
False Positive Rate (Alarm Fatigue): {metrics["false_positive_rate"]:.4f}

Confusion Matrix:
  TP: {metrics["true_positives"]}
  FP: {metrics["false_positives"]}
  TN: {metrics["true_negatives"]}
  FN: {metrics["false_negatives"]}
"""
    output_path.write_text(text, encoding="utf-8")
    print(text)


def shap_values_for_positive_class(explainer, X: pd.DataFrame) -> np.ndarray:
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        return shap_values[1]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def write_shap_plots(model: Pipeline, X_test: pd.DataFrame, report_dir: Path) -> None:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for this analysis. Install it in the active environment "
            "with `pip install shap` and rerun this script."
        ) from exc

    imputer = model.named_steps["imputer"]
    rf = model.named_steps["model"]
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    explainer = shap.TreeExplainer(rf)
    positive_shap_values = shap_values_for_positive_class(explainer, X_test_imputed)

    plt.figure()
    shap.summary_plot(positive_shap_values, X_test_imputed, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(report_dir / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(positive_shap_values, X_test_imputed, plot_type="bar", show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(report_dir / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RF SHAP plots and calibration metrics.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_supervised_train(args.data_dir)
    X_test, y_test = load_test(args.data_dir)
    model = train_random_forest(X_train, y_train)

    scores = model.predict_proba(X_test)[:, 1]
    calibration = calculate_operating_point(y_test, scores)
    write_calibration_report(calibration, args.report_dir / "clinical_calibration.txt")
    write_shap_plots(model, X_test, args.report_dir)


if __name__ == "__main__":
    main()
