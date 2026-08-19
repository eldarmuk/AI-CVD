"""
Classical machine learning and anomaly-detection baselines.

Inputs are produced by 06_flatten_tabular_features.py. The validation split is
balanced for threshold selection and headline validation AUPRC, while the full
test split preserves its natural class imbalance.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "anomaly_detection"
REPORT_DIR = PROJECT_ROOT / "reports" / "classical_baselines"
MODEL_DIR = PROJECT_ROOT / "models" / "classical_baselines"

RANDOM_SEED = 42
N_BOOTSTRAPS = int(os.getenv("N_BOOTSTRAPS", "1000"))
MAX_OCSVM_TRAIN = int(os.getenv("MAX_OCSVM_TRAIN", "10000"))
MAX_IFOREST_TRAIN = int(os.getenv("MAX_IFOREST_TRAIN", "100000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_split(data_dir: Path, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.read_parquet(data_dir / f"X_{split}_flat.parquet")
    y = np.load(data_dir / f"y_{split}_l3.npy").astype(np.int8)
    if len(X) != len(y):
        raise ValueError(f"Length mismatch for {split}: X={len(X)} y={len(y)}")
    return X, y


def load_supervised_train(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    X_path = data_dir / "X_train_supervised_flat.parquet"
    y_path = data_dir / "y_train_supervised.npy"
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Missing mixed supervised training artifacts. Run "
            "python -m src.pipelines.06_flatten_tabular_features first."
        )
    X = pd.read_parquet(X_path)
    y = np.load(y_path).astype(np.int8)
    if len(X) != len(y):
        raise ValueError(f"Supervised train length mismatch: X={len(X)} y={len(y)}")
    return X, y


def load_pure_mask(data_dir: Path, split: str, y: np.ndarray) -> np.ndarray:
    path = data_dir / f"{split}_pure_healthy_mask.npy"
    if path.exists():
        return np.load(path).astype(bool)
    logger.warning("%s not found; falling back to y == 0 for healthy mask", path)
    return y == 0


def feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    drop_cols = {"senior_id", "target_timestamp"}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X, list(X.columns)


def balanced_indices(y: np.ndarray, seed: int = RANDOM_SEED) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Balanced evaluation requires both positive and negative examples.")
    n = min(len(pos), len(neg))
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    rng.shuffle(idx)
    return idx


def sample_rows(X: pd.DataFrame, y_mask: np.ndarray, max_rows: int, seed: int) -> pd.DataFrame:
    idx = np.flatnonzero(y_mask)
    if max_rows > 0 and len(idx) > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, max_rows, replace=False))
    return X.iloc[idx]


def bootstrap_auc_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bootstraps: int = N_BOOTSTRAPS,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
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


def evaluate_scores(
    model_name: str,
    y_val_bal: np.ndarray,
    val_bal_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, Any]:
    threshold, val_gmean = gmean_threshold(y_val_bal, val_bal_scores)
    y_pred = (test_scores >= threshold).astype(np.int8)
    ci_low, ci_high = bootstrap_auc_ci(y_test, test_scores)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    f1, f1_threshold = optimal_f1(y_test, test_scores)

    return {
        "model": model_name,
        "val_balanced_auroc": float(roc_auc_score(y_val_bal, val_bal_scores)),
        "val_balanced_auprc": float(average_precision_score(y_val_bal, val_bal_scores)),
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


def supervised_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression_l2": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        class_weight="balanced",
                        solver="saga",
                        max_iter=2000,
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "logistic_regression_elasticnet": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="elasticnet",
                        l1_ratio=0.25,
                        class_weight="balanced",
                        solver="saga",
                        max_iter=2000,
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
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
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=600,
                        max_depth=4,
                        learning_rate=0.03,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective="binary:logistic",
                        eval_metric="aucpr",
                        tree_method="hist",
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    else:
        logger.warning("xgboost is not installed; skipping XGBoost baseline.")
    return models


def anomaly_models() -> dict[str, Any]:
    return {
        "isolation_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    IsolationForest(
                        n_estimators=400,
                        contamination="auto",
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "one_class_svm": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)),
            ]
        ),
    }


def score_model(model: Any, X: pd.DataFrame, anomaly: bool = False) -> np.ndarray:
    if anomaly:
        return -model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def save_pickle(path: Path, obj: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def write_xgb_importance(model: Pipeline, feature_names: list[str], report_dir: Path) -> None:
    clf = model.named_steps.get("model")
    if clf is None or not hasattr(clf, "feature_importances_"):
        return
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": clf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(report_dir / "xgboost_feature_importance.csv", index=False)


def skipped_supervised_row(model_name: str, y_train: np.ndarray, reason: str) -> dict[str, Any]:
    classes, counts = np.unique(y_train, return_counts=True)
    return {
        "model": model_name,
        "model_family": "Supervised Classifier",
        "status": "skipped",
        "skip_reason": reason,
        "train_classes": json.dumps({str(cls): int(count) for cls, count in zip(classes, counts)}),
        "val_balanced_auroc": np.nan,
        "val_balanced_auprc": np.nan,
        "val_gmean_threshold": np.nan,
        "val_gmean": np.nan,
        "test_auroc": np.nan,
        "test_auroc_ci_low": np.nan,
        "test_auroc_ci_high": np.nan,
        "test_auprc": np.nan,
        "test_optimal_f1": np.nan,
        "test_optimal_f1_threshold": np.nan,
        "test_precision_at_80_recall": np.nan,
        "test_confusion_tn": np.nan,
        "test_confusion_fp": np.nan,
        "test_confusion_fn": np.nan,
        "test_confusion_tp": np.nan,
        "test_threshold_source": np.nan,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classical baseline benchmark suite.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--skip-ocsvm", action="store_true", help="Skip One-Class SVM for very large runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    train_df, y_train = load_split(args.data_dir, "train")
    train_supervised_df, y_train_supervised = load_supervised_train(args.data_dir)
    val_df, y_val = load_split(args.data_dir, "val")
    test_df, y_test = load_split(args.data_dir, "test")
    train_pure_mask = load_pure_mask(args.data_dir, "train", y_train)

    X_train, feature_names = feature_matrix(train_df)
    X_train_supervised, supervised_feature_names = feature_matrix(train_supervised_df)
    X_val, _ = feature_matrix(val_df)
    X_test, _ = feature_matrix(test_df)
    if supervised_feature_names != feature_names:
        raise ValueError("Supervised and pure-healthy flattened feature columns do not match.")
    val_bal_idx = balanced_indices(y_val)

    metrics_rows: list[dict[str, Any]] = []

    supervised_classes = np.unique(y_train_supervised)
    if len(supervised_classes) < 2:
        reason = (
            "Supervised classifier requires at least two classes in y_train_supervised; "
            "the mixed supervised cohort is single-class."
        )
        logger.warning(
            "Skipping supervised baselines because y_train_supervised has one class only: %s. "
            "Proceeding with healthy-only anomaly detectors.",
            supervised_classes.tolist(),
        )
        metrics_rows.extend(skipped_supervised_row(name, y_train_supervised, reason) for name in supervised_models())
    else:
        for name, model in supervised_models().items():
            logger.info("Training %s on mixed supervised cohort (%s windows)", name, len(X_train_supervised))
            fit_kwargs: dict[str, Any] = {}
            if name == "xgboost":
                fit_kwargs["model__sample_weight"] = compute_sample_weight("balanced", y_train_supervised)
            model.fit(X_train_supervised, y_train_supervised, **fit_kwargs)
            val_scores = score_model(model, X_val)
            test_scores = score_model(model, X_test)
            row = evaluate_scores(name, y_val[val_bal_idx], val_scores[val_bal_idx], y_test, test_scores)
            row["status"] = "fit"
            row["model_family"] = "Tree Ensemble" if name in {"random_forest", "xgboost"} else "Regularized Linear"
            row["train_windows"] = int(len(X_train_supervised))
            row["train_source"] = "mixed supervised few-shot cohort"
            metrics_rows.append(row)
            np.savez_compressed(
                args.report_dir / f"predictions_{name}.npz",
                y_val_balanced=y_val[val_bal_idx],
                val_balanced_scores=val_scores[val_bal_idx],
                y_test=y_test,
                test_scores=test_scores,
            )
            save_pickle(args.model_dir / f"{name}.pkl", model)
            if name == "xgboost":
                write_xgb_importance(model, feature_names, args.report_dir)

    healthy_train = sample_rows(X_train, train_pure_mask, MAX_IFOREST_TRAIN, RANDOM_SEED)
    for name, model in anomaly_models().items():
        if args.skip_ocsvm and name == "one_class_svm":
            logger.info("Skipping one_class_svm by request.")
            continue
        train_subset = healthy_train
        if name == "one_class_svm":
            train_subset = sample_rows(X_train, train_pure_mask, MAX_OCSVM_TRAIN, RANDOM_SEED)
        logger.info("Training %s on %s pure-healthy windows", name, len(train_subset))
        model.fit(train_subset)
        val_scores = score_model(model, X_val, anomaly=True)
        test_scores = score_model(model, X_test, anomaly=True)
        row = evaluate_scores(name, y_val[val_bal_idx], val_scores[val_bal_idx], y_test, test_scores)
        row["model_family"] = "Classical Anomaly Detector"
        row["status"] = "fit"
        row["train_windows"] = int(len(train_subset))
        metrics_rows.append(row)
        np.savez_compressed(
            args.report_dir / f"predictions_{name}.npz",
            y_val_balanced=y_val[val_bal_idx],
            val_balanced_scores=val_scores[val_bal_idx],
            y_test=y_test,
            test_scores=test_scores,
        )
        save_pickle(args.model_dir / f"{name}.pkl", model)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("test_auprc", ascending=False)
    metrics_df.to_csv(args.report_dir / "classical_baseline_metrics.csv", index=False)
    with open(args.report_dir / "classical_baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, indent=2)
    logger.info("Classical benchmark complete: %s", args.report_dir)


if __name__ == "__main__":
    main()
