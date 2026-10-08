"""Validated binary metrics and whole-tie priority threshold selection.

Release adapter preserves the Study B selection rule while validating public inputs.
"""

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def validate_scores(y, p):
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Nonempty aligned one-dimensional labels and scores required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Binary labels and finite probabilities in [0,1] required")
    return y.astype(int), p


def binary_metrics(y, p):
    y, p = validate_scores(y, p)
    prevalence = float(y.mean())
    ap = float(average_precision_score(y, p)) if y.any() else None
    return {
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "ap": ap,
        "prevalence": prevalence,
        "ap_over_prevalence": ap / prevalence if prevalence else None,
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def operating_metrics(y, p, threshold):
    y, p = validate_scores(y, p)
    if not np.isfinite(threshold):
        raise ValueError("Finite threshold required")
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))

    def ratio(a, b):
        return a / b if b else None

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "PPV": ratio(tp, tp + fp),
        "NPV": ratio(tn, tn + fn),
        "fraction_prioritized": float(pred.mean()),
    }


def choose_threshold(y, p, target=0.8):
    """Maximize specificity subject to sensitivity; break ties at largest threshold."""
    y, p = validate_scores(y, p)
    if not 0 < target <= 1:
        raise ValueError("Target sensitivity must lie in (0,1]")
    if len(np.unique(y)) != 2:
        return {"available": False, "reason": "undefined_sensitivity_or_specificity"}
    candidates = np.r_[np.nextafter(float(p.max()), np.inf), np.unique(p)]
    feasible = []
    for threshold in candidates:
        metrics = operating_metrics(y, p, threshold)
        if metrics["sensitivity"] >= target:
            feasible.append((metrics["specificity"], float(threshold), metrics))
    _, threshold, metrics = max(feasible, key=lambda x: (x[0], x[1]))
    return {"available": True, "threshold": threshold, "target": target, **metrics}
