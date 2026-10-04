"""Positive-class metrics and validation-only threshold selection."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def checked_arrays(labels, probabilities):
    if any(isinstance(v, bool | np.bool_) or not isinstance(v, int | np.integer) for v in labels):
        raise ValueError("Labels must be strict integers")
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Labels and probabilities require equal nonempty one-dimensional arrays")
    if any(type(v) not in (int, np.int32, np.int64) for v in y.tolist()):
        raise ValueError("Labels must be strict integers")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Require binary labels and finite probabilities in [0, 1]")
    return y.astype(int), p


def select_threshold(labels, probabilities):
    y, p = checked_arrays(labels, probabilities)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both classes")
    precision, recall, thresholds = precision_recall_curve(y, p)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    # Stable tie rule: largest threshold among the F1 maxima.
    return float(thresholds[np.flatnonzero(scores == scores.max())[-1]])


def classification_metrics(labels, probabilities, threshold):
    y, p = checked_arrays(labels, probabilities)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid classification threshold")
    if len(np.unique(y)) != 2:
        raise ValueError("AP/ROC comparison requires both classes")
    predicted = (p >= threshold).astype(int)
    edges = np.linspace(0, 1, 11)
    bins = np.minimum(np.digitize(p, edges) - 1, 9)
    calibration = []
    for i in range(10):
        selected = bins == i
        if selected.any():
            calibration.append(
                {
                    "count": int(selected.sum()),
                    "mean_probability": float(p[selected].mean()),
                    "fraction_positive": float(y[selected].mean()),
                }
            )
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y, p)),
        "sample_count": int(len(y)),
        "positive_label_count": int(y.sum()),
        "negative_label_count": int((y == 0).sum()),
        "threshold": float(threshold),
        "calibration": calibration,
    }
