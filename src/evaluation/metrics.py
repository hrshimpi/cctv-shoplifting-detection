"""
Reusable anomaly-detection evaluator - precision, recall, F1,
specificity, false positive rate, ROC-AUC, PR-AUC - built once so
experiments C-E (later steps) reuse it unchanged. Only the (labels,
scores) pairs fed in differ between experiments, never this module.

Conventions used throughout:
  - Anomaly score: HIGHER = more anomalous (mean squared reconstruction
    error from the autoencoder - see src/models/bigru_autoencoder.py).
  - Labels: 1 = anomaly, 0 = normal.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


@dataclass
class EvaluationResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    specificity: float
    false_positive_rate: float
    roc_auc: float
    pr_auc: float
    num_normal: int
    num_anomaly: int

    def as_dict(self) -> dict:
        return asdict(self)


def choose_threshold(normal_scores: np.ndarray, percentile: float = 95.0) -> float:
    """Pick a decision threshold from the TRAINING set's normal-only
    reconstruction-error distribution (the given percentile of it).
    Scores at or above this threshold are classified anomalous. Only
    normal scores are used to set the threshold, matching the one-class
    setup - there are no anomaly examples available during training to
    tune against."""
    return float(np.percentile(normal_scores, percentile))


def evaluate(labels: np.ndarray, scores: np.ndarray, threshold: float) -> EvaluationResult:
    """labels: 0/1 array (1 = anomaly). scores: anomaly score per sample
    (higher = more anomalous), same length as labels."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores)
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    if len(np.unique(labels)) == 2:
        roc_auc = float(roc_auc_score(labels, scores))
        pr_auc = float(average_precision_score(labels, scores))
    else:
        # ROC-AUC/PR-AUC are undefined with only one class present.
        roc_auc = float("nan")
        pr_auc = float("nan")

    return EvaluationResult(
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        false_positive_rate=false_positive_rate,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        num_normal=int((labels == 0).sum()),
        num_anomaly=int((labels == 1).sum()),
    )


def plot_reconstruction_error_histogram(
    labels: np.ndarray,
    scores: np.ndarray,
    out_path: str | Path,
    threshold: float | None = None,
) -> None:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores)

    # A handful of degenerate windows can have reconstruction error orders
    # of magnitude above the bulk of the distribution, which (with
    # range=full-data-min/max) crams every informative bin into a sliver
    # near 0 and makes the plot unreadable. Scope the plotted range to the
    # 99th percentile instead - a presentation choice (zoom to where the
    # distribution and threshold actually are), not a change to what's
    # counted in precision/recall/ROC-AUC/etc., which use the raw scores.
    upper = float(np.percentile(scores, 99)) if len(scores) else 1.0
    if threshold is not None:
        upper = max(upper, threshold * 1.2)
    if upper <= 0:
        upper = float(scores.max()) if len(scores) and scores.max() > 0 else 1.0
    n_clipped = int((scores > upper).sum())

    plt.figure(figsize=(8, 5))
    plt.hist(scores[labels == 0], bins=50, range=(0, upper), alpha=0.6, label="normal", density=True)
    if (labels == 1).any():
        plt.hist(scores[labels == 1], bins=50, range=(0, upper), alpha=0.6, label="anomaly", density=True)
    if threshold is not None:
        plt.axvline(threshold, color="black", linestyle="--", label=f"threshold ({threshold:.4g})")
    plt.xlabel("Reconstruction error (MSE)")
    plt.ylabel("Density")
    title = "Reconstruction error: normal vs. anomaly"
    if n_clipped:
        title += f"\n(x-axis clipped at p99={upper:.4g}; {n_clipped}/{len(scores)} point(s) beyond it not shown)"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
