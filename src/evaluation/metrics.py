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


# --------------------------------------------------------------------------------------------
# research_v2 additions (the functions above are unchanged so research_v1 stays reproducible)
# --------------------------------------------------------------------------------------------

def fast_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) ROC-AUC with tie averaging; NaN if one class is absent."""
    from scipy.stats import rankdata
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def recall_at_fpr(labels, scores, fpr_target=0.05) -> float:
    from sklearn.metrics import roc_curve
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return float('nan')
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.max(tpr[fpr <= fpr_target])) if np.any(fpr <= fpr_target) else 0.0


def specificity_at_recall(labels, scores, recall_target=0.9) -> float:
    from sklearn.metrics import roc_curve
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return float('nan')
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = tpr >= recall_target
    return float(1 - np.min(fpr[ok])) if np.any(ok) else 0.0


def balanced_accuracy(labels, scores, threshold) -> float:
    labels = np.asarray(labels).astype(int)
    pred = np.asarray(scores) >= threshold
    tpr = pred[labels == 1].mean() if (labels == 1).any() else float('nan')
    tnr = (~pred[labels == 0]).mean() if (labels == 0).any() else float('nan')
    return float(np.nanmean([tpr, tnr]))


def grouped_bootstrap(labels, scores, groups, metric=None, draws=1000, seed=0, alpha=0.05):
    """Percentile CI that resamples whole groups (videos) with replacement.

    Frames of one video are strongly correlated, so resampling frames would make
    the interval far too narrow. Returns (low, high, samples)."""
    metric = metric or fast_roc_auc
    labels, scores, groups = np.asarray(labels), np.asarray(scores), np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    bounds = np.concatenate([[0], np.cumsum(np.bincount(inverse, minlength=len(uniq)))])
    members = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        value = metric(labels[idx], scores[idx])
        if np.isfinite(value):
            out.append(value)
    if not out:
        return float('nan'), float('nan'), np.array([])
    out = np.asarray(out)
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2)), out


def detection_metrics(labels, scores, threshold, cameras=None, groups=None, draws=1000, seed=0) -> dict:
    """All research_v2 metrics for one set of (frame or window) scores."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    has_both = len(np.unique(labels)) == 2
    pred = scores >= threshold
    out = dict(n=int(len(labels)), num_normal=int((labels == 0).sum()), num_anomaly=int((labels == 1).sum()),
               base_rate=float(labels.mean()) if len(labels) else float('nan'), threshold=float(threshold),
               roc_auc=fast_roc_auc(labels, scores),
               pr_auc=float(average_precision_score(labels, scores)) if has_both else float('nan'),
               balanced_accuracy=balanced_accuracy(labels, scores, threshold),
               recall=float(pred[labels == 1].mean()) if (labels == 1).any() else float('nan'),
               specificity=float((~pred[labels == 0]).mean()) if (labels == 0).any() else float('nan'),
               recall_at_fpr5=recall_at_fpr(labels, scores, 0.05),
               specificity_at_recall90=specificity_at_recall(labels, scores, 0.9))
    out['pr_auc_lift'] = out['pr_auc'] / out['base_rate'] if has_both else float('nan')
    if cameras is not None:
        cameras = np.asarray(cameras)
        per = {}
        for c in np.unique(cameras):
            m = cameras == c
            per[int(c)] = fast_roc_auc(labels[m], scores[m])
        vals = np.array([v for v in per.values() if np.isfinite(v)])
        out['per_camera_auc'] = per
        out['camera_auc_mean'] = float(vals.mean()) if len(vals) else float('nan')
        out['camera_auc_std'] = float(vals.std()) if len(vals) else float('nan')
    if groups is not None and draws and has_both:
        lo, hi, _ = grouped_bootstrap(labels, scores, groups, draws=draws, seed=seed)
        out['roc_auc_ci'] = [lo, hi]
    return out
