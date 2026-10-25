"""Compare saved A/B checkpoints on identical RetailS windows and export errors.

No training or threshold search occurs here. Each checkpoint retains its
normal-only p95 calibration and the existing video-frame-mask window labels.
"""

import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

from scripts.evaluate import score_windows
from scripts.analyze_false_positives import COCO17_BONES, find_track_window
from src.data.retails import discover_retails_root
from src.evaluation.metrics import choose_threshold, evaluate
from src.experiments.dataset import WindowMeta, build_windows_for_subset
from src.features.motion_features import feature_dim
from src.models.bigru_autoencoder import BiGRUAutoencoder


def classify_errors(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("Nonempty, aligned one-dimensional labels and scores required")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("Binary labels and finite scores/threshold required")
    predicted = scores >= threshold
    outcomes = np.where(labels == 1, np.where(predicted, "TP", "FN"), np.where(predicted, "FP", "TN"))
    counts = {key: int((outcomes == key).sum()) for key in ("TN", "FP", "FN", "TP")}
    return outcomes, counts


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def render_error(root, row, cfg, out):
    """Reconstruct a scored window from its source for pose-quality inspection."""
    meta = WindowMeta(**{key: row[key] for key in WindowMeta.__dataclass_fields__})
    points = find_track_window(root, row["subset"], meta, cfg)
    offsets = [0, len(points) // 3, 2 * len(points) // 3, len(points) - 1]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), sharex=True, sharey=True)
    for ax, offset in zip(axes, offsets):
        xy = points[offset, :, :2]
        visible = points[offset, :, 2] >= cfg["confidence_threshold"]
        ax.scatter(xy[visible, 0], -xy[visible, 1], s=16)
        for a, b in COCO17_BONES:
            if visible[a] and visible[b]:
                ax.plot(xy[[a, b], 0], -xy[[a, b], 1], linewidth=1.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Source frame {meta.start_frame + offset}", fontsize=9)
        ax.set_xlabel("Normalized x")
    axes[0].set_ylabel("Normalized -y")
    fig.suptitle(f"{row['experiment'].upper()} {row['outcome']} | {row['subset']} | {meta.pose_file} person {meta.person_id}\n"
                 f"Score {row['score']:.4g}, threshold {row['threshold']:.4g}; video-mask label {row['label']}", fontsize=10)
    fig.tight_layout()
    path = out / f"{row['experiment']}_{row['subset']}_{row['outcome'].lower()}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def main():
    out = ROOT / "outputs/logs/ab_comparison"
    out.mkdir(parents=True, exist_ok=True)
    root = discover_retails_root(ROOT / "data/retails")
    if root is None:
        raise FileNotFoundError("RetailS dataset is required")
    checkpoints, splits = {}, {}
    for name in ("a", "b"):
        checkpoints[name] = torch.load(ROOT / f"outputs/checkpoints/experiment_{name}/model.pt", weights_only=False, map_location="cpu")
        splits[name] = json.loads((ROOT / f"outputs/logs/experiment_{name}/split_manifest.json").read_text())
        if checkpoints[name]["split"] != splits[name]:
            raise ValueError(f"Checkpoint and split manifest disagree for {name}")
    if splits["a"] != splits["b"]:
        raise ValueError("A/B source-video splits must match")
    clean_cfg = lambda c: {k: v for k, v in c.items() if k not in ("experiment_name", "use_velocity")}
    if clean_cfg(checkpoints["a"]["config"]) != clean_cfg(checkpoints["b"]["config"]):
        raise ValueError("A/B differ in settings beyond velocity features")
    if checkpoints["a"]["config"]["use_velocity"] or not checkpoints["b"]["config"]["use_velocity"]:
        raise ValueError("Expected position-only A and position+velocity B")

    reference_keys = None
    rows, results, plot_data = [], {}, {}
    for name, checkpoint in checkpoints.items():
        cfg = checkpoint["config"]
        model = BiGRUAutoencoder(input_dim=feature_dim(cfg["use_velocity"], cfg["use_acceleration"]),
                                 hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
        model.load_state_dict(checkpoint["model_state_dict"])
        calibration = np.load(ROOT / f"outputs/checkpoints/experiment_{name}/train_reconstruction_errors.npy")
        if (len(calibration) != splits[name]["train_windows"] + splits[name]["validation_windows"]
                or not np.isfinite(calibration).all()):
            raise ValueError("Invalid normal calibration errors")
        threshold = choose_threshold(calibration, cfg["threshold_percentile"])
        saved_metrics = json.loads((ROOT / f"outputs/logs/experiment_{name}/eval_metrics.json").read_text())
        all_labels, all_scores, keys = [], [], []
        subsets = {}
        for subset in ("test_realworld", "test_staged"):
            features, labels, metadata = build_windows_for_subset(root, subset, cfg,
                use_velocity=cfg["use_velocity"], use_acceleration=cfg["use_acceleration"],
                max_pose_files=None, seed=cfg["seed"], return_metadata=True)
            scores, _ = score_windows(model, features, cfg["batch_size"])
            outcomes, counts = classify_errors(labels, scores, threshold)
            metrics = evaluate(labels, scores, threshold).as_dict()
            for key, value in metrics.items():
                if not np.isclose(value, saved_metrics[subset][key], rtol=1e-5, atol=1e-7, equal_nan=True):
                    raise ValueError(f"Saved evaluation differs: {name}/{subset}/{key}")
            subsets[subset] = {**metrics, "confusion": counts, "anomaly_fraction": float(labels.mean())}
            for meta, label, score, outcome in zip(metadata, labels, scores, outcomes):
                info = asdict(meta)
                keys.append((subset, *info.values(), int(label)))
                rows.append({"experiment": name, "subset": subset, **info,
                             "label": int(label), "score": float(score), "threshold": threshold,
                             "prediction": int(score >= threshold), "outcome": str(outcome)})
            all_labels.append(labels)
            all_scores.append(scores)
        if reference_keys is not None and keys != reference_keys:
            raise ValueError("A/B evaluation window identities or labels do not match")
        reference_keys = keys
        labels, scores = np.concatenate(all_labels), np.concatenate(all_scores)
        outcomes, counts = classify_errors(labels, scores, threshold)
        combined = evaluate(labels, scores, threshold).as_dict()
        quantiles = {label_name: dict(zip(["p05", "p50", "p95"],
                     np.quantile(scores[labels == value], [.05, .5, .95]).tolist()))
                     for value, label_name in ((0, "normal"), (1, "anomaly"))}
        experiment_rows = [r for r in rows if r["experiment"] == name]
        camera_counts = []
        for subset, camera in sorted({(r["subset"], r["camera_id"]) for r in experiment_rows}):
            group = [r for r in experiment_rows if r["subset"] == subset and r["camera_id"] == camera]
            camera_counts.append({"subset": subset, "camera_id": camera,
                                  **{k: sum(r["outcome"] == k for r in group) for k in counts}})
        results[name] = {"feature_dim": feature_dim(cfg["use_velocity"], cfg["use_acceleration"]),
            "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_sha256": digest(ROOT / f"outputs/checkpoints/experiment_{name}/model.pt"),
            "calibration_windows": len(calibration),
            "calibration_score_quantiles": dict(zip(["p50", "p90", "p95", "p99", "max"],
                np.quantile(calibration, [.5, .9, .95, .99, 1]).tolist())), "subsets": subsets,
            "combined": {**combined, "confusion": counts, "anomaly_fraction": float(labels.mean())},
            "score_quantiles": quantiles, "per_camera": camera_counts,
            "missed_anomaly_examples": sorted([r for r in experiment_rows if r["outcome"] == "FN"], key=lambda r: r["score"])[:10],
            "false_positive_examples": [r for r in experiment_rows if r["outcome"] == "FP"]}
        example_plots = []
        for subset in subsets:
            for outcome in ("FN", "FP"):
                candidates = [r for r in experiment_rows if r["subset"] == subset and r["outcome"] == outcome]
                if candidates:
                    selected = sorted(candidates, key=lambda r: r["score"], reverse=outcome == "FP")[0]
                    example_plots.append(render_error(root, selected, cfg, out))
        results[name]["error_pose_plots"] = example_plots
        plot_data[name] = labels, scores, counts
        print(f"{name.upper()}: {json.dumps(results[name]['combined'])}", flush=True)

    with (out / "window_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"matched_training_split": True, "matched_test_windows": True, "test_windows": len(reference_keys),
              "threshold_protocol": "p95 of saved normal-only training+validation reconstruction errors; no test-label tuning",
              "label_protocol": "Any anomalous video frame labels every person's overlapping window anomalous; not person-level ground truth",
              "training_scope": "50 sampled pose files, 15 epochs, one seed; final-epoch checkpoint",
              "experiments": results}
    (out / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Matched 2D model comparison", "",
             "Both models use 50 sampled normal pose files, 15 epochs and the same source-video split.",
             "Evaluation uses identical windows from both complete labeled test subsets. Thresholds retain normal-only p95 calibration.", "",
             "| Experiment | Features | Subset | Normal | Anomaly | Precision | Recall | F1 | ROC-AUC | AP | FN | FP |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        for subset, m in {**result["subsets"], "combined": result["combined"]}.items():
            lines.append(f"| {name.upper()} | {result['feature_dim']} | {subset} | {m['num_normal']} | {m['num_anomaly']} | "
                         f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | "
                         f"{m['confusion']['FN']} | {m['confusion']['FP']} |")
    lines += ["", "## Interpretation limits", "",
              "- High precision or AP must be read alongside anomaly prevalence and recall.",
              "- Window labels describe video-mask overlap, not person-specific theft actions.",
              "- One seed, overlapping windows and sampled training files limit generalization claims.",
              "- Input/output projections grow with the feature dimension; hidden width/depth remain fixed.",
              "- Final-epoch checkpoints are used; test labels did not select an epoch or threshold.",
              "- Error strips are normalized skeletons, not source RGB or proof of an individual's action.", "",
              "See `window_predictions.csv` for every source-linked prediction and `comparison.json` for score distributions and camera counts."]
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for i, (name, (labels, scores, counts)) in enumerate(plot_data.items()):
        matrix = np.array([[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]])
        ax = axes[0, i]
        ax.imshow(matrix, cmap="Blues")
        for (y, x), value in np.ndenumerate(matrix):
            ax.text(x, y, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"],
               xlabel="Predicted", ylabel="Video-mask window label", title=f"Experiment {name.upper()}: confusion counts")
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        axes[1, 0].plot(fpr, tpr, label=f"{name.upper()} (AUC {results[name]['combined']['roc_auc']:.3f})")
        axes[1, 1].plot(recall, precision, label=f"{name.upper()} (AP {results[name]['combined']['pr_auc']:.3f})")
    axes[1, 0].plot([0, 1], [0, 1], "k--", label="Chance")
    axes[1, 0].set(xlabel="False positive rate", ylabel="Recall", title="ROC: identical held-out windows")
    axes[1, 1].axhline(labels.mean(), color="black", linestyle="--", label=f"Anomaly prevalence {labels.mean():.3f}")
    axes[1, 1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall: class imbalance context", ylim=(0, 1.02))
    for ax in axes[1]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved comparison and traceable window predictions to {out}")


if __name__ == "__main__":
    main()
