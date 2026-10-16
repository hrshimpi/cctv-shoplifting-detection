#!/usr/bin/env python
"""
Evaluate a trained experiment's checkpoint on the two labeled RetailS
test subsets (test_realworld, test_staged, and combined). Standalone,
runs locally - no notebook required.

Usage:
    python scripts/evaluate.py --experiment a
    python scripts/evaluate.py --experiment b
    python scripts/evaluate.py --experiment c
    python scripts/evaluate.py --experiment d
    python scripts/evaluate.py --experiment e

Loads the checkpoint + config scripts/train.py saved, rebuilds labeled
evaluation windows with the SAME feature flags used in training (via
src/experiments/dataset.py, so train/eval windows are built identically),
scores them with the model's reconstruction error, sets a decision
threshold from the saved training-set error distribution, and reports
precision/recall/F1/specificity/FPR/ROC-AUC/PR-AUC plus inference time
(ms/window, the trained autoencoder's own forward+scoring pass - not
including the one-time, shared MotionBERT lift for C/D/E, which isn't a
property of the autoencoder architecture being compared) via
src/evaluation/metrics.py - the same evaluator every experiment (A-E)
reuses unchanged.

A/B evaluate on the FULL, unsampled test_realworld/test_staged. C/D/E
(their config's "max_eval_pose_files") sample a fixed number of pose
files from each instead - a MotionBERT forward pass per window makes the
full test subsets infeasible on CPU (~12h estimated); see README.md
"Limitations" for this tradeoff stated plainly.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.data.retails import discover_retails_root  # noqa: E402
from src.evaluation.metrics import choose_threshold, evaluate, plot_reconstruction_error_histogram  # noqa: E402
from src.experiments.dataset import build_windows_for_subset, build_windows_for_subset_3d  # noqa: E402
from src.features.motion_features import feature_dim, feature_dim_3d  # noqa: E402
from src.models.bigru_attention import BiGRUAttentionAutoencoder  # noqa: E402
from src.models.bigru_autoencoder import BiGRUAutoencoder  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "retails"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "third_party" / "MotionBERT" / "checkpoint" / "pose3d" / "MB_ft_h36m" / "best_epoch.bin"

EXPERIMENT_CONFIGS = {
    "a": PROJECT_ROOT / "configs" / "experiment_a.yaml",
    "b": PROJECT_ROOT / "configs" / "experiment_b.yaml",
    "c": PROJECT_ROOT / "configs" / "experiment_c.yaml",
    "d": PROJECT_ROOT / "configs" / "experiment_d.yaml",
    "e": PROJECT_ROOT / "configs" / "experiment_e.yaml",
}
EXPERIMENTS_3D = {"c", "d", "e"}
MODEL_CLASSES = {"bigru": BiGRUAutoencoder, "bigru_attention": BiGRUAttentionAutoencoder}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENT_CONFIGS), required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="MotionBERT checkpoint path (experiments c/d/e only)")
    return parser.parse_args()


def score_windows(model: BiGRUAutoencoder, features: np.ndarray, batch_size: int) -> tuple[np.ndarray, float]:
    """Returns (scores, ms_per_window) - timing covers only the trained
    autoencoder's own forward+reconstruction-error pass (model.eval(),
    no_grad, CPU), the same measurement for every experiment regardless
    of whether its features came from the cheap 2D path or the cached
    3D lift, so inference-time comparisons across A-E are apples to
    apples."""
    loader = DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=batch_size, shuffle=False)
    model.eval()
    scores = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for (batch,) in loader:
            scores.append(model.reconstruction_error(batch).numpy())
    elapsed = time.perf_counter() - t0
    scores = np.concatenate(scores) if scores else np.empty((0,))
    ms_per_window = (elapsed / len(scores) * 1000) if len(scores) else float("nan")
    return scores, ms_per_window


def main() -> None:
    args = parse_args()
    with open(EXPERIMENT_CONFIGS[args.experiment], encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = OUTPUTS_DIR / "checkpoints" / cfg["experiment_name"]
    log_dir = OUTPUTS_DIR / "logs" / cfg["experiment_name"]
    plot_dir = OUTPUTS_DIR / "plots" / cfg["experiment_name"]
    checkpoint_path = out_dir / "model.pt"
    if not checkpoint_path.exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint_path} - run `python scripts/train.py --experiment {args.experiment}` first."
        )

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    train_cfg = checkpoint["config"]

    model_3d = None
    if args.experiment in EXPERIMENTS_3D:
        checkpoint_3d_path = Path(args.checkpoint)
        if not checkpoint_3d_path.exists():
            raise SystemExit(f"No MotionBERT checkpoint at {checkpoint_3d_path} - see README.md 'MotionBERT setup'.")
        from src.motionbert.lift import load_motionbert

        model_3d = load_motionbert(checkpoint_3d_path)
        input_dim = feature_dim_3d(
            train_cfg["use_velocity"], train_cfg["use_acceleration"], train_cfg["use_geometry"], train_cfg["use_angles"]
        )
    else:
        input_dim = feature_dim(train_cfg["use_velocity"], train_cfg["use_acceleration"])
    model_cls = MODEL_CLASSES[train_cfg.get("model", "bigru")]
    model = model_cls(
        input_dim=input_dim,
        hidden_size=train_cfg["hidden_size"],
        num_layers=train_cfg["num_layers"],
        dropout=train_cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_errors = np.load(out_dir / "train_reconstruction_errors.npy")
    threshold = choose_threshold(train_errors, percentile=train_cfg["threshold_percentile"])
    print(f"Decision threshold (p{train_cfg['threshold_percentile']} of training reconstruction error): {threshold:.6f}")

    root = discover_retails_root(DATA_DIR)
    if root is None:
        raise SystemExit(f"Could not find RetailS under {DATA_DIR}.")

    results = {}
    all_features, all_labels = [], []
    for subset_key in ("test_realworld", "test_staged"):
        if args.experiment in EXPERIMENTS_3D:
            features, labels = build_windows_for_subset_3d(
                root,
                subset_key,
                train_cfg,
                model_3d,
                use_velocity=train_cfg["use_velocity"],
                use_acceleration=train_cfg["use_acceleration"],
                use_geometry=train_cfg["use_geometry"],
                use_angles=train_cfg["use_angles"],
                max_pose_files=train_cfg.get("max_eval_pose_files"),
                seed=train_cfg["seed"],
            )
        else:
            features, labels = build_windows_for_subset(
                root,
                subset_key,
                train_cfg,
                use_velocity=train_cfg["use_velocity"],
                use_acceleration=train_cfg["use_acceleration"],
                max_pose_files=None,
                seed=train_cfg["seed"],
            )
        if len(features) == 0:
            print(f"No windows built for {subset_key} (tracks too short for window_length={train_cfg['window_length']}) - skipping.")
            continue

        scores, ms_per_window = score_windows(model, features, train_cfg["batch_size"])
        result = evaluate(labels, scores, threshold)
        results[subset_key] = {**result.as_dict(), "inference_ms_per_window": ms_per_window}
        print(f"\n--- {subset_key} ---")
        for k, v in results[subset_key].items():
            print(f"  {k}: {v}")

        plot_reconstruction_error_histogram(
            labels, scores, plot_dir / f"reconstruction_error_histogram_{subset_key}.png", threshold=threshold
        )

        all_features.append(features)
        all_labels.append(labels)

    if all_features:
        combined_features = np.concatenate(all_features)
        combined_labels = np.concatenate(all_labels)
        combined_scores, combined_ms_per_window = score_windows(model, combined_features, train_cfg["batch_size"])
        combined_result = evaluate(combined_labels, combined_scores, threshold)
        results["combined"] = {**combined_result.as_dict(), "inference_ms_per_window": combined_ms_per_window}
        print("\n--- combined (test_realworld + test_staged) ---")
        for k, v in results["combined"].items():
            print(f"  {k}: {v}")
        plot_reconstruction_error_histogram(
            combined_labels, combined_scores, plot_dir / "reconstruction_error_histogram_combined.png", threshold=threshold
        )

    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {log_dir / 'eval_metrics.json'}")
    print(f"Histograms saved under {plot_dir}/")


if __name__ == "__main__":
    main()
