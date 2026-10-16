#!/usr/bin/env python
"""
Per-camera analysis (step 4, requirement 5): break the test-set metrics
down by RetailS camera_id for the best 2D experiment and the best 3D
experiment, and check the spec's hypothesis that 3D pose may be less
sensitive to camera viewpoint than 2D pose - against the real per-camera
numbers, not assumed. Standalone, runs locally - no notebook required.

Usage:
    python scripts/analyze_per_camera.py --exp-2d a --exp-3d d

"Best" (by combined ROC-AUC, the least threshold-dependent metric used
throughout README.md "Results") is a CLI default, not hardcoded, so this
can be re-run if a later run changes which experiment is best.

The two experiments are NOT on equal footing here, and this script
states that plainly in its own output: A evaluates on the FULL,
unsampled test sets (all 6 cameras, thousands of windows per camera);
D evaluates on only max_eval_pose_files sampled files per subset (see
README.md "Limitations"), so its per-camera breakdown may have far
fewer windows for some cameras, or miss some cameras' normal windows
entirely (already observed: zero normal windows in D's test_staged
sample at all). Variance-by-camera comparisons are only meaningful for
cameras where both experiments have enough windows of both classes to
compute ROC-AUC - reported explicitly, not glossed over.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.data.retails import discover_retails_root  # noqa: E402
from src.evaluation.metrics import choose_threshold, evaluate  # noqa: E402
from src.experiments.dataset import build_windows_for_subset, build_windows_for_subset_3d  # noqa: E402
from src.features.motion_features import feature_dim, feature_dim_3d  # noqa: E402
from src.models.bigru_attention import BiGRUAttentionAutoencoder  # noqa: E402
from src.models.bigru_autoencoder import BiGRUAutoencoder  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "retails"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "third_party" / "MotionBERT" / "checkpoint" / "pose3d" / "MB_ft_h36m" / "best_epoch.bin"
EXPERIMENTS_3D = {"c", "d", "e"}
MODEL_CLASSES = {"bigru": BiGRUAutoencoder, "bigru_attention": BiGRUAttentionAutoencoder}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-2d", default="a", help="best 2D experiment letter")
    parser.add_argument("--exp-3d", default="d", help="best 3D experiment letter")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    return parser.parse_args()


def load_model_and_threshold(experiment: str, checkpoint_3d_path: Path):
    checkpoint = torch.load(OUTPUTS_DIR / "checkpoints" / f"experiment_{experiment}" / "model.pt", weights_only=False)
    cfg = checkpoint["config"]
    model_3d = None
    if experiment in EXPERIMENTS_3D:
        from src.motionbert.lift import load_motionbert

        model_3d = load_motionbert(checkpoint_3d_path)
        input_dim = feature_dim_3d(cfg["use_velocity"], cfg["use_acceleration"], cfg["use_geometry"], cfg["use_angles"])
    else:
        input_dim = feature_dim(cfg["use_velocity"], cfg["use_acceleration"])
    model_cls = MODEL_CLASSES[cfg.get("model", "bigru")]
    model = model_cls(input_dim=input_dim, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    train_errors = np.load(OUTPUTS_DIR / "checkpoints" / f"experiment_{experiment}" / "train_reconstruction_errors.npy")
    threshold = choose_threshold(train_errors, percentile=cfg["threshold_percentile"])
    return model, model_3d, cfg, threshold


def gather_windows(experiment: str, cfg: dict, model_3d, root: Path, realworld_max_override: str = "default"):
    """All windows across both test subsets, with per-window camera_id,
    for one experiment.

    realworld_max_override: "default" uses the experiment's own config
    (cfg["max_eval_pose_files"], None for 2D experiments - the full
    subset). Pass None explicitly to force the FULL test_realworld
    subset regardless of config - used for the per-camera comparison
    specifically, since test_realworld is far more class-balanced than
    test_staged (whose normal windows are too rare, at any feasible
    sample size, to support a per-camera breakdown - see module
    docstring) and only 47 pose files total, so lifting it in full for a
    3D experiment is a one-time, cacheable cost worth paying for a
    properly-powered viewpoint comparison, rather than leaving 5 of 6
    cameras without a single normal window."""
    all_features, all_labels, all_cameras = [], [], []
    for subset_key in ("test_realworld", "test_staged"):
        if subset_key == "test_realworld" and realworld_max_override != "default":
            max_pose_files = realworld_max_override
        else:
            max_pose_files = cfg.get("max_eval_pose_files") if experiment in EXPERIMENTS_3D else None
        if experiment in EXPERIMENTS_3D:
            features, labels, meta = build_windows_for_subset_3d(
                root, subset_key, cfg, model_3d,
                use_velocity=cfg["use_velocity"], use_acceleration=cfg["use_acceleration"],
                use_geometry=cfg["use_geometry"], use_angles=cfg["use_angles"],
                max_pose_files=max_pose_files, seed=cfg["seed"], return_metadata=True,
            )
        else:
            features, labels, meta = build_windows_for_subset(
                root, subset_key, cfg, use_velocity=cfg["use_velocity"], use_acceleration=cfg["use_acceleration"],
                max_pose_files=None, seed=cfg["seed"], return_metadata=True,
            )
        if len(features) == 0:
            continue
        all_features.append(features)
        all_labels.append(labels)
        # test_realworld and test_staged use different camera_id
        # zero-padding in their filenames (confirmed: "01_0222.json" vs
        # "1_1050000.json" - same physical camera, different string) -
        # normalize via int() before grouping, or "camera 1" and
        # "camera 01" silently fragment into two buckets when the two
        # test subsets are combined, as they are here.
        all_cameras.extend(str(int(m.camera_id)) for m in meta)
    return np.concatenate(all_features), np.concatenate(all_labels), np.array(all_cameras)


def score(model, features, batch_size) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=batch_size, shuffle=False)
    model.eval()
    scores = []
    with torch.no_grad():
        for (batch,) in loader:
            scores.append(model.reconstruction_error(batch).numpy())
    return np.concatenate(scores) if scores else np.empty((0,))


def per_camera_table(labels: np.ndarray, scores: np.ndarray, cameras: np.ndarray, threshold: float) -> dict:
    out = {}
    for cam in sorted(set(cameras), key=lambda c: int(c)):
        mask = cameras == cam
        cam_labels, cam_scores = labels[mask], scores[mask]
        n_normal, n_anomaly = int((cam_labels == 0).sum()), int((cam_labels == 1).sum())
        if n_normal == 0 or n_anomaly == 0:
            out[cam] = {"n_normal": n_normal, "n_anomaly": n_anomaly, "roc_auc": None, "note": "only one class present - ROC-AUC undefined"}
            continue
        result = evaluate(cam_labels, cam_scores, threshold)
        out[cam] = {"n_normal": n_normal, "n_anomaly": n_anomaly, "roc_auc": result.roc_auc, "precision": result.precision, "recall": result.recall}
    return out


def main() -> None:
    args = parse_args()
    root = discover_retails_root(DATA_DIR)
    if root is None:
        raise SystemExit(f"Could not find RetailS under {DATA_DIR}.")

    for label, exp in (("2D", args.exp_2d), ("3D", args.exp_3d)):
        model, model_3d, cfg, threshold = load_model_and_threshold(exp, Path(args.checkpoint))
        # Force the full test_realworld subset for the 3D experiment -
        # see gather_windows' docstring for why (class balance/camera
        # coverage, not the config's CPU-feasibility eval cap).
        override = None if (label == "3D" and exp in EXPERIMENTS_3D) else "default"
        features, labels, cameras = gather_windows(exp, cfg, model_3d, root, realworld_max_override=override)
        scores = score(model, features, cfg["batch_size"])
        cams_present = sorted(set(cameras), key=lambda c: int(c))
        print(f"\n=== Best {label}: experiment_{exp} ===")
        print(f"Total windows: {len(labels)}, cameras present: {cams_present}")
        table = per_camera_table(labels, scores, cameras, threshold)
        for cam, row in table.items():
            print(f"  camera {cam}: {row}")

        defined = {c: r["roc_auc"] for c, r in table.items() if r["roc_auc"] is not None}
        if len(defined) >= 2:
            vals = list(defined.values())
            print(f"  ROC-AUC across cameras with both classes present ({list(defined.keys())}): "
                  f"min={min(vals):.3f} max={max(vals):.3f} range={max(vals) - min(vals):.3f} std={np.std(vals):.3f}")
        else:
            print(f"  Only {len(defined)} camera(s) have both classes present - cross-camera variance not computable.")


if __name__ == "__main__":
    main()
