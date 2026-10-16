#!/usr/bin/env python
"""
False-positive analysis (step 4, requirement 4): for a trained
experiment, find every window the model flags anomalous but RetailS
labels normal, trace each one back to its real track/frames, compute
simple descriptive pose statistics, and render it so what's actually
happening can be inspected - not just counted. Standalone, runs
locally - no notebook required.

Usage:
    python scripts/analyze_false_positives.py --experiment a

RetailS has no per-window action label (only binary normal/anomaly), so
there is no ground truth to check false positives against for the
scenarios the course spec flags as risky (adjusting clothing, reaching
into a personal bag, phone into pocket, crouching, holding an item near
the torso, normal product comparison). This script does not fabricate
one - it reports simple, transparent geometric signals (hand-to-hip and
hand-to-wrist proximity, hip-height range as a crouch proxy) and a
rendered skeleton strip for every false positive found, so a real
judgment call about which scenario (if any) a case resembles can be
made by looking at it, with the reasoning spelled out rather than
asserted.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.data.preprocessing import build_windows, preprocess_track  # noqa: E402
from src.data.retails import SUBSET_DIRS, discover_retails_root, find_pose_files, load_pose_file  # noqa: E402
from src.evaluation.metrics import choose_threshold  # noqa: E402
from src.experiments.dataset import build_windows_for_subset  # noqa: E402
from src.features.motion_features import feature_dim  # noqa: E402
from src.models.bigru_autoencoder import BiGRUAutoencoder  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "retails"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots" / "false_positive_analysis"

LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

COCO17_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="a", help="2D experiment to analyze (a or b; 3D experiments evaluate on a small sample - check for zero false positives first)")
    return parser.parse_args()


def find_track_window(root: Path, subset_key: str, meta, preprocessing_cfg: dict):
    """Re-derive the exact normalized window a WindowMeta refers to, by
    reloading its source pose file and re-running the same
    preprocess_track/build_windows pipeline used to build it."""
    subset_dir = root / SUBSET_DIRS[subset_key]
    pose_path = next(p for p in find_pose_files(subset_dir) if p.name == meta.pose_file)
    for track in load_pose_file(pose_path):
        if track.person_id != meta.person_id:
            continue
        processed = preprocess_track(
            track,
            confidence_threshold=preprocessing_cfg["confidence_threshold"],
            max_interpolation_gap=preprocessing_cfg["max_interpolation_gap"],
            smoothing_window=preprocessing_cfg["smoothing_window"],
        )
        windows = build_windows(
            processed["frame_ids"], processed["keypoints"], processed["visibility_mask"],
            window_length=preprocessing_cfg["window_length"], stride=preprocessing_cfg["stride"],
        )
        for w in windows:
            if w["start_frame"] == meta.start_frame and w["end_frame"] == meta.end_frame:
                return w["keypoints"]
    raise RuntimeError(f"Could not re-derive window for {meta}")


def _angle_at_vertex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at vertex b formed by points a-b-c, in radians. a, b, c: (T, 2)."""
    v1, v2 = a - b, c - b
    denom = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8
    return np.arccos(np.clip(np.sum(v1 * v2, axis=-1) / denom, -1.0, 1.0))


def describe_window(keypoints: np.ndarray) -> dict:
    """Simple, transparent descriptive stats from a normalized
    (pelvis-centered, torso-scaled) window - all in torso-length units
    (distances) or radians (angles), so comparable across people/cameras.
    Not a classifier - just surfaced numbers to reason from alongside
    the render.

    Note: src/data/preprocessing.py's normalize_position/normalize_scale
    re-center and re-scale EVERY FRAME independently on its own pelvis/
    torso length, which erases any whole-body height or size change
    across a window by construction - a "hip height range" computed
    this way is trivially always ~0 and was dropped as a crouch proxy
    for exactly that reason (caught by actually computing it and getting
    0.000 for every window, not assumed). Knee angle survives this
    normalization (invariant to translation/uniform scale) and is used
    instead - a smaller min angle means a more bent knee at some point
    in the window, consistent with crouching/bending down.
    """
    xy = keypoints[:, :, :2]
    lw, rw = xy[:, LEFT_WRIST], xy[:, RIGHT_WRIST]
    lh, rh = xy[:, LEFT_HIP], xy[:, RIGHT_HIP]
    lk, rk = xy[:, 13], xy[:, 14]
    la, ra = xy[:, 15], xy[:, 16]
    ls, rs = xy[:, LEFT_SHOULDER], xy[:, RIGHT_SHOULDER]
    torso_mid = (ls + rs + lh + rh) / 4.0

    wrist_wrist_dist = np.linalg.norm(lw - rw, axis=1)
    lw_hip_dist = np.linalg.norm(lw - lh, axis=1)
    rw_hip_dist = np.linalg.norm(rw - rh, axis=1)
    lw_torso_dist = np.linalg.norm(lw - torso_mid, axis=1)
    rw_torso_dist = np.linalg.norm(rw - torso_mid, axis=1)
    left_knee_angle = np.degrees(_angle_at_vertex(lh, lk, la))
    right_knee_angle = np.degrees(_angle_at_vertex(rh, rk, ra))

    return {
        "min_wrist_wrist_dist": float(wrist_wrist_dist.min()),
        "min_lw_hip_dist": float(lw_hip_dist.min()),
        "min_rw_hip_dist": float(rw_hip_dist.min()),
        "min_lw_torso_dist": float(lw_torso_dist.min()),
        "min_rw_torso_dist": float(rw_torso_dist.min()),
        "min_left_knee_angle_deg": float(left_knee_angle.min()),
        "min_right_knee_angle_deg": float(right_knee_angle.min()),
        "knee_angle_range_deg": float(
            max(left_knee_angle.max() - left_knee_angle.min(), right_knee_angle.max() - right_knee_angle.min())
        ),
    }


def plot_fp_window(keypoints: np.ndarray, meta, idx: int, score: float, threshold: float) -> Path:
    T = keypoints.shape[0]
    frame_idxs = sorted(set([0, T // 3, 2 * T // 3, T - 1]))
    fig, axes = plt.subplots(1, len(frame_idxs), figsize=(4 * len(frame_idxs), 4))
    if len(frame_idxs) == 1:
        axes = [axes]
    for ax, fi in zip(axes, frame_idxs):
        x, y = keypoints[fi, :, 0], -keypoints[fi, :, 1]
        ax.scatter(x, y, s=20, c="tab:red")
        for a, b in COCO17_BONES:
            ax.plot([x[a], x[b]], [y[a], y[b]], c="tab:red", linewidth=1.5)
        ax.set_title(f"frame offset {fi}/{T - 1}", fontsize=9)
        ax.set_aspect("equal")
    fig.suptitle(
        f"FP {idx}: {meta.pose_file} person {meta.person_id} cam {meta.camera_id} "
        f"frames {meta.start_frame}-{meta.end_frame}  (score={score:.4f}, threshold={threshold:.4f})",
        fontsize=10,
    )
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / f"fp_{idx}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(OUTPUTS_DIR / "checkpoints" / f"experiment_{args.experiment}" / "model.pt", weights_only=False)
    cfg = checkpoint["config"]
    model = BiGRUAutoencoder(
        input_dim=feature_dim(cfg["use_velocity"], cfg["use_acceleration"]),
        hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"], dropout=cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_errors = np.load(OUTPUTS_DIR / "checkpoints" / f"experiment_{args.experiment}" / "train_reconstruction_errors.npy")
    threshold = choose_threshold(train_errors, percentile=cfg["threshold_percentile"])
    print(f"experiment_{args.experiment}, threshold={threshold:.6f}")

    root = discover_retails_root(DATA_DIR)
    if root is None:
        raise SystemExit(f"Could not find RetailS under {DATA_DIR}.")

    fp_idx = 0
    for subset_key in ("test_realworld", "test_staged"):
        features, labels, meta = build_windows_for_subset(
            root, subset_key, cfg, use_velocity=cfg["use_velocity"], use_acceleration=cfg["use_acceleration"],
            max_pose_files=None, seed=cfg["seed"], return_metadata=True,
        )
        if len(features) == 0:
            continue
        loader = DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=cfg["batch_size"], shuffle=False)
        scores = []
        with torch.no_grad():
            for (batch,) in loader:
                scores.append(model.reconstruction_error(batch).numpy())
        scores = np.concatenate(scores)
        preds = (scores >= threshold).astype(int)
        fp_positions = np.where((labels == 0) & (preds == 1))[0]

        print(f"\n{subset_key}: {len(fp_positions)} false positive(s) out of {(labels == 0).sum()} normal windows")
        for pos in fp_positions:
            m = meta[pos]
            window_keypoints = find_track_window(root, subset_key, m, cfg)
            stats = describe_window(window_keypoints)
            render_path = plot_fp_window(window_keypoints, m, fp_idx, scores[pos], threshold)
            print(f"\n  FP {fp_idx}: {m.pose_file} person {m.person_id} camera {m.camera_id} frames {m.start_frame}-{m.end_frame}")
            print(f"    score={scores[pos]:.4f} (threshold {threshold:.4f})")
            for k, v in stats.items():
                print(f"    {k}: {v:.3f}")
            print(f"    render: {render_path}")
            fp_idx += 1


if __name__ == "__main__":
    main()
