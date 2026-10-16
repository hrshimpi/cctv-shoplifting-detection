#!/usr/bin/env python
"""
Visually validate the COCO-17 -> H36M-17 joint mapping (src/utils/
joint_mapping.py) and, if a MotionBERT checkpoint is available, the full
2D -> 3D lift (src/motionbert/lift.py). Standalone, runs locally - no
notebook required.

Usage:
    python scripts/validate_joint_mapping.py
    python scripts/validate_joint_mapping.py --num-sequences 5
    python scripts/validate_joint_mapping.py --checkpoint path/to/best_epoch.bin

For each of a handful of real RetailS tracks, picks the highest-mean-
confidence window (same "pick the cleanest representative example"
approach as scripts/run_eda.py's skeleton plot - preprocess_track()/
build_windows() behave identically regardless of which window gets
picked for the picture) and renders, side by side:
  1. The original COCO-17 2D skeleton.
  2. The same frame reindexed into H36M-17 order (src/utils/
     joint_mapping.py), confirming the remapping puts shoulders, elbows,
     wrists, hips, knees, and ankles in the right place before anything
     is handed to MotionBERT.
  3. The full MotionBERT 2D->3D lift of the whole window, IF a
     checkpoint is found (see README.md "MotionBERT setup") - otherwise
     this panel is skipped with an explicit message, not silently
     dropped.

Saves one PNG per sequence under outputs/plots/joint_mapping_validation/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401

from src.data.preprocessing import build_windows, preprocess_track  # noqa: E402
from src.data.retails import (  # noqa: E402
    SUBSET_DIRS,
    discover_retails_root,
    find_pose_files,
    load_pose_file,
)
from src.utils.joint_mapping import COCO17_NAMES, H36M17_BONES, H36M17_NAMES, coco17_to_h36m17  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "retails"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "preprocessing.yaml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "third_party" / "MotionBERT" / "checkpoint" / "pose3d" / "MB_ft_h36m" / "best_epoch.bin"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots" / "joint_mapping_validation"

# COCO-17 skeleton bone connections, by keypoint index (same as scripts/run_eda.py).
COCO17_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Short labels for the joints the user specifically wants to visually
# confirm (shoulders, elbows, wrists, hips, knees, ankles) - other joints
# (face/head/spine) are left unlabeled to keep the plot legible.
_LABELED = {"shoulder", "elbow", "wrist", "hip", "knee", "ankle"}


def _abbrev(name: str) -> str:
    side = "L" if name.startswith("left_") else "R" if name.startswith("right_") else ""
    part = name.split("_", 1)[-1]
    return f"{side}.{part}" if side else part


COCO17_LABELS = {i: _abbrev(n) for i, n in enumerate(COCO17_NAMES) if any(k in n for k in _LABELED)}
H36M17_LABELS = {i: _abbrev(n) for i, n in enumerate(H36M17_NAMES) if any(k in n for k in _LABELED)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--num-sequences", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=40, help="Tracks scanned per sequence to pick the cleanest window from")
    return parser.parse_args()


def find_best_window(pose_files, cfg: dict, num_candidates: int, skip: int):
    """Scan pose files (starting after `skip` already-used ones) for the
    highest-mean-confidence window among up to num_candidates tracks.
    Returns (raw_keypoints_for_window (T,17,3), track) or None."""
    candidates = []
    seen = 0
    for pose_path in pose_files:
        for track in load_pose_file(pose_path):
            if len(track.frames) < cfg["window_length"]:
                continue
            seen += 1
            if seen <= skip:
                continue

            processed = preprocess_track(
                track,
                confidence_threshold=cfg["confidence_threshold"],
                max_interpolation_gap=cfg["max_interpolation_gap"],
                smoothing_window=cfg["smoothing_window"],
                apply_normalization=False,
            )
            windows = build_windows(
                processed["frame_ids"], processed["keypoints"], processed["visibility_mask"],
                window_length=cfg["window_length"], stride=cfg["stride"],
            )
            if not windows:
                continue
            w = windows[0]
            mean_conf = w["keypoints"][:, :, 2].mean()
            candidates.append((mean_conf, w["keypoints"], track))
            if len(candidates) >= num_candidates:
                break
        if len(candidates) >= num_candidates:
            break

    if not candidates:
        return None, seen
    best_conf, best_kp, best_track = max(candidates, key=lambda c: c[0])
    return (best_kp, best_track), seen


def plot_2d_skeleton(ax, keypoints: np.ndarray, bones, labels: dict, title: str) -> None:
    x, y = keypoints[:, 0], -keypoints[:, 1]
    ax.scatter(x, y, s=25, c="tab:blue")
    for a, b in bones:
        ax.plot([x[a], x[b]], [y[a], y[b]], c="tab:blue", linewidth=1.5)
    for idx, label in labels.items():
        ax.annotate(label, (x[idx], y[idx]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")


def plot_3d_skeleton(ax, keypoints_xyz: np.ndarray, bones, labels: dict, title: str) -> None:
    x, y, z = keypoints_xyz[:, 0], keypoints_xyz[:, 1], keypoints_xyz[:, 2]
    px, py, pz = x, z, -y  # (x, depth, -height) so the figure stands upright
    ax.scatter(px, py, pz, s=25, c="tab:red")
    for a, b in bones:
        ax.plot([px[a], px[b]], [py[a], py[b]], [pz[a], pz[b]], c="tab:red", linewidth=1.5)
    for idx, label in labels.items():
        ax.text(px[idx], py[idx], pz[idx], label, fontsize=7)
    ax.set_title(title, fontsize=10)

    # matplotlib's default 3D axes auto-scale each axis independently to
    # fill the same box, which (given MotionBERT's depth/z output is
    # consistently a much smaller range than x/y here - a real property,
    # not a bug, see README.md "Limitations") would stretch the depth
    # axis far more than x/y and visually exaggerate 3D structure that
    # isn't really there. Force equal, undistorted scaling instead so the
    # rendered proportions reflect the actual geometry.
    all_pts = np.stack([px, py, pz])
    centers = all_pts.mean(axis=1)
    half_range = max((all_pts.max(axis=1) - all_pts.min(axis=1)).max() / 2.0, 1e-6)
    ax.set_xlim(centers[0] - half_range, centers[0] + half_range)
    ax.set_ylim(centers[1] - half_range, centers[1] + half_range)
    ax.set_zlim(centers[2] - half_range, centers[2] + half_range)
    ax.set_box_aspect((1, 1, 1))


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = discover_retails_root(DATA_DIR)
    if root is None:
        raise SystemExit(f"Could not find RetailS under {DATA_DIR}. Download/extract it first (see README.md).")
    pose_files = find_pose_files(root / SUBSET_DIRS["train"])
    if not pose_files:
        raise SystemExit(f"No pose JSON files found under {root / SUBSET_DIRS['train']}/pose/.")

    checkpoint_path = Path(args.checkpoint)
    model = None
    if checkpoint_path.exists():
        from src.motionbert.lift import lift_2d_to_3d, load_motionbert

        print(f"Found MotionBERT checkpoint at {checkpoint_path} - 3D lift panel will be included.")
        model = load_motionbert(checkpoint_path)
    else:
        print(
            f"No MotionBERT checkpoint at {checkpoint_path} - skipping the 3D lift panel.\n"
            "See README.md 'MotionBERT setup' for how to download it (manual, OneDrive - "
            "not scriptable). Rendering the COCO-17 / H36M-17 2D reindexing panels only.\n"
        )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    skip = 0
    produced = 0
    for seq_idx in range(args.num_sequences):
        result, scanned = find_best_window(pose_files, cfg, args.num_candidates, skip)
        skip += scanned
        if result is None:
            print(f"Sequence {seq_idx}: ran out of tracks long enough for one window - stopping.")
            break
        raw_window, track = result
        h36m_window = coco17_to_h36m17(raw_window)

        # Pick the single highest-confidence frame within the window for
        # a clean, legible 2D/3D still (same rationale as run_eda.py's
        # skeleton plot: a presentation choice, not a pipeline change).
        best_frame_idx = int(raw_window[:, :, 2].mean(axis=1).argmax())

        ncols = 3 if model is not None else 2
        fig = plt.figure(figsize=(5 * ncols, 5))
        ax1 = fig.add_subplot(1, ncols, 1)
        plot_2d_skeleton(ax1, raw_window[best_frame_idx], COCO17_BONES, COCO17_LABELS, f"COCO-17 (person {track.person_id}, cam {track.camera_id})")
        ax2 = fig.add_subplot(1, ncols, 2)
        plot_2d_skeleton(ax2, h36m_window[best_frame_idx], H36M17_BONES, H36M17_LABELS, "H36M-17 (reindexed)")

        if model is not None:
            lifted = lift_2d_to_3d(model, raw_window)
            ax3 = fig.add_subplot(1, ncols, 3, projection="3d")
            plot_3d_skeleton(ax3, lifted[best_frame_idx], H36M17_BONES, H36M17_LABELS, "MotionBERT 3D lift")

        plt.tight_layout()
        out_path = PLOTS_DIR / f"sequence_{seq_idx}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Sequence {seq_idx}: saved {out_path} (person {track.person_id}, camera {track.camera_id}, video {track.video_id})")
        produced += 1

    print(f"\n{produced} sequence render(s) saved under {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
