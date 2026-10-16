#!/usr/bin/env python
"""
Short EDA over the real RetailS dataset. Runs locally, no notebook
required.

Usage:
    python scripts/run_eda.py
    python scripts/run_eda.py --config configs/preprocessing.yaml

Reports, all computed from the actual downloaded/extracted files (never
assumed):
  - the real schema found in a sample pose file and a sample label file,
    and how it differs from the course spec's summary (see
    src/data/retails.py's docstring for the full writeup)
  - person tracks and (person, frame) pose entries per camera, per subset
  - normal vs. anomaly frame balance in the two labeled test subsets
  - a before/after normalization skeleton plot for two sample windows,
    saved to outputs/plots/

Train has no anomaly labels at all (RetailS_train ships no gt/
directory) - that's real, not a bug in this script.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data.preprocessing import build_windows, preprocess_track  # noqa: E402
from src.data.retails import (  # noqa: E402
    SUBSET_DIRS,
    discover_retails_root,
    find_pose_files,
    gt_dir_for_subset,
    label_path_for_pose_file,
    load_anomaly_labels,
    load_pose_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "retails"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "preprocessing.yaml"

# COCO-17 skeleton bone connections, by keypoint index, for the plot.
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def print_real_schema_sample(root: Path) -> None:
    train_files = find_pose_files(root / SUBSET_DIRS["train"])
    staged_dir = root / SUBSET_DIRS["test_staged"]
    staged_files = find_pose_files(staged_dir)
    gt_dir = gt_dir_for_subset(staged_dir)

    print("=" * 70)
    print("REAL SCHEMA CHECK (against the actual downloaded files)")
    print("=" * 70)

    if train_files:
        with open(train_files[0], encoding="utf-8") as f:
            sample = json.load(f)
        pid = next(iter(sample))
        fid = next(iter(sample[pid]))
        frame_record = sample[pid][fid]
        print(f"\nSample pose file: {train_files[0].name}")
        print(f"  top-level keys (person_ids), first few: {list(sample.keys())[:5]}")
        print(f"  person '{pid}' frame_ids, first few: {list(sample[pid].keys())[:5]}")
        print(f"  frame record keys: {list(frame_record.keys())}")
        print(f"  keypoints length: {len(frame_record['keypoints'])} (expect 51 = 17 x 3)")
        print(f"  'scores' value: {frame_record['scores']!r}")

    if staged_files and gt_dir:
        label_path = label_path_for_pose_file(staged_files[0], gt_dir)
        if label_path:
            labels = load_anomaly_labels(label_path)
            print(f"\nSample label file: {label_path.name}")
            print(f"  shape: {labels.shape}, dtype: {labels.dtype}, unique values: {np.unique(labels)}")

    print("\nConfirmed differences from the course spec summary:")
    print("  - No `bounding_box` field anywhere in the pose JSON.")
    print("  - No `camera_id` field inside the JSON - it's the first segment of the filename.")
    print("  - Anomaly label is a separate per-video .npy array (frame-indexed), not an inline field.")
    print("  - Train filenames have a trailing '_<part>' segment; test filenames don't (2 vs 3 underscore-separated segments).")
    print("  - RetailS_train has no gt/ directory at all - 0 anomaly labels in the train subset by design.")


def summarize_subset(root: Path, subset_key: str) -> None:
    subset_dir = root / SUBSET_DIRS[subset_key]
    pose_files = find_pose_files(subset_dir)
    gt_dir = gt_dir_for_subset(subset_dir)

    tracks_per_camera: Counter = Counter()
    pose_entries_per_camera: Counter = Counter()
    videos_per_camera: defaultdict[str, set] = defaultdict(set)
    normal_frames = 0
    anomaly_frames = 0

    for pose_path in tqdm(pose_files, desc=f"Scanning {subset_key}"):
        tracks = load_pose_file(pose_path)
        for track in tracks:
            tracks_per_camera[track.camera_id] += 1
            pose_entries_per_camera[track.camera_id] += len(track.frames)
            videos_per_camera[track.camera_id].add((track.video_id, track.part))

        if gt_dir is not None:
            label_path = label_path_for_pose_file(pose_path, gt_dir)
            labels = load_anomaly_labels(label_path)
            if labels is not None:
                normal_frames += int((labels == 0).sum())
                anomaly_frames += int((labels == 1).sum())

    print(f"\n--- {subset_key} ({subset_dir.name}) ---")
    print(f"Pose files (videos/parts): {len(pose_files)}")
    print(f"{'camera_id':>10} {'videos':>8} {'tracks':>8} {'pose entries (person-frames)':>30}")
    for cam in sorted(tracks_per_camera, key=int):
        print(
            f"{cam:>10} {len(videos_per_camera[cam]):>8} {tracks_per_camera[cam]:>8} "
            f"{pose_entries_per_camera[cam]:>30}"
        )
    print(f"{'TOTAL':>10} {sum(len(v) for v in videos_per_camera.values()):>8} "
          f"{sum(tracks_per_camera.values()):>8} {sum(pose_entries_per_camera.values()):>30}")

    if gt_dir is not None:
        total = normal_frames + anomaly_frames
        pct = (anomaly_frames / total * 100) if total else 0.0
        print(f"\nLabel balance: {normal_frames} normal / {anomaly_frames} anomaly frames "
              f"({pct:.2f}% anomalous, {total} total labeled frames)")
    else:
        print("\nNo gt/ directory for this subset - no anomaly labels (expected for train).")


def plot_before_after_normalization(root: Path, cfg: dict, num_candidates: int = 40) -> None:
    """Picks the cleanest-looking example frames it can for the
    illustration: within each candidate window, uses the single frame
    with the highest mean keypoint confidence, then keeps the overall
    best 2 such frames across several candidate tracks. This is a
    presentation choice (pick a representative, legible example), not a
    change to the pipeline itself - preprocess_track()/build_windows()
    run identically either way, confidence-threshold and all."""
    subset_dir = root / SUBSET_DIRS["train"]
    pose_files = find_pose_files(subset_dir)

    candidates = []
    for pose_path in pose_files:
        for track in load_pose_file(pose_path):
            if len(track.frames) < cfg["window_length"]:
                continue
            raw_frame_ids, raw_keypoints = _track_raw_arrays(track)
            processed = preprocess_track(
                track,
                confidence_threshold=cfg["confidence_threshold"],
                max_interpolation_gap=cfg["max_interpolation_gap"],
                smoothing_window=cfg["smoothing_window"],
            )
            windows = build_windows(
                processed["frame_ids"], processed["keypoints"], processed["visibility_mask"],
                window_length=cfg["window_length"], stride=cfg["stride"],
            )
            if not windows:
                continue
            w = windows[0]
            mean_conf_per_frame = raw_keypoints[:, :, 2].mean(axis=1)
            frame_id_to_raw_idx = {fid: i for i, fid in enumerate(raw_frame_ids)}
            window_frame_ids = range(w["start_frame"], w["end_frame"] + 1)
            best_offset, best_conf = max(
                (
                    (offset, mean_conf_per_frame[frame_id_to_raw_idx[fid]])
                    for offset, fid in enumerate(window_frame_ids)
                    if fid in frame_id_to_raw_idx
                ),
                key=lambda t: t[1],
            )
            raw_idx = frame_id_to_raw_idx[w["start_frame"] + best_offset]
            candidates.append((best_conf, raw_keypoints[raw_idx], w["keypoints"][best_offset], track))
        if len(candidates) >= num_candidates:
            break

    windows_to_plot = [c[1:] for c in sorted(candidates, key=lambda c: -c[0])[:2]]

    if not windows_to_plot:
        print("\nNo track long enough to plot a sample window - skipping skeleton plot.")
        return

    fig, axes = plt.subplots(2, len(windows_to_plot[:2]), figsize=(5 * len(windows_to_plot[:2]), 10))
    if len(windows_to_plot[:2]) == 1:
        axes = axes.reshape(2, 1)

    for col, (raw_kp, norm_kp, track) in enumerate(windows_to_plot[:2]):
        _plot_skeleton(axes[0, col], raw_kp, f"Before normalization\nperson {track.person_id}, cam {track.camera_id}")
        _plot_skeleton(axes[1, col], norm_kp, "After centering + scaling")

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "skeleton_before_after_normalization.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSkeleton before/after plot saved to {out_path}")


def _track_raw_arrays(track):
    from src.data.preprocessing import track_to_array
    return track_to_array(track)


def _plot_skeleton(ax, keypoints: np.ndarray, title: str) -> None:
    x, y = keypoints[:, 0], -keypoints[:, 1]  # flip y for a natural upright view
    ax.scatter(x, y, s=20, c="tab:blue")
    for a, b in SKELETON_EDGES:
        ax.plot([x[a], x[b]], [y[a], y[b]], c="tab:blue", linewidth=1.5)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = discover_retails_root(DATA_DIR)
    if root is None:
        raise SystemExit(
            f"Could not find a {SUBSET_DIRS['train']}/ directory under {DATA_DIR}. "
            "Download and extract the RetailS archive first (see README.md)."
        )

    print_real_schema_sample(root)

    for subset_key in ("train", "test_realworld", "test_staged"):
        summarize_subset(root, subset_key)

    plot_before_after_normalization(root, cfg)


if __name__ == "__main__":
    main()
