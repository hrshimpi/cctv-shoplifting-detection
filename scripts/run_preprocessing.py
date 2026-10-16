#!/usr/bin/env python
"""
Standalone CLI entry point for the pose preprocessing pipeline. Runs
locally, no notebook required.

Usage:
    python scripts/run_preprocessing.py
    python scripts/run_preprocessing.py --config configs/preprocessing.yaml --subset train --limit 5

Runs src/data/preprocessing.py's per-track pipeline over every person
track found in the given RetailS subset, builds temporal windows, and
reports how many tracks/windows were produced - an end-to-end smoke test
on real data, not just a library import.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.data.preprocessing import build_windows, preprocess_track  # noqa: E402
from src.data.retails import (  # noqa: E402
    SUBSET_DIRS,
    discover_retails_root,
    find_pose_files,
    load_pose_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "retails"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "preprocessing.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--subset", default="train", choices=sorted(SUBSET_DIRS))
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N pose files (quick smoke test)"
    )
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

    subset_dir = root / SUBSET_DIRS[args.subset]
    pose_files = find_pose_files(subset_dir)
    if not pose_files:
        raise SystemExit(f"No pose JSON files found under {subset_dir}/pose/.")
    if args.limit:
        pose_files = pose_files[: args.limit]

    print(f"Processing {len(pose_files)} pose file(s) from {subset_dir}")
    print(f"Config: {cfg}")

    total_tracks = 0
    total_windows = 0
    skipped_too_short = 0

    for pose_path in pose_files:
        for track in load_pose_file(pose_path):
            total_tracks += 1
            processed = preprocess_track(
                track,
                confidence_threshold=cfg["confidence_threshold"],
                max_interpolation_gap=cfg["max_interpolation_gap"],
                smoothing_window=cfg["smoothing_window"],
            )
            windows = build_windows(
                processed["frame_ids"],
                processed["keypoints"],
                processed["visibility_mask"],
                window_length=cfg["window_length"],
                stride=cfg["stride"],
            )
            if not windows and len(processed["frame_ids"]) > 0:
                skipped_too_short += 1
            total_windows += len(windows)

    print(f"\nPerson tracks processed: {total_tracks}")
    print(f"Tracks too short for even one window (< {cfg['window_length']} contiguous frames): {skipped_too_short}")
    print(f"Temporal windows built: {total_windows}")


if __name__ == "__main__":
    main()
