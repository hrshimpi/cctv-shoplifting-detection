#!/usr/bin/env python
"""
Extract a sample of frames per class from the external video package
(UCF-Crime shoplifting/normal clips), for external validation of the
RetailS-trained pipeline. Adapted from the original
Shoplifting_Video_Project_Package/scripts/extract_frames.py into this
project's merged data/external_video/ layout - sampling logic unchanged,
frame count is now a CLI argument instead of a hardcoded 100, and both
video class folders are accepted even while empty (real videos get added
later - this prints a clear message instead of failing).

Usage:
    python scripts/extract_frames.py
    python scripts/extract_frames.py --frames-per-class 150
"""

import argparse
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIRS = {
    "shoplifting": PROJECT_ROOT / "data" / "external_video" / "videos" / "shoplifting",
    "normal": PROJECT_ROOT / "data" / "external_video" / "videos" / "normal",
}
FRAME_DIRS = {
    "shoplifting": PROJECT_ROOT / "data" / "external_video" / "frames" / "shoplifting",
    "normal": PROJECT_ROOT / "data" / "external_video" / "frames" / "normal",
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def collect_videos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def count_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 0)


def extract_class(label: str, target_per_class: int) -> None:
    videos = collect_videos(VIDEO_DIRS[label])
    out_dir = FRAME_DIRS[label]
    out_dir.mkdir(parents=True, exist_ok=True)

    if not videos:
        print(f"[{label}] No videos found yet in {VIDEO_DIRS[label]} - skipping (add videos there and re-run).")
        return

    counts = [(v, count_frames(v)) for v in videos]
    total = sum(n for _, n in counts)

    if total == 0:
        print(f"[{label}] Videos were found but frame counts could not be read.")
        return

    # Sample approximately uniformly across all available frames.
    n_targets = min(target_per_class, total)
    global_targets = [round(i * (total - 1) / max(n_targets - 1, 1)) for i in range(n_targets)]

    offsets = []
    running = 0
    for v, n in counts:
        offsets.append((running, running + n, v))
        running += n

    saved = 0
    for target in global_targets:
        for start, end, video in offsets:
            if start <= target < end:
                local_idx = target - start
                cap = cv2.VideoCapture(str(video))
                cap.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
                ok, frame = cap.read()
                cap.release()
                if ok:
                    out = out_dir / f"{label}_{saved + 1:03d}.jpg"
                    cv2.imwrite(str(out), frame)
                    saved += 1
                break

    print(f"[{label}] Extracted {saved} frames into {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-per-class", type=int, default=100, help="Target number of frames to sample per class (default: 100)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for label in ("shoplifting", "normal"):
        extract_class(label, args.frames_per_class)


if __name__ == "__main__":
    main()
