"""
Local-only, interactive helper to pick real ROI coordinates by hand.

Uses cv2.selectROI, which needs a display - this script is never
imported by motion_baseline.py or anything meant to run headlessly in
Colab. Run it once on your own machine against a representative frame or
video, drag a box over the shelf/display case you want to watch, and it
writes the result to a small JSON config that motion_baseline.py's
load_roi_config() reads.

Usage:
    python src/classical_cv/select_roi_interactive.py <image_or_video_path> [output_json_path] [frame_index]
"""

import json
import sys
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _grab_frame(video_path: str | Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        for _ in range(frame_index + 1):
            ok, frame = cap.read()
            if not ok:
                raise IndexError(f"Video has fewer than {frame_index + 1} frames")
    finally:
        cap.release()
    return frame


def select_roi_from_image(image_path: str | Path) -> tuple[int, int, int, int]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return _select_roi_from_frame(frame)


def select_roi_from_video(video_path: str | Path, frame_index: int = 0) -> tuple[int, int, int, int]:
    frame = _grab_frame(video_path, frame_index)
    return _select_roi_from_frame(frame)


def _select_roi_from_frame(frame) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.selectROI("Select ROI - drag a box, press ENTER/SPACE to confirm, C to cancel", frame)
    cv2.destroyAllWindows()
    return int(x), int(y), int(w), int(h)


def save_roi_config(roi: tuple[int, int, int, int], out_path: str | Path) -> None:
    x, y, w, h = roi
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"x": x, "y": y, "w": w, "h": h}, f, indent=2)
    print(f"Saved ROI {roi} to {out_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    source_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/roi_config.json")
    frame_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    if source_path.suffix.lower() in VIDEO_EXTS:
        roi = select_roi_from_video(source_path, frame_index)
    else:
        roi = select_roi_from_image(source_path)

    save_roi_config(roi, out_path)


if __name__ == "__main__":
    main()
