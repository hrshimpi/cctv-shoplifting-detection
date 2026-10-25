"""
Raw-video person detection, pose estimation, and tracking for the demo
pipeline (scripts/run_demo.py) - Ultralytics YOLO Pose (pretrained) for
per-frame person detection + 2D pose, its built-in ByteTrack for
cross-frame identity, nothing trained from scratch here.

YOLO Pose's keypoint order (confirmed against
third_party installed via pip: ultralytics/cfg/datasets/coco-pose.yaml)
is the standard COCO-17 order - nose, eyes, ears, shoulders, elbows,
wrists, hips, knees, ankles - identical to RetailS's own convention (see
src/data/retails.py), so no joint reindexing is needed before feeding
YOLO's output into this project's existing preprocessing pipeline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data.retails import PersonTrack, PoseFrame

DEFAULT_YOLO_CHECKPOINT = Path(__file__).resolve().parents[2] / "third_party" / "yolo" / "yolov8n-pose.pt"


def detect_and_track(
    video_path: str | Path,
    checkpoint_path: str | Path = DEFAULT_YOLO_CHECKPOINT,
    conf_threshold: float = 0.3,
    img_size: int = 640,
) -> tuple[dict[int, PersonTrack], float, tuple[int, int]]:
    """Runs YOLO Pose + ByteTrack over every frame of video_path, grouping
    detections by ByteTrack's persistent track id into PersonTrack
    objects (the same dataclass src/data/retails.py uses for RetailS, so
    every downstream function - preprocess_track, build_windows, the
    MotionBERT lift, the feature builders - works unchanged on either
    source).

    Returns (tracks, fps, frame_size): tracks keyed by YOLO's integer
    track id; frame_size is (width, height), needed by
    scripts/run_demo.py to open a matching VideoWriter for the
    annotated output.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(str(video_path))
    opened = cap.isOpened()
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    if not opened or not np.isfinite(fps) or fps <= 0 or min(frame_size) <= 0:
        raise ValueError(f"Cannot read video dimensions/FPS: {video_path}")

    from ultralytics import YOLO  # imported lazily - heavy, optional dependency only the demo needs

    model = YOLO(str(checkpoint_path))
    tracks: dict[int, PersonTrack] = {}

    results_stream = model.track(
        source=str(video_path), persist=True, tracker="bytetrack.yaml", conf=conf_threshold,
        imgsz=img_size, verbose=False, stream=True
    )

    for frame_idx, result in enumerate(results_stream):
        if result.boxes is None or result.boxes.id is None or result.keypoints is None:
            continue  # no tracked detections this frame

        track_ids = result.boxes.id.int().tolist()
        keypoints = result.keypoints.data.cpu().numpy()  # (N, 17, 3) - x, y, confidence

        for track_id, kpts in zip(track_ids, keypoints):
            if track_id not in tracks:
                tracks[track_id] = PersonTrack(
                    person_id=str(track_id),
                    camera_id="demo",
                    video_id=video_path.stem,
                    part="1",
                    source_file=video_path,
                    frames={},
                )
            tracks[track_id].frames[frame_idx] = PoseFrame(frame_id=frame_idx, keypoints=kpts.astype(np.float32), scores=None)

    return tracks, fps, frame_size
