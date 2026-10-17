"""
Draws the annotated output video for scripts/run_demo.py: each tracked
person's skeleton plus a running anomaly-score label, color-coded
against the trained model's own decision threshold.
"""

from __future__ import annotations

import cv2
import numpy as np

# COCO-17 bone connectivity (same convention as scripts/run_eda.py / validate_joint_mapping.py).
COCO17_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

NORMAL_COLOR = (0, 200, 0)  # BGR - green
ANOMALY_COLOR = (0, 0, 255)  # BGR - red
PENDING_COLOR = (200, 200, 200)  # BGR - gray, not enough frames yet for a window


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, color: tuple[int, int, int], conf_threshold: float = 0.3) -> None:
    """keypoints: (17, 3) raw-pixel x, y, confidence for one person in
    one frame. Draws only bones/joints above conf_threshold - in place."""
    pts = keypoints[:, :2].astype(int)
    conf = keypoints[:, 2]
    for a, b in COCO17_BONES:
        if conf[a] >= conf_threshold and conf[b] >= conf_threshold:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 2)
    for i in range(17):
        if conf[i] >= conf_threshold:
            cv2.circle(frame, tuple(pts[i]), 3, color, -1)


def draw_label(frame: np.ndarray, keypoints: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    """Places a text label near a person's head (nose, falling back to
    the topmost visible keypoint if the nose itself isn't confident)."""
    conf = keypoints[:, 2]
    if conf[0] >= 0.3:
        x, y = keypoints[0, :2].astype(int)
    else:
        visible = np.where(conf >= 0.3)[0]
        if len(visible) == 0:
            return
        topmost = visible[np.argmin(keypoints[visible, 1])]
        x, y = keypoints[topmost, :2].astype(int)
    cv2.putText(frame, text, (x - 10, max(y - 15, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def status_color(is_anomalous: bool | None) -> tuple[int, int, int]:
    if is_anomalous is None:
        return PENDING_COLOR
    return ANOMALY_COLOR if is_anomalous else NORMAL_COLOR


def status_text(track_id: str, score: float | None, is_anomalous: bool | None) -> str:
    if score is None:
        return f"id {track_id}: building window..."
    return f"id {track_id}: {'ANOMALY' if is_anomalous else 'normal'} ({score:.3f})"
