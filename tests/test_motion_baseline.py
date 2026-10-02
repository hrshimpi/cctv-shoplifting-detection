"""Unit tests for src/classical_cv/motion_baseline.py.

Uses small synthetic BGR frames instead of real video - the point is to
verify the SSIM/ROI detection logic itself (reference handling, fixed vs
rolling behavior, alert thresholding), not to re-validate OpenCV/skimage.
"""

import json

import numpy as np
import pytest

from src.classical_cv.motion_baseline import (
    MotionROIDetector,
    compute_ssim_score,
    load_roi_config,
)

ROI = (0, 0, 20, 20)


def _solid_frame(value: int, size: int = 40) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_compute_ssim_score_identical_frames_is_one():
    frame = _solid_frame(100)
    assert compute_ssim_score(frame, frame) == 1.0


def test_compute_ssim_score_different_frames_is_lower():
    black = _solid_frame(0)
    white = _solid_frame(255)
    assert compute_ssim_score(black, white) < 1.0


def test_first_frame_is_never_flagged_and_becomes_the_reference():
    detector = MotionROIDetector(roi=ROI, alerts_dir=None)
    score, flagged = detector.process_frame(_solid_frame(50), frame_index=0)

    assert score == 1.0
    assert flagged is False
    assert detector.frame_count == 1


def test_fixed_reference_flags_a_clear_change():
    detector = MotionROIDetector(roi=ROI, threshold=0.8, reference_mode="fixed", alerts_dir=None)
    detector.process_frame(_solid_frame(0), frame_index=0)  # reference frame

    score, flagged = detector.process_frame(_solid_frame(255), frame_index=1)

    assert flagged is True
    assert len(detector.alerts) == 1
    assert detector.alerts[0].frame_index == 1


def test_fixed_reference_stays_quiet_on_unchanged_frames():
    detector = MotionROIDetector(roi=ROI, threshold=0.8, reference_mode="fixed", alerts_dir=None)
    for i in range(5):
        _, flagged = detector.process_frame(_solid_frame(100), frame_index=i)

    assert flagged is False
    assert detector.alerts == []
    assert detector.average_similarity == 1.0


def test_rolling_reference_does_not_alert_on_slow_drift():
    """A rolling reference should tolerate a sequence of small, gradual
    changes without alerting on every single step - that's the whole
    point of the EMA update over the naive fixed reference."""
    detector = MotionROIDetector(roi=ROI, threshold=0.5, reference_mode="rolling", ema_alpha=0.5, alerts_dir=None)
    for value in (100, 105, 110, 115, 120):
        detector.process_frame(_solid_frame(value), frame_index=value)

    assert detector.alerts == []


def test_invalid_reference_mode_raises():
    with pytest.raises(ValueError):
        MotionROIDetector(roi=ROI, reference_mode="not-a-real-mode")


def test_load_roi_config_reads_xywh(tmp_path):
    config_path = tmp_path / "roi.json"
    config_path.write_text(json.dumps({"x": 10, "y": 20, "w": 30, "h": 40}), encoding="utf-8")

    assert load_roi_config(config_path) == (10, 20, 30, 40)
