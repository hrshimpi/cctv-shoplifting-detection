"""
Classical, non-deep-learning motion baseline: watches a fixed
Region-of-Interest (ROI) on a shelf/display case and flags frames where
the ROI has visibly changed from a reference, using the Structural
Similarity Index (SSIM) on denoised grayscale crops.

Headless by design: this module has to run inside Google Colab, which
has no display, so it never calls cv2.imshow / cv2.selectROI. ROI
coordinates come in as a plain (x, y, w, h) tuple or a small JSON config
file - see select_roi_interactive.py for the local-only, GUI tool that
produces that config file.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

ROI = tuple[int, int, int, int]


def compute_ssim_score(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """SSIM between two BGR crops, after grayscale conversion and a
    median blur to denoise sensor/compression noise before comparing."""
    gray_a = cv2.medianBlur(cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY), 5)
    gray_b = cv2.medianBlur(cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY), 5)
    return float(ssim(gray_a, gray_b))


def load_roi_config(path: str | Path) -> ROI:
    """Load an ROI (x, y, w, h) from the JSON config select_roi_interactive.py writes."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["x"], cfg["y"], cfg["w"], cfg["h"]


@dataclass
class Alert:
    timestamp: str
    frame_index: int
    similarity_score: float
    roi: ROI


class MotionROIDetector:
    """
    Watches one fixed ROI across a frame sequence and flags frames whose
    SSIM against a reference crop drops below `threshold`.

    reference_mode:
      - "fixed": the ROI crop from the very first frame is the permanent
        reference, for the life of the detector. This is the textbook
        version of the technique - simple, but it drifts: if ambient
        light shifts later in the sequence (a cloud passing outside a
        window, a store dimming its lights toward closing), the whole
        ROI reads as "different" from a reference lit under yesterday's
        conditions, and every subsequent frame false-alarms on nothing
        but a brightness change, not an actual object change.
      - "rolling": the reference is an exponential moving average (EMA)
        of the ROI crop, updated after every frame that does NOT trigger
        an alert. This lets the reference "age" along with slow scene
        drift (lighting, minor camera noise) while still reacting sharply
        to a sudden change (an item removed) - alert frames are
        deliberately excluded from the update, so a real change never
        gets quietly absorbed into the new "normal". This is the
        deliberate upgrade over the naive fixed-reference version.
    """

    def __init__(
        self,
        roi: ROI,
        threshold: float = 0.8,
        reference_mode: str = "fixed",
        ema_alpha: float = 0.05,
        alerts_dir: str | Path | None = "outputs/alerts",
        alerts_log_path: str | Path | None = None,
    ) -> None:
        if reference_mode not in ("fixed", "rolling"):
            raise ValueError(f"reference_mode must be 'fixed' or 'rolling', got {reference_mode!r}")

        self.roi = roi
        self.threshold = threshold
        self.reference_mode = reference_mode
        self.ema_alpha = ema_alpha
        self.alerts_dir = Path(alerts_dir) if alerts_dir else None
        self.alerts_log_path = Path(alerts_log_path) if alerts_log_path else None
        if self.alerts_log_path is not None:
            self.alerts_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.alerts_log_path.write_text("", encoding="utf-8")

        self._reference: np.ndarray | None = None
        self.alerts: list[Alert] = []
        self.similarity_scores: list[float] = []

    @property
    def frame_count(self) -> int:
        return len(self.similarity_scores)

    @property
    def average_similarity(self) -> float:
        if not self.similarity_scores:
            return 0.0
        return sum(self.similarity_scores) / len(self.similarity_scores)

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        x, y, w, h = self.roi
        return frame[y : y + h, x : x + w]

    def process_frame(self, frame: np.ndarray, frame_index: int) -> tuple[float, bool]:
        """Feed one frame in. Returns (similarity_score, flagged)."""
        crop = self._crop(frame)

        if self._reference is None:
            self._reference = crop.astype(np.float32)
            self.similarity_scores.append(1.0)
            return 1.0, False

        score = compute_ssim_score(crop, self._reference.astype(np.uint8))
        flagged = score < self.threshold

        if flagged:
            self._log_alert(frame, frame_index, score)
        elif self.reference_mode == "rolling":
            self._reference = self.ema_alpha * crop.astype(np.float32) + (1 - self.ema_alpha) * self._reference

        self.similarity_scores.append(score)
        return score, flagged

    def _log_alert(self, frame: np.ndarray, frame_index: int, score: float) -> None:
        alert = Alert(
            timestamp=datetime.now(timezone.utc).isoformat(),
            frame_index=frame_index,
            similarity_score=score,
            roi=self.roi,
        )
        self.alerts.append(alert)

        if self.alerts_log_path is not None:
            with open(self.alerts_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(alert)) + "\n")

        if self.alerts_dir is not None:
            self.alerts_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.alerts_dir / f"frame_{frame_index:05d}.jpg"), frame)


def draw_roi_overlay(frame: np.ndarray, roi: ROI, score: float, flagged: bool) -> np.ndarray:
    """Return a copy of `frame` with the ROI box drawn - green while
    quiet, red on an alert - and the SSIM score printed above it. Shared
    by run_on_video() and compare_pipeline.py so both draw it identically."""
    x, y, w, h = roi
    annotated = frame.copy()
    color = (0, 0, 255) if flagged else (0, 200, 0)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        annotated,
        f"SSIM={score:.3f}",
        (x, max(y - 8, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )
    return annotated


def run_on_video(
    video_path: str | Path,
    roi: ROI,
    threshold: float = 0.8,
    reference_mode: str = "fixed",
    output_video_path: str | Path | None = None,
    alerts_dir: str | Path | None = "outputs/alerts",
    alerts_log_path: str | Path | None = None,
) -> MotionROIDetector:
    """
    Process a full video headlessly - no cv2.imshow, no GUI. Writes an
    annotated copy (ROI box drawn green when quiet, red on an alert) via
    cv2.VideoWriter if `output_video_path` is given. Returns the detector
    so callers can read frame_count / alerts / average_similarity.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_video_path is not None:
        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    detector = MotionROIDetector(
        roi=roi,
        threshold=threshold,
        reference_mode=reference_mode,
        alerts_dir=alerts_dir,
        alerts_log_path=alerts_log_path,
    )

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            score, flagged = detector.process_frame(frame, frame_index)

            if writer is not None:
                writer.write(draw_roi_overlay(frame, roi, score, flagged))

            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return detector
