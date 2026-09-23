"""
Run both detectors from this project - the step 2 classical SSIM/ROI
baseline and the step 3 fine-tuned YOLOv8 model - on the *same* sample
video, frame by frame, and write one combined side-by-side annotated
video: classical on the left, YOLO on the right.

Also prints a short table of how many frames each one flagged, so the
two approaches can be compared directly rather than just described.

Run after both `python src/classical_cv/motion_baseline.py`-based setup
(step 2) and `python src/detection/train_yolo.py` (step 3) exist.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO  # noqa: E402

from src.classical_cv.motion_baseline import MotionROIDetector, draw_roi_overlay  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "CCTV_Shoplifting_Dataset" / "videos"
RUNS_DIR = PROJECT_ROOT / "outputs" / "yolo_runs"
DEFAULT_WEIGHTS = RUNS_DIR / "train" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compare_pipeline"

# Same ROI run_baseline_demo.py uses for this scene (see README) - the
# classical detector needs a hand-picked ROI regardless of which video
# it's pointed at.
ELECTRONICS_TABLE_ROI = (175, 343, 75, 65)


def _label_bar(width: int, text: str) -> np.ndarray:
    bar = np.zeros((24, width, 3), dtype=np.uint8)
    cv2.putText(bar, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return bar


def compare_on_video(
    video_path: str | Path,
    roi: tuple[int, int, int, int] = ELECTRONICS_TABLE_ROI,
    weights: str | Path = DEFAULT_WEIGHTS,
    ssim_threshold: float = 0.8,
    yolo_conf: float = 0.25,
    output_video_path: str | Path | None = None,
) -> dict:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    classical = MotionROIDetector(roi=roi, threshold=ssim_threshold, alerts_dir=None)
    yolo_model = YOLO(str(weights))

    output_video_path = Path(output_video_path) if output_video_path else OUTPUT_DIR / f"{video_path.stem}_side_by_side.mp4"
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    combined_height = height + 24
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width * 2, combined_height))

    classical_label = _label_bar(width, "Classical SSIM/ROI")
    yolo_label = _label_bar(width, "YOLOv8 (fine-tuned)")

    yolo_flagged_frames = 0
    yolo_confidences = []
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            score, flagged = classical.process_frame(frame, frame_index)
            classical_annotated = draw_roi_overlay(frame, roi, score, flagged)

            result = yolo_model.predict(source=frame, conf=yolo_conf, verbose=False)[0]
            yolo_annotated = result.plot()
            if len(result.boxes):
                yolo_flagged_frames += 1
                yolo_confidences.extend(float(b.conf[0]) for b in result.boxes)

            left = np.vstack([classical_label, classical_annotated])
            right = np.vstack([yolo_label, yolo_annotated])
            writer.write(np.hstack([left, right]))

            frame_index += 1
    finally:
        cap.release()
        writer.release()

    summary = {
        "video": video_path.name,
        "frames": frame_index,
        "classical_alerts": len(classical.alerts),
        "classical_avg_similarity": classical.average_similarity,
        "yolo_flagged_frames": yolo_flagged_frames,
        "yolo_avg_confidence": sum(yolo_confidences) / len(yolo_confidences) if yolo_confidences else 0.0,
        "output_video": str(output_video_path),
    }
    return summary


def print_summary_table(summaries: list[dict]) -> None:
    header = f"{'video':<20}{'frames':>8}{'classical_alerts':>18}{'yolo_flagged':>14}{'yolo_avg_conf':>15}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['video']:<20}{s['frames']:>8}{s['classical_alerts']:>18}"
            f"{s['yolo_flagged_frames']:>14}{s['yolo_avg_confidence']:>15.3f}"
        )


if __name__ == "__main__":
    videos = ["shoplifting1", "not_shoplifting1"]
    summaries = [compare_on_video(VIDEOS_DIR / f"{v}.mp4") for v in videos]
    print_summary_table(summaries)
    for s in summaries:
        print(f"\nCombined side-by-side video: {s['output_video']}")
