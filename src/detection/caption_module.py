"""
Pair YOLO detections with a caption for the same frame, for a short
qualitative demo: "YOLO: shoplifting_person, 0.87 confidence" next to
"Caption: ...".

Step 1's EDA already found real, ground-truth VLM captions shipped with
this dataset (VLM-labels/*.json - one file per source video, each with a
global description plus 3 chronological segments, every segment carrying
its own frame_range + scene_description - see src/data/eda.py). Per the
brief: when real captions exist, use them instead of running a
pretrained captioning model - so this module loads and pairs the real
ones; it does not load BLIP or any other model.

Each demo entry is logged as one JSON line (reusing step 2's
alert-log pattern from motion_baseline.py: a small dataclass, timestamped,
appended to a JSONL file) with {timestamp, image, detections, caption}.
"""

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO  # noqa: E402

from src.utils.dataset_discovery import (  # noqa: E402
    find_annotation_files,
    frame_number,
    load_vlm_video_labels,
    video_stem_for_image,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "outputs" / "yolo_runs"
DEFAULT_WEIGHTS = RUNS_DIR / "train" / "weights" / "best.pt"
VAL_IMAGES_DIR = PROJECT_ROOT / "outputs" / "yolo_dataset" / "images" / "val"


@dataclass
class Detection:
    class_name: str
    confidence: float


@dataclass
class CaptionedFrame:
    timestamp: str
    image: str
    detections: list[Detection]
    caption: str


def get_caption_for_image(img_path: Path, video_labels: dict[str, dict]) -> str:
    """Find the ground-truth VLM scene_description whose frame_range
    covers this specific frame; fall back to the video's overall
    description if no segment matches."""
    record = video_labels.get(video_stem_for_image(img_path))
    if record is None:
        return "(no VLM-labels record found for this frame's source video)"

    frame_num = frame_number(img_path)
    for segment in record.get("chronological_sequence_segments", []):
        try:
            lo, hi = (int(v) for v in segment["frame_range"].split("-"))
        except (KeyError, ValueError):
            continue
        if lo <= frame_num <= hi:
            return segment.get("scene_description", record.get("global_sequence_description", ""))

    return record.get("global_sequence_description", "(no caption available)")


def build_demo(
    weights: str | Path,
    image_paths: list[Path],
    conf: float = 0.25,
    log_path: str | Path | None = None,
) -> list[CaptionedFrame]:
    model = YOLO(str(weights))
    video_labels = load_vlm_video_labels(find_annotation_files(DATA_DIR))

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")

    entries = []
    for img_path in image_paths:
        result = model.predict(source=str(img_path), conf=conf, verbose=False)[0]
        detections = [
            Detection(class_name=result.names[int(box.cls[0])], confidence=float(box.conf[0]))
            for box in result.boxes
        ]
        caption = get_caption_for_image(img_path, video_labels)
        entry = CaptionedFrame(
            timestamp=datetime.now(timezone.utc).isoformat(),
            image=img_path.name,
            detections=detections,
            caption=caption,
        )
        entries.append(entry)

        det_str = (
            "; ".join(f"{d.class_name}, {d.confidence:.2f} confidence" for d in detections)
            if detections
            else "no detections"
        )
        print(f"\n{img_path.name}")
        print(f"  YOLO: {det_str}")
        print(f"  Caption: {caption}")

        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry)) + "\n")

    return entries


def pick_sample_images(n_per_class: int = 3) -> list[Path]:
    """Deterministically pick a small, class-balanced sample of held-out
    val images for the demo (val, not train - frames the model wasn't
    fine-tuned on)."""
    all_val = sorted(VAL_IMAGES_DIR.glob("*.png"))
    not_shoplifting = [p for p in all_val if p.name.startswith("not_shoplifting")]
    shoplifting = [p for p in all_val if p.name.startswith("shoplifting")]
    step_a = max(1, len(not_shoplifting) // n_per_class)
    step_b = max(1, len(shoplifting) // n_per_class)
    return not_shoplifting[::step_a][:n_per_class] + shoplifting[::step_b][:n_per_class]


if __name__ == "__main__":
    samples = pick_sample_images(n_per_class=3)
    build_demo(
        weights=DEFAULT_WEIGHTS,
        image_paths=samples,
        log_path=PROJECT_ROOT / "outputs" / "yolo_runs" / "caption_demo.jsonl",
    )
