"""
Shared dataset-discovery helpers for the CCTV shoplifting dataset.

Both src/data/eda.py and src/detection/prepare_yolo_dataset.py need the
same "figure out what's actually in data/" logic - class config, VLM
per-video labels, image/label directories. This module is the one place
that logic lives, so nothing downstream re-guesses the dataset's
structure independently (see src/data/download_dataset.py's docstring for
why this matters: the real download ships no data.yaml/classes.txt, so
this is discovered at runtime, not assumed).
"""

import json
import re
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_NAMES = ("train", "val", "valid", "test")
ANNOTATION_KEYWORDS = ("caption", "annotation", "vlm")
FRAME_NUM_RE = re.compile(r"_f(\d+)$")


def find_file(root: Path, names: set[str]) -> Path | None:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name in names:
            return path
    return None


def find_annotation_files(root: Path) -> list[Path]:
    found = []
    for path in root.rglob("*.json"):
        if any(keyword in path.name.lower() for keyword in ANNOTATION_KEYWORDS):
            found.append(path)
    return sorted(found)


def find_images_dir(root: Path) -> Path | None:
    """Locate a directory literally named 'images' anywhere under root.

    Used when no data.yaml/classes.txt declares split paths to resolve a
    layout from - this dataset ships one flat images/ directory with no
    train/val/test split.
    """
    candidates = sorted(p for p in root.rglob("images") if p.is_dir())
    return candidates[0] if candidates else None


def load_class_config(
    data_yaml_path: Path | None, classes_txt_path: Path | None
) -> tuple[list[str] | None, dict]:
    """Load class names from data.yaml/classes.txt if the download shipped
    one. Returns (None, {}) rather than raising when neither exists, so
    the caller can fall back to another discovered class signal."""
    if data_yaml_path is not None:
        with open(data_yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        names = cfg.get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names, key=lambda x: int(x))]
        if names is None:
            raise ValueError(f"{data_yaml_path} has no 'names' field")
        return list(names), cfg
    if classes_txt_path is not None:
        names = [
            line.strip()
            for line in classes_txt_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return names, {}
    return None, {}


def load_vlm_video_labels(vlm_files: list[Path]) -> dict[str, dict]:
    """Parse each per-video VLM-labels JSON into {video_stem: record}."""
    video_labels = {}
    for path in vlm_files:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        video_name = record.get("video_filename")
        if not video_name:
            continue
        video_labels[Path(video_name).stem] = record
    return video_labels


def video_stem_for_image(img_path: Path) -> str:
    """Strip the '_f<frame_number>' suffix a sampled frame's filename adds
    to its source video's name, e.g. 'shoplifting3_f0102' -> 'shoplifting3'."""
    return re.sub(r"_f\d+$", "", img_path.stem)


def classify_image_by_video(img_path: Path, video_labels: dict[str, dict]) -> str | None:
    """Map a frame filename to its source video's shoplifting /
    not_shoplifting label via VLM-labels metadata (join key: the video-name
    filename prefix shared by every frame sampled from that video)."""
    record = video_labels.get(video_stem_for_image(img_path))
    if record is None:
        return None
    return "shoplifting" if record.get("is_anomaly_sequence") else "not_shoplifting"


def resolve_split_dirs(data_dir: Path, data_yaml_path: Path | None, cfg: dict) -> dict[str, Path]:
    """Resolve {split_name: image_dir} from data.yaml, falling back to
    scanning for common YOLO directory layouts under data_dir."""
    splits: dict[str, Path] = {}

    if data_yaml_path is not None:
        base = Path(cfg.get("path", str(data_yaml_path.parent)))
        if not base.is_absolute():
            base = (data_yaml_path.parent / base).resolve()

        for split in SPLIT_NAMES:
            val = cfg.get(split)
            if val is None:
                continue
            p = Path(val)
            if not p.is_absolute():
                p = (base / p).resolve()
            if p.exists() and p.is_dir():
                splits[split] = p

    if splits:
        return splits

    # Fallback: scan for common layouts anywhere under data_dir
    for split in SPLIT_NAMES:
        for candidate in (
            data_dir / "images" / split,
            data_dir / split / "images",
        ):
            if candidate.exists() and candidate.is_dir():
                splits.setdefault(split, candidate)
    return splits


def labels_dir_for_images(images_dir: Path) -> Path:
    """YOLO convention: an 'images' path segment mirrors to 'labels'."""
    parts = list(images_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return images_dir.parent / "labels"


def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse class + bbox from each line. Trailing values beyond the first
    5 fields (this dataset's labels carry 17 COCO keypoint (x, y,
    visibility) triplets after the bbox) are intentionally ignored."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls_id = int(float(parts[0]))
        xc, yc, w, h = (float(x) for x in parts[1:5])
        boxes.append((cls_id, xc, yc, w, h))
    return boxes


def frame_number(path: Path) -> int:
    m = FRAME_NUM_RE.search(path.stem)
    return int(m.group(1)) if m else -1
