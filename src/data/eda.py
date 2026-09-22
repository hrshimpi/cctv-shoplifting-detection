"""
Exploratory data analysis for the CCTV shoplifting detection dataset.

This script does NOT hardcode the dataset's folder layout or class list —
everything below reflects what `download_dataset.py`'s tree/config dump
actually showed when run against the real Kaggle download:

  data/CCTV_Shoplifting_Dataset/
    images/            - 456 flat PNG frames, "{video}_f{frame:04d}.png"
    labels/            - matching YOLO-pose .txt files: class 0 + bbox +
                         17 COCO keypoint (x, y, visibility) triplets per
                         detected person
    VLM-labels/        - one JSON per source video (not a single shared
                         caption file): {video_filename, total_video_frames,
                         is_anomaly_sequence, global_sequence_description,
                         chronological_sequence_segments: [...]}
    annotated_images/, annotated_videos/, videos/ - renders/source videos,
                         not used here

There is no data.yaml or classes.txt anywhere in the download, and no
train/val/test split. The one class signal this dataset actually ships is
per source video - "shoplifting" vs "not_shoplifting" - recorded in each
VLM-labels JSON's is_anomaly_sequence field. This script prefers an
explicit data.yaml/classes.txt if a future re-download ever includes one
(load_class_config), and otherwise falls back to deriving a class per
frame from the VLM-labels metadata, joined to frames by filename prefix.

Run after `python src/data/download_dataset.py` has populated data/.

Outputs (saved to outputs/eda/):
  - class_distribution.png
  - sample_grid.png (6-9 sample images with YOLO boxes drawn)
  - resolution_histogram.png
Console output includes per-split image counts and, if VLM-labels are
found, example scene-description segments tied to real sampled frames.
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_NAMES = ("train", "val", "valid", "test")
ANNOTATION_KEYWORDS = ("caption", "annotation", "vlm")
FRAME_NUM_RE = re.compile(r"_f(\d+)$")

random.seed(0)


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


def classify_image_by_video(img_path: Path, video_labels: dict[str, dict]) -> str | None:
    """Map a frame filename to its source video's shoplifting /
    not_shoplifting label via VLM-labels metadata (join key: the video-name
    filename prefix shared by every frame sampled from that video)."""
    video_stem = re.sub(r"_f\d+$", "", img_path.stem)
    record = video_labels.get(video_stem)
    if record is None:
        return None
    return "shoplifting" if record.get("is_anomaly_sequence") else "not_shoplifting"


def resolve_split_dirs(data_yaml_path: Path | None, cfg: dict) -> dict[str, Path]:
    """Resolve {split_name: image_dir} from data.yaml, falling back to
    scanning for common YOLO directory layouts under DATA_DIR."""
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

    # Fallback: scan for common layouts anywhere under DATA_DIR
    for split in SPLIT_NAMES:
        for candidate in (
            DATA_DIR / "images" / split,
            DATA_DIR / split / "images",
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


def representative_frame(
    split_dirs: dict[str, Path], video_stem: str, lo: int, hi: int
) -> Path | None:
    """Pick a real sampled frame file for `video_stem` whose frame number
    falls inside [lo, hi], closest to the middle of the range."""
    candidates = []
    for img_dir in split_dirs.values():
        candidates.extend(
            p for p in img_dir.glob(f"{video_stem}_f*")
            if p.suffix.lower() in IMAGE_EXTS and lo <= frame_number(p) <= hi
        )
    if not candidates:
        return None
    mid = (lo + hi) / 2
    return min(candidates, key=lambda p: abs(frame_number(p) - mid))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        raise SystemExit("data/ is empty. Run `python src/data/download_dataset.py` first.")

    data_yaml_path = find_file(DATA_DIR, {"data.yaml"})
    classes_txt_path = find_file(DATA_DIR, {"classes.txt"})
    class_names, cfg = load_class_config(data_yaml_path, classes_txt_path)

    vlm_files = find_annotation_files(DATA_DIR)
    video_labels = load_vlm_video_labels(vlm_files) if vlm_files else {}

    split_dirs = resolve_split_dirs(data_yaml_path, cfg)
    if not split_dirs:
        images_dir = find_images_dir(DATA_DIR)
        if images_dir is None:
            raise SystemExit("Could not resolve any train/val/test image directories under data/.")
        split_dirs = {"all": images_dir}
        print(f"No data.yaml/classes.txt split config found under {DATA_DIR} - "
              f"treating all images under {images_dir.relative_to(DATA_DIR)} as a "
              "single unsplit pool (this dataset ships no train/val/test split).")

    use_video_classes = class_names is None
    if use_video_classes and not video_labels:
        raise SystemExit(
            "No data.yaml/classes.txt AND no VLM-labels JSON found under data/ - "
            "no class signal to discover. Inspect data/ manually."
        )
    if use_video_classes:
        print("No data.yaml/classes.txt found - deriving class labels from "
              "VLM-labels' is_anomaly_sequence field instead (shoplifting / "
              "not_shoplifting per source video).")
    else:
        print(f"Discovered {len(class_names)} classes from config: {class_names}")
    print(f"Discovered splits: {list(split_dirs.keys())}")

    # 1. Per-split image counts + class counts
    split_counts: dict[str, int] = {}
    class_counter: Counter = Counter()
    all_samples: list[tuple[Path, Path, str | None]] = []
    total_detections = 0

    for split, img_dir in split_dirs.items():
        images = list_images(img_dir)
        split_counts[split] = len(images)
        lbl_dir = labels_dir_for_images(img_dir)
        for img_path in tqdm(images, desc=f"Scanning {split}"):
            label_path = lbl_dir / (img_path.stem + ".txt")
            boxes = parse_yolo_label(label_path)
            total_detections += len(boxes)

            if use_video_classes:
                img_class = classify_image_by_video(img_path, video_labels)
                if img_class is not None:
                    class_counter[img_class] += 1
            else:
                img_class = None
                for cls_id, *_ in boxes:
                    name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                    class_counter[name] += 1
            all_samples.append((img_path, label_path, img_class))

    print("\nImages per split:")
    for split, n in split_counts.items():
        print(f"  {split}: {n}")
    print(f"  TOTAL: {sum(split_counts.values())}")

    if use_video_classes:
        print(f"\nPerson detections across all frames (YOLO class 0, pose-format "
              f"labels with 17 keypoints each): {total_detections}")

    # 2. Class distribution plot
    plt.figure(figsize=(8, 5))
    names_for_plot = list(class_counter.keys())
    counts = [class_counter[n] for n in names_for_plot]
    plt.bar(names_for_plot, counts, color="steelblue")
    plt.xlabel("Class" if not use_video_classes else "Sequence label")
    plt.ylabel("Instance count" if not use_video_classes else "Frame count")
    plt.title("Class distribution (bounding box instances)" if not use_video_classes
               else "Class distribution (frames per sequence label)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print(f"\nClass counts: {dict(class_counter)}")

    # 3. Sample grid with YOLO boxes drawn
    labeled_samples = [s for s in all_samples if s[1].exists()] or all_samples
    n_samples = min(9, max(6, min(9, len(labeled_samples))), len(labeled_samples))
    sample_choices = random.sample(labeled_samples, n_samples) if labeled_samples else []

    if sample_choices:
        ncols = 3
        nrows = (len(sample_choices) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)
        for ax, (img_path, label_path, img_class) in zip(axes, sample_choices):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            for cls_id, xc, yc, bw, bh in parse_yolo_label(label_path):
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                box_label = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, box_label, (x1, max(y1 - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            ax.imshow(img)
            title = img_path.name if img_class is None else f"{img_path.name}\n[{img_class}]"
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        for ax in axes[len(sample_choices):]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "sample_grid.png", dpi=150)
        plt.close()

    # 4. Image resolution histogram (PIL header read, no full decode)
    resolutions = []
    for img_path, _, _ in tqdm(all_samples, desc="Reading resolutions"):
        try:
            with Image.open(img_path) as im:
                resolutions.append(im.size)  # (width, height)
        except Exception:
            continue

    if resolutions:
        widths, heights = zip(*resolutions)
        plt.figure(figsize=(8, 5))
        plt.hist(widths, bins=20, alpha=0.6, label="width")
        plt.hist(heights, bins=20, alpha=0.6, label="height")
        plt.xlabel("Pixels")
        plt.ylabel("Count")
        plt.title("Image resolution histogram")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "resolution_histogram.png", dpi=150)
        plt.close()
        unique_res = Counter(resolutions)
        print(f"\nUnique resolutions (top 5): {unique_res.most_common(5)}")

    # 5. VLM scene-description segments, if present
    if vlm_files and video_labels:
        print(f"\nFound {len(vlm_files)} VLM-labels file(s): "
              f"{[str(p.relative_to(DATA_DIR)) for p in vlm_files]}")

        by_class: dict[str, list[str]] = {"not_shoplifting": [], "shoplifting": []}
        for video_stem, record in video_labels.items():
            cls = "shoplifting" if record.get("is_anomaly_sequence") else "not_shoplifting"
            by_class[cls].append(video_stem)

        example_videos = []
        for cls in ("not_shoplifting", "shoplifting"):
            example_videos.extend(sorted(by_class[cls])[:3])
        example_videos = example_videos[:5]

        print("\nExample image/caption pairs (scene-description segments from "
              "VLM-labels, each tied to a real sampled frame):")
        for video_stem in example_videos:
            record = video_labels[video_stem]
            segments = record.get("chronological_sequence_segments", [])
            if not segments:
                continue
            seg = segments[len(segments) // 2]  # a middle segment
            frame_range = seg.get("frame_range", "")
            try:
                lo, hi = (int(x) for x in frame_range.split("-"))
            except ValueError:
                lo, hi = 0, 0
            rep_frame = representative_frame(split_dirs, video_stem, lo, hi)
            cls = "shoplifting" if record.get("is_anomaly_sequence") else "not_shoplifting"
            print(f"\n  video: {video_stem} [{cls}]")
            print(f"  image: {rep_frame.name if rep_frame else '(no sampled frame in range)'}")
            print(f"  frame_range: {frame_range} ({seg.get('temporal_phase', '')})")
            print(f"  caption: {seg.get('scene_description', '')}")
    else:
        print("\nNo VLM-labels JSON found under data/.")

    print(f"\nPlots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
