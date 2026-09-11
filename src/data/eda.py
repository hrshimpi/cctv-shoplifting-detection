"""
Exploratory data analysis for the CCTV shoplifting detection dataset.

This script does NOT hardcode the dataset's folder layout or class list.
Everything is discovered at runtime:
  - the class list is read from whichever data.yaml / classes.txt actually
    exists under data/ (see src/data/download_dataset.py, which prints the
    real folder tree and config contents after downloading)
  - split (train/val/test) image directories are resolved from data.yaml's
    path/train/val/test fields, falling back to common YOLO layouts
    (images/<split>/ or <split>/images/) only if data.yaml doesn't specify
    explicit paths
  - YOLO label .txt files are located by mirroring each images/ directory
    to a labels/ directory (standard YOLO convention)
  - any VLM-style caption/annotation JSON file is located by filename
    keyword, not a hardcoded path

Run after `python src/data/download_dataset.py` has populated data/.

Outputs (saved to outputs/eda/):
  - class_distribution.png
  - sample_grid.png (6-9 sample images with YOLO boxes drawn)
  - resolution_histogram.png
Console output includes per-split image counts and, if found, example
VLM caption/image pairs.
"""

import json
import random
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


def load_classes(data_yaml_path: Path | None, classes_txt_path: Path | None):
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
    raise FileNotFoundError(
        "No data.yaml or classes.txt found under data/. "
        "Run src/data/download_dataset.py first."
    )


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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        raise SystemExit("data/ is empty. Run `python src/data/download_dataset.py` first.")

    data_yaml_path = find_file(DATA_DIR, {"data.yaml"})
    classes_txt_path = find_file(DATA_DIR, {"classes.txt"})
    class_names, cfg = load_classes(data_yaml_path, classes_txt_path)
    print(f"Discovered {len(class_names)} classes: {class_names}")

    split_dirs = resolve_split_dirs(data_yaml_path, cfg)
    if not split_dirs:
        raise SystemExit("Could not resolve any train/val/test image directories under data/.")
    print(f"Discovered splits: {list(split_dirs.keys())}")

    # 1. Per-split image counts + label-derived class counts
    split_counts: dict[str, int] = {}
    class_counter: Counter = Counter()
    all_samples: list[tuple[Path, Path]] = []

    for split, img_dir in split_dirs.items():
        images = list_images(img_dir)
        split_counts[split] = len(images)
        lbl_dir = labels_dir_for_images(img_dir)
        for img_path in tqdm(images, desc=f"Scanning {split}"):
            label_path = lbl_dir / (img_path.stem + ".txt")
            for cls_id, *_ in parse_yolo_label(label_path):
                name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                class_counter[name] += 1
            all_samples.append((img_path, label_path))

    print("\nImages per split:")
    for split, n in split_counts.items():
        print(f"  {split}: {n}")
    print(f"  TOTAL: {sum(split_counts.values())}")

    # 2. Class distribution plot
    plt.figure(figsize=(8, 5))
    names = [n for n in class_names if n in class_counter] or list(class_counter.keys())
    counts = [class_counter[n] for n in names]
    plt.bar(names, counts, color="steelblue")
    plt.xlabel("Class")
    plt.ylabel("Instance count")
    plt.title("Class distribution (bounding box instances)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print(f"\nClass instance counts: {dict(class_counter)}")

    # 3. Sample grid with YOLO boxes drawn
    labeled_samples = [s for s in all_samples if s[1].exists()] or all_samples
    n_samples = min(9, max(6, min(9, len(labeled_samples))), len(labeled_samples))
    sample_choices = random.sample(labeled_samples, n_samples) if labeled_samples else []

    if sample_choices:
        ncols = 3
        nrows = (len(sample_choices) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)
        for ax, (img_path, label_path) in zip(axes, sample_choices):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            for cls_id, xc, yc, bw, bh in parse_yolo_label(label_path):
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, name, (x1, max(y1 - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            ax.imshow(img)
            ax.set_title(img_path.name, fontsize=8)
            ax.axis("off")
        for ax in axes[len(sample_choices):]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "sample_grid.png", dpi=150)
        plt.close()

    # 4. Image resolution histogram (PIL header read, no full decode)
    resolutions = []
    for img_path, _ in tqdm(all_samples, desc="Reading resolutions"):
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

    # 5. VLM captions, if present
    annotation_files = find_annotation_files(DATA_DIR)
    if annotation_files:
        print(f"\nFound {len(annotation_files)} caption/annotation file(s): "
              f"{[str(p.relative_to(DATA_DIR)) for p in annotation_files]}")
        ann_path = annotation_files[0]
        with open(ann_path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = list(data.values()) if not any(
                k in data for k in ("image", "file_name", "caption", "text")
            ) else [data]
        else:
            records = []

        print(f"\nExample image/caption pairs from {ann_path.name}:")
        for record in records[:5]:
            print(json.dumps(record, indent=2, ensure_ascii=False)[:500])
            print("-" * 40)
    else:
        print("\nNo VLM caption/annotation JSON file found under data/.")

    print(f"\nPlots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
