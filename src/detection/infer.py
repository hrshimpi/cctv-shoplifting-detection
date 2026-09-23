"""
Run the fine-tuned YOLOv8 detector on sample images or a video and save
annotated outputs (boxes + class + confidence) to outputs/.

`source` can be a single image, a folder of images, or a video path -
ultralytics' predict() dispatches on file type automatically, drawing
boxes and writing the annotated result itself (save=True) rather than
this script re-implementing box drawing.

Run after `python src/detection/train_yolo.py`.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "outputs" / "yolo_runs"
DEFAULT_WEIGHTS = RUNS_DIR / "train" / "weights" / "best.pt"
DEFAULT_SOURCE = PROJECT_ROOT / "outputs" / "yolo_dataset" / "images" / "val"


def run_inference(
    weights: str | Path = DEFAULT_WEIGHTS,
    source: str | Path = DEFAULT_SOURCE,
    conf: float = 0.25,
    imgsz: int = 320,
    device: str = "cpu",
    project: str | Path = RUNS_DIR,
    name: str = "infer",
):
    model = YOLO(str(weights))
    results = model.predict(
        source=str(source),
        conf=conf,
        imgsz=imgsz,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
        save=True,
    )

    print(f"\n=== Inference on {source} ===")
    for r in results:
        boxes = r.boxes
        print(f"{Path(r.path).name}: {len(boxes)} detection(s)")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            print(f"    {r.names[cls_id]}: {conf_score:.2f} confidence")

    save_dir = Path(results[0].save_dir) if results else Path(project) / name
    print(f"\nAnnotated output(s) saved to {save_dir}/")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Image, folder of images, or video path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="infer")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
    )
