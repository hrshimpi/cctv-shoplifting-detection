"""
Fine-tune a pretrained YOLOv8 model on the 2-class dataset
prepare_yolo_data.py built (outputs/yolo_dataset/data.yaml).

Model size, epochs, batch size, and image size are all configurable -
either via CLI args or by calling train() directly with a config dict -
so this scales from a quick CPU smoke-test up to a real run on a better
GPU without editing code. Defaults are deliberately small (yolov8n,
modest imgsz/epochs) so this completes on a free Colab GPU - or, as
verified locally, on a plain CPU in a few minutes on this tiny dataset.

Run after `python src/detection/prepare_yolo_data.py`.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "outputs" / "yolo_dataset" / "data.yaml"
WEIGHTS_DIR = PROJECT_ROOT / "outputs" / "weights"
RUNS_DIR = PROJECT_ROOT / "outputs" / "yolo_runs"


def resolve_pretrained_weights(model_name: str) -> str:
    """Return a local path to `model_name`, caching it under
    outputs/weights/ instead of letting ultralytics drop it at the repo
    root (its default when given a bare model name)."""
    cached = WEIGHTS_DIR / model_name
    if cached.exists():
        return str(cached)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    YOLO(model_name)  # triggers ultralytics' own download, into the CWD
    downloaded = PROJECT_ROOT / model_name
    if downloaded.exists():
        shutil.move(str(downloaded), str(cached))
    return str(cached) if cached.exists() else model_name


def train(
    model: str = "yolov8n.pt",
    epochs: int = 30,
    batch: int = 8,
    imgsz: int = 416,
    data: str | Path = DATA_YAML,
    device: str = "cpu",
    project: str | Path = RUNS_DIR,
    name: str = "train",
    **kwargs,
):
    """Fine-tune `model` on `data` and return the trained ultralytics YOLO
    object. Extra ultralytics train() kwargs (patience, lr0, workers, ...)
    can be passed straight through via **kwargs."""
    if not Path(data).exists():
        raise SystemExit(f"{data} not found. Run `python src/detection/prepare_yolo_data.py` first.")

    weights_path = resolve_pretrained_weights(model)
    yolo_model = YOLO(weights_path)

    yolo_model.train(
        data=str(data),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
        **kwargs,
    )
    return yolo_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained YOLOv8 checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--data", default=str(DATA_YAML))
    parser.add_argument("--device", default="cpu", help="'cpu', '0' for first GPU, etc.")
    parser.add_argument("--project", default=str(RUNS_DIR))
    parser.add_argument("--name", default="train")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        data=args.data,
        device=args.device,
        project=args.project,
        name=args.name,
    )
