"""
Evaluate a fine-tuned YOLOv8 checkpoint on the val split and report real
metrics for the README.

Runs model.val() (which also saves confusion_matrix.png, PR_curve.png,
and friends into its own run directory - that's ultralytics' built-in
plotting, not reimplemented here), prints precision / recall / mAP50 /
mAP50-95, and separately reloads the training run's results.csv to plot
loss curves with matplotlib (results.csv only exists after train_yolo.py
has been run - val() alone doesn't produce it).

Run after `python src/detection/train_yolo.py`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "outputs" / "yolo_dataset" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "outputs" / "yolo_runs"


def evaluate(weights: str | Path, data: str | Path = DATA_YAML, name: str = "eval", **val_kwargs):
    model = YOLO(str(weights))
    metrics = model.val(data=str(data), project=str(RUNS_DIR), name=name, exist_ok=True, plots=True, **val_kwargs)

    box = metrics.box
    summary = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
    }

    print("\n=== Validation metrics ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    print("\nPer-class mAP50:")
    for cls_id, name_ in metrics.names.items():
        print(f"  {name_}: {box.ap50[cls_id] if cls_id < len(box.ap50) else float('nan'):.4f}")

    save_dir = Path(metrics.save_dir)
    print(f"\nConfusion matrix + PR/F1/P/R curves saved to {save_dir}/")

    return metrics, summary


def plot_loss_curves(train_run_dir: str | Path, out_path: str | Path | None = None) -> Path | None:
    """Reload results.csv from a training run and plot train/val loss curves."""
    results_csv = Path(train_run_dir) / "results.csv"
    if not results_csv.exists():
        print(f"No results.csv found at {results_csv} - skipping loss curve plot.")
        return None

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    loss_cols = [c for c in df.columns if c.endswith("box_loss") or c.endswith("cls_loss") or c.endswith("dfl_loss")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    loss_kinds = ["box_loss", "cls_loss", "dfl_loss"]
    for ax, kind in zip(axes, loss_kinds):
        for col in loss_cols:
            if col.endswith(kind):
                label = "train" if col.startswith("train/") else "val"
                ax.plot(df["epoch"], df[col], label=label)
        ax.set_title(kind)
        ax.set_xlabel("epoch")
        ax.legend()
    plt.tight_layout()

    out_path = Path(out_path) if out_path else Path(train_run_dir) / "loss_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Loss curves saved to {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(RUNS_DIR / "train" / "weights" / "best.pt"))
    parser.add_argument("--data", default=str(DATA_YAML))
    parser.add_argument("--name", default="eval")
    parser.add_argument("--train-run-dir", default=str(RUNS_DIR / "train"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(weights=args.weights, data=args.data, name=args.name)
    plot_loss_curves(args.train_run_dir)
