#!/usr/bin/env python
"""
Train the BiGRU sequence autoencoder for one experiment (A-E) on RetailS
normal sequences only. Standalone, runs locally - no notebook required.

Usage:
    python scripts/train.py --experiment a
    python scripts/train.py --experiment b
    python scripts/train.py --experiment c
    python scripts/train.py --experiment d
    python scripts/train.py --experiment e
    python scripts/train.py --experiment a --config configs/experiment_a.yaml
    python scripts/train.py --experiment c --checkpoint path/to/best_epoch.bin

A/B use 2D pose (src/data/preprocessing.py's pelvis/torso-normalized
output) directly. C/D additionally lift through a frozen, pretrained
MotionBERT (src/motionbert/lift.py - see README.md "MotionBERT setup"
for how to obtain the checkpoint) to get 3D pose first.

Anomaly score convention: mean squared reconstruction error (higher =
more anomalous). Trained one-class - RetailS_train ships no anomaly
labels at all, so "normal sequences only" is automatically satisfied by
training on that subset in full (see README.md "Dataset").

Saves, under outputs/:
  checkpoints/<experiment_name>/model.pt
  checkpoints/<experiment_name>/train_reconstruction_errors.npy  (consumed by scripts/evaluate.py to set the decision threshold)
  logs/<experiment_name>/training_log.csv
  logs/<experiment_name>/config_used.yaml
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.data.retails import discover_retails_root  # noqa: E402
from src.experiments.dataset import build_windows_for_subset, build_windows_for_subset_3d  # noqa: E402
from src.features.motion_features import feature_dim, feature_dim_3d  # noqa: E402
from src.models.bigru_attention import BiGRUAttentionAutoencoder  # noqa: E402
from src.models.bigru_autoencoder import BiGRUAutoencoder  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "retails"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "third_party" / "MotionBERT" / "checkpoint" / "pose3d" / "MB_ft_h36m" / "best_epoch.bin"

EXPERIMENT_CONFIGS = {
    "a": PROJECT_ROOT / "configs" / "experiment_a.yaml",
    "b": PROJECT_ROOT / "configs" / "experiment_b.yaml",
    "c": PROJECT_ROOT / "configs" / "experiment_c.yaml",
    "d": PROJECT_ROOT / "configs" / "experiment_d.yaml",
    "e": PROJECT_ROOT / "configs" / "experiment_e.yaml",
}
EXPERIMENTS_3D = {"c", "d", "e"}
MODEL_CLASSES = {"bigru": BiGRUAutoencoder, "bigru_attention": BiGRUAttentionAutoencoder}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENT_CONFIGS), required=True)
    parser.add_argument("--config", default=None, help="Override the default config path for this experiment")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="MotionBERT checkpoint path (experiments c/d/e only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config) if args.config else EXPERIMENT_CONFIGS[args.experiment]
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    root = discover_retails_root(DATA_DIR)
    if root is None:
        raise SystemExit(f"Could not find RetailS under {DATA_DIR}. Download/extract it first (see README.md).")

    print(f"=== Training {cfg['experiment_name']} ===")
    print(f"Config: {cfg}")

    t0 = time.time()
    if args.experiment in EXPERIMENTS_3D:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise SystemExit(f"No MotionBERT checkpoint at {checkpoint_path} - see README.md 'MotionBERT setup'.")
        from src.motionbert.lift import load_motionbert

        model_3d = load_motionbert(checkpoint_path)
        features, labels = build_windows_for_subset_3d(
            root,
            "train",
            cfg,
            model_3d,
            use_velocity=cfg["use_velocity"],
            use_acceleration=cfg["use_acceleration"],
            use_geometry=cfg["use_geometry"],
            use_angles=cfg["use_angles"],
            max_pose_files=cfg["max_train_pose_files"],
            seed=cfg["seed"],
        )
    else:
        features, labels = build_windows_for_subset(
            root,
            "train",
            cfg,
            use_velocity=cfg["use_velocity"],
            use_acceleration=cfg["use_acceleration"],
            max_pose_files=cfg["max_train_pose_files"],
            seed=cfg["seed"],
        )
    print(
        f"Built {len(features)} training windows in {time.time() - t0:.1f}s "
        f"(feature dim {features.shape[-1]}, from {cfg['max_train_pose_files']} sampled pose files)"
    )
    assert (labels == 0).all(), "RetailS_train unexpectedly has non-zero labels - investigate before trusting this run"

    n = len(features)
    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(n)
    n_val = int(n * cfg["val_fraction"])
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(features[train_idx])), batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(features[val_idx])), batch_size=cfg["batch_size"], shuffle=False)

    if args.experiment in EXPERIMENTS_3D:
        input_dim = feature_dim_3d(cfg["use_velocity"], cfg["use_acceleration"], cfg["use_geometry"], cfg["use_angles"])
    else:
        input_dim = feature_dim(cfg["use_velocity"], cfg["use_acceleration"])
    assert input_dim == features.shape[-1], f"feature_dim mismatch: {input_dim} vs {features.shape[-1]}"

    model_cls = MODEL_CLASSES[cfg.get("model", "bigru")]
    model = model_cls(
        input_dim=input_dim, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"], dropout=cfg["dropout"]
    )
    assert cfg["optimizer"].lower() == "adam", f"Only adam is wired up, got {cfg['optimizer']!r}"
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = torch.nn.MSELoss()

    out_dir = OUTPUTS_DIR / "checkpoints" / cfg["experiment_name"]
    log_dir = OUTPUTS_DIR / "logs" / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_rows = []
    train_start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                val_losses.append(loss_fn(model(batch), batch).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        elapsed = time.time() - train_start
        print(f"epoch {epoch:3d}/{cfg['epochs']}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  ({elapsed:.1f}s elapsed)")
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "elapsed_sec": elapsed})

    with open(log_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "elapsed_sec"])
        writer.writeheader()
        writer.writerows(log_rows)

    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, out_dir / "model.pt")

    # Reconstruction error over ALL normal training windows (train+val
    # together) - evaluate.py uses this to set the decision threshold, so
    # it should reflect the full normal-sequence distribution the model
    # was fit to, not just the train split.
    model.eval()
    all_loader = DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=cfg["batch_size"], shuffle=False)
    errors = []
    with torch.no_grad():
        for (batch,) in all_loader:
            errors.append(model.reconstruction_error(batch).numpy())
    train_errors = np.concatenate(errors)
    np.save(out_dir / "train_reconstruction_errors.npy", train_errors)

    with open(log_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print(f"\nDone in {time.time() - t0:.1f}s total.")
    print(f"Checkpoint: {out_dir / 'model.pt'}")
    print(
        f"Training-set reconstruction error: mean={train_errors.mean():.6f}, "
        f"p{cfg['threshold_percentile']}={np.percentile(train_errors, cfg['threshold_percentile']):.6f} "
        "(this percentile becomes the eval threshold)"
    )


if __name__ == "__main__":
    main()
