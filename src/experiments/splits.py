"""Group temporal windows by physical source video before validation splitting."""

import numpy as np


def split_by_video(metadata, val_fraction: float, seed: int):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    # Ignore pose-file parts: parts of one camera/video must stay together.
    keys = [(str(int(m.camera_id)) if str(m.camera_id).isdigit() else str(m.camera_id), str(m.video_id))
            for m in metadata]
    groups = sorted(set(keys))
    if len(groups) < 2:
        raise ValueError("At least two source-video groups are needed for disjoint validation")
    order = np.random.default_rng(seed).permutation(len(groups))
    n_val = min(len(groups) - 1, max(1, round(len(groups) * val_fraction)))
    val_groups = {groups[i] for i in order[:n_val]}
    train_idx = np.array([i for i, key in enumerate(keys) if key not in val_groups], dtype=np.int64)
    val_idx = np.array([i for i, key in enumerate(keys) if key in val_groups], dtype=np.int64)
    report = {"method": "source_video_grouped", "seed": seed, "requested_val_fraction": val_fraction,
              "train_groups": [list(g) for g in groups if g not in val_groups],
              "validation_groups": [list(g) for g in groups if g in val_groups],
              "train_windows": len(train_idx), "validation_windows": len(val_idx),
              "actual_validation_window_fraction": len(val_idx) / len(keys)}
    return train_idx, val_idx, report
