"""
2D -> 3D pose lifting via a frozen, pretrained MotionBERT (DSTformer).

MotionBERT is used exactly as published - no fine-tuning, no gradient
updates. See README.md's "MotionBERT setup" section for how to obtain
the checkpoint this module loads, and
third_party/MotionBERT/configs/pose3d/MB_ft_h36m.yaml for the source of
every hyperparameter/flag below (confirmed against the real file, not
assumed).
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOTIONBERT_ROOT = PROJECT_ROOT / "third_party" / "MotionBERT"
if str(MOTIONBERT_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONBERT_ROOT))

from lib.model.DSTformer import DSTformer  # noqa: E402

from src.utils.joint_mapping import coco17_to_h36m17  # noqa: E402

# Matches MB_ft_h36m.yaml / MB_train_h36m.yaml exactly: dim_feat, mlp_ratio,
# depth, dim_rep, num_heads, maxlen, num_joints. att_fuse defaults to True
# in DSTformer.__init__, matching the yaml's att_fuse: True.
DSTFORMER_KWARGS = dict(
    dim_in=3,
    dim_out=3,
    dim_feat=512,
    dim_rep=512,
    depth=5,
    num_heads=8,
    mlp_ratio=2,
    maxlen=243,
    num_joints=17,
)
# MB_ft_h36m.yaml: rootrel: True - output is root(pelvis)-relative, so the
# model's raw pelvis prediction is discarded and zeroed, matching
# third_party/MotionBERT/infer_wild.py's own post-processing.
ROOTREL = True


def load_motionbert(checkpoint_path: str | Path) -> nn.Module:
    """Build DSTformer with MB_ft_h36m's real hyperparameters and load the
    official pretrained weights (checkpoint['model_pos']). Returns the
    model in eval() mode with all parameters frozen.

    The published checkpoint was saved from a model wrapped in
    nn.DataParallel for multi-GPU training (third_party/MotionBERT/
    infer_wild.py only re-wraps in DataParallel "if torch.cuda.is_available()"
    before loading, which silently also assumes a GPU machine) - every
    key in checkpoint['model_pos'] is therefore prefixed "module." (e.g.
    "module.blocks_ts.4...."). This project always builds a plain,
    unwrapped DSTformer (CPU-only - see README.md), so that prefix is
    stripped before loading rather than wrapping/unwrapping in
    DataParallel just to match it.
    """
    model = DSTformer(**DSTFORMER_KWARGS, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_pos"]
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def crop_scale(motion_h36m: np.ndarray) -> np.ndarray:
    """
    Replicates third_party/MotionBERT/lib/utils/utils_data.py:crop_scale
    with scale_range=[1, 1] (i.e. no random augmentation) exactly - the
    normalization MotionBERT's own inference pipeline
    (lib/data/dataset_wild.py) applies before the model ever sees 2D
    input: fit one bounding box over every confidence>0 keypoint across
    the WHOLE window, then center+scale that box to [-1, 1].

    This is deliberately NOT this project's own pelvis-centered/
    torso-scaled convention (src/data/preprocessing.py normalize_position/
    normalize_scale) - that normalization is for the BiGRU branch; this
    one is specifically what MotionBERT expects, so call
    preprocess_track(..., apply_normalization=False) upstream of this.

    motion_h36m: (T, 17, 3) - x, y, confidence, H36M-17 joint order.
    Returns (T, 17, 3) with x, y remapped into [-1, 1]; confidence
    unchanged.
    """
    motion = motion_h36m.copy()
    valid_coords = motion[motion[..., 2] != 0][:, :2]
    if len(valid_coords) < 4:
        return np.zeros_like(motion)

    xmin, xmax = valid_coords[:, 0].min(), valid_coords[:, 0].max()
    ymin, ymax = valid_coords[:, 1].min(), valid_coords[:, 1].max()
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros_like(motion)

    xs, ys = (xmin + xmax - scale) / 2, (ymin + ymax - scale) / 2
    motion[..., :2] = (motion[..., :2] - np.array([xs, ys])) / scale
    motion[..., :2] = (motion[..., :2] - 0.5) * 2
    # The official utility clips all channels, including HRNet confidence
    # scores that can slightly exceed one.
    return np.clip(motion, -1, 1)


def lift_2d_to_3d(model: nn.Module, keypoints_coco17: np.ndarray) -> np.ndarray:
    """
    keypoints_coco17: (T, 17, 3) - x, y, confidence, COCO-17 order. Should
    be cleaned (confidence-masked, gap-interpolated, optionally smoothed)
    but NOT pelvis-centered/torso-scaled - see preprocess_track(...,
    apply_normalization=False) in src/data/preprocessing.py.

    Returns (T, 17, 3) - x, y, z in H36M-17 order (see
    src/utils/joint_mapping.py), in MotionBERT's own [-1, 1]-normalized,
    root(pelvis)-relative space.
    """
    return lift_2d_to_3d_batch(model, keypoints_coco17[None, ...])[0]


def lift_2d_to_3d_batch(model: nn.Module, keypoints_coco17_batch: np.ndarray) -> np.ndarray:
    """
    Same as lift_2d_to_3d, batched over N windows in a single forward
    pass - a single CPU forward call over a batch is substantially faster
    per-window than N separate calls (better matmul utilization), which
    matters here: a DSTformer forward pass is the dominant cost of
    building the 3D-pose experiments' (C/D) training/eval windows (see
    README.md "Training and evaluating an experiment").

    keypoints_coco17_batch: (N, T, 17, 3). Returns (N, T, 17, 3), same
    convention as lift_2d_to_3d.
    """
    h36m = coco17_to_h36m17(keypoints_coco17_batch)
    normalized = np.stack([crop_scale(window) for window in h36m])
    x = torch.from_numpy(normalized).float()  # (N, T, 17, 3)
    with torch.no_grad():
        pred = model(x)  # (N, T, 17, 3)
    pred = pred.numpy()
    if ROOTREL:
        pred[:, :, 0, :] = 0.0
    return pred
