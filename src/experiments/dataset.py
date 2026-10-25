"""
Shared window + feature assembly for training and evaluation. Used
unchanged by both scripts/train.py and scripts/evaluate.py so every
experiment builds its data the same way - only the feature flags
(use_velocity/use_acceleration), the RetailS subset, and how many pose
files to sample differ between calls.

return_metadata=True (used by scripts/analyze_false_positives.py and
scripts/analyze_per_camera.py, step 4) additionally returns, per
window, which real track/frames it came from (camera_id, video_id,
person_id, start_frame, end_frame, pose_file) - needed to trace a
window back to something renderable/groupable, not needed by
train.py/evaluate.py's own metrics and off by default so their return
shape is unchanged.
"""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.preprocessing import build_windows, preprocess_track
from src.data.retails import (
    SUBSET_DIRS,
    find_pose_files,
    gt_dir_for_subset,
    label_path_for_pose_file,
    load_anomaly_labels,
    load_pose_file,
)
from src.features.motion_features import build_feature_vector, build_feature_vector_3d

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIFT_CACHE_DIR = PROJECT_ROOT / "outputs" / "cache" / "motionbert_lift"


@dataclass
class WindowMeta:
    pose_file: str
    person_id: str
    camera_id: str
    video_id: str
    start_frame: int
    end_frame: int


def _window_label(labels_array: np.ndarray | None, start_frame: int, end_frame: int) -> int:
    """A window is anomalous if ANY of its frames are labeled anomalous
    in the per-video frame mask. This is a conservative convention - a
    real modeling choice, not an obvious default - documented here since
    both train.py and evaluate.py rely on it being applied consistently."""
    if labels_array is None:
        return 0
    lo = max(start_frame, 0)
    hi = min(end_frame, len(labels_array) - 1)
    if lo > hi:
        return 0
    return int((labels_array[lo : hi + 1] == 1).any())


def _sample_pose_files(pose_files: list[Path], max_pose_files: int | None, seed: int) -> list[Path]:
    if max_pose_files is not None and len(pose_files) > max_pose_files:
        rng = random.Random(seed)
        return sorted(rng.sample(pose_files, max_pose_files))
    return pose_files


def _collect_windows(
    root: Path,
    subset_key: str,
    preprocessing_cfg: dict,
    max_pose_files: int | None,
    seed: int,
    apply_normalization: bool,
    desc: str,
) -> tuple[list[np.ndarray], list[int], list[WindowMeta]]:
    """Shared per-track windowing pass used by both the 2D path
    (build_windows_for_subset) and the 3D path's pre-lift collection
    step (lifted_windows_for_subset) - everything except feature
    engineering and (for 3D) the MotionBERT lift itself."""
    subset_dir = root / SUBSET_DIRS[subset_key]
    pose_files = _sample_pose_files(find_pose_files(subset_dir), max_pose_files, seed)
    gt_dir = gt_dir_for_subset(subset_dir)

    window_keypoints: list[np.ndarray] = []
    window_labels: list[int] = []
    window_meta: list[WindowMeta] = []

    for pose_path in tqdm(pose_files, desc=desc):
        label_path = label_path_for_pose_file(pose_path, gt_dir)
        if subset_key != "train" and (label_path is None or not label_path.is_file()):
            raise FileNotFoundError(f"Missing test ground truth for {pose_path}; refusing to assume normal labels")
        video_labels = load_anomaly_labels(label_path)

        for track in load_pose_file(pose_path):
            processed = preprocess_track(
                track,
                confidence_threshold=preprocessing_cfg["confidence_threshold"],
                max_interpolation_gap=preprocessing_cfg["max_interpolation_gap"],
                smoothing_window=preprocessing_cfg["smoothing_window"],
                apply_normalization=apply_normalization,
            )
            windows = build_windows(
                processed["frame_ids"],
                processed["keypoints"],
                processed["visibility_mask"],
                window_length=preprocessing_cfg["window_length"],
                stride=preprocessing_cfg["stride"],
            )
            for w in windows:
                window_keypoints.append(w["keypoints"])
                window_labels.append(_window_label(video_labels, w["start_frame"], w["end_frame"]))
                window_meta.append(
                    WindowMeta(
                        pose_file=pose_path.name,
                        person_id=track.person_id,
                        camera_id=track.camera_id,
                        video_id=track.video_id,
                        start_frame=w["start_frame"],
                        end_frame=w["end_frame"],
                    )
                )

    return window_keypoints, window_labels, window_meta


def build_windows_for_subset(
    root: Path,
    subset_key: str,
    preprocessing_cfg: dict,
    use_velocity: bool,
    use_acceleration: bool,
    max_pose_files: int | None = None,
    seed: int = 0,
    return_metadata: bool = False,
):
    """Returns (features, labels), or (features, labels, metadata) if
    return_metadata=True:
      features: (N, window_length, D) float32
      labels:   (N,) int64 - all zero for the train subset (it ships no
                gt/ at all), real 0/1 per window for the two test subsets.
      metadata: list[WindowMeta], len N
    """
    window_keypoints, window_labels, window_meta = _collect_windows(
        root, subset_key, preprocessing_cfg, max_pose_files, seed,
        apply_normalization=True, desc=f"Building windows ({subset_key})",
    )

    if not window_keypoints:
        features = np.empty((0, preprocessing_cfg["window_length"], 0), dtype=np.float32)
        labels = np.empty((0,), dtype=np.int64)
        return (features, labels, window_meta) if return_metadata else (features, labels)

    features = np.stack(
        [build_feature_vector(kp[:, :, :2], use_velocity, use_acceleration) for kp in window_keypoints]
    ).astype(np.float32)
    labels = np.array(window_labels, dtype=np.int64)
    return (features, labels, window_meta) if return_metadata else (features, labels)


def _lift_cache_path(subset_key: str, preprocessing_cfg: dict, max_pose_files: int | None, seed: int) -> Path:
    p = preprocessing_cfg
    name = (
        f"{subset_key}_max{max_pose_files}_seed{seed}"
        f"_conf{p['confidence_threshold']}_gap{p['max_interpolation_gap']}"
        f"_smooth{p['smoothing_window']}_win{p['window_length']}_stride{p['stride']}.npz"
    )
    return LIFT_CACHE_DIR / name


def _meta_to_arrays(meta: list[WindowMeta]) -> dict:
    return {
        "pose_files": np.array([m.pose_file for m in meta]),
        "person_ids": np.array([m.person_id for m in meta]),
        "camera_ids": np.array([m.camera_id for m in meta]),
        "video_ids": np.array([m.video_id for m in meta]),
        "start_frames": np.array([m.start_frame for m in meta], dtype=np.int64),
        "end_frames": np.array([m.end_frame for m in meta], dtype=np.int64),
    }


def _meta_from_arrays(cached) -> list[WindowMeta]:
    return [
        WindowMeta(
            pose_file=str(cached["pose_files"][i]),
            person_id=str(cached["person_ids"][i]),
            camera_id=str(cached["camera_ids"][i]),
            video_id=str(cached["video_ids"][i]),
            start_frame=int(cached["start_frames"][i]),
            end_frame=int(cached["end_frames"][i]),
        )
        for i in range(len(cached["camera_ids"]))
    ]


def lifted_windows_for_subset(
    root: Path,
    subset_key: str,
    preprocessing_cfg: dict,
    motionbert_model,
    max_pose_files: int | None = None,
    seed: int = 0,
    lift_batch_size: int = 32,
    return_metadata: bool = False,
):
    """The expensive, feature-flag-independent half of the 3D pipeline:
    builds every window for a subset (apply_normalization=False, since
    MotionBERT wants its own bounding-box normalization - see
    src/motionbert/lift.py:crop_scale, not this project's pelvis/torso
    one) and lifts each to 3D via MotionBERT, batched
    (lift_2d_to_3d_batch) rather than one window per forward call.

    Experiments C/D/E use the *same* pose-file sample and therefore the
    *same* lifted windows - only the feature engineering (and, for E,
    the model) on top differs (build_windows_for_subset_3d). Re-running
    a DSTformer forward pass per window for every experiment would
    multiply the already-expensive lifting cost for no reason, so the
    result is cached to disk (outputs/cache/motionbert_lift/, gitignored
    - pure derived data, never committed) keyed by subset + sampling +
    preprocessing parameters, and reused across experiments/reruns
    automatically.

    return_metadata=True on a cache written before metadata tracking
    existed triggers a cheap one-time backfill: metadata is
    deterministic given (subset, sampling, preprocessing), so it's
    recomputed via the fast collection pass only (no re-lifting) and
    saved back into the same cache file.

    Returns (lifted_keypoints, labels), or (lifted_keypoints, labels,
    metadata) if return_metadata=True:
      lifted_keypoints: (N, window_length, 17, 3) float32, H36M-17 order
      labels: (N,) int64
      metadata: list[WindowMeta], len N
    """
    cache_path = _lift_cache_path(subset_key, preprocessing_cfg, max_pose_files, seed)
    if cache_path.exists():
        cached = np.load(cache_path)
        if not return_metadata:
            return cached["keypoints"], cached["labels"]
        if "camera_ids" in cached:
            return cached["keypoints"], cached["labels"], _meta_from_arrays(cached)

        # Legacy cache (predates metadata) - backfill without re-lifting.
        _, _, window_meta = _collect_windows(
            root, subset_key, preprocessing_cfg, max_pose_files, seed,
            apply_normalization=False, desc=f"Backfilling window metadata ({subset_key})",
        )
        np.savez(cache_path, keypoints=cached["keypoints"], labels=cached["labels"], **_meta_to_arrays(window_meta))
        return cached["keypoints"], cached["labels"], window_meta

    window_keypoints, window_labels, window_meta = _collect_windows(
        root, subset_key, preprocessing_cfg, max_pose_files, seed,
        apply_normalization=False, desc=f"Collecting windows ({subset_key})",
    )

    if not window_keypoints:
        empty = np.empty((0, preprocessing_cfg["window_length"], 17, 3), dtype=np.float32)
        labels = np.empty((0,), dtype=np.int64)
        return (empty, labels, window_meta) if return_metadata else (empty, labels)

    # A/B must work before the optional MotionBERT code is installed.
    from src.motionbert.lift import lift_2d_to_3d_batch

    lifted_all = []
    for start in tqdm(range(0, len(window_keypoints), lift_batch_size), desc=f"Lifting 2D->3D ({subset_key})"):
        batch = np.stack(window_keypoints[start : start + lift_batch_size])
        lifted_all.append(lift_2d_to_3d_batch(motionbert_model, batch))
    lifted = np.concatenate(lifted_all).astype(np.float32)
    labels = np.array(window_labels, dtype=np.int64)

    LIFT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, keypoints=lifted, labels=labels, **_meta_to_arrays(window_meta))
    return (lifted, labels, window_meta) if return_metadata else (lifted, labels)


def build_windows_for_subset_3d(
    root: Path,
    subset_key: str,
    preprocessing_cfg: dict,
    motionbert_model,
    use_velocity: bool,
    use_acceleration: bool,
    use_geometry: bool,
    use_angles: bool,
    max_pose_files: int | None = None,
    seed: int = 0,
    return_metadata: bool = False,
):
    """The 3D analogue of build_windows_for_subset, for Experiments C/D/E:
    gets (cached) lifted 3D windows from lifted_windows_for_subset, then
    applies the feature flags on top (cheap - pure numpy, see
    src/features/motion_features.py's *_3d functions). Returns
    (features, labels), or (features, labels, metadata) if
    return_metadata=True - same shapes/semantics as
    build_windows_for_subset.
    """
    result = lifted_windows_for_subset(
        root, subset_key, preprocessing_cfg, motionbert_model, max_pose_files, seed, return_metadata=return_metadata
    )
    lifted, labels = result[0], result[1]
    meta = result[2] if return_metadata else None

    if len(lifted) == 0:
        features = np.empty((0, preprocessing_cfg["window_length"], 0), dtype=np.float32)
        return (features, labels, meta) if return_metadata else (features, labels)

    features = np.stack(
        [build_feature_vector_3d(window, use_velocity, use_acceleration, use_geometry, use_angles) for window in lifted]
    ).astype(np.float32)
    return (features, labels, meta) if return_metadata else (features, labels)
