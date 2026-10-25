"""
Pose preprocessing: per-person-track cleaning, normalization, and
temporal windowing, operating on the PersonTrack objects src/data/retails.py
loads from the real RetailS pose JSON.

COCO-17 keypoint indices (standard order, matches the dataset's paper):
  0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
  5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
  9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
  13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
"""

from __future__ import annotations

import numpy as np

from src.data.retails import PersonTrack

NUM_KEYPOINTS = 17
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6


def track_to_array(track: PersonTrack) -> tuple[np.ndarray, np.ndarray]:
    """Sort a track's frames by frame_id and stack into arrays.
    Returns (frame_ids (T,), keypoints (T, 17, 3))."""
    frame_ids = np.array(track.frame_ids_sorted, dtype=np.int64)
    keypoints = np.stack([track.frames[fid].keypoints for fid in frame_ids])
    return frame_ids, keypoints


def visibility_mask(keypoints: np.ndarray, confidence_threshold: float) -> np.ndarray:
    """(T, 17) bool mask: True where a joint's confidence clears the threshold."""
    return keypoints[..., 2] >= confidence_threshold


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges of consecutive True values in `flags`."""
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(flags):
        if v and not in_run:
            start, in_run = i, True
        elif not v and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(flags)))
    return runs


def interpolate_short_gaps(
    frame_ids: np.ndarray, keypoints: np.ndarray, mask: np.ndarray, max_gap: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fill short runs of low-confidence/missing frames (<= max_gap
    consecutive real frame_ids, not array positions) per joint via linear
    interpolation. Gaps longer than max_gap, and leading/trailing gaps
    with no two-sided support, are left out of the returned mask rather
    than extrapolated.

    Returns (keypoints_filled, still_missing_mask).
    """
    T = len(frame_ids)
    filled = keypoints.copy()
    still_missing = ~mask.copy()

    for j in range(NUM_KEYPOINTS):
        valid = mask[:, j]
        if valid.sum() < 2:
            continue  # not enough support to interpolate at all

        for gap_start, gap_end in _runs(~valid):
            left = gap_start - 1
            right = gap_end
            if left < 0 or right >= T:
                continue  # leading/trailing gap: no two-sided support

            frame_gap = frame_ids[right] - frame_ids[left]
            if frame_gap > max_gap + 1:
                continue  # gap too long in real frame terms, leave missing

            for coord in range(2):  # interpolate x, y only - not confidence
                filled[gap_start:gap_end, j, coord] = np.interp(
                    frame_ids[gap_start:gap_end],
                    [frame_ids[left], frame_ids[right]],
                    [keypoints[left, j, coord], keypoints[right, j, coord]],
                )
            still_missing[gap_start:gap_end, j] = False

    return filled, still_missing


def smooth(keypoints: np.ndarray, window: int = 3) -> np.ndarray:
    """Optional light temporal smoothing: centered moving average over
    `window` frames, per joint/coordinate along the frame axis."""
    if window <= 1:
        return keypoints
    kernel = np.ones(window) / window
    pad = window // 2
    smoothed = keypoints.copy()
    for j in range(keypoints.shape[1]):
        for coord in range(2):
            padded = np.pad(keypoints[:, j, coord], (pad, pad), mode="edge")
            smoothed[:, j, coord] = np.convolve(padded, kernel, mode="valid")[: len(keypoints)]
    return smoothed


def normalize_position(keypoints: np.ndarray) -> np.ndarray:
    """Center every frame's keypoints on the pelvis (midpoint of the two hips)."""
    pelvis = (keypoints[:, LEFT_HIP, :2] + keypoints[:, RIGHT_HIP, :2]) / 2.0
    centered = keypoints.copy()
    centered[:, :, :2] -= pelvis[:, None, :]
    return centered


def normalize_scale(keypoints: np.ndarray, min_scale: float = 1e-3) -> np.ndarray:
    """Scale each frame so torso length (shoulder-midpoint to
    hip-midpoint) is 1.0. Falls back to shoulder width, then hip width,
    if torso length is degenerate, and leaves a frame unscaled if all
    three are unusable (keypoints too close together / zeroed out)."""
    shoulder_mid = (keypoints[:, LEFT_SHOULDER, :2] + keypoints[:, RIGHT_SHOULDER, :2]) / 2.0
    hip_mid = (keypoints[:, LEFT_HIP, :2] + keypoints[:, RIGHT_HIP, :2]) / 2.0
    torso_length = np.linalg.norm(shoulder_mid - hip_mid, axis=1)

    shoulder_width = np.linalg.norm(
        keypoints[:, LEFT_SHOULDER, :2] - keypoints[:, RIGHT_SHOULDER, :2], axis=1
    )
    hip_width = np.linalg.norm(keypoints[:, LEFT_HIP, :2] - keypoints[:, RIGHT_HIP, :2], axis=1)

    scale = torso_length.copy()
    use_shoulder = scale < min_scale
    scale[use_shoulder] = shoulder_width[use_shoulder]
    use_hip = scale < min_scale
    scale[use_hip] = hip_width[use_hip]
    scale[scale < min_scale] = 1.0

    scaled = keypoints.copy()
    scaled[:, :, :2] /= scale[:, None, None]
    return scaled


def preprocess_track(
    track: PersonTrack,
    confidence_threshold: float = 0.3,
    max_interpolation_gap: int = 5,
    smoothing_window: int = 3,
    apply_normalization: bool = True,
) -> dict:
    """Run the per-track cleaning pipeline: sort, mask, interpolate,
    smooth, and (unless apply_normalization=False) pelvis-center + scale.
    Returns frame_ids, keypoints, and a visibility mask (True = usable
    joint after interpolation).

    apply_normalization=False skips the pelvis-centering/torso-scaling
    steps, leaving keypoints in the same (pixel) coordinate space the
    pose file used. The BiGRU branch (src/experiments/dataset.py) always
    wants normalization on (the default); src/motionbert/lift.py wants it
    off, since MotionBERT expects its own whole-sequence bounding-box
    normalization (see src/motionbert/lift.py:crop_scale), not this
    project's pelvis/torso convention.
    """
    frame_ids, keypoints = track_to_array(track)

    # RetailS occasionally ships a literal NaN confidence value (x, y
    # still present and finite) rather than a low-but-real number -
    # confirmed, not hypothetical: e.g. RetailS_test_realworld's
    # 01_0257.json, person 1, frame_ids 9-11, every joint. Treat NaN the
    # same as RetailS's own explicit zero-confidence entries (no
    # information) rather than letting it propagate: `>= confidence_threshold`
    # below already treats NaN as invisible (NaN comparisons are always
    # False), so this only changes behavior for consumers that read the
    # confidence value itself downstream - notably MotionBERT
    # (src/motionbert/lift.py), which takes confidence as a real input
    # channel and, unlike the 2D BiGRU branch (which discards this column
    # before building features), has no tolerance for NaN: its temporal
    # attention mixes all frames in a window together, so a NaN at even
    # one frame silently contaminates the model's output for the entire
    # window, not just that frame.
    keypoints[:, :, 2] = np.nan_to_num(keypoints[:, :, 2], nan=0.0)

    mask = visibility_mask(keypoints, confidence_threshold)
    keypoints, still_missing = interpolate_short_gaps(frame_ids, keypoints, mask, max_interpolation_gap)
    # Adjacent array rows can be far apart in video time. Never let a
    # smoothing kernel mix poses from separate contiguous track segments.
    for start, end in _contiguous_runs(frame_ids):
        keypoints[start:end] = smooth(keypoints[start:end], smoothing_window)
    if apply_normalization:
        keypoints = normalize_position(keypoints)
        keypoints = normalize_scale(keypoints)

    return {
        "person_id": track.person_id,
        "camera_id": track.camera_id,
        "video_id": track.video_id,
        "part": track.part,
        "frame_ids": frame_ids,
        "keypoints": keypoints,
        "visibility_mask": ~still_missing,
    }


def _contiguous_runs(frame_ids: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) array-index ranges where frame_ids increase by
    exactly 1 each step - i.e. no gap in the underlying track."""
    if len(frame_ids) == 0:
        return []
    runs = []
    start = 0
    for i in range(1, len(frame_ids)):
        if frame_ids[i] != frame_ids[i - 1] + 1:
            runs.append((start, i))
            start = i
    runs.append((start, len(frame_ids)))
    return runs


def build_windows(
    frame_ids: np.ndarray,
    keypoints: np.ndarray,
    visibility_mask: np.ndarray,
    window_length: int,
    stride: int,
) -> list[dict]:
    """Slice a preprocessed track into fixed-length temporal windows over
    *contiguous* frame_id runs only - a window is never built across a
    real gap in the track (e.g. a person leaving and re-entering frame)."""
    windows = []
    for start_idx, end_idx in _contiguous_runs(frame_ids):
        run_len = end_idx - start_idx
        if run_len < window_length:
            continue
        for w_start in range(start_idx, end_idx - window_length + 1, stride):
            w_end = w_start + window_length
            windows.append(
                {
                    "start_frame": int(frame_ids[w_start]),
                    "end_frame": int(frame_ids[w_end - 1]),
                    "keypoints": keypoints[w_start:w_end],
                    "visibility_mask": visibility_mask[w_start:w_end],
                }
            )
    return windows
