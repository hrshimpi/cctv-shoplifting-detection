"""Versioned, visibility-aware pose preparation; legacy experiments stay reproducible."""
import numpy as np
from src.data.preprocessing import track_to_array, interpolate_short_gaps, _contiguous_runs
from src.features.motion_features import MOTION_JOINT_INDICES, MOTION_JOINT_INDICES_3D
from src.utils.joint_mapping import coco17_to_h36m17


def prepare_track(track, cfg, normalize=True, coordinate_order=None):
    frame_ids, points = track_to_array(track)
    points = points.astype(np.float32, copy=True)
    order = coordinate_order or cfg['coordinate_order']
    if order not in ('xy', 'yx'):
        raise ValueError('coordinate_order must be explicit: xy or yx')
    if order == 'yx':
        points[..., :2] = points[..., [1, 0]]
    threshold = cfg['confidence_threshold']
    visible = np.isfinite(points).all(-1) & (points[..., 2] >= threshold)
    points[..., 2] = np.where(visible, np.clip(points[..., 2], 0, 1), 0)
    points[..., :2] = np.where(visible[..., None], points[..., :2], 0)
    points, missing = interpolate_short_gaps(frame_ids, points, visible, cfg['max_interpolation_gap'])
    usable = ~missing
    points[..., 2] = np.where(usable, np.maximum(points[..., 2], threshold), 0)
    # Smooth with available support only. Invalid observations never influence a neighbor.
    width = cfg['smoothing_window']
    if width < 1 or width % 2 != 1:
        raise ValueError('smoothing_window must be positive and odd')
    kernel, pad = np.ones(width), width // 2
    for start, end in _contiguous_runs(frame_ids):
        for joint in range(17):
            weights = usable[start:end, joint].astype(float)
            denominator = np.convolve(np.pad(weights, (pad, pad), mode='edge'), kernel, 'valid')
            for axis in range(2):
                values = points[start:end, joint, axis] * weights
                numerator = np.convolve(np.pad(values, (pad, pad), mode='edge'), kernel, 'valid')
                points[start:end, joint, axis] = numerator / np.maximum(denominator, 1)
    xy = points[..., :2]
    pelvis = xy[:, [11, 12]].mean(1)
    torso = np.linalg.norm(xy[:, [5, 6]].mean(1) - pelvis, axis=1)
    anchor_valid = usable[:, [5, 6, 11, 12]].all(1) & (torso > 1e-3)
    scaled = (xy - pelvis[:, None]) / np.maximum(torso[:, None, None], 1e-3)
    scaled[~usable] = 0
    anchor_valid &= np.max(np.abs(scaled), axis=(1, 2)) <= cfg['max_normalized_coordinate']
    points[..., :2][~usable] = 0
    points[..., 2][~usable] = 0
    normalized = points.copy()
    normalized[..., :2] = scaled
    return dict(frame_ids=frame_ids, keypoints=normalized if normalize else points,
                normalized_keypoints=normalized, visibility_mask=usable, anchor_valid=anchor_valid)


def accepted_window(window, anchor_valid, cfg):
    return (bool(np.all(anchor_valid)) and
            float(window['visibility_mask'].mean()) >= cfg['min_visible_fraction'])


def h36m_visibility(visible):
    # Mapping averages XY, but availability requires every contributing joint.
    full = np.repeat(visible[..., None], 3, axis=-1).astype(np.float32)
    mapped = coco17_to_h36m17(full)[..., 0]
    return mapped >= 1


def feature_mask(visible, experiment):
    """Mask missing positions and differences without two-sided temporal support."""
    is3d = experiment in ('c', 'd', 'e')
    dim = 3 if is3d else 2
    joints = MOTION_JOINT_INDICES_3D if is3d else MOTION_JOINT_INDICES
    parts = [np.repeat(visible[..., None], dim, -1).reshape(len(visible), -1)]
    if experiment in ('b', 'd', 'e'):
        valid = np.zeros((len(visible), len(joints)), dtype=bool)
        valid[1:] = visible[1:, joints] & visible[:-1, joints]
        parts.append(np.repeat(valid[..., None], dim, -1).reshape(len(visible), -1))
    if experiment in ('d', 'e'):
        # build_feature_vector_3d: left/right wrist-pelvis, wrist-thorax, wrist-wrist.
        parts.append(np.stack([visible[:, 13] & visible[:, 0], visible[:, 16] & visible[:, 0],
            visible[:, 13] & visible[:, 8], visible[:, 16] & visible[:, 8],
            visible[:, 13] & visible[:, 16]], axis=-1))
    return np.concatenate(parts, -1)


def masked_error(reconstruction, target, mask):
    """Per-window MSE over supported target coordinates only."""
    if reconstruction.shape != target.shape or mask.shape != target.shape:
        raise ValueError('Reconstruction, target and mask must have identical shapes')
    denominator = mask.sum(dim=(1, 2))
    if (denominator == 0).any():
        raise ValueError('Cannot score an entirely unsupported window')
    return (((reconstruction - target) ** 2) * mask).sum(dim=(1, 2)) / denominator
