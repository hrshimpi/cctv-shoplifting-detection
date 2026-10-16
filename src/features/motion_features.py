"""
Motion features derived from a preprocessed pose window: per-joint
velocity (and, optionally, acceleration) for the limb joints - wrists,
elbows, shoulders, hips, knees, ankles. Operates on the (x, y) output of
src/data/preprocessing.py (already centered on the pelvis and scaled by
torso length), never on raw/unnormalized coordinates.

COCO-17 indices (see src/data/preprocessing.py for how these were
verified, not just assumed):
  5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
  9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
  13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
The 5 facial keypoints (nose, eyes, ears) carry no shoplifting-relevant
motion signal and are excluded from the velocity/acceleration joint set
- deliberately, not an oversight.

The *_3d functions below are the same idea applied to MotionBERT's
lifted (x, y, z) output (src/motionbert/lift.py), which is in H36M-17
order (src/utils/joint_mapping.py), not COCO-17 - the joint index
constants are therefore separate (MOTION_JOINT_INDICES_3D vs.
MOTION_JOINT_INDICES). See README.md "Limitations" for a real, checked
finding: MotionBERT's z (depth) output on this dataset is weaker and
less evenly distributed across the body than x/y - z-derived values here
(the z component of position/velocity/acceleration, and any geometry
distance/angle that involves a depth difference) should be read with
that in mind, not as a fully trustworthy full-depth reconstruction.
"""

import numpy as np

from src.utils.joint_mapping import H36M17

MOTION_JOINTS = {
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}
MOTION_JOINT_INDICES = list(MOTION_JOINTS.values())
NUM_MOTION_JOINTS = len(MOTION_JOINT_INDICES)


def compute_velocity(keypoints_xy: np.ndarray) -> np.ndarray:
    """position(t) - position(t-1) for the motion joints.

    keypoints_xy: (T, 17, 2) array of (x, y) only (no confidence column).
    Returns (T, len(MOTION_JOINT_INDICES), 2). The first frame has no
    t-1, so it's zero-filled rather than dropped, to keep the sequence
    length equal to the input window length.
    """
    joints = keypoints_xy[:, MOTION_JOINT_INDICES, :]
    velocity = np.zeros_like(joints)
    velocity[1:] = joints[1:] - joints[:-1]
    return velocity


def compute_acceleration(velocity: np.ndarray) -> np.ndarray:
    """velocity(t) - velocity(t-1), same zero-fill convention as
    compute_velocity for the first frame."""
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    return acceleration


def build_feature_vector(
    keypoints_xy: np.ndarray,
    use_velocity: bool = False,
    use_acceleration: bool = False,
) -> np.ndarray:
    """Assemble the per-frame feature vector for one window.

    keypoints_xy: (T, 17, 2) normalized pose (see module docstring).
    Returns (T, D) where D depends on which flags are set:
      pose only:                17*2 = 34
      + velocity:                + 12*2 = 24  -> 58
      + acceleration (needs velocity): + 12*2 = 24 -> 82
    Acceleration without velocity isn't a meaningful combination (there's
    no velocity to difference), so use_acceleration implies velocity is
    computed and included even if use_velocity was left False.
    """
    T = keypoints_xy.shape[0]
    parts = [keypoints_xy.reshape(T, -1)]

    velocity = None
    if use_velocity or use_acceleration:
        velocity = compute_velocity(keypoints_xy)
        parts.append(velocity.reshape(T, -1))

    if use_acceleration:
        acceleration = compute_acceleration(velocity)
        parts.append(acceleration.reshape(T, -1))

    return np.concatenate(parts, axis=1)


def feature_dim(use_velocity: bool = False, use_acceleration: bool = False) -> int:
    dim = 17 * 2
    if use_velocity or use_acceleration:
        dim += NUM_MOTION_JOINTS * 2
    if use_acceleration:
        dim += NUM_MOTION_JOINTS * 2
    return dim


# --- 3D (MotionBERT-lifted, H36M-17 order) ----------------------------

MOTION_JOINTS_3D = {
    "left_shoulder": H36M17["left_shoulder"], "right_shoulder": H36M17["right_shoulder"],
    "left_elbow": H36M17["left_elbow"], "right_elbow": H36M17["right_elbow"],
    "left_wrist": H36M17["left_wrist"], "right_wrist": H36M17["right_wrist"],
    "left_hip": H36M17["left_hip"], "right_hip": H36M17["right_hip"],
    "left_knee": H36M17["left_knee"], "right_knee": H36M17["right_knee"],
    "left_ankle": H36M17["left_ankle"], "right_ankle": H36M17["right_ankle"],
}
MOTION_JOINT_INDICES_3D = list(MOTION_JOINTS_3D.values())

NUM_GEOMETRY_DISTANCES = 5  # left/right wrist-to-pelvis, left/right wrist-to-thorax, wrist-to-wrist
NUM_JOINT_ANGLES = 4  # left/right elbow, left/right knee


def compute_velocity_3d(keypoints_xyz: np.ndarray) -> np.ndarray:
    """Same idea as compute_velocity, in 3D: position(t) - position(t-1)
    for the motion joints.

    keypoints_xyz: (T, 17, 3) H36M-17 order (MotionBERT's lifted output).
    Returns (T, len(MOTION_JOINT_INDICES_3D), 3), first frame zero-filled.
    """
    joints = keypoints_xyz[:, MOTION_JOINT_INDICES_3D, :]
    velocity = np.zeros_like(joints)
    velocity[1:] = joints[1:] - joints[:-1]
    return velocity


def compute_acceleration_3d(velocity: np.ndarray) -> np.ndarray:
    """velocity(t) - velocity(t-1), same zero-fill convention as compute_acceleration_3d's first frame."""
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    return acceleration


def _body_scale_3d(keypoints_xyz: np.ndarray, min_scale: float = 1e-3) -> np.ndarray:
    """Per-frame torso length (pelvis-to-thorax 3D distance), the 3D
    analogue of src/data/preprocessing.py's 2D torso-length scale - used
    to normalize geometry distances by body size. Returns (T,)."""
    pelvis = keypoints_xyz[:, H36M17["pelvis"], :]
    thorax = keypoints_xyz[:, H36M17["thorax"], :]
    scale = np.linalg.norm(thorax - pelvis, axis=-1)
    return np.maximum(scale, min_scale)


def compute_geometry_distances(keypoints_xyz: np.ndarray) -> np.ndarray:
    """Body-relative distances, each normalized by that frame's torso
    length so they're comparable across people/distance-from-camera:
    left_wrist-to-pelvis, right_wrist-to-pelvis, left_wrist-to-thorax,
    right_wrist-to-thorax, left_wrist-to-right_wrist. These are meant to
    capture hands approaching the body/pockets (concealment) and hands
    coming together.

    keypoints_xyz: (T, 17, 3) H36M-17 order. Returns (T, 5).
    """
    scale = _body_scale_3d(keypoints_xyz)
    pelvis = keypoints_xyz[:, H36M17["pelvis"], :]
    thorax = keypoints_xyz[:, H36M17["thorax"], :]
    left_wrist = keypoints_xyz[:, H36M17["left_wrist"], :]
    right_wrist = keypoints_xyz[:, H36M17["right_wrist"], :]

    distances = np.stack(
        [
            np.linalg.norm(left_wrist - pelvis, axis=-1),
            np.linalg.norm(right_wrist - pelvis, axis=-1),
            np.linalg.norm(left_wrist - thorax, axis=-1),
            np.linalg.norm(right_wrist - thorax, axis=-1),
            np.linalg.norm(left_wrist - right_wrist, axis=-1),
        ],
        axis=-1,
    )
    return distances / scale[:, None]


def _angle_at_vertex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at vertex b formed by points a-b-c, in radians. a, b, c: (T, 3)."""
    v1 = a - b
    v2 = c - b
    denom = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8
    cos_theta = np.clip(np.sum(v1 * v2, axis=-1) / denom, -1.0, 1.0)
    return np.arccos(cos_theta)


def compute_joint_angles(keypoints_xyz: np.ndarray) -> np.ndarray:
    """Elbow angles (shoulder-elbow-wrist) and knee angles
    (hip-knee-ankle), both sides, in radians - scale-invariant by
    construction, so not normalized by body size.

    keypoints_xyz: (T, 17, 3) H36M-17 order.
    Returns (T, 4): [left_elbow, right_elbow, left_knee, right_knee].
    """
    H = H36M17
    left_elbow = _angle_at_vertex(
        keypoints_xyz[:, H["left_shoulder"], :], keypoints_xyz[:, H["left_elbow"], :], keypoints_xyz[:, H["left_wrist"], :]
    )
    right_elbow = _angle_at_vertex(
        keypoints_xyz[:, H["right_shoulder"], :], keypoints_xyz[:, H["right_elbow"], :], keypoints_xyz[:, H["right_wrist"], :]
    )
    left_knee = _angle_at_vertex(
        keypoints_xyz[:, H["left_hip"], :], keypoints_xyz[:, H["left_knee"], :], keypoints_xyz[:, H["left_ankle"], :]
    )
    right_knee = _angle_at_vertex(
        keypoints_xyz[:, H["right_hip"], :], keypoints_xyz[:, H["right_knee"], :], keypoints_xyz[:, H["right_ankle"], :]
    )
    return np.stack([left_elbow, right_elbow, left_knee, right_knee], axis=-1)


def build_feature_vector_3d(
    keypoints_xyz: np.ndarray,
    use_velocity: bool = False,
    use_acceleration: bool = False,
    use_geometry: bool = False,
    use_angles: bool = False,
) -> np.ndarray:
    """Assemble the per-frame 3D feature vector for one window (the 3D
    analogue of build_feature_vector).

    keypoints_xyz: (T, 17, 3) H36M-17 order, MotionBERT's lifted output.
    Returns (T, D) where D depends on which flags are set:
      3D pose only:                        17*3 = 51
      + velocity:                           + 12*3 = 36  -> 87
      + acceleration (needs velocity):      + 12*3 = 36  -> 123
      + geometry (5 body-relative distances, normalized by torso length): + 5
      + angles (4 joint angles, radians):   + 4
    Same convention as build_feature_vector: use_acceleration implies
    velocity is computed and included even if use_velocity was left False.
    """
    T = keypoints_xyz.shape[0]
    parts = [keypoints_xyz.reshape(T, -1)]

    velocity = None
    if use_velocity or use_acceleration:
        velocity = compute_velocity_3d(keypoints_xyz)
        parts.append(velocity.reshape(T, -1))

    if use_acceleration:
        acceleration = compute_acceleration_3d(velocity)
        parts.append(acceleration.reshape(T, -1))

    if use_geometry:
        parts.append(compute_geometry_distances(keypoints_xyz))

    if use_angles:
        parts.append(compute_joint_angles(keypoints_xyz))

    return np.concatenate(parts, axis=1)


def feature_dim_3d(
    use_velocity: bool = False,
    use_acceleration: bool = False,
    use_geometry: bool = False,
    use_angles: bool = False,
) -> int:
    dim = 17 * 3
    if use_velocity or use_acceleration:
        dim += NUM_MOTION_JOINTS * 3
    if use_acceleration:
        dim += NUM_MOTION_JOINTS * 3
    if use_geometry:
        dim += NUM_GEOMETRY_DISTANCES
    if use_angles:
        dim += NUM_JOINT_ANGLES
    return dim
