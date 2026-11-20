"""
Explicit joint-index mapping between RetailS's COCO-17 keypoint order and
the Human3.6M (H36M) 17-joint order MotionBERT is trained on. The two
conventions are different skeletons, not just a relabeling - this module
is the single place that reconciles them, used by src/motionbert/lift.py
and scripts/validate_joint_mapping.py.

COCO-17 (RetailS, see src/data/retails.py / src/data/preprocessing.py):
    0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
    5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
    9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
    13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle

H36M-17 (MotionBERT). This order is NOT assumed - it's reverse-engineered
from two independent, real sources inside third_party/MotionBERT:
  1. lib/utils/vismo.py's own rendering code, which defines the skeleton
     bones to draw as
       joint_pairs = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],
                       [8,9],[8,11],[8,14],[9,10],[11,12],[12,13],
                       [14,15],[15,16]]
       joint_pairs_left  = [[8,11],[11,12],[12,13],[0,4],[4,5],[5,6]]
       joint_pairs_right = [[8,14],[14,15],[15,16],[0,1],[1,2],[2,3]]
     i.e. two leg chains and two arm chains off a central spine, with
     1-3/14-16 labeled "right" and 4-6/11-13 labeled "left".
  2. lib/data/dataset_wild.py's halpe2h36m() function, which maps the
     (textually-labeled) Halpe26 skeleton onto H36M-17 index-by-index.
     Halpe26 has explicit Hip-center/Neck/Head points COCO-17 lacks, so
     this confirms the exact semantic identity of every H36M-17 index:
         0 pelvis (<- Halpe Hip-center)   8  thorax (<- Halpe Neck)
         1 right_hip                      9  nose   (<- Halpe Nose)
         2 right_knee                     10 head   (<- Halpe Head)
         3 right_ankle                    11 left_shoulder
         4 left_hip                       12 left_elbow
         5 left_knee                      13 left_wrist
         6 left_ankle                     14 right_shoulder
         7 spine (<- midpoint(Neck, Hip)) 15 right_elbow
                                           16 right_wrist
     Both sources agree with each other.

13 of the 17 H36M joints are a direct, point-for-point reuse of a COCO-17
joint (all limb joints, plus the nose -> H36M 9). The remaining 4 have no
COCO-17 equivalent at all and are estimated the same way halpe2h36m()
estimates them from Halpe26's extra points:
    pelvis (H36M 0) = midpoint(left_hip, right_hip)
    thorax (H36M 8) = midpoint(left_shoulder, right_shoulder)   # neck proxy
    spine  (H36M 7) = midpoint(pelvis, thorax)                  # same formula halpe2h36m uses
    head   (H36M 10) = midpoint(left_eye, right_eye)            # COCO has no dedicated head/head-top point; closest available proxy
"""

from __future__ import annotations

import numpy as np

COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
COCO17 = {name: i for i, name in enumerate(COCO17_NAMES)}

H36M17_NAMES = [
    "pelvis", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle", "spine", "thorax",
    "nose", "head", "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
]
H36M17 = {name: i for i, name in enumerate(H36M17_NAMES)}

# H36M index -> COCO17 index, for the 13 directly-reused joints.
DIRECT_H36M_TO_COCO = {
    H36M17["right_hip"]: COCO17["right_hip"],
    H36M17["right_knee"]: COCO17["right_knee"],
    H36M17["right_ankle"]: COCO17["right_ankle"],
    H36M17["left_hip"]: COCO17["left_hip"],
    H36M17["left_knee"]: COCO17["left_knee"],
    H36M17["left_ankle"]: COCO17["left_ankle"],
    H36M17["nose"]: COCO17["nose"],
    H36M17["left_shoulder"]: COCO17["left_shoulder"],
    H36M17["left_elbow"]: COCO17["left_elbow"],
    H36M17["left_wrist"]: COCO17["left_wrist"],
    H36M17["right_shoulder"]: COCO17["right_shoulder"],
    H36M17["right_elbow"]: COCO17["right_elbow"],
    H36M17["right_wrist"]: COCO17["right_wrist"],
}

# H36M joints with no COCO-17 equivalent - estimated in coco17_to_h36m17().
DERIVED_H36M_JOINTS = {H36M17["pelvis"], H36M17["thorax"], H36M17["spine"], H36M17["head"]}

# Bone connectivity in H36M-17 order, for rendering - taken verbatim from
# third_party/MotionBERT/lib/utils/vismo.py so the skeleton drawn here
# matches the skeleton MotionBERT was trained on.
H36M17_BONES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8),
    (8, 9), (8, 11), (8, 14), (9, 10), (11, 12), (12, 13), (14, 15), (15, 16),
]
H36M17_BONES_LEFT = {(8, 11), (11, 12), (12, 13), (0, 4), (4, 5), (5, 6)}
H36M17_BONES_RIGHT = {(8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3)}


def coco17_to_h36m17(keypoints: np.ndarray) -> np.ndarray:
    """
    keypoints: (..., 17, C), C in {2, 3} (xy, or xy+confidence), COCO-17
    joint order.

    Returns (..., 17, C) in H36M-17 order. Derived confidence is the
    minimum of contributing confidences: a missing endpoint cannot create
    an apparently supported synthetic joint.
    """
    if keypoints.shape[-2] != 17:
        raise ValueError(f"Expected 17 joints, got shape {keypoints.shape}")

    out = np.zeros_like(keypoints)
    for h36m_idx, coco_idx in DIRECT_H36M_TO_COCO.items():
        out[..., h36m_idx, :] = keypoints[..., coco_idx, :]

    left_hip = keypoints[..., COCO17["left_hip"], :]
    right_hip = keypoints[..., COCO17["right_hip"], :]
    left_shoulder = keypoints[..., COCO17["left_shoulder"], :]
    right_shoulder = keypoints[..., COCO17["right_shoulder"], :]
    left_eye = keypoints[..., COCO17["left_eye"], :]
    right_eye = keypoints[..., COCO17["right_eye"], :]

    pelvis = (left_hip + right_hip) / 2.0
    thorax = (left_shoulder + right_shoulder) / 2.0
    spine = (pelvis + thorax) / 2.0
    head = (left_eye + right_eye) / 2.0
    if keypoints.shape[-1] == 3:
        pelvis[..., 2] = np.minimum(left_hip[..., 2], right_hip[..., 2])
        thorax[..., 2] = np.minimum(left_shoulder[..., 2], right_shoulder[..., 2])
        spine[..., 2] = np.minimum(pelvis[..., 2], thorax[..., 2])
        head[..., 2] = np.minimum(left_eye[..., 2], right_eye[..., 2])

    out[..., H36M17["pelvis"], :] = pelvis
    out[..., H36M17["thorax"], :] = thorax
    out[..., H36M17["spine"], :] = spine
    out[..., H36M17["head"], :] = head
    return out
