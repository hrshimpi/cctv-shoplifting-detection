"""Window scoring variants and label-free score calibration for research_v2 (step 2).

All variants are computed from one saved per-window, per-frame, per-joint error
tensor ``err`` (N, T, 17) and its support ``mask`` (N, T, 17), so no extra model
inference is needed. Calibration uses *normal training-validation* windows only:

* ``log`` (reconstruction errors are positive and heavy tailed; monotone, so
  window AUC is unchanged, but smoothing/averaging behaves better);
* standardisation by the pooled validation mean/std, or, with ``camnorm``, by the
  validation mean/std of the window's own camera.

Calibrated scores are in "normal-validation standard deviations", so scores of
different variants and folds are on a comparable scale.
"""
from __future__ import annotations

import numpy as np

UPPER_BODY = list(range(5, 11))      # COCO shoulders, elbows, wrists
NO_FACE = list(range(5, 17))         # drop nose, eyes, ears (noisy, little action signal)


def _masked_mean(err, mask, axis):
    num = (err * mask).sum(axis)
    den = mask.sum(axis)
    return num, den


def window_score(err: np.ndarray, mask: np.ndarray, variant: str) -> np.ndarray:
    err = np.asarray(err, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if err.shape != mask.shape or err.ndim != 3:
        raise ValueError('err and mask must both be (N, T, 17)')
    m = mask.astype(np.float32)
    num, den = _masked_mean(err, m, (1, 2))
    mean = num / np.maximum(den, 1)
    if variant == 'mean':
        return mean
    if variant == 'max_joint':
        jn, jd = _masked_mean(err, m, 1)                       # (N, 17)
        per_joint = np.where(jd > 0, jn / np.maximum(jd, 1), -np.inf)
        return per_joint.max(1)
    if variant.startswith('topk'):
        k = int(variant[4:])
        fn, fd = _masked_mean(err, m, 2)                       # (N, T)
        per_frame = np.where(fd > 0, fn / np.maximum(fd, 1), -np.inf)
        top = -np.sort(-per_frame, axis=1)[:, :k]
        valid = np.isfinite(top)
        return np.where(valid.any(1), np.where(valid, top, 0).sum(1) / np.maximum(valid.sum(1), 1), mean)
    if variant in ('upper_body', 'no_face'):
        joints = UPPER_BODY if variant == 'upper_body' else NO_FACE
        sn, sd = _masked_mean(err[:, :, joints], m[:, :, joints], (1, 2))
        return np.where(sd > 0, sn / np.maximum(sd, 1), mean)   # fall back if no supported joint
    raise ValueError(f'Unknown scoring variant {variant}')


VARIANTS = ('mean', 'max_joint', 'topk3', 'topk5', 'upper_body', 'no_face')


class Calibrator:
    """Fit on normal validation window scores (+ their cameras); apply to any scores."""

    def __init__(self, log: bool = True, camnorm: bool = False, min_camera_windows: int = 30):
        self.log, self.camnorm, self.min_camera_windows = log, camnorm, min_camera_windows

    def _pre(self, s):
        s = np.asarray(s, dtype=np.float64)
        return np.log(np.maximum(s, 0) + 1e-8) if self.log else s

    def fit(self, val_scores, val_cameras):
        v = self._pre(val_scores)
        self.mu, self.sd = float(v.mean()), float(v.std() + 1e-8)
        self.per_camera = {}
        if self.camnorm:
            cams = np.asarray(val_cameras)
            for c in np.unique(cams):
                x = v[cams == c]
                if len(x) >= self.min_camera_windows:
                    self.per_camera[int(c)] = (float(x.mean()), float(x.std() + 1e-8))
        return self

    def transform(self, scores, cameras):
        s = self._pre(scores)
        out = (s - self.mu) / self.sd
        if self.camnorm:
            cams = np.asarray(cameras)
            for c, (mu, sd) in self.per_camera.items():
                sel = cams == c
                out[sel] = (s[sel] - mu) / sd
        return out
