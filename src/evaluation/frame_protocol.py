"""Frame-level scoring protocol for research_v2 (step 0b, smoothing for step 3).

Standard pose-VAD protocol (STG-NF, MoCoDAD, GEPC style), adapted to RetailS:

1. each person-window score is written to every frame the window covers; where
   windows of one person overlap, the frame keeps the maximum;
2. optional Gaussian smoothing of each person's frame series, only inside
   contiguous covered runs (never across a track gap);
3. frame score = maximum over persons;
4. optional Gaussian smoothing of the frame series inside contiguous scored runs;
5. frames with no scored person get ``fill`` (the subset minimum) so every
   labelled frame of the full test set is counted.
"""
from __future__ import annotations

import numpy as np


def _smooth_runs(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian filter over each contiguous finite run of ``values`` (NaN = gap)."""
    if not sigma or sigma <= 0:
        return values
    from scipy.ndimage import gaussian_filter1d
    out = values.copy()
    finite = np.isfinite(values)
    if not finite.any():
        return out
    edges = np.flatnonzero(np.diff(np.concatenate([[0], finite.astype(np.int8), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        out[a:b] = gaussian_filter1d(values[a:b].astype(np.float64), sigma, mode='nearest')
    return out


def frame_scores(scores, pose_file, track, start, end, lengths=None, sigma=0.0):
    """Return {pose_file: float array (NaN = unscored)}.

    ``lengths`` gives the number of frames per pose file (the label mask length);
    files without a given length use their last covered frame + 1.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite window scores')
    pose_file, track = np.asarray(pose_file), np.asarray(track)
    start, end = np.asarray(start, np.int64), np.asarray(end, np.int64)
    if np.any(end < start) or np.any(start < 0):
        raise ValueError('Invalid window frame range')
    lengths = dict(lengths or {})
    out = {}
    order = np.lexsort((start, track, pose_file))
    files = pose_file[order]
    boundaries = np.flatnonzero(files[1:] != files[:-1]) + 1
    for chunk in np.split(order, boundaries):
        if not len(chunk):
            continue
        stem = str(pose_file[chunk[0]])
        n = int(lengths.get(stem, end[chunk].max() + 1))
        if end[chunk].max() >= n:
            raise ValueError(f'Window beyond the label mask of {stem}')
        video = np.full(n, -np.inf)
        for t in np.unique(track[chunk]):
            sel = chunk[track[chunk] == t]
            lo, hi = start[sel].min(), end[sel].max() + 1
            person = np.full(hi - lo, -np.inf)
            for i in sel:
                seg = slice(start[i] - lo, end[i] - lo + 1)
                np.maximum(person[seg], scores[i], out=person[seg])
            person[~np.isfinite(person)] = np.nan
            person = _smooth_runs(person, sigma)
            person = np.where(np.isfinite(person), person, -np.inf)
            np.maximum(video[lo:hi], person, out=video[lo:hi])
        video[~np.isfinite(video)] = np.nan
        out[stem] = _smooth_runs(video, sigma)
    for stem, n in lengths.items():
        if stem not in out:
            out[stem] = np.full(int(n), np.nan)
    return out


def flatten(frame_dict, frame_labels, cameras, fill=None):
    """Concatenate per-video frame scores with labels, camera and video ids.

    ``fill`` replaces unscored frames; default = minimum scored value."""
    stems = sorted(frame_labels)
    scores = np.concatenate([frame_dict[s] for s in stems]) if stems else np.zeros(0)
    labels = np.concatenate([np.asarray(frame_labels[s]) for s in stems]).astype(int) if stems else np.zeros(0, int)
    if len(scores) != len(labels):
        raise ValueError('Frame scores and labels differ in length')
    scored = np.isfinite(scores)
    if fill is None:
        fill = float(np.min(scores[scored])) if scored.any() else 0.0
    camera = np.concatenate([np.full(len(frame_labels[s]), cameras[s]) for s in stems]) if stems else np.zeros(0, int)
    video = np.concatenate([np.full(len(frame_labels[s]), i) for i, s in enumerate(stems)]) if stems else np.zeros(0, int)
    return dict(scores=np.where(scored, scores, fill), labels=labels, camera=camera, video=video,
                scored=scored, stems=stems, fill=fill)
