"""Describe video-mask support without assigning actions to individual people."""
import numpy as np


def positive_intervals(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError('A nonempty binary video mask is required')
    edges = np.diff(np.r_[0, labels.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    (np.flatnonzero(edges == -1) - 1).tolist()))


def window_label_support(labels, start, end, people_per_frame):
    positive_intervals(labels)  # Validate the full source mask, including outside this window.
    labels = np.asarray(labels)
    people = np.asarray(people_per_frame)
    if people.shape != labels.shape or not np.isfinite(people).all() or (people < 0).any():
        raise ValueError('Aligned nonnegative person counts are required')
    if not 0 <= start <= end < len(labels):
        raise ValueError('Inclusive window is outside the video mask')
    mask = labels[start:end + 1] == 1
    counts = people[start:end + 1]
    return dict(positive_frames=int(mask.sum()), window_frames=int(len(mask)),
                positive_fraction=float(mask.mean()),
                positive_frames_with_multiple_people=int((mask & (counts > 1)).sum()),
                actor_identity_verified=False)
