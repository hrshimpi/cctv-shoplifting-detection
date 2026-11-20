"""Exploratory video-frame aggregation; unsupported frames remain NaN."""
import numpy as np


def aggregate_windows(scores, metadata, lengths=None):
    if len(scores) != len(metadata) or not np.isfinite(scores).all():
        raise ValueError('Aligned finite window scores are required')
    if lengths is None:
        lengths = {}
        for meta in metadata:
            name = meta['pose_file']
            lengths[name] = max(lengths.get(name, 0), int(meta['end_frame']) + 1)
    arrays = {name: np.full(length, np.nan, dtype=np.float32) for name, length in lengths.items()}
    for value, meta in zip(scores, metadata):
        name, start, end = meta['pose_file'], int(meta['start_frame']), int(meta['end_frame'])
        if name not in arrays or start < 0 or end < start or end >= len(arrays[name]):
            raise ValueError('Window outside its source frame range')
        arrays[name][start:end + 1] = np.fmax(arrays[name][start:end + 1], float(value))
    return arrays
