"""Frozen clip descriptors and duplicate-aware exploratory split validation."""
import numpy as np


def clip_descriptor(embeddings, include_changes=True):
    values = np.asarray(embeddings)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Finite chronological embeddings for at least two frames are required')
    parts = [values.mean(0), values.std(0)]
    if include_changes:
        parts.append(np.abs(np.diff(values, axis=0)).mean(0))
    return np.concatenate(parts).astype(np.float32)


def verify_split(rows):
    seen = {}
    for row in rows:
        split = row['split']
        if split not in ['train', 'validation', 'test']:
            raise ValueError('Unknown split')
        for key in ['clip_id', 'source_sha256', 'sampled_sha256', 'duplicate_group']:
            token = (key, row[key])
            if token in seen and seen[token] != split:
                raise ValueError('Shared source/duplicate group crosses splits')
            seen[token] = split
    for split in ['train', 'validation', 'test']:
        if {int(r['label']) for r in rows if r['split'] == split} != {0, 1}:
            raise ValueError('Each split requires both classes')


def portable_probability(features, mean, scale, coefficient, intercept):
    if not np.isfinite(features).all() or (np.asarray(scale) <= 0).any():
        raise ValueError('Finite features and positive feature scales are required')
    logits = ((features - mean) / scale) @ np.asarray(coefficient).reshape(-1) + float(np.asarray(intercept).reshape(-1)[0])
    # Stable sigmoid for portable inference without loading a pickle.
    result = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    result[positive] = 1 / (1 + np.exp(-logits[positive]))
    negative = np.exp(logits[~positive]); result[~positive] = negative / (1 + negative)
    return result
