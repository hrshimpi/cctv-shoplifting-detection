"""research_v2 data layer: a per-file track cache plus fast, vectorised windowing.

Expensive work (JSON parsing + quality cleaning via ``prepare_track``) runs once
per pose file and is cached as ``.npz`` under ``outputs/cache/research_v2``.
Windows of any length/stride are then cut from the cached tracks in seconds, so
step 5 (window length) never re-parses the dataset.

Window construction and acceptance replicate ``quality_dataset.collect``
(research_v1) exactly: contiguous frame runs only, every frame anchor-valid,
mean joint visibility >= ``min_visible_fraction``; window label = 1 if any frame
of the source video mask is positive. Only real RetailS files are read.
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.data.quality import prepare_track
from src.data.retails import (SUBSET_DIRS, find_pose_files, gt_dir_for_subset, label_path_for_pose_file,
                              load_anomaly_labels, load_pose_file, parse_filename)
from src.experiments.dataset import _sample_pose_files

ROOT = Path(__file__).resolve().parents[2]
CLEAN_KEYS = ['coordinate_order', 'confidence_threshold', 'max_interpolation_gap', 'smoothing_window',
              'max_normalized_coordinate']
IMPLEMENTATION_FILES = ['src/data/quality.py', 'src/data/preprocessing.py', 'src/data/retails.py',
                        'src/experiments/v2_data.py']


def _sha(path: Path) -> str:
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def clean_key(cfg: dict) -> str:
    payload = dict(cfg={k: cfg[k] for k in CLEAN_KEYS},
                   impl={p: _sha(ROOT / p) for p in IMPLEMENTATION_FILES if p != 'src/experiments/v2_data.py'})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def camera_int(camera_id) -> int:
    return int(camera_id) if str(camera_id).isdigit() else -1


# --------------------------------------------------------------------------------------------
# Track cache
# --------------------------------------------------------------------------------------------

def _clean_file(job):
    source, target, cfg, dtype = job
    target = Path(target)
    if target.exists():
        return str(target)
    frame_ids, xy, vis, anchor, offsets, persons = [], [], [], [], [0], []
    for track in load_pose_file(Path(source)):
        if not track.frames:
            continue
        p = prepare_track(track, cfg, normalize=False)
        frame_ids.append(p['frame_ids'].astype(np.int32))
        xy.append(p['normalized_keypoints'][..., :2].astype(dtype))
        vis.append(p['visibility_mask'].astype(bool))
        anchor.append(p['anchor_valid'].astype(bool))
        offsets.append(offsets[-1] + len(p['frame_ids']))
        persons.append(str(track.person_id))
    empty = not frame_ids
    arrays = dict(
        frame_ids=np.concatenate(frame_ids) if not empty else np.zeros(0, np.int32),
        xy=np.concatenate(xy) if not empty else np.zeros((0, 17, 2), dtype),
        vis=np.concatenate(vis) if not empty else np.zeros((0, 17), bool),
        anchor=np.concatenate(anchor) if not empty else np.zeros(0, bool),
        offsets=np.asarray(offsets, np.int64), persons=np.asarray(persons, dtype='<U32'))
    tmp = target.with_name(target.stem + '.partial.npz')
    np.savez(tmp, **arrays)
    tmp.replace(target)
    return str(target)


def build_track_cache(root: Path, subset: str, cfg: dict, cache_dir: Path, files=None, workers=None,
                      progress=True) -> list[Path]:
    """Clean every pose file of ``subset`` once; returns the per-file cache paths (sorted by source)."""
    files = sorted(files if files is not None else find_pose_files(root / SUBSET_DIRS[subset]))
    folder = Path(cache_dir) / f'tracks_{clean_key(cfg)}' / subset
    folder.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if subset == 'train' else np.float32
    jobs, targets = [], []
    for f in files:
        target = folder / f'{f.stem}__{f.stat().st_size}.npz'
        targets.append(target)
        if not target.exists():
            jobs.append((str(f), str(target), cfg, dtype))
    if jobs:
        workers = workers or max(1, min(8, os.cpu_count() or 1))
        it = None
        if workers > 1:
            with ProcessPoolExecutor(workers) as pool:
                it = pool.map(_clean_file, jobs, chunksize=1)
                _drain(it, len(jobs), f'clean {subset}', progress)
        else:
            _drain(map(_clean_file, jobs), len(jobs), f'clean {subset}', progress)
    return targets


def _drain(iterator, total, desc, progress):
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=total, desc=desc)
        except ImportError:
            pass
    for _ in iterator:
        pass


# --------------------------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------------------------

@dataclass
class WindowSet:
    xy: np.ndarray            # (N, L, 17, 2) float
    vis: np.ndarray           # (N, L, 17) bool
    labels: np.ndarray        # (N,) int64, any-positive rule (research_v1)
    pos_frac: np.ndarray      # (N,) float32 fraction of positive frames
    pose_file: np.ndarray     # (N,) str, source file stem (one timeline)
    person: np.ndarray        # (N,) str
    camera: np.ndarray        # (N,) int
    video: np.ndarray         # (N,) str
    start: np.ndarray         # (N,) int, first frame id
    end: np.ndarray           # (N,) int, last frame id (inclusive)
    track: np.ndarray         # (N,) int, globally unique track id
    window_length: int
    stride: int
    subset: str
    frame_labels: dict = field(default_factory=dict)   # pose_file stem -> per-frame mask (test only)
    counts: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.labels)

    def take(self, idx):
        idx = np.asarray(idx)
        keep = {k: getattr(self, k)[idx] for k in ('xy', 'vis', 'labels', 'pos_frac', 'pose_file', 'person',
                                                     'camera', 'video', 'start', 'end', 'track')}
        files = set(keep['pose_file'].tolist())
        return WindowSet(**keep, window_length=self.window_length, stride=self.stride, subset=self.subset,
                         frame_labels={k: v for k, v in self.frame_labels.items() if k in files},
                         counts=dict(self.counts))

    def meta_records(self):
        from types import SimpleNamespace
        return [SimpleNamespace(pose_file=f, person_id=p, camera_id=str(c), video_id=v, start_frame=int(s), end_frame=int(e))
                for f, p, c, v, s, e in zip(self.pose_file, self.person, self.camera, self.video, self.start, self.end)]


def window_starts(frame_ids, anchor, vis, length, stride, min_visible):
    """Accepted window start indices inside one track (array positions)."""
    n = len(frame_ids)
    if n < length:
        return np.zeros(0, np.int64)
    breaks = np.flatnonzero(np.diff(frame_ids) != 1) + 1
    bounds = np.concatenate([[0], breaks, [n]])
    bad = np.concatenate([[0], np.cumsum(~anchor)])
    seen = np.concatenate([[0], np.cumsum(vis.sum(1))])
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < length:
            continue
        s = np.arange(a, b - length + 1, stride)
        ok = (bad[s + length] - bad[s] == 0) & ((seen[s + length] - seen[s]) / (length * 17.0) >= min_visible)
        out.append(s[ok])
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def _candidate_count(frame_ids, offsets, length, stride):
    total = 0
    for t in range(len(offsets) - 1):
        f = frame_ids[offsets[t]:offsets[t + 1]]
        if len(f) < length:
            continue
        breaks = np.flatnonzero(np.diff(f) != 1) + 1
        bounds = np.concatenate([[0], breaks, [len(f)]])
        runs = bounds[1:] - bounds[:-1]
        total += int(np.sum(np.where(runs >= length, (runs - length) // stride + 1, 0)))
    return total


def _file_windows(cache_path, length, stride, min_visible):
    with np.load(cache_path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    per_track = []
    for t in range(len(d['offsets']) - 1):
        a, b = d['offsets'][t], d['offsets'][t + 1]
        s = window_starts(d['frame_ids'][a:b], d['anchor'][a:b], d['vis'][a:b], length, stride, min_visible)
        per_track.append((a + s, t))
    return d, per_track


def build_windows_v2(root: Path, subset: str, cfg: dict, cache_dir: Path, length: int, stride: int,
                     max_files=None, max_windows=None, seed=0, workers=None, progress=True) -> WindowSet:
    """Windows for ``subset``. ``max_files`` reproduces research_v1's file sample (same seed and sampler).
    ``max_windows`` uniformly subsamples accepted windows across *all* chosen files (memory cap)."""
    all_files = find_pose_files(root / SUBSET_DIRS[subset])
    files = sorted(_sample_pose_files(all_files, max_files, cfg['sample_seed']))
    caches = build_track_cache(root, subset, cfg, cache_dir, files, workers, progress)
    gt = gt_dir_for_subset(root / SUBSET_DIRS[subset])
    min_visible = cfg['min_visible_fraction']

    # Pass 1: accepted window positions per file (cheap), so a cap can sample uniformly.
    plan = []
    for f, c in zip(files, caches):
        d, per_track = _file_windows(c, length, stride, min_visible)
        plan.append((f, c, [(s, t) for s, t in per_track if len(s)], _candidate_count(d['frame_ids'], d['offsets'], length, stride)))
    sizes = np.array([sum(len(s) for s, _ in p[2]) for p in plan], np.int64)
    total = int(sizes.sum())
    chosen = None
    if max_windows is not None and total > max_windows:
        rng = np.random.default_rng(seed)
        chosen = np.zeros(total, bool)
        chosen[rng.choice(total, size=max_windows, replace=False)] = True
    dtype = np.float16 if subset == 'train' else np.float32
    xy, vis, labels, pos_frac, cols = [], [], [], [], {k: [] for k in ('pose_file', 'person', 'camera', 'video', 'start', 'end', 'track')}
    frame_labels, cursor, track_base = {}, 0, 0
    counts = dict(candidate=0, retained=0, rejected=0, capped_to=max_windows)
    for (f, c, tracks, n_candidates), size in zip(plan, sizes):
        camera, video, _ = parse_filename(f)
        mask = load_anomaly_labels(label_path_for_pose_file(f, gt)) if subset != 'train' else None
        if subset != 'train':
            if mask is None:
                raise FileNotFoundError(f'Missing frame mask for {f}')
            frame_labels[f.stem] = np.asarray(mask).astype(np.int8)
        counts['candidate'] += n_candidates
        counts['retained'] += int(size)
        if size == 0:
            track_base += 1000
            continue
        with np.load(c, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        L = length
        offs = np.arange(L)
        for starts, t in tracks:
            keep = np.ones(len(starts), bool) if chosen is None else chosen[cursor:cursor + len(starts)]
            cursor += len(starts)
            starts = starts[keep]
            if not len(starts):
                continue
            idx = starts[:, None] + offs[None]
            xy.append(d['xy'][idx].astype(dtype))
            vis.append(d['vis'][idx])
            first, last = d['frame_ids'][starts], d['frame_ids'][starts + L - 1]
            if mask is not None:
                if first.min() < 0 or last.max() >= len(mask):
                    raise ValueError(f'Out-of-range labels in {f}')
                cum = np.concatenate([[0], np.cumsum(np.asarray(mask) == 1)])
                npos = cum[last + 1] - cum[first]
                labels.append((npos > 0).astype(np.int64))
                pos_frac.append((npos / (last - first + 1)).astype(np.float32))
            else:
                labels.append(np.zeros(len(starts), np.int64))
                pos_frac.append(np.zeros(len(starts), np.float32))
            n = len(starts)
            cols['pose_file'] += [f.stem] * n
            cols['person'] += [str(d['persons'][t])] * n
            cols['camera'] += [camera_int(camera)] * n
            cols['video'] += [str(video)] * n
            cols['start'].append(first.astype(np.int64))
            cols['end'].append(last.astype(np.int64))
            cols['track'] += [track_base + t] * n
        track_base += max(1000, len(d['offsets']))
    if subset != 'train':
        # Every labelled video is part of the frame-level universe, even with zero retained windows.
        for f in files:
            if f.stem not in frame_labels:
                frame_labels[f.stem] = np.asarray(load_anomaly_labels(label_path_for_pose_file(f, gt))).astype(np.int8)
    if not xy:
        raise ValueError(f'No accepted windows for {subset}')
    labels = np.concatenate(labels)
    counts['rejected'] = counts['candidate'] - counts['retained']
    counts['kept'] = int(len(labels))
    counts['kept_anomaly'] = int(labels.sum())
    return WindowSet(xy=np.concatenate(xy), vis=np.concatenate(vis), labels=labels, pos_frac=np.concatenate(pos_frac),
                     pose_file=np.asarray(cols['pose_file']), person=np.asarray(cols['person']),
                     camera=np.asarray(cols['camera'], np.int64), video=np.asarray(cols['video']),
                     start=np.concatenate(cols['start']), end=np.concatenate(cols['end']),
                     track=np.asarray(cols['track'], np.int64), window_length=length, stride=stride,
                     subset=subset, frame_labels=frame_labels, counts=counts)


def video_cameras(windows: WindowSet) -> dict:
    """pose_file stem -> camera int for every labelled video (from the filename)."""
    out = {}
    for stem in windows.frame_labels:
        camera, _, _ = parse_filename(Path(stem + '.json'))
        out[stem] = camera_int(camera)
    return out
