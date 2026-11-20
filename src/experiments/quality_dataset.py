"""Source-hashed, quality-gated windows shared by every new experiment."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from tqdm import tqdm
from src.data.preprocessing import build_windows
from src.data.quality import prepare_track, accepted_window, feature_mask, h36m_visibility
from src.data.retails import (SUBSET_DIRS, find_pose_files, load_pose_file, gt_dir_for_subset,
                              label_path_for_pose_file, load_anomaly_labels)
from src.experiments.dataset import WindowMeta, _sample_pose_files, _window_label
from src.features.motion_features import build_feature_vector, build_feature_vector_3d

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def implementation():
    return {p: sha256(ROOT / p) for p in ['src/data/quality.py', 'src/data/preprocessing.py',
        'src/data/retails.py', 'src/utils/joint_mapping.py', 'src/features/motion_features.py',
        'src/experiments/quality_dataset.py', 'src/motionbert/lift.py']}


def collect(root, subset, cfg, max_files):
    files = _sample_pose_files(find_pose_files(root / SUBSET_DIRS[subset]), max_files, cfg['sample_seed'])
    sources = {}
    for path in files:
        sources[str(path.relative_to(root))] = sha256(path)
        label_path = label_path_for_pose_file(path, gt_dir_for_subset(root / SUBSET_DIRS[subset]))
        if subset != 'train':
            if label_path is None:
                raise FileNotFoundError(path)
            sources[str(label_path.relative_to(root))] = sha256(label_path)
    key_cfg = {k: cfg[k] for k in ['coordinate_order', 'confidence_threshold', 'max_interpolation_gap',
        'smoothing_window', 'window_length', 'stride', 'max_normalized_coordinate', 'min_visible_fraction', 'sample_seed']}
    key = fingerprint(dict(sources=sources, config=key_cfg, implementation=implementation()))
    folder = ROOT / 'outputs/cache/quality_v1'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{subset}_{max_files}_{key[:16]}.npz'
    manifest_path = path.with_suffix('.json')
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['fingerprint'] != key or manifest['array_sha256'] != sha256(path):
            raise ValueError('Quality cache integrity mismatch')
        with np.load(path, allow_pickle=False) as cached:
            arrays = {k: cached[k] for k in cached.files}
        return arrays, manifest
    raw, normalized, masks, metadata = [], [], [], []
    counts = dict(candidate=0, retained=0, rejected=0, candidate_anomaly=0, rejected_anomaly=0)
    for source in tqdm(files, desc=f'Quality windows {subset}'):
        label_path = label_path_for_pose_file(source, gt_dir_for_subset(root / SUBSET_DIRS[subset]))
        labels = load_anomaly_labels(label_path)
        for track in load_pose_file(source):
            if not track.frames:
                continue
            prepared = prepare_track(track, cfg, normalize=False)
            for w in build_windows(prepared['frame_ids'], prepared['keypoints'], prepared['visibility_mask'], cfg['window_length'], cfg['stride']):
                if labels is not None and (w['start_frame'] < 0 or w['end_frame'] >= len(labels)):
                    raise ValueError(f'Out-of-range labels in {source}')
                label = _window_label(labels, w['start_frame'], w['end_frame'])
                counts['candidate'] += 1
                counts['candidate_anomaly'] += label
                start = int(np.searchsorted(prepared['frame_ids'], w['start_frame']))
                end = start + cfg['window_length']
                if not accepted_window(w, prepared['anchor_valid'][start:end], cfg):
                    counts['rejected'] += 1
                    counts['rejected_anomaly'] += label
                    continue
                raw.append(w['keypoints'].copy())
                normalized.append(prepared['normalized_keypoints'][start:end].copy())
                masks.append(w['visibility_mask'].copy())
                metadata.append(dict(pose_file=source.name, person_id=track.person_id, camera_id=track.camera_id,
                    video_id=track.video_id, start_frame=w['start_frame'], end_frame=w['end_frame'], label=label))
                counts['retained'] += 1
    if not raw:
        raise ValueError(f'No accepted windows for {subset}')
    arrays = dict(raw=np.stack(raw), normalized=np.stack(normalized), visibility=np.stack(masks),
                  labels=np.array([m['label'] for m in metadata], dtype=np.int64))
    assert np.isfinite(arrays['raw']).all() and np.isfinite(arrays['normalized']).all()
    assert np.max(np.abs(arrays['normalized'][..., :2])) <= cfg['max_normalized_coordinate']
    temp = path.with_suffix('.partial.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(path)
    manifest = dict(fingerprint=key, config=key_cfg, sources=sources, implementation=implementation(),
                    metadata=metadata, counts=counts, array_sha256=sha256(path), array_path=str(path))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(subset, counts, flush=True)
    return arrays, manifest


def lift(arrays, manifest, checkpoint, model, batch_size=8):
    from src.motionbert.lift import lift_2d_to_3d_batch
    key = fingerprint(dict(data=manifest['fingerprint'], checkpoint=sha256(checkpoint), implementation=implementation()))
    path = ROOT / f'outputs/cache/quality_v1/lift_{key[:20]}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            points = cached['points']
            if str(cached['fingerprint']) != key or sha256_array(points) != str(cached['array_hash']):
                raise ValueError('Lift cache integrity mismatch')
    else:
        chunks = []
        for start in tqdm(range(0, len(arrays['raw']), batch_size), desc='MotionBERT lift'):
            chunks.append(lift_2d_to_3d_batch(model, arrays['raw'][start:start + batch_size]))
        points = np.concatenate(chunks).astype(np.float32)
        temp = path.with_suffix('.partial.npz')
        np.savez_compressed(temp, points=points, fingerprint=np.array(key), array_hash=np.array(sha256_array(points)))
        temp.replace(path)
    if points.shape != arrays['raw'].shape or not np.isfinite(points).all():
        raise ValueError('Invalid lifted points')
    return points


def sha256_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def features(arrays, experiment, points_3d=None):
    visible = arrays['visibility']
    if experiment in ('a', 'b'):
        values = np.stack([build_feature_vector(p[..., :2], experiment == 'b') for p in arrays['normalized']])
    else:
        visible = h36m_visibility(visible)
        values = np.stack([build_feature_vector_3d(p, use_velocity=experiment in ('d', 'e'),
                           use_geometry=experiment in ('d', 'e')) for p in points_3d])
    masks = np.stack([feature_mask(v, experiment) for v in visible])
    if masks.shape != values.shape or not np.isfinite(values).all():
        raise ValueError('Invalid feature arrays')
    values = np.where(masks, values, 0).astype(np.float32)
    return values, masks


def window_metadata(manifest):
    return [WindowMeta(**{k: m[k] for k in WindowMeta.__dataclass_fields__}) for m in manifest['metadata']]
