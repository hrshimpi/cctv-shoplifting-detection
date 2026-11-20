"""Audit the exact A/B input sample and full test windows without changing data."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import yaml
from src.data.preprocessing import preprocess_track, build_windows
from src.data.retails import (SUBSET_DIRS, discover_retails_root, find_pose_files,
    load_pose_file, gt_dir_for_subset, label_path_for_pose_file, load_anomaly_labels)
from src.experiments.dataset import _sample_pose_files, _window_label


def main():
    cfg = yaml.safe_load((ROOT / 'configs/experiment_a.yaml').read_text())
    root = discover_retails_root(ROOT / 'data/retails')
    out = ROOT / 'outputs/logs/pose_quality_audit'
    out.mkdir(parents=True, exist_ok=True)
    rows, summary = [], {}
    for subset, folder in SUBSET_DIRS.items():
        files = _sample_pose_files(find_pose_files(root / folder), cfg['max_train_pose_files'] if subset == 'train' else None, cfg['seed'])
        start = len(rows)
        for path in files:
            mask_path = label_path_for_pose_file(path, gt_dir_for_subset(root / folder))
            if subset != 'train' and mask_path is None:
                raise FileNotFoundError(path)
            labels = load_anomaly_labels(mask_path)
            for track in load_pose_file(path):
                if not track.frames:
                    continue
                processed = preprocess_track(track, cfg['confidence_threshold'], cfg['max_interpolation_gap'], cfg['smoothing_window'])
                for w in build_windows(processed['frame_ids'], processed['keypoints'], processed['visibility_mask'], cfg['window_length'], cfg['stride']):
                    xy, visible = w['keypoints'][..., :2], w['visibility_mask']
                    if labels is not None and (w['start_frame'] < 0 or w['end_frame'] >= len(labels)):
                        raise ValueError(f'Window outside mask: {path}')
                    label = _window_label(labels, w['start_frame'], w['end_frame'])
                    if labels is not None:
                        assert label == int(np.any(labels[w['start_frame']:w['end_frame'] + 1] == 1))
                    torso = (xy[:, [5, 6]].mean(1) - xy[:, [11, 12]].mean(1))
                    anchors = visible[:, [5, 6, 11, 12]].all(1)
                    rows.append(dict(subset=subset, pose_file=path.name, person_id=track.person_id,
                        camera_id=track.camera_id, video_id=track.video_id,
                        start_frame=w['start_frame'], end_frame=w['end_frame'], label=label,
                        visible_fraction=float(visible.mean()), anchor_fraction=float(anchors.mean()),
                        missing_coordinate_absmax=float(np.abs(xy[~visible]).max()) if (~visible).any() else 0.,
                        coordinate_absmax=float(np.abs(xy).max()),
                        visible_coordinate_absmax=float(np.abs(xy[visible]).max()) if visible.any() else 0.,
                        torso_first_axis_dominant_fraction=float((np.abs(torso[anchors, 0]) > np.abs(torso[anchors, 1])).mean()) if anchors.any() else None))
        group = rows[start:]
        summary[subset] = dict(pose_files=len(files), windows=len(group),
            low_visibility_windows=sum(r['visible_fraction'] < .5 for r in group),
            incomplete_anchor_windows=sum(r['anchor_fraction'] < 1 for r in group),
            coordinate_over_10_windows=sum(r['coordinate_absmax'] > 10 for r in group),
            missing_coordinate_over_10_windows=sum(r['missing_coordinate_absmax'] > 10 for r in group),
            anomaly_windows=sum(r['label'] for r in group),
            window_label_alignment='All window endpoints and ANY-frame labels verified')
        print(subset, json.dumps(summary[subset]), flush=True)
    training = [r for r in rows if r['subset'] == 'train']
    errors = np.load(ROOT / 'outputs/checkpoints/experiment_a/train_reconstruction_errors.npy')
    assert len(training) == len(errors)
    for row, error in zip(training, errors):
        row['saved_a_reconstruction_error'] = float(error)
    fields = list(rows[0])
    with (out / 'window_quality.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    worst = np.argsort(errors)[-10:][::-1]
    report = dict(scope='Exact 50-file A/B train sample and complete test subsets; audit only',
        subsets=summary, worst_normal_windows=[training[i] for i in worst],
        normalized_coordinate_quantiles=dict(zip(['p50', 'p95', 'p99', 'max'],
            np.quantile([r['coordinate_absmax'] for r in training], [.5, .95, .99, 1]).tolist())),
        finding='The legacy model feature path discards visibility masks; missing coordinates still affect smoothing, normalization and reconstruction loss.',
        axis_caution='Axis dominance is descriptive. The authors specify XYC; no axis swap is inferred or applied from poses alone.',
        label_caution='Video-frame masks are not person-level action annotations; an overlapping bystander window may receive label 1.')
    (out / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Audit saved:', out)


if __name__ == '__main__':
    main()
