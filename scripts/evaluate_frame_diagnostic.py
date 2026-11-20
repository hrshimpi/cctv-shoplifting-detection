"""Secondary frame-level diagnostic, retaining the predeclared window evaluation.

Aggregate by maximum over retrospective overlapping windows and people. Calibrate
with the same aggregation on NORMAL VALIDATION windows; never tune on test masks.
"""
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from src.data.retails import discover_retails_root, SUBSET_DIRS, gt_dir_for_subset
from src.evaluation.frame_scores import aggregate_windows
from src.evaluation.metrics import choose_threshold, evaluate
from src.experiments.quality_dataset import window_metadata, sha256
from src.experiments.splits import split_by_video
from src.experiments.artifacts import run_paths, research_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    root = discover_retails_root(ROOT / 'data/retails')
    required = research_runs()
    reports, missing, camera_rows = {}, [], []
    output = ROOT / 'outputs/logs/research_v1/frame_diagnostic'
    output.mkdir(parents=True, exist_ok=True)
    for name in required:
        folder, log = run_paths(name)
        evaluation = log / 'evaluation.json'
        if not evaluation.exists():
            missing.append(name); continue
        checkpoint = torch.load(folder / 'model.pt', weights_only=False, map_location='cpu')
        original = json.loads(evaluation.read_text())
        if sha256(folder / 'model.pt') != original['checkpoint_sha256']:
            raise ValueError('Stale model evaluation')
        limit = checkpoint['config']['expanded_train_files'] if name.startswith('expanded') else checkpoint['config']['matched_train_files']
        key = checkpoint['training_data_fingerprint']
        path = ROOT / f'outputs/cache/quality_v1/train_{limit}_{key[:16]}.json'
        training = json.loads(path.read_text())
        _, indices, split = split_by_video(window_metadata(training), checkpoint['config']['val_fraction'], checkpoint['config']['split_seed'])
        if split != checkpoint['split']:
            raise ValueError('Normal calibration split differs from checkpoint')
        calibration_errors = np.load(folder / 'validation_errors.npy')
        calibration_metadata = [training['metadata'][i] for i in indices]
        aggregated = aggregate_windows(calibration_errors, calibration_metadata)
        normal_scores = np.concatenate([a[np.isfinite(a)] for a in aggregated.values()])
        threshold = choose_threshold(normal_scores, checkpoint['config']['threshold_percentile'])
        predictions = pd.read_csv(evaluation.parent / 'window_predictions.csv', dtype={'pose_file': str})
        results, labels_all, scores_all, subset_all = {}, [], [], []
        for subset in ['test_realworld', 'test_staged']:
            labels_by_file = {}
            for p in sorted(gt_dir_for_subset(root / SUBSET_DIRS[subset]).glob('*.npy')):
                labels = np.load(p)
                if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
                    raise ValueError('Invalid frame ground truth')
                labels_by_file[p.stem + '.json'] = labels
            group = predictions[predictions.subset == subset]
            arrays = aggregate_windows(group.score.to_numpy(), group.to_dict('records'),
                {k: len(v) for k, v in labels_by_file.items()})
            for camera in sorted({p.split('_')[0] for p in labels_by_file}):
                names = [p for p in labels_by_file if p.split('_')[0] == camera]
                cy = np.concatenate([labels_by_file[p] for p in names])
                cs = np.concatenate([arrays[p] for p in names])
                cv = np.isfinite(cs)
                camera_rows.append(dict(run=name, subset=subset, camera_id=camera,
                    **evaluate(cy[cv], cs[cv], threshold).as_dict(), total_annotated_frames=len(cy),
                    scored_frames=int(cv.sum()), score_coverage=float(cv.mean()),
                    unscored_anomaly_frames=int(((cy == 1) & ~cv).sum()),
                    all_annotated_anomaly_recall=float(((cy == 1) & cv & (cs >= threshold)).sum() / max(1, (cy == 1).sum()))))
            labels = np.concatenate(list(labels_by_file.values()))
            scores = np.concatenate([arrays[k] for k in labels_by_file])
            valid = np.isfinite(scores)
            metrics = evaluate(labels[valid], scores[valid], threshold).as_dict()
            detected = int(((labels == 1) & valid & (scores >= threshold)).sum())
            results[subset] = dict(**metrics, total_annotated_frames=len(labels), scored_frames=int(valid.sum()),
                score_coverage=float(valid.mean()), unscored_normal_frames=int(((labels == 0) & ~valid).sum()),
                unscored_anomaly_frames=int(((labels == 1) & ~valid).sum()),
                all_annotated_anomaly_recall=detected / int((labels == 1).sum()))
            labels_all.append(labels); scores_all.append(scores); subset_all.extend([subset] * len(labels))
        labels, scores = np.concatenate(labels_all), np.concatenate(scores_all)
        valid = np.isfinite(scores)
        metrics = evaluate(labels[valid], scores[valid], threshold).as_dict()
        detected = int(((labels == 1) & valid & (scores >= threshold)).sum())
        results['combined'] = dict(**metrics, total_annotated_frames=len(labels), scored_frames=int(valid.sum()),
            score_coverage=float(valid.mean()), annotated_normal_frames=int((labels == 0).sum()),
            annotated_anomaly_frames=int((labels == 1).sum()),
            unscored_normal_frames=int(((labels == 0) & ~valid).sum()),
            unscored_anomaly_frames=int(((labels == 1) & ~valid).sum()),
            all_annotated_anomaly_recall=detected / int((labels == 1).sum()))
        np.savez_compressed(output / f'{name}.npz', labels=labels, scores=scores, subsets=np.array(subset_all))
        reports[name] = dict(metrics=results, normal_validation_scored_frames=len(normal_scores), threshold=threshold,
                             checkpoint_sha256=original['checkpoint_sha256'])
        print(name, json.dumps(results['combined']), flush=True)
    if missing and not args.allow_incomplete:
        raise FileNotFoundError('Missing research evaluations: ' + ', '.join(missing))
    report = dict(complete=not missing, missing=missing, runs=reports,
        protocol='Exploratory diagnostic: retrospective maximum over overlapping windows and people; p95 calibrated on aggregated NORMAL VALIDATION scores.',
        limitations=[
            'Secondary diagnostic added to expose window-selection bias; it does not replace the predeclared window results.',
            'Unscored frames remain unknown; conditional metrics exclude them, while all-annotated-anomaly recall treats them as misses.',
            'Scores apply to whole windows retrospectively, not causal instant-by-instant alarms.',
            'Frame masks describe video-level anomaly intervals, not the identity or action of an individual.'])
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(camera_rows).to_csv(output / 'per_camera_metrics.csv', index=False)


if __name__ == '__main__':
    main()
