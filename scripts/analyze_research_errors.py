"""Describe imported-run errors without tuning on held-out labels."""
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/matplotlib'))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.experiments.artifacts import run_paths
from src.experiments.quality_dataset import sha256


def main():
    base = ROOT / 'outputs/logs/research_v1'
    frame_report = json.loads((base / 'frame_diagnostic/report.json').read_text())
    if not frame_report['complete']:
        raise ValueError('Finish frame diagnostics first')
    out = base / 'error_analysis'
    out.mkdir(parents=True, exist_ok=True)
    rows, quantiles, histories, calibration_by_run, model_hashes = [], [], {}, {}, {}
    for seed in (0, 1, 2):
        name = f'expanded_a_seed{seed}'
        model, log = run_paths(name)
        report = json.loads((log / 'evaluation.json').read_text())
        if frame_report['runs'][name]['checkpoint_sha256'] != report['checkpoint_sha256']:
            raise ValueError('Frame and window diagnostics differ')
        model_hashes[name] = sha256(model / 'model.pt')
        frame = pd.read_csv(log / 'window_predictions.csv', dtype={'pose_file':str, 'camera_id':str})
        frame['run'] = name
        rows.append(frame)
        histories[name] = pd.read_csv(log / 'training_log.csv')
        calibration = np.load(model / 'validation_errors.npy', allow_pickle=False)
        calibration_by_run[name] = calibration
        threshold = report['metrics']['combined']['threshold']
        groups = [('normal_validation', calibration)]
        for subset in ['test_realworld', 'test_staged']:
            for label, kind in [(0, 'normal'), (1, 'anomaly')]:
                groups.append((f'{subset}_{kind}', frame[(frame.subset == subset) & (frame.label == label)].score.to_numpy()))
        for cohort, values in groups:
            quantiles.append(dict(run=name, cohort=cohort, windows=len(values), threshold=threshold,
                fraction_above_threshold=float((values >= threshold).mean()),
                **{f'p{int(q*100)}':float(np.quantile(values, q)) for q in [.05,.25,.5,.75,.95,.99]}))
    windows = pd.concat(rows, ignore_index=True)
    pd.DataFrame(quantiles).to_csv(out / 'score_quantiles.csv', index=False)
    windows.groupby(['run','subset','camera_id','outcome']).size().rename('windows').reset_index().to_csv(out / 'window_confusion_by_camera.csv', index=False)
    cameras = pd.read_csv(base / 'frame_diagnostic/per_camera_metrics.csv', dtype={'camera_id':str})
    cameras = cameras[cameras.run.str.startswith('expanded_a_seed')]
    camera_summary = cameras.groupby(['subset','camera_id']).agg(
        score_coverage=('score_coverage','first'), scored_normal_frames=('num_normal','first'),
        scored_anomaly_frames=('num_anomaly','first'), unscored_anomaly_frames=('unscored_anomaly_frames','first'),
        recall_mean=('recall','mean'), recall_sd=('recall','std'), false_positive_rate_mean=('false_positive_rate','mean'),
        roc_auc_mean=('roc_auc','mean'), all_annotated_anomaly_recall_mean=('all_annotated_anomaly_recall','mean')).reset_index()
    camera_summary.to_csv(out / 'frame_camera_summary.csv', index=False)
    # Reconstruct simple pose support/movement diagnostics in exact window order.
    support_rows = []
    seed0 = windows[windows.run == 'expanded_a_seed0']
    for subset in ['test_realworld', 'test_staged']:
        _, log = run_paths('expanded_a_seed0')
        report = json.loads((log / 'evaluation.json').read_text())
        fingerprint = report['metrics'][subset]['data_fingerprint']
        path = ROOT / f'outputs/cache/quality_v1/{subset}_None_{fingerprint[:16]}.json'
        manifest = json.loads(path.read_text())
        array_path = path.with_suffix('.npz')
        if sha256(array_path) != manifest['array_sha256']:
            raise ValueError('Changed pose cache')
        group = seed0[seed0.subset == subset]
        identity = ['pose_file','person_id','start_frame','end_frame']
        for row, meta in zip(group.to_dict('records'), manifest['metadata']):
            if any(str(row[k]) != str(meta[k]) for k in identity):
                raise ValueError('Pose diagnostics window order differs')
        with np.load(array_path, allow_pickle=False) as arrays:
            visible = arrays['visibility']
            xy = arrays['normalized'][..., :2]
            valid = visible[:,1:] & visible[:,:-1]
            difference = np.linalg.norm(xy[:,1:] - xy[:,:-1], axis=-1)
            movement = (difference * valid).sum((1,2)) / np.maximum(valid.sum((1,2)), 1)
        for row, visibility, motion in zip(group.to_dict('records'), visible.mean((1,2)), movement):
            support_rows.append(dict(**row, visible_fraction=float(visibility), normalized_motion_per_frame=float(motion)))
    support = pd.DataFrame(support_rows)
    support.to_csv(out / 'seed0_pose_diagnostics.csv', index=False)
    support_summary = support.groupby(['subset','outcome']).agg(windows=('score','size'),
        median_score=('score','median'), median_visible_fraction=('visible_fraction','median'),
        median_normalized_motion_per_frame=('normalized_motion_per_frame','median')).reset_index()
    support_summary.to_csv(out / 'pose_diagnostics_summary.csv', index=False)
    plots = []
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for ax, seed in zip(axes, (0,1,2)):
        name = f'expanded_a_seed{seed}'
        group = windows[windows.run == name]
        cohorts = [('Normal validation', calibration_by_run[name]),
                   ('Test normal',group[group.label == 0].score.to_numpy()),
                   ('Test anomaly',group[group.label == 1].score.to_numpy())]
        upper = float(np.quantile(np.concatenate([v for _,v in cohorts]), .99))
        for label, values in cohorts:
            values = np.sort(values)
            ax.step(values, np.arange(1,len(values)+1)/len(values), label=label)
        ax.axvline(group.threshold.iloc[0], color='black', linestyle='--', label='Normal-validation p95')
        ax.set(xlim=(0,upper), ylim=(0,1.01), xlabel='Masked reconstruction MSE', ylabel='Cumulative fraction', title=f'Seed {seed}; x-axis clipped at pooled p99')
    axes[0].legend(fontsize=8); fig.tight_layout()
    path = out / 'score_distributions.png'; fig.savefig(path,dpi=150); plt.close(fig)
    plots.append(str(path.relative_to(ROOT)))
    fig, axes = plt.subplots(1,3,figsize=(14,3.5))
    for ax,(name,history) in zip(axes,histories.items()):
        ax.plot(history.epoch,history.train_loss,label='Train')
        ax.plot(history.epoch,history.validation_loss,label='Normal validation')
        ax.set(title=name,xlabel='Epoch',ylabel='Masked MSE')
    axes[0].legend(); fig.tight_layout()
    path = out / 'expanded_learning_curves.png'; fig.savefig(path,dpi=150); plt.close(fig)
    plots.append(str(path.relative_to(ROOT)))
    metrics = [frame_report['runs'][f'expanded_a_seed{s}']['metrics']['combined'] for s in (0,1,2)]
    reference = metrics[0]
    recall = float(np.mean([r['recall'] for r in metrics]))
    all_recall = float(np.mean([r['all_annotated_anomaly_recall'] for r in metrics]))
    auc = float(np.mean([r['roc_auc'] for r in metrics]))
    test_fpr = float(windows[windows.label == 0].prediction.mean())
    findings = [
        f"Expanded A scored {reference['scored_frames']:,}/{reference['total_annotated_frames']:,} annotated frames ({reference['score_coverage']:.1%}); {reference['unscored_anomaly_frames']:,} anomalous frames remain unscored.",
        f"Across three seeds, mean frame recall on scored frames is {recall:.1%}, recall over all annotated positive frames is {all_recall:.1%}, and mean frame ROC-AUC is {auc:.3f}.",
        f"The normal-validation p95 window threshold flags {test_fpr:.1%} of retained test-normal windows, versus approximately 5% of calibration windows. This documents calibration mismatch, not its cause.",
        'Window and frame rankings remain weak; lowering the threshold can trade false alarms for recall but cannot improve ROC-AUC for fixed scores.',
        'Per-camera metrics and pose-motion summaries are descriptive test diagnostics. Camera differences, occlusion and pose-only limitations are hypotheses, not established causes.',
        'Source RGB/action identities are unavailable for RetailS error strips. A video anomaly mask does not establish which person is acting anomalously.'
    ]
    next_experiment = ('Compare expanded A against expanded B (2D positions plus joint velocities) on the identical 100-file cohort, masks, source-video split, 15 epochs and seeds 0/1/2. Fix the protocol before running; use normal validation for checkpoint/calibration. Retain held-out results as diagnostics and obtain a fresh labeled validation/confirmation set before iterative anomaly-based tuning. No new training is started by this analysis.')
    if all(f'expanded_b_seed{s}' in frame_report['runs'] for s in (0,1,2)):
        next_experiment = ('Expanded B is completed. Review expanded_ab_comparison/comparison.md, then audit pose/label alignment and obtain independently reviewed person/time annotations on separate retail clips before further architecture or threshold tuning. Preserve A/B as baselines; no new training is started.')
    report = dict(checkpoint_sha256=model_hashes, findings=findings, next_experiment=next_experiment,
        frame_summary=dict(mean_recall=recall, mean_all_annotated_anomaly_recall=all_recall, mean_roc_auc=auc,
                           scored_frames=reference['scored_frames'], annotated_frames=reference['total_annotated_frames']),
        plots=plots, limitations=['Three seeds share one data split; overlapping windows and frames are correlated.',
            'Test-guided repeated experimentation risks overfitting the benchmark; fresh confirmation data is required.',
            'Movement is displacement per source frame, not calibrated physical velocity.'])
    (out / 'report.json').write_text(json.dumps(report,indent=2))
    lines = ['# Expanded model error analysis',''] + ['- '+f for f in findings]
    lines += ['', '## Frame metrics by camera (mean over seeds)', '', camera_summary.to_string(index=False),
              '', '## Pose support and motion (seed 0)', '', support_summary.to_string(index=False),
              '', '## Next controlled experiment', '', next_experiment]
    (out / 'analysis.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(findings)); print(camera_summary.to_string(index=False))


if __name__ == '__main__':
    main()
