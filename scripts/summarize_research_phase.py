"""Verify completed research artifacts and write measured comparisons and limitations."""
import argparse
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/matplotlib'))
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from src.experiments.quality_dataset import sha256
from src.evaluation.metrics import evaluate
from src.experiments.artifacts import run_paths, research_runs

IDENTITY = ['subset', 'pose_file', 'person_id', 'camera_id', 'video_id', 'start_frame', 'end_frame']


def predictions(path):
    return pd.read_csv(path, dtype={k: str for k in IDENTITY[:5]}).set_index(IDENTITY).sort_index()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def error_strips(runs, frames, out):
    """Trace selected FP/FN rows to the exact quality-cache window, without RGB/action claims."""
    from scripts.validate_research_inputs import plot_supported, COCO17_BONES
    examples = []
    for name in ['matched_d_seed0', 'expanded_a_seed0', 'expanded_b_seed0']:
        if name not in frames:
            continue
        report, frame = runs[name], frames[name].reset_index()
        for subset in ['test_realworld', 'test_staged']:
            key = report['metrics'][subset]['data_fingerprint']
            manifest_path = ROOT / f'outputs/cache/quality_v1/{subset}_None_{key[:16]}.json'
            manifest = json.loads(manifest_path.read_text())
            array_path = Path(manifest['array_path'])
            if manifest['array_sha256'] != sha256(array_path):
                raise ValueError('Changed quality cache during error reconstruction')
            lookup = {(m['pose_file'], str(m['person_id']), m['start_frame'], m['end_frame']): i
                      for i, m in enumerate(manifest['metadata'])}
            with np.load(array_path, allow_pickle=False) as arrays:
                normalized = arrays['normalized']
            for outcome in ['FN', 'FP']:
                selected = frame[(frame.subset == subset) & (frame.outcome == outcome)]
                if selected.empty:
                    continue
                row = selected.sort_values('score', ascending=outcome == 'FN').iloc[0]
                index = lookup[(row.pose_file, str(row.person_id), int(row.start_frame), int(row.end_frame))]
                points = normalized[index]
                offsets = [0, len(points) // 3, 2 * len(points) // 3, len(points) - 1]
                fig, axes = plt.subplots(1, 4, figsize=(12, 4))
                for ax, offset in zip(axes, offsets):
                    plot_supported(ax, points[offset], COCO17_BONES, {}, f'Source frame {int(row.start_frame) + offset}')
                fig.suptitle(f"{name}: {outcome}, {row.pose_file}, person {row.person_id}\n"
                    f"score {row.score:.4g}, threshold {row.threshold:.4g}; video-mask label {row.label}", fontsize=10)
                fig.tight_layout()
                path = out / f'{name}_{subset}_{outcome.lower()}.png'
                fig.savefig(path, dpi=150); plt.close(fig)
                examples.append(dict(run=name, subset=subset, outcome=outcome, pose_file=row.pose_file,
                    person_id=str(row.person_id), camera_id=str(row.camera_id), start_frame=int(row.start_frame),
                    end_frame=int(row.end_frame), score=float(row.score), threshold=float(row.threshold),
                    video_mask_label=int(row.label), plot=str(path.relative_to(ROOT))))
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'outputs/logs/research_v1'
    out.mkdir(parents=True, exist_ok=True)
    required = research_runs()
    runs, frames, missing = {}, {}, []
    table, cameras = [], []
    reference = None
    for name in required:
        checkpoint_folder, log = run_paths(name)
        path = log / 'evaluation.json'
        if not path.exists():
            missing.append(name); continue
        report = json.loads(path.read_text())
        model_path = checkpoint_folder / 'model.pt'
        if report['checkpoint_sha256'] != sha256(model_path):
            raise ValueError(f'Stale evaluation: {name}')
        frame = predictions(path.parent / 'window_predictions.csv')
        if not frame.index.is_unique or not np.isfinite(frame.score).all():
            raise ValueError(f'Duplicate windows or nonfinite scores in {name}')
        if reference is None:
            reference = frame
        if not frame.index.equals(reference.index) or not np.array_equal(frame.label, reference.label):
            raise ValueError('Research models do not use identical test windows/labels')
        runs[name], frames[name] = report, frame
        combined = report['metrics']['combined']
        candidate_anomaly = sum(report['metrics'][s]['coverage']['candidate_anomaly'] for s in ['test_realworld', 'test_staged'])
        detected = int(((frame.label == 1) & (frame.prediction == 1)).sum())
        table.append(dict(run=name, experiment=report['experiment'], seed=report['seed'],
            training_files=report['config']['expanded_train_files'] if name.startswith('expanded') else report['config']['matched_train_files'],
            train_windows=report['split']['train_windows'], validation_windows=report['split']['validation_windows'],
            feature_dim=report['feature_dim'], best_epoch=report['best_epoch'],
            **{k: combined[k] for k in ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'num_normal', 'num_anomaly']},
            candidate_window_recall=detected / candidate_anomaly, test_windows=len(frame),
            unscored_candidate_windows=sum(report['metrics'][s]['coverage']['rejected'] for s in ['test_realworld', 'test_staged'])))
        for (subset, camera), group in frame.reset_index().groupby(['subset', 'camera_id']):
            metrics = evaluate(group.label.to_numpy(), group.score.to_numpy(), combined['threshold']).as_dict()
            cameras.append(dict(run=name, subset=subset, camera_id=camera, **metrics))
    matched_splits = [r['split'] for name, r in runs.items() if name.startswith('matched')]
    if matched_splits and any(s != matched_splits[0] for s in matched_splits):
        raise ValueError('A-E training splits differ')
    verification = ROOT / 'outputs/video_inference/matched_d_seed0/verification.json'
    videos = json.loads(verification.read_text()) if verification.exists() else []
    if len(videos) != 7:
        missing.append('seven verified external videos')
    for video in videos:
        if video['decoded_output_frames'] != video['source_frames'] or not np.isclose(video['source_fps'], video['output_fps']):
            raise ValueError('Video timing/count check failed')
    frame_path = out / 'frame_diagnostic/report.json'
    frame_diagnostic = json.loads(frame_path.read_text()) if frame_path.exists() else None
    if not frame_diagnostic or not frame_diagnostic['complete'] or set(frame_diagnostic['runs']) != set(required):
        missing.append('frame-level diagnostic for all registered research runs')
    if frame_diagnostic:
        for name, result in frame_diagnostic['runs'].items():
            if name in runs and result['checkpoint_sha256'] != runs[name]['checkpoint_sha256']:
                raise ValueError('Frame diagnostic model hash differs from window evaluation')
    if missing and not args.allow_incomplete:
        raise FileNotFoundError('Required work remains: ' + ', '.join(missing))
    df = pd.DataFrame(table)
    if not df.empty:
        df.to_csv(out / 'comparison.csv', index=False)
    if frame_diagnostic:
        pd.DataFrame([dict(run=name, **r['metrics']['combined']) for name, r in frame_diagnostic['runs'].items()]).to_csv(out / 'frame_comparison.csv', index=False)
    pd.DataFrame(cameras).to_csv(out / 'per_camera_metrics.csv', index=False)
    expanded = df[df.run.str.startswith('expanded_a_seed')] if not df.empty else pd.DataFrame()
    seed_summary = {m: dict(mean=float(expanded[m].mean()), std=float(expanded[m].std(ddof=1)),
                           minimum=float(expanded[m].min()), maximum=float(expanded[m].max()))
                    for m in ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc']} if len(expanded) == 3 else None
    legacy_restricted = []
    legacy_path = ROOT / 'outputs/logs/ab_comparison/window_predictions.csv'
    if reference is not None and legacy_path.exists():
        legacy = pd.read_csv(legacy_path, dtype={k: str for k in IDENTITY[:5]})
        for experiment in ['a', 'b']:
            old = legacy[legacy.experiment == experiment].set_index(IDENTITY).sort_index()
            selected = old.loc[reference.index]
            if not np.array_equal(selected.label, reference.label):
                raise ValueError('Legacy and new labels differ on the intersection')
            legacy_restricted.append(dict(experiment=experiment,
                **evaluate(selected.label.to_numpy(), selected.score.to_numpy(), float(selected.threshold.iloc[0])).as_dict(),
                scope='Original 50-file model/threshold restricted to the same retained test windows; multiple protocol changes prevent causal attribution'))
    plots = []
    if all(f'matched_{e}_seed0' in frames for e in 'abcde'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for e in 'abcde':
            f = frames[f'matched_{e}_seed0']
            fpr, tpr, _ = roc_curve(f.label, f.score)
            precision, recall, _ = precision_recall_curve(f.label, f.score)
            axes[0].plot(fpr, tpr, label=e.upper())
            axes[1].plot(recall, precision, label=e.upper())
        axes[0].plot([0, 1], [0, 1], 'k--', label='Chance')
        axes[0].set(xlabel='False positive rate', ylabel='Recall', title='Matched A-E: retained-window ROC')
        axes[1].axhline(reference.label.mean(), color='black', linestyle='--', label='Anomaly prevalence')
        axes[1].set(xlabel='Recall', ylabel='Precision', title='Matched A-E: precision-recall', ylim=(0, 1.02))
        for ax in axes:
            ax.legend()
        fig.tight_layout(); fig.savefig(out / 'matched_curves.png', dpi=150); plt.close(fig)
        plots.append(str((out / 'matched_curves.png').relative_to(ROOT)))
        fig, axes = plt.subplots(1, 5, figsize=(17, 3.5))
        for ax, e in zip(axes, 'abcde'):
            history = pd.read_csv(out / f'matched_{e}_seed0/training_log.csv')
            ax.plot(history.epoch, history.train_loss, label='Train')
            ax.plot(history.epoch, history.validation_loss, label='Validation')
            ax.set(title=e.upper(), xlabel='Epoch', ylabel='Masked MSE')
        axes[0].legend(); fig.tight_layout(); fig.savefig(out / 'learning_curves.png', dpi=150); plt.close(fig)
        plots.append(str((out / 'learning_curves.png').relative_to(ROOT)))
    findings = []
    matched = [r for r in table if r['run'].startswith('matched')]
    if len(matched) == 5:
        findings.append(f"Matched A-E window ROC-AUC ranges from {min(r['roc_auc'] for r in matched):.3f} to {max(r['roc_auc'] for r in matched):.3f}; retained-window recall ranges from {min(r['recall'] for r in matched):.1%} to {max(r['recall'] for r in matched):.1%}.")
        best_2d = max(r['roc_auc'] for r in matched if r['experiment'] in 'ab')
        best_3d = max(r['roc_auc'] for r in matched if r['experiment'] in 'cde')
        findings.append(f"The highest matched 2D ROC-AUC is {best_2d:.3f}, versus {best_3d:.3f} for 3D. This one-seed comparison does not establish a 3D advantage.")
        findings.append(f"The retained-window cohort is {float(reference.label.mean()):.1%} anomaly-labeled; high precision/AP alone is not evidence of a reliable detector.")
    if seed_summary:
        findings.append(f"Expanded A, three seeds: window recall {seed_summary['recall']['mean']:.3f} +/- {seed_summary['recall']['std']:.3f}, ROC-AUC {seed_summary['roc_auc']['mean']:.3f} +/- {seed_summary['roc_auc']['std']:.3f} (sample SD; fixed data split).")
        findings.append('Expanded runs were trained on Colab CUDA and imported with model/data hashes; matched runs used local CPU training. Different sample sizes and backends prevent attributing differences to data size alone.')
    expanded_b = df[df.run.str.startswith('expanded_b_seed')] if not df.empty else pd.DataFrame()
    b_summary = {m:dict(mean=float(expanded_b[m].mean()),std=float(expanded_b[m].std(ddof=1)))
                 for m in ['precision','recall','f1','roc_auc','pr_auc']} if len(expanded_b)==3 else None
    if b_summary:
        findings.append(f"Expanded B, three seeds: window recall {b_summary['recall']['mean']:.3f} +/- {b_summary['recall']['std']:.3f}, ROC-AUC {b_summary['roc_auc']['mean']:.3f} +/- {b_summary['roc_auc']['std']:.3f}. This exploratory motion-feature follow-up was performed after inspecting test results; fresh confirmation is needed.")
    if frame_diagnostic:
        values = [r['metrics']['combined'] for r in frame_diagnostic['runs'].values()]
        if values:
            findings.append(f"Secondary frame diagnostic: score coverage is {values[0]['score_coverage']:.1%}; ROC-AUC across completed runs ranges from {min(m['roc_auc'] for m in values):.3f} to {max(m['roc_auc'] for m in values):.3f}. Window metrics must not be presented as frame-level accuracy.")
    examples = error_strips(runs, frames, out)
    report = dict(complete=not missing, missing=missing, matched_splits_identical=bool(matched_splits),
        error_examples=examples,
        findings=findings,
        environment=dict(python=sys.version, packages={name: version(name) for name in ['torch', 'numpy', 'pandas', 'scikit-learn', 'opencv-python', 'ultralytics']}),
        run_artifacts={name:dict(checkpoint_dir=str(run_paths(name)[0].relative_to(ROOT)), log_dir=str(run_paths(name)[1].relative_to(ROOT)), backend=runs[name].get('backend', {'device':'cpu'})) for name in runs},
        motionbert_provenance=json.loads((ROOT / 'outputs/logs/motionbert_setup.json').read_text()),
        input_validation=json.loads((ROOT / 'outputs/logs/research_input_validation.json').read_text()),
        frame_level_diagnostic=frame_diagnostic,
        retained_test_windows=len(reference) if reference is not None else None, comparisons=table,
        runs=runs, expanded_three_seed_summary=seed_summary, expanded_b_three_seed_summary=b_summary, legacy_on_retained_windows=legacy_restricted,
        video_verification=videos, plots=plots,
        limitations=[
            'RetailS axis conversion is an explicit, empirically supported dataset interpretation; author text says XYC and source RGB/calibration is unavailable.',
            'A-E use 10 normal training pose files with full test pose-file coverage; only technically retained windows are scored.',
            'Expanded position-only A uses 100 training files and three model seeds with a fixed source-video split; the family was declared before these test results.',
            'Candidate-window recall includes rejected positive windows as misses; conditional recall alone can conceal abstention.',
            'Video masks label temporal overlap for every person; these are not person-specific theft annotations.',
            'Overlapping windows are correlated. Three seeds are descriptive, not a significance or unseen-store study.',
            'MotionBERT has no RetailS 3D ground truth here. Structural checks and plausible skeletons do not establish 3D accuracy.',
            'Visual review is mixed: camera 5 contains an implausibly elevated inferred leg. The 30-frame single-view lift uses no flip-test averaging.',
            'UCF videos are external-domain inputs without temporal/person ground truth. Inference is offline, not a demonstrated real-time system.',
            'Legacy and new protocols differ in masking, coordinates, filtering, sample size and calibration; comparison is not a single-factor ablation.'])
    (out / 'final_report.json').write_text(json.dumps(clean(report), indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Quality-corrected model development', '', f"Status: {'complete' if not missing else 'in progress'}", '',
        'Thresholds use p95 of normal validation errors; epochs are selected by normal validation loss. No test labels tune either.', '',
        '| Run | Train files | Test windows | Precision | Recall (retained) | Recall (candidate) | F1 | ROC-AUC | AP |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['run']} | {r['training_files']} | {r['test_windows']} | {r['precision']:.4f} | {r['recall']:.4f} | "
                     f"{r['candidate_window_recall']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} |")
    if frame_diagnostic:
        lines += ['', '## Secondary frame-level diagnostic', '', frame_diagnostic['protocol'], '',
            '| Run | Scored frame coverage | Precision | Recall (scored) | Recall (all annotated positives) | F1 | ROC-AUC | AP |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name, r in frame_diagnostic['runs'].items():
            m = r['metrics']['combined']
            lines.append(f"| {name} | {m['score_coverage']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
                f"{m['all_annotated_anomaly_recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} |")
        lines += [''] + ['- ' + s for s in frame_diagnostic['limitations']]
    lines += ['', '## Interpretation limits', ''] + ['- ' + s for s in report['limitations']]
    lines += ['', '## Measured findings', ''] + ['- ' + s for s in findings]
    if missing:
        lines += ['', 'Remaining: ' + ', '.join(missing)]
    (out / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(df.to_string(index=False) if not df.empty else 'No completed model evaluations yet')
    print('Report complete:', not missing)


if __name__ == '__main__':
    main()
