"""Audit frozen A/B errors and weak label support; never tune or create labels."""
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
from src.data.retails import discover_retails_root, SUBSET_DIRS, gt_dir_for_subset
from src.evaluation.frame_scores import aggregate_windows
from src.evaluation.label_audit import positive_intervals, window_label_support
from src.experiments.artifacts import run_paths
from src.experiments.quality_dataset import sha256

RUNS = [f'expanded_{feature}_seed{seed}' for feature in 'ab' for seed in (0, 1, 2)]
IDENTITY = ['subset', 'pose_file', 'person_id', 'start_frame', 'end_frame']


def main():
    out = ROOT / 'outputs/model_failure_audit'
    out.mkdir(parents=True, exist_ok=True)
    root = discover_retails_root(ROOT / 'data/retails')
    if root is None:
        raise FileNotFoundError('RetailS pose/mask sources are unavailable')
    bundle = ROOT / 'outputs/colab_handoff/retail_training'
    bundle_manifest = json.loads((bundle / 'bundle_manifest.json').read_text())
    frame_path = ROOT / 'outputs/logs/research_v1/frame_diagnostic/report.json'
    frames = json.loads(frame_path.read_text())
    alignment = json.loads((ROOT / 'outputs/validation_alignment/report.json').read_text())
    if not alignment['structural_checks_passed'] or not frames['complete']:
        raise ValueError('Existing structural and frame checks must pass first')
    predictions, provenance = {}, {}
    for name in RUNS:
        model, log = run_paths(name)
        evaluation = json.loads((log / 'evaluation.json').read_text())
        digest = sha256(model / 'model.pt')
        if digest != evaluation['checkpoint_sha256'] or digest != frames['runs'][name]['checkpoint_sha256']:
            raise ValueError('Checkpoint/evaluation identities differ')
        pred = pd.read_csv(log / 'window_predictions.csv', dtype={k: str for k in ['pose_file', 'person_id', 'camera_id', 'video_id']})
        if pred.duplicated(IDENTITY).any() or not np.isfinite(pred[['score', 'threshold']]).all().all():
            raise ValueError('Duplicate windows or nonfinite predictions')
        if not np.array_equal(pred.prediction.to_numpy(), (pred.score >= pred.threshold).astype(int)):
            raise ValueError('Stored predictions disagree with frozen thresholds')
        if not np.allclose(pred.threshold, evaluation['metrics']['combined']['threshold'], rtol=1e-7, atol=1e-10):
            raise ValueError('Prediction thresholds differ from the completed evaluation')
        for subset in ['test_realworld', 'test_staged']:
            group = pred[pred.subset == subset]
            with np.load(model / f'{subset}_scores.npz', allow_pickle=False) as saved:
                if not np.allclose(group.score, saved['scores'], rtol=1e-7, atol=1e-10) or not np.array_equal(group.label, saved['labels']):
                    raise ValueError('Prediction CSV differs from original scored arrays')
        if predictions:
            reference = next(iter(predictions.values()))
            if not pred[IDENTITY + ['label']].equals(reference[IDENTITY + ['label']]):
                raise ValueError('A/B window identities differ')
        predictions[name] = pred
        provenance[name] = dict(checkpoint_sha256=digest, predictions_sha256=sha256(log / 'window_predictions.csv'),
                                evaluation_sha256=sha256(log / 'evaluation.json'))
    reference = predictions[RUNS[0]]
    detail, intervals, source_counts, run_rows, quality_counts = [], [], [], [], []
    for subset in ['test_realworld', 'test_staged']:
        manifest_path = bundle / f'data/{subset}.json'
        manifest = json.loads(manifest_path.read_text())
        for extension in ['json', 'npz']:
            relative = f'data/{subset}.{extension}'
            if sha256(bundle / relative) != bundle_manifest['files'][relative]:
                raise ValueError('Prepared evaluation artifact changed')
        for relative, digest in manifest['sources'].items():
            if sha256(root / relative) != digest:
                raise ValueError('Pose/mask source changed')
        group = reference[reference.subset == subset]
        quality_counts.append(dict(subset=subset, **manifest['counts']))
        with np.load(manifest_path.with_suffix('.npz'), allow_pickle=False) as arrays:
            support_by_index = dict(zip(group.index, arrays['masks'].mean(axis=(1, 2)).tolist()))
        for row, meta in zip(group.to_dict('records'), manifest['metadata']):
            if any(str(row[k]) != str(meta[k]) for k in IDENTITY if k != 'subset') or int(row['label']) != meta['label']:
                raise ValueError('Prepared window identity changed')
        if len(group) != len(manifest['metadata']):
            raise ValueError('Prepared window count changed')
        # Coverage uses original masks even for clips without a retained window.
        labels = {p.stem + '.json': np.load(p, allow_pickle=False)
                  for p in sorted(gt_dir_for_subset(root / SUBSET_DIRS[subset]).glob('*.npy'))}
        aggregated = {name: aggregate_windows(pred[pred.subset == subset].score.to_numpy(),
                      pred[pred.subset == subset].to_dict('records'), {k: len(v) for k, v in labels.items()})
                      for name, pred in predictions.items()}
        for filename, mask in labels.items():
            events = positive_intervals(mask)
            raw = json.loads((root / SUBSET_DIRS[subset] / 'pose/test' / filename).read_text())
            people = np.zeros(len(mask), dtype=np.int32)
            for track in raw.values():
                ids = np.array([int(fid) for fid in track], dtype=int)
                if ((ids < 0) | (ids >= len(mask))).any():
                    raise ValueError('Pose frame outside mask')
                people[ids] += 1
            source_counts.append(dict(subset=subset, pose_file=filename, mask_frames=len(mask),
                positive_frames=int(mask.sum()), mask_positive_intervals=len(events),
                positive_frames_without_raw_person=int(((mask == 1) & (people == 0)).sum()),
                positive_frames_with_multiple_people=int(((mask == 1) & (people > 1)).sum())))
            for idx, row in group[group.pose_file == filename].iterrows():
                support = window_label_support(mask, int(row.start_frame), int(row.end_frame), people)
                if int(support['positive_frames'] > 0) != int(row.label):
                    raise ValueError('Mask-derived window label changed')
                record = {k: row[k] for k in IDENTITY + ['label']}
                record.update(support)
                record['retained_feature_mask_fraction'] = support_by_index[idx]
                for name, pred in predictions.items():
                    record[name + '_score'] = float(pred.loc[idx, 'score'])
                    record[name + '_prediction'] = int(pred.loc[idx, 'prediction'])
                votes = sum(record[name + '_prediction'] for name in RUNS)
                record.update(alert_votes=votes, all_six_miss=bool(row.label == 1 and votes == 0),
                    all_six_false_alarm=bool(row.label == 0 and votes == 6))
                detail.append(record)
            for start, end in events:
                for name in RUNS:
                    scores = aggregated[name][filename][start:end + 1]
                    valid = np.isfinite(scores)
                    threshold = frames['runs'][name]['threshold']
                    hits = valid & (scores >= threshold)
                    intervals.append(dict(subset=subset, pose_file=filename, start_frame=start, end_frame=end,
                        run=name, annotated_frames=len(scores), scored_frames=int(valid.sum()),
                        flagged_frames=int(hits.sum()), any_model_overlap=bool(hits.any()),
                        fully_unscored=bool(not valid.any()), frame_threshold=threshold,
                        actor_identity_verified=False))
    table = pd.DataFrame(detail)
    source_table = pd.DataFrame(source_counts)
    interval_table = pd.DataFrame(intervals)
    for name, pred in predictions.items():
        for subset, group in [('combined', pred), *list(pred.groupby('subset'))]:
            y, p = group.label.to_numpy(), group.prediction.to_numpy()
            tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
            fp = int(((y == 0) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
            event = interval_table[interval_table.run == name]
            if subset != 'combined': event = event[event.subset == subset]
            run_rows.append(dict(run=name, subset=subset, tp=tp, fn=fn, fp=fp, tn=tn,
                recall=tp / max(tp + fn, 1), false_positive_rate=fp / max(fp + tn, 1),
                mask_intervals=len(event), intervals_with_model_overlap=int(event.any_model_overlap.sum()),
                fully_unscored_intervals=int(event.fully_unscored.sum())))
    run_table = pd.DataFrame(run_rows)
    overlap_rows = []
    for (subset, count), cohort in table.groupby(['subset', 'positive_frames']):
        for name in RUNS:
            overlap_rows.append(dict(subset=subset, positive_frames=count, windows=len(cohort), run=name,
                alert_fraction=float(cohort[name + '_prediction'].mean()),
                mean_feature_mask_fraction=float(cohort.retained_feature_mask_fraction.mean())))
    pd.DataFrame(overlap_rows).to_csv(out / 'alerts_by_positive_frame_count.csv', index=False)
    pd.DataFrame(quality_counts).to_csv(out / 'quality_rejection_counts.csv', index=False)
    # Prioritize diagnostic windows by agreement only, never for threshold/model selection.
    table['review_priority'] = np.where(table.all_six_false_alarm, 'shared_false_alarm',
                              np.where(table.all_six_miss, 'shared_miss', 'other'))
    table.to_csv(out / 'window_label_support_and_predictions.csv', index=False)
    table[table.review_priority != 'other'].to_csv(out / 'diagnostic_queue.csv', index=False)
    interval_table.to_csv(out / 'mask_interval_coverage.csv', index=False)
    source_table.to_csv(out / 'source_label_inventory.csv', index=False)
    run_table.to_csv(out / 'frozen_run_errors.csv', index=False)
    positive = table[table.label == 1]
    full_positive = positive[positive.positive_fraction == 1]
    partial_positive = positive[positive.positive_fraction < 1]
    totals = dict(source_pose_mask_pairs=len(source_table), retained_windows=len(table),
        positive_windows=len(positive), negative_windows=int((table.label == 0).sum()),
        partially_positive_windows=int(((positive.positive_fraction > 0) & (positive.positive_fraction < 1)).sum()),
        fully_positive_windows=len(full_positive),
        positive_windows_with_multiple_people=int((positive.positive_frames_with_multiple_people > 0).sum()),
        shared_miss_windows=int(table.all_six_miss.sum()), shared_false_alarm_windows=int(table.all_six_false_alarm.sum()),
        mask_positive_intervals=int(source_table.mask_positive_intervals.sum()),
        positive_frames=int(source_table.positive_frames.sum()),
        positive_frames_without_raw_person=int(source_table.positive_frames_without_raw_person.sum()),
        positive_frames_with_multiple_people=int(source_table.positive_frames_with_multiple_people.sum()),
        rejected_candidate_windows=sum(q['rejected'] for q in quality_counts),
        rejected_positive_candidate_windows=sum(q['rejected_anomaly'] for q in quality_counts),
        independently_verified_new_action_labels=0)
    findings = [
        f"All six frozen A/B runs were compared on the same {len(table):,} retained windows; checkpoints and original source hashes pass validation.",
        f"{totals['shared_miss_windows']:,}/{len(positive):,} positive windows are missed by all six runs; {totals['shared_false_alarm_windows']} normal windows trigger every run. Shared failures suggest testing different evidence rather than averaging these six models; this is a hypothesis, not a measured ensemble result.",
        f"{totals['partially_positive_windows']} positive windows include a mixture of normal and positive mask frames. ANY-positive window labels do not identify exact action timing.",
        f"A seed 0 alerts on {full_positive[RUNS[0] + '_prediction'].mean():.1%} of {len(full_positive)} fully positive windows, versus {partial_positive[RUNS[0] + '_prediction'].mean():.1%} of mixed windows. Misses persist without mixed window labels; label overlap alone does not explain weak recall.",
        f"{totals['positive_windows_with_multiple_people']} positive windows overlap a positive video-mask frame with multiple raw person tracks present. Video masks do not identify the acting person, including in single-person clips.",
        f"{totals['positive_frames_without_raw_person']:,} positive mask frames have no raw person observation; {totals['positive_frames_with_multiple_people']:,} contain multiple raw tracks. These counts describe available pose support, not verified detector failures.",
        f"Existing quality gates excluded {totals['rejected_positive_candidate_windows']} positive candidate windows out of {totals['rejected_candidate_windows']} rejected windows. Raw pose absence, quality rejection, and retained-window misses are separate diagnostics and must not be added together.",
        'Contiguous positive-mask intervals are not independent theft events. Their retrospective overlap diagnostics cannot measure causal detection delay or person-level action accuracy.',
        'RetailS provides local pose tracks and video masks but no source RGB/person action labels here. The seven external clips have sampled AI observations only. No new verified action labels can be established from these artifacts.',
        'The test sets are diagnostic only: no model, threshold, smoothing, window, or training split was changed.'
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    combined = run_table[run_table.subset == 'combined']
    axes[0].bar(combined.run.str.replace('expanded_', ''), combined.recall, color='#307ab0')
    axes[0].set(title='Frozen window recall', ylim=(0, 1), ylabel='Fraction of mask-positive windows')
    axes[1].bar(['All-six misses', 'Other positive windows'],
                [totals['shared_miss_windows'], len(positive) - totals['shared_miss_windows']], color=['#ba4a45', '#307ab0'])
    axes[1].set(title='Agreement does not establish action truth', ylabel='Positive windows')
    fig.tight_layout(); fig.savefig(out / 'failure_summary.png', dpi=150); plt.close(fig)
    report = dict(phase='frozen_label_and_failure_audit', complete=True, totals=totals, findings=findings,
        models=provenance, bundle_manifest_sha256=sha256(bundle / 'bundle_manifest.json'),
        frame_report_sha256=sha256(frame_path), changes_to_models_or_thresholds=False,
        independent_action_validation_complete=False, action_label_status='unverified_without_RGB_and_independent_person_time_labels',
        next_experiment='Preserve A/B baselines. Acquire an independent person/time-labeled validation and locked confirmation set with RGB before anomaly-based tuning. Evaluate action/context evidence separately from pose reconstruction.',
        limitations=['Overlapping windows and shared seeds are not independent observations.',
                     'Multiple raw track IDs may include duplicate or fragmented tracks.',
                     'No accuracy improvement or causal failure attribution is established by this audit.'])
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out / 'report.md').write_text('# Frozen model label and failure audit\n\n' + '\n\n'.join(findings) +
        '\n\n## Per-run diagnostics\n\n' + run_table.to_string(index=False) + '\n', encoding='utf-8')
    print(json.dumps(totals, indent=2)); print('\n'.join(findings))


if __name__ == '__main__':
    main()
