"""Score cached retail-video tracks with the quality-corrected research models.

Outputs are retrospective anomaly scores, not person-specific theft labels.
The predeclared default is matched D, independent of its held-out test metrics.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
import yaml
from src.data.retails import PersonTrack, PoseFrame
from src.data.preprocessing import build_windows
from src.data.quality import prepare_track, accepted_window
from src.experiments.quality_dataset import features, lift, sha256, fingerprint, implementation
from src.evaluation.metrics import choose_threshold
from src.demo.render import draw_skeleton, draw_label
from scripts.extract_video_poses import verify_export
from scripts.run_research_phase import make_model, score, CHECKPOINT


def active_score(windows, frame_id, stride):
    """Expire completed-window scores before a later gap can inherit them."""
    matches = [w for w in windows if w['end_frame'] <= frame_id < w['end_frame'] + stride]
    return max(matches, key=lambda w: w['end_frame']) if matches else None


def readable_label(frame, points, text, color):
    visible = points[:, 2] >= .3
    if not visible.any():
        return
    joint = 0 if visible[0] else np.flatnonzero(visible)[np.argmin(points[visible, 1])]
    font, size, thickness = cv2.FONT_HERSHEY_SIMPLEX, .35, 1
    (width, height), baseline = cv2.getTextSize(text, font, size, thickness)
    x = int(np.clip(points[joint, 0] - 10, 2, max(2, frame.shape[1] - width - 3)))
    y = int(np.clip(points[joint, 1] - 12, 40, frame.shape[0] - baseline - 2))
    cv2.rectangle(frame, (x - 1, y - height - 2), (x + width + 1, y + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, size, color, thickness, cv2.LINE_AA)


def load_tracks(path, source):
    records = json.loads(path.read_text())
    return {int(tid): PersonTrack(str(tid), 'external', source.stem, '1', source,
        {int(fid): PoseFrame(int(fid), np.asarray(record['keypoints'], dtype=np.float32).reshape(17, 3), record.get('scores'))
         for fid, record in frames.items()}) for tid, frames in records.items()}


def export_video(video, run_name, model, saved, calibration, model_3d, pose_dir):
    summary = verify_export(pose_dir)
    cfg, experiment = saved['config'], saved['experiment']
    if Path(summary['source_video']).resolve() != video.resolve():
        raise ValueError('Cached poses belong to a different source video')
    tracks = load_tracks(pose_dir / 'poses.json', video)
    raw, normalized, masks, metadata = [], [], [], []
    candidates = 0
    for tid, track in tracks.items():
        prepared = prepare_track(track, cfg, normalize=False, coordinate_order='xy')
        for window in build_windows(prepared['frame_ids'], prepared['keypoints'], prepared['visibility_mask'], cfg['window_length'], cfg['stride']):
            candidates += 1
            start = int(np.searchsorted(prepared['frame_ids'], window['start_frame']))
            end = start + cfg['window_length']
            if not accepted_window(window, prepared['anchor_valid'][start:end], cfg):
                continue
            raw.append(window['keypoints'])
            normalized.append(prepared['normalized_keypoints'][start:end])
            masks.append(window['visibility_mask'])
            metadata.append(dict(track_id=tid, start_frame=window['start_frame'], end_frame=window['end_frame']))
    threshold = choose_threshold(calibration, cfg['threshold_percentile'])
    if raw:
        arrays = dict(raw=np.stack(raw), normalized=np.stack(normalized), visibility=np.stack(masks))
        points = None
        if experiment in 'cde':
            torch.set_num_threads(cfg['lift_threads'])
            manifest = dict(fingerprint=fingerprint(dict(video=sha256(video), poses=sha256(pose_dir / 'poses.json'),
                config=cfg, inference_coordinate_order='xy', implementation=implementation())))
            points = lift(arrays, manifest, CHECKPOINT, model_3d)
            torch.set_num_threads(2)
        values, valid = features(arrays, experiment, points)
        scores, _ = score(model, values, valid, cfg['batch_size'])
        for meta, value in zip(metadata, scores):
            meta.update(score=float(value), above_threshold=bool(value >= threshold),
                        start_seconds=meta['start_frame'] / summary['fps'], end_seconds=meta['end_frame'] / summary['fps'])
    folder = ROOT / 'outputs/video_inference' / run_name / video.stem
    folder.mkdir(parents=True, exist_ok=True)
    by_track = {tid: [m for m in metadata if m['track_id'] == tid] for tid in tracks}
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if not cap.isOpened() or not np.isclose(fps, summary['fps']):
        raise ValueError('Source video metadata mismatch')
    intermediate = folder / 'annotated_mp4v.mp4'
    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        raise RuntimeError('Cannot open video encoder')
    frame_id, scored_frames = 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            any_score = False
            for tid, track in tracks.items():
                if frame_id not in track.frames:
                    continue
                active = active_score(by_track[tid], frame_id, cfg['stride'])
                color = (160, 160, 160) if active is None else ((0, 0, 255) if active['above_threshold'] else (0, 190, 0))
                kpts = track.frames[frame_id].keypoints
                draw_skeleton(frame, kpts, color, cfg['confidence_threshold'])
                text = f'id {tid}: unscored' if active is None else f"id {tid}: {'ABOVE' if active['above_threshold'] else 'below'} {active['score']:.3f}"
                readable_label(frame, kpts, text, color)
                any_score |= active is not None
            cv2.rectangle(frame, (0, 0), (size[0], 28), (0, 0, 0), -1)
            cv2.putText(frame, 'Offline pose anomaly scores', (4, 11), cv2.FONT_HERSHEY_SIMPLEX, .34, (255, 255, 255), 1)
            cv2.putText(frame, f'{frame_id / fps:.2f}s | threshold {threshold:.4f} | gray: unscored', (4, 24), cv2.FONT_HERSHEY_SIMPLEX, .30, (255, 255, 255), 1)
            writer.write(frame)
            scored_frames += int(any_score)
            frame_id += 1
    finally:
        cap.release(); writer.release()
    if frame_id != summary['source_frames']:
        raise ValueError('Source decode count differs from pose export')
    import imageio_ffmpeg
    output = folder / 'annotated.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', str(intermediate),
        '-an', '-c:v', 'libx264', '-threads', '2', '-pix_fmt', 'yuv420p', '-crf', '22', '-movflags', '+faststart', str(output)], check=True)
    check = cv2.VideoCapture(str(output))
    decoded, output_fps = 0, check.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = check.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != size:
            raise ValueError('Output video size changed')
        if decoded == min(300, frame_id - 1):
            cv2.imwrite(str(folder / 'preview.jpg'), frame)
        decoded += 1
    check.release()
    if decoded != frame_id or not np.isclose(output_fps, fps):
        raise ValueError('Output decode count/FPS mismatch')
    report = dict(video=str(video), source_sha256=sha256(video), pose_sha256=sha256(pose_dir / 'poses.json'),
        pose_directory=str(pose_dir), run=run_name, experiment=experiment, threshold=threshold,
        checkpoint_sha256=saved['checkpoint_sha256'] if 'checkpoint_sha256' in saved else None,
        source_coordinate_order='xy (YOLO); no RetailS axis conversion on external videos',
        candidate_windows=candidates, retained_windows=len(metadata), rejected_windows=candidates - len(metadata),
        flagged_windows=sum(m['above_threshold'] for m in metadata), source_frames=frame_id, decoded_output_frames=decoded,
        source_fps=fps, output_fps=output_fps, frame_size=list(size), frames_with_a_current_score=scored_frames,
        video_frame_score_coverage=scored_frames / frame_id, output=str(output),
        protocol='Offline full-track preprocessing; completed-window scores expire after one stride. No streaming/causal claim.',
        limitation='External video labels do not supply person-level or interval ground truth; no detection accuracy is inferred.', windows=metadata)
    (folder / 'scores.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(video.name, f'{len(metadata)}/{candidates} windows; verified {decoded} frames at {fps} fps', flush=True)
    return {k: v for k, v in report.items() if k != 'windows'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='matched_d_seed0')
    parser.add_argument('--video', type=Path, help='Optional single video; default processes all seven inputs')
    args = parser.parse_args()
    folder = ROOT / 'outputs/checkpoints/research_v1' / args.run
    saved = torch.load(folder / 'model.pt', weights_only=False, map_location='cpu')
    if saved['implementation'] != implementation():
        raise ValueError('Processing implementation changed since model training; use a matching model version')
    saved['checkpoint_sha256'] = sha256(folder / 'model.pt')
    model = make_model(saved['experiment'], saved['feature_dim'], saved['config'])
    model.load_state_dict(saved['model_state_dict']); model.eval()
    calibration = np.load(folder / 'validation_errors.npy')
    model_3d = None
    if saved['experiment'] in 'cde':
        if sha256(CHECKPOINT) != saved['motionbert_sha256']:
            raise ValueError('MotionBERT checkpoint differs from training')
        from src.motionbert.lift import load_motionbert
        model_3d = load_motionbert(CHECKPOINT)
    profiles = yaml.safe_load((ROOT / 'configs/external_video_profiles.yaml').read_text())['videos']
    videos = [args.video.resolve()] if args.video else sorted((ROOT / 'data/external_video/videos').rglob('*.mp4'))
    reports = []
    for video in videos:
        relative = video.relative_to(ROOT / 'data/external_video/videos')
        pose_dir = ROOT / profiles[video.name]['pose_dir'] if video.name in profiles else ROOT / 'outputs/poses_batch/conf05' / relative.parent / video.stem
        reports.append(export_video(video, args.run, model, saved, calibration, model_3d, pose_dir))
    out = ROOT / 'outputs/video_inference' / args.run / 'verification.json'
    if args.video and out.exists():
        existing = {r['video']: r for r in json.loads(out.read_text())
                    if r.get('checkpoint_sha256') == saved['checkpoint_sha256']}
        existing.update({r['video']: r for r in reports})
        reports = [existing[k] for k in sorted(existing)]
    out.write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print('Verified output videos:', len(reports), flush=True)


if __name__ == '__main__':
    main()
