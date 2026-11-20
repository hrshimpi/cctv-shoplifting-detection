"""Prepare neutral, score-free retail clips for human observations."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import imageio_ffmpeg
import yaml
from src.experiments.quality_dataset import sha256
from scripts.extract_video_poses import verify_export


def video_info(path):
    cap=cv2.VideoCapture(str(path)); count=0
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    fps=cap.get(cv2.CAP_PROP_FPS)
    size=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    while True:
        ok,_=cap.read()
        if not ok: break
        count+=1
    cap.release()
    return dict(frame_count=count,fps=fps,size=size)


def main():
    out=ROOT / 'outputs/manual_validation'; media=out / 'media'; media.mkdir(parents=True,exist_ok=True)
    profiles=yaml.safe_load((ROOT / 'configs/external_video_profiles.yaml').read_text())['videos']
    videos=sorted((ROOT / 'data/external_video/videos').rglob('*.mp4'),key=lambda p:hashlib.sha256(p.name.encode()).hexdigest())
    cases,private=[],[]
    for i,source in enumerate(videos,1):
        case_id=f'RV{i:03d}'
        pose_dir=ROOT / profiles[source.name]['pose_dir'] if source.name in profiles else ROOT / 'outputs/poses_batch/conf05' / source.parent.name / source.stem
        summary=verify_export(pose_dir)
        if Path(summary['source_video']).resolve()!=source.resolve():
            raise ValueError('Pose export refers to a different video')
        original=media / f'{case_id}_source.mp4'
        timing=['-vf',f"setpts=N/({summary['fps']}*TB)",'-r',str(summary['fps']),'-fps_mode','cfr']
        # Both displays start at frame zero; container timestamp offsets must
        # not shift reviewer frame IDs between source and skeleton views.
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(source),
            *timing,'-an','-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(original)],check=True)
        overlay=media / f'{case_id}_overlay.mp4'
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(pose_dir/'skeletons.mp4'),
            *timing,'-an','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(overlay)],check=True)
        info=video_info(original); overlay_info=video_info(overlay)
        if info!=overlay_info or info['frame_count']!=summary['source_frames'] or abs(info['fps']-summary['fps'])>1e-6:
            raise ValueError('Video and overlay timing/count/size differ')
        tracks=json.loads((pose_dir / 'poses.json').read_text())
        cases.append(dict(case_id=case_id,source_sha256=sha256(source),**info,
            track_ids=sorted(tracks,key=int),source_file=f'media/{original.name}',overlay_file=f'media/{overlay.name}',
            source_media_sha256=sha256(original),overlay_sha256=sha256(overlay),pose_sha256=sha256(pose_dir/'poses.json'),
            display_timing='Zero-based constant-FPS timestamps normalized; source frame order/count/resolution preserved'))
        private.append(dict(case_id=case_id,source_path=str(source.relative_to(ROOT)),pose_dir=str(pose_dir.relative_to(ROOT))))
        print('Prepared:',case_id,info['frame_count'],'frames',flush=True)
    manifest_id=hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()
    manifest=dict(schema=1,manifest_id=manifest_id,cases=cases,
        role='Previously inspected pilot clips; not an independent confirmation set',
        label_status='pending human review; source filenames/categories and model scores omitted from page')
    (out / 'review_manifest.json').write_text(json.dumps(manifest,indent=2))
    (out / '_private_sources.json').write_text(json.dumps(private,indent=2))
    template=(ROOT / 'scripts/templates/retail_validation_review.html').read_text(encoding='utf-8')
    (out / 'index.html').write_text(template.replace('__MANIFEST__',json.dumps(manifest).replace('<','\\u003c')),encoding='utf-8')
    pending=dict(schema=1,manifest_id=manifest_id,reviewer='',reviews=[dict(case_id=c['case_id'],source_sha256=c['source_sha256'],
        status='pending',reviewed_whole_video=False,retail_scene='',tracking_quality='',notes='',events=[]) for c in cases])
    (out / 'pending_review_template.json').write_text(json.dumps(pending,indent=2))
    report=dict(cases=len(cases),frames=sum(c['frame_count'] for c in cases),
        video_seconds=sum(c['frame_count']/c['fps'] for c in cases),labels_created=0,manifest_id=manifest_id,
        page=str(out/'index.html'),status='ready_for_human_review',media_timing_verified=True)
    (out / 'preparation_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
