"""Render verified A/B scores and pose diagnostics onto all seven retail videos."""
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from scripts.run_video_inference import load_tracks
from src.demo.render import draw_skeleton
from src.evaluation.video_status import active_window,model_status,PRIMARY_RUN
from src.experiments.artifacts import run_paths
from src.experiments.quality_dataset import sha256

RUNS=[f'expanded_{family}_seed{seed}' for family in ('a','b') for seed in (0,1,2)]
W,H=1280,960
WHITE=(232,238,247);MUTED=(170,181,197);GREEN=(110,205,150);AMBER=(45,184,255);GRAY=(145,153,165)


def text(canvas,value,x,y,scale=.48,color=WHITE):
    cv2.putText(canvas,str(value),(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def fmt(value):return '--' if value is None else f'{value:.5f}'


def intervals(flags,fps):
    result=[];start=None
    for frame,flag in enumerate([*flags,False]):
        if flag and start is None:start=frame
        if not flag and start is not None:
            result.append(dict(start_frame=start,end_frame=frame-1,start_seconds=start/fps,end_seconds=(frame-1)/fps,
                               label='unverified_model_alert'))
            start=None
    return result


def draw_frame(frame,case,fid,rows,thresholds,moving,benchmark,guard_intervals):
    canvas=np.full((H,W,3),(28,23,19),np.uint8)
    primary_alert=any(r['scores'][PRIMARY_RUN] is not None and r['scores'][PRIMARY_RUN]>=thresholds[PRIMARY_RUN] for r in rows)
    primary_scored=any(r['scores'][PRIMARY_RUN] is not None for r in rows)
    if moving: headline='CAMERA MOTION - SCORING PAUSED';color=GRAY
    elif primary_alert:headline='POSSIBLE SHOPLIFTING / SUSPICIOUS ACTIVITY - UNVERIFIED';color=AMBER
    elif primary_scored:headline='NO CURRENT MODEL ALERT - NOT PROOF OF NORMAL ACTIVITY';color=GREEN
    else:headline='UNSCORED - WAITING FOR A VALID POSE SEQUENCE';color=GRAY
    text(canvas,f"{case['case_id']} | {headline}",20,34,.60,color)
    text(canvas,f"Frame {fid}/{case['frame_count']-1} | Time {fid/case['fps']:.2f}s | Source {case['fps']:.0f} FPS | People tracked {len(rows)}",20,65,.51)
    text(canvas,'Offline pose-anomaly screening. Model scores are NOT theft probabilities.',20,90,.48,MUTED)
    view=cv2.resize(frame,(640,480),interpolation=cv2.INTER_CUBIC)
    for row in rows:
        points=np.array(row['keypoints_xy_confidence'],dtype=np.float32);points[:,:2]*=2
        score=row['scores'][PRIMARY_RUN]
        color=GRAY if score is None else (AMBER if score>=thresholds[PRIMARY_RUN] else GREEN)
        draw_skeleton(view,points,color,.3)
        visible=(points[:,2]>=.3)&np.isfinite(points).all(1)
        if visible.any():
            xy=points[visible,:2];left,top=np.maximum(xy.min(0)-6,0).astype(int);right,bottom=np.minimum(xy.max(0)+6,[639,479]).astype(int)
            cv2.rectangle(view,(left,top),(right,bottom),color,1)
            label=f"ID {row['track_id']} | {'?' if score is None else ('ALERT?' if score>=thresholds[PRIMARY_RUN] else 'below')}"
            text(view,label,min(left,440),max(16,top-5),.45,color)
    canvas[112:592,20:660]=view
    text(canvas,'FROZEN NORMAL-VALIDATION THRESHOLDS',20,625,.54)
    for j,family in enumerate(('a','b')):
        line='   '.join(f'{family.upper()}{seed}: {thresholds[f"expanded_{family}_seed{seed}"]:.6f}' for seed in (0,1,2))
        text(canvas,line,20,653+j*25,.48)
    text(canvas,'Primary display decision: A0. Other five runs are comparisons.',20,715,.44,MUTED)
    text(canvas,'Window: 30 frames | Stride: 15 | Joint confidence gate: 0.30',20,739,.44)
    text(canvas,'Gray: unscored | Green: below threshold | Amber: model alert',20,763,.43)
    text(canvas,'Camera motion: '+('paused' if moving else 'outside exclusion intervals'),20,787,.44)
    text(canvas,'ACTION REVIEW: UNCERTAIN. Shoplifting is not confirmed.',20,815,.46,AMBER)
    text(canvas,'RetailS held-out means - benchmark only, NOT video accuracy:',20,853,.43,MUTED)
    for j,family in enumerate(('a','b')):
        b=benchmark[family]
        text(canvas,f"{family.upper()}: P {b['precision']:.1%}  R {b['recall']:.1%}  F1 {b['f1']:.1%}  ROC-AUC {b['roc_auc']:.3f}",20,878+24*j,.46)
    text(canvas,'Track IDs are estimates; occlusion and ID changes remain possible.',20,935,.42,MUTED)
    text(canvas,'PER-PERSON VALUES | A/B = positions / positions+motion',690,119,.43)
    for index,row in enumerate(rows):
        y=135+index*153
        if y+145>H:raise ValueError('Too many simultaneous tracks for the video dashboard')
        cv2.rectangle(canvas,(682,y),(1264,y+143),(46,39,32),-1)
        score=row['scores'][PRIMARY_RUN];color=GRAY if score is None else (AMBER if score>=thresholds[PRIMARY_RUN] else GREEN)
        text(canvas,f"ID {row['track_id']} | visible {row['visible_joints']}/17 | pose conf {row['mean_joint_confidence']:.2f} | obs {row['observations_so_far']}",695,y+21,.43,color)
        for j,family in enumerate(('a','b')):
            values='   '.join(f'{family.upper()}{seed} {fmt(row["scores"][f"expanded_{family}_seed{seed}"])}' for seed in (0,1,2))
            text(canvas,values,695,y+43+20*j,.44)
        ratio=' / '.join(' '.join('--' if row['score_ratios'][f'expanded_{family}_seed{s}'] is None else f"{row['score_ratios'][f'expanded_{family}_seed{s}']:.2f}" for s in (0,1,2)) for family in ('a','b'))
        text(canvas,'xThreshold A / B: '+ratio,695,y+84,.39)
        text(canvas,f"Wrists {row['visible_wrists']}/2 | Anchors {row['visible_anchors']}/4 | Track span {row['track_span_seconds']:.2f}s",695,y+103,.41)
        window=row['primary_window'];window_text='none' if window is None else f"{window['start_frame']}-{window['end_frame']} | scene {window['scene_id']} | score age {fid-window['end_frame']}f"
        text(canvas,'Window '+window_text,695,y+122,.39)
        label='UNSCORED' if score is None else ('POSSIBLE SUSPICIOUS ACTIVITY - UNVERIFIED' if score>=thresholds[PRIMARY_RUN] else 'BELOW THRESHOLD')
        text(canvas,label,695,y+138,.39,color)
    return canvas,primary_alert,primary_scored


def main():
    folder=ROOT/'outputs/manual_validation'
    manifest=json.loads((folder/'review_manifest.json').read_text())
    private={x['case_id']:x for x in json.loads((folder/'_private_sources.json').read_text())}
    analysis=json.loads((ROOT/'outputs/retail_pilot_analysis/report.json').read_text())
    motion_path=ROOT/'configs/retail_camera_motion.json';motion=json.loads(motion_path.read_text())
    if analysis['manifest_id']!=manifest['manifest_id'] or analysis['camera_guard_sha256']!=sha256(motion_path):raise ValueError('Stale camera-aware scoring report')
    comparison=json.loads((ROOT/'outputs/logs/research_v1/expanded_ab_comparison/comparison.json').read_text())
    benchmark={r['family']:r for r in comparison['means'] if r['scope']=='window'}
    values=pd.read_csv(ROOT/'outputs/retail_pilot_analysis/window_scores.csv')
    values=values[values['mode']=='camera_guard']
    if not np.isfinite(values.score).all():raise ValueError('Nonfinite stored model score')
    thresholds={}
    for run in RUNS:
        model,_=run_paths(run)
        if sha256(model/'model.pt')!=analysis['checkpoint_sha256'][run]:raise ValueError('Model checkpoint changed')
        calibration=np.load(model/'validation_errors.npy',allow_pickle=False)
        thresholds[run]=float(np.percentile(calibration,95))
    output_root=ROOT/'outputs/video_analysis';output_root.mkdir(parents=True,exist_ok=True)
    reports=[]
    for case in manifest['cases']:
        cid=case['case_id'];source=folder/case['source_file'];pose_path=ROOT/private[cid]['pose_dir']/'poses.json'
        if sha256(source)!=case['source_media_sha256'] or sha256(pose_path)!=case['pose_sha256']:raise ValueError('Source media or poses changed')
        tracks=load_tracks(pose_path,source)
        windows=defaultdict(list)
        for row in values[values.case_id==cid].to_dict('records'):
            windows[(str(row['run']),int(row['track_id']))].append(row)
        frame_tracks=defaultdict(list)
        for tid,track in tracks.items():
            for fid in track.frames:frame_tracks[fid].append(tid)
        if max(map(len,frame_tracks.values()),default=0)>5:raise ValueError('Dashboard requires layout expansion')
        exclusions=motion['cases'].get(cid,{}).get('excluded_intervals',[])
        out=output_root/cid;out.mkdir(parents=True,exist_ok=True)
        video_path=folder/'media'/f'{cid}_model_analysis.mp4'
        temporary=video_path.with_suffix('.partial.mp4')
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}',
            '-r',str(case['fps']),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        cap=cv2.VideoCapture(str(source));fid=0;flags=[];scored=[];observations=defaultdict(int);first_alert=None
        try:
            with (out/'frame_values.jsonl').open('w',encoding='utf-8') as sidecar:
                while True:
                    ok,frame=cap.read()
                    if not ok:break
                    moving=any(start<=fid<=end for start,end in exclusions)
                    rows=[]
                    for tid in sorted(frame_tracks[fid]):
                        track=tracks[tid];points=track.frames[fid].keypoints
                        if not np.isfinite(points).all():raise ValueError('Nonfinite raw pose')
                        visible=points[:,2]>=.3;observations[tid]+=1
                        active={run:None if moving else active_window(windows[(run,tid)],fid,15) for run in RUNS}
                        scores={run:None if active[run] is None else float(active[run]['score']) for run in RUNS}
                        primary=active[PRIMARY_RUN]
                        rows.append(dict(track_id=tid,keypoints_xy_confidence=points.tolist(),scores=scores,
                            score_ratios={run:None if scores[run] is None else scores[run]/thresholds[run] for run in RUNS},
                            status=model_status(scores,thresholds,moving),visible_joints=int(visible.sum()),
                            mean_joint_confidence=float(np.clip(points[:,2],0,1).mean()),
                            visible_wrists=int(visible[[9,10]].sum()),visible_anchors=int(visible[[5,6,11,12]].sum()),
                            observations_so_far=observations[tid],track_span_seconds=(fid-min(track.frames)+1)/case['fps'],
                            primary_window=None if primary is None else {k:int(primary[k]) for k in ['start_frame','end_frame','scene_id']}))
                    canvas,alert,has_score=draw_frame(frame,case,fid,rows,thresholds,moving,benchmark,exclusions)
                    process.stdin.write(canvas.tobytes())
                    sidecar.write(json.dumps(dict(frame_id=fid,seconds=fid/case['fps'],camera_motion=moving,
                        primary_model_alert=alert,action_confirmed=False,people=rows),allow_nan=False)+'\n')
                    if fid==min(300,case['frame_count']-1):cv2.imwrite(str(out/'preview.jpg'),canvas)
                    if alert and first_alert is None:
                        first_alert=fid;cv2.imwrite(str(out/'first_alert.jpg'),canvas)
                    flags.append(alert);scored.append(has_score);fid+=1
            process.stdin.close();error=process.stderr.read().decode(errors='replace');code=process.wait()
            if code:raise RuntimeError('Video encoder failed: '+error)
        except BaseException:
            if process.poll() is None:process.kill();process.wait()
            raise
        finally:cap.release()
        if fid!=case['frame_count']:raise ValueError('Input frame count differs')
        temporary.replace(video_path)
        check=cv2.VideoCapture(str(video_path));output_fps=check.get(cv2.CAP_PROP_FPS);decoded=0
        while True:
            ok,frame=check.read()
            if not ok:break
            if frame.shape[:2]!=(H,W):raise ValueError('Unexpected output size')
            decoded+=1
        check.release()
        if decoded!=fid or abs(output_fps-case['fps'])>1e-6:raise ValueError('Output count/FPS differs')
        expected=next(r for r in analysis['comparisons'] if r['case_id']==cid and r['run']==PRIMARY_RUN and r['mode']=='camera_guard')
        if sum(flags)!=expected['flagged_frames'] or sum(scored)!=expected['scored_frames']:raise ValueError('Displayed scores differ from verified inference')
        poster=folder/'media'/f'{cid}_analysis_preview.jpg'
        shutil.copy2(out/('first_alert.jpg' if first_alert is not None else 'preview.jpg'),poster)
        alert_intervals=intervals(flags,case['fps'])
        pd.DataFrame(alert_intervals,columns=['start_frame','end_frame','start_seconds','end_seconds','label']).to_csv(out/'model_alert_intervals.csv',index=False)
        report=dict(case_id=cid,source_frames=fid,decoded_output_frames=decoded,source_fps=case['fps'],output_fps=output_fps,
            output_size=[W,H],video=str(video_path.relative_to(ROOT)),poster=str(poster.relative_to(ROOT)),
            source_media_sha256=case['source_media_sha256'],pose_sha256=case['pose_sha256'],checkpoint_sha256=analysis['checkpoint_sha256'],
            score_csv_sha256=sha256(ROOT/'outputs/retail_pilot_analysis/window_scores.csv'),thresholds=thresholds,
            scored_frames=sum(scored),unverified_alert_frames=sum(flags),first_unverified_alert_frame=first_alert,
            alert_intervals=alert_intervals,primary_run=PRIMARY_RUN,action_confirmed=False,
            protocol='Offline completed-window scores, one-stride expiry, camera-motion guard. Pose confidence and reconstruction error are not theft probabilities.',
            frame_values=str((out/'frame_values.jsonl').relative_to(ROOT)),video_sha256=sha256(video_path))
        (out/'verification.json').write_text(json.dumps(report,indent=2));reports.append(report)
        (output_root/'verification.json').write_text(json.dumps(reports,indent=2))
        print(cid,'verified:',decoded,'frames;',sum(flags),'unverified alert frames',flush=True)
    print('COMPLETE:',len(reports),'annotated videos;',sum(r['decoded_output_frames'] for r in reports),'frames verified')


if __name__=='__main__':main()
