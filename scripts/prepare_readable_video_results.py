"""Render simple model labels and export traceable per-video activity timelines."""
import argparse
import csv
import json
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
from scripts.prepare_mnnit_clips import sha256
from src.evaluation.video_status import PRIMARY_RUN,active_window
from src.evaluation.video_timeline import assess_frame,summarize_timeline
from src.experiments.artifacts import run_paths

SCENES={
 'RV001':('Shop entrance','People pass through a crowded entrance. Overlap makes tracking difficult.'),
 'RV002':('Electronics counter','A customer handles a display device, later moves a hand toward the waist, and leaves the view. The item movement is unresolved.'),
 'RV003':('Overhead checkout','A customer moves near the shelf and counter while the attendant stays behind the counter. Hand visibility is limited.'),
 'RV004':('Overhead checkout','A shopper reaches around the display, moves hands toward the body, and leaves. Item placement is unclear.'),
 'RV005':('Clothing store counter','A customer browses near the counter and later moves a hand toward a trouser pocket. An item transfer is unresolved.'),
 'RV006':('Grocery aisle','Customers browse items and walk through the aisle. Shelves obscure parts of their bodies.'),
 'RV007':('Electronics checkout','Customers stand near counters with small items or bags. The camera pans between views.')}
COLORS={'alert':(45,150,245),'normal':(115,200,115),'unknown':(175,180,188)}
W,H=1280,800


def clock(seconds):
    return f'{int(seconds)//60:02d}:{seconds%60:05.2f}'


def text(canvas,value,x,y,size=.6,color=(235,239,245),thickness=1):
    cv2.putText(canvas,str(value),(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,thickness,cv2.LINE_AA)


def wrap(canvas,value,x,y,width=270,size=.52,color=(210,220,230)):
    line=''
    for word in value.split():
        candidate=(line+' '+word).strip()
        if line and cv2.getTextSize(candidate,cv2.FONT_HERSHEY_SIMPLEX,size,1)[0][0]>width:
            text(canvas,line,x,y,size,color);y+=23;line=word
        else:line=candidate
    if line:text(canvas,line,x,y,size,color);y+=23
    return y


def draw_frame(frame,record,case,assessment,threshold):
    canvas=np.full((H,W,3),(27,24,21),np.uint8)
    state=assessment['state'];color=COLORS[state]
    headline={'alert':'POSSIBLE SHOPLIFTING','normal':'NORMAL ACTIVITY PREDICTED','unknown':'UNABLE TO ASSESS'}[state]
    cv2.rectangle(canvas,(0,0),(W,79),(42,36,29),-1)
    text(canvas,headline,20,34,.85,color,2)
    text(canvas,f"{case['case_id']} | Model prediction",20,64,.52)
    text(canvas,f"{clock(record['seconds'])} / {clock(case['frame_count']/case['fps'])}",930,38,.64)
    text(canvas,'Source frame '+str(record['frame_id']),1050,65,.44)
    # The source occupies most of the canvas. Boxes are extents of visible pose
    # joints, not newly inferred detector boxes or independently verified actors.
    view=cv2.resize(frame,(960,720),interpolation=cv2.INTER_CUBIC)
    for person in record['people']:
        score=person['scores'].get(PRIMARY_RUN)
        pstate='unknown' if score is None or record['camera_motion'] else ('alert' if score>=threshold else 'normal')
        c=COLORS[pstate];points=np.asarray(person['keypoints_xy_confidence'],dtype=float)
        visible=(points[:,2]>=.3)&np.isfinite(points).all(1)
        if visible.any():
            xy=points[visible,:2]*[960/frame.shape[1],720/frame.shape[0]]
            low=np.clip(xy.min(0)-8,[0,0],[959,719]).astype(int)
            high=np.clip(xy.max(0)+8,[0,0],[959,719]).astype(int)
            cv2.rectangle(view,tuple(low),tuple(high),c,2)
            label='Person '+str(person['track_id'])
            x=min(int(low[0]),760);y=max(25,int(low[1])-9)
            size=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.6,2)[0]
            cv2.rectangle(view,(x,max(0,y-23)),(min(959,x+size[0]+9),min(719,y+5)),(30,30,30),-1)
            text(view,label,x+3,y,.6,c,2)
    canvas[80:,0:960]=view
    cv2.line(canvas,(960,80),(960,H),(80,82,85),1)
    text(canvas,'Activity summary',982,116,.64,thickness=1)
    text(canvas,SCENES[case['case_id']][0],982,145,.48)
    explanation={'alert':'The model flagged unusual movement. Review the highlighted person and time.',
        'normal':'No alert was raised for the people the model could assess.',
        'unknown':'The camera moved or tracking did not provide enough information.'}[state]
    wrap(canvas,explanation,982,181)
    text(canvas,f"People tracked: {len(record['people'])}",982,281,.53)
    text(canvas,f"People assessed: {len(assessment['assessed_people'])}",982,307,.53)
    y=344
    for person in record['people']:
        score=person['scores'].get(PRIMARY_RUN)
        pstate='unknown' if score is None or record['camera_motion'] else ('alert' if score>=threshold else 'normal')
        label={'alert':'Possible shoplifting','normal':'Normal predicted','unknown':'Unable to assess'}[pstate]
        text(canvas,'Person '+str(person['track_id']),982,y,.55,COLORS[pstate],1)
        text(canvas,label,982,y+23,.48,COLORS[pstate]);y+=66
    if not record['people']:text(canvas,'No person currently tracked',982,y,.43)
    wrap(canvas,'Labels are automated predictions, not confirmed action annotations.',982,720,size=.43,color=(175,185,197))
    return canvas


def conclusion(summary):
    if summary['alert_periods']:
        people=', '.join('Person '+str(p) for p in summary['alert_people'])
        return (f"Possible shoplifting was flagged in {summary['alert_periods']} model-alert periods, totaling {summary['alert_seconds']:.2f} seconds, involving {people}. "
                f"The model assessed {summary['assessment_coverage']:.1%} of the video. Review the marked times; the alerts do not confirm that shoplifting occurred.")
    return (f"Normal activity was predicted during assessed portions ({summary['assessment_coverage']:.1%} of the video), with no shoplifting alert. "
            f"The remaining {summary['unknown_seconds']:.2f} seconds could not be assessed. No alert does not establish that shoplifting was absent.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true',help='Check existing outputs against frozen scores without rendering')
    args=parser.parse_args()
    folder=ROOT/'outputs/manual_validation';out=ROOT/'outputs/readable_video_results';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((folder/'review_manifest.json').read_text())
    verified={r['case_id']:r for r in json.loads((ROOT/'outputs/video_analysis/verification.json').read_text())}
    model,_=run_paths(PRIMARY_RUN)
    scores_path=ROOT/'outputs/retail_pilot_analysis/window_scores.csv'
    windows=defaultdict(list)
    values=pd.read_csv(scores_path)
    for row in values[(values['mode']=='camera_guard') & (values['run']==PRIMARY_RUN)].to_dict('records'):
        windows[(row['case_id'],int(row['track_id']))].append(row)
    reports=[]
    for case in manifest['cases']:
        cid=case['case_id'];old=verified[cid];sidecar=ROOT/old['frame_values']
        if sha256(model/'model.pt')!=old['checkpoint_sha256'][PRIMARY_RUN] or sha256(scores_path)!=old['score_csv_sha256']:
            raise ValueError('Frozen scores or primary checkpoint changed')
        source=folder/case['source_file']
        if sha256(source)!=old['source_media_sha256'] or sha256(folder/'media'/f'{cid}_model_analysis.mp4')!=old['video_sha256']:
            raise ValueError('Source or original diagnostic video changed')
        records=[json.loads(line) for line in sidecar.read_text().splitlines()]
        if len(records)!=case['frame_count']:raise ValueError('Incomplete per-frame evidence')
        for frame in records:
            for person in frame['people']:
                active=None if frame['camera_motion'] else active_window(windows[(cid,int(person['track_id']))],frame['frame_id'],15)
                expected=None if active is None else float(active['score'])
                if person['scores'][PRIMARY_RUN]!=expected:
                    raise ValueError('Readable input differs from original per-person window inference')
        threshold=old['thresholds'][PRIMARY_RUN]
        summary=summarize_timeline(records,case['fps'],threshold)
        if summary['alert_frames']!=old['unverified_alert_frames'] or summary['assessed_frames']!=old['scored_frames']:
            raise ValueError('Readable timeline differs from verified inference')
        summary.update(case_id=cid,scene=SCENES[cid][0],scene_summary=SCENES[cid][1],
            conclusion=conclusion(summary),frame_values_sha256=sha256(sidecar),source_sha256=sha256(source),
            primary_model=PRIMARY_RUN,model_sha256=old['checkpoint_sha256'][PRIMARY_RUN],threshold=threshold,
            video=f'media/{cid}_activity.mp4',poster=f'media/{cid}_activity_preview.jpg')
        if args.verify_only:
            saved=json.loads((out/f'{cid}_timeline.json').read_text())
            if any(saved.get(key)!=value for key,value in summary.items()):
                raise ValueError('Saved activity timeline differs from recomputed frozen inference')
            if sha256(folder/summary['video'])!=saved['video_sha256']:
                raise ValueError('Rendered activity video changed')
            reports.append(saved)
            print(cid,'verified against frozen checkpoint and original window scores',flush=True)
            continue
        (out/f'{cid}_timeline.json').write_text(json.dumps(summary,indent=2))
        with (out/f'{cid}_timeline.csv').open('w',newline='',encoding='utf-8') as stream:
            columns=['start_seconds','end_seconds_exclusive','label','people','assessed_frames','frames','alert_frames','unknown_frames']
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
            for row in summary['seconds']:writer.writerow({k:row[k] for k in columns})
        target=folder/summary['video'];temporary=target.with_suffix('.partial.mp4')
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}',
            '-r',str(case['fps']),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        cap=cv2.VideoCapture(str(source));fid=0
        preview_frame=summary['alert_intervals'][0]['start_frame'] if summary['alert_intervals'] else min(300,len(records)-1)
        try:
            for record in records:
                ok,frame=cap.read()
                if not ok:raise ValueError('Source decode failed')
                canvas=draw_frame(frame,record,case,assess_frame(record,threshold),threshold)
                process.stdin.write(canvas.tobytes())
                if fid==preview_frame:cv2.imwrite(str(folder/summary['poster']),canvas)
                fid+=1
            if cap.read()[0]:raise ValueError('Unexpected extra source frames')
            process.stdin.close();error=process.stderr.read().decode(errors='replace');code=process.wait()
            if code:raise RuntimeError(error)
        except BaseException:
            if process.poll() is None:process.kill();process.wait()
            raise
        finally:cap.release()
        temporary.replace(target)
        check=cv2.VideoCapture(str(target));fps=check.get(cv2.CAP_PROP_FPS);count=0
        while True:
            ok,frame=check.read()
            if not ok:break
            if frame.shape[:2]!=(H,W):raise ValueError('Wrong readable output dimensions')
            count+=1
        check.release()
        if count!=case['frame_count'] or not np.isclose(fps,case['fps']):raise ValueError('Readable video count/FPS differs')
        summary.update(decoded_output_frames=count,output_fps=fps,output_size=[W,H],video_sha256=sha256(target))
        (out/f'{cid}_timeline.json').write_text(json.dumps(summary,indent=2))
        reports.append(summary)
        print(cid,'verified',count,'frames;',summary['alert_periods'],'alert periods;',len(summary['seconds']),'timeline rows',flush=True)
    report=dict(manifest_id=manifest['manifest_id'],cases=reports,complete=True,total_verified_frames=sum(r['decoded_output_frames'] for r in reports))
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE:',len(reports),'readable videos and timelines; original diagnostic videos preserved',flush=True)


if __name__=='__main__':main()
