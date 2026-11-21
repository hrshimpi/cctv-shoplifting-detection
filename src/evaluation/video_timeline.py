"""Readable activity timelines derived from model scores, never action truth."""
import math
from collections import Counter
from src.evaluation.video_status import PRIMARY_RUN

LABELS={'alert':'Possible shoplifting','normal':'Normal activity predicted','unknown':'Unable to assess'}


def assess_frame(frame,threshold):
    if not math.isfinite(threshold) or threshold<=0:raise ValueError('Positive finite threshold required')
    people=frame['people'];assessed=[];flagged=[];unknown=[]
    for person in people:
        score=person['scores'].get(PRIMARY_RUN)
        if score is not None and not math.isfinite(score):raise ValueError('Nonfinite score')
        tid=int(person['track_id'])
        if score is None or frame['camera_motion']:unknown.append(tid)
        else:
            assessed.append(tid)
            if score>=threshold:flagged.append(tid)
    if frame['camera_motion']:state,reason='unknown','camera_motion'
    elif flagged:state,reason='alert','threshold_crossing'
    elif assessed:state,reason='normal','below_threshold'
    elif people:state,reason='unknown','insufficient_tracking'
    else:state,reason='unknown','no_person_tracked'
    if bool(frame['primary_model_alert'])!=bool(flagged):
        raise ValueError('Readable status differs from stored primary model alert')
    return dict(state=state,label=LABELS[state],reason=reason,alert_people=sorted(flagged),
                assessed_people=sorted(assessed),unassessed_people=sorted(unknown))


def summarize_timeline(frames,fps,threshold):
    if not frames or not math.isfinite(fps) or fps<=0:raise ValueError('Frames and positive finite FPS required')
    assessed=[];segments=[]
    for fid,frame in enumerate(frames):
        if frame['frame_id']!=fid:raise ValueError('Frame IDs must be contiguous and zero based')
        if not math.isclose(frame['seconds'],fid/fps,abs_tol=1e-6):raise ValueError('Frame time differs from source FPS')
        row=assess_frame(frame,threshold);assessed.append(row)
        token=(row['state'],row['reason'])
        if not segments or token!=(segments[-1]['state'],segments[-1]['reason']):
            segments.append(dict(state=row['state'],label=row['label'],reason=row['reason'],start_frame=fid,end_frame=fid,
                people=set(),unassessed_people=set()))
        segment=segments[-1];segment['end_frame']=fid
        segment['people'].update(row['alert_people'] if row['state']=='alert' else row['assessed_people'])
        segment['unassessed_people'].update(row['unassessed_people'])
    for segment in segments:
        segment.update(start_seconds=segment['start_frame']/fps,end_seconds_exclusive=(segment['end_frame']+1)/fps,
                       duration_seconds=(segment['end_frame']-segment['start_frame']+1)/fps,
                       people=sorted(segment['people']),unassessed_people=sorted(segment['unassessed_people']))
    # Fixed one-second overview rows keep short pose gaps from producing an
    # unreadable table. Exact subsecond alert spans remain separate and intact.
    bins=[]
    for second in range(math.ceil(len(frames)/fps)):
        start=math.ceil(second*fps);end=min(len(frames),math.ceil((second+1)*fps))
        rows=assessed[start:end]
        if not rows:continue
        counts=Counter(r['state'] for r in rows)
        state='alert' if counts['alert'] else ('normal' if counts['normal'] else 'unknown')
        alert_ids=sorted({p for r in rows for p in r['alert_people']})
        people=alert_ids if alert_ids else sorted({p for r in rows for p in r['assessed_people']})
        bin_alerts=[]
        for segment in segments:
            if segment['state']=='alert' and segment['start_frame']<end and segment['end_frame']>=start:
                a=max(start,segment['start_frame']);b=min(end,segment['end_frame']+1)
                bin_alerts.append(dict(start_seconds=a/fps,end_seconds_exclusive=b/fps,people=segment['people']))
        bins.append(dict(start_seconds=start/fps,end_seconds_exclusive=end/fps,start_frame=start,end_frame=end-1,
            state=state,label=LABELS[state],people=people,frames=end-start,assessed_frames=counts['normal']+counts['alert'],
            alert_frames=counts['alert'],unknown_frames=counts['unknown'],alert_spans=bin_alerts,
            seek_seconds=bin_alerts[0]['start_seconds'] if bin_alerts else start/fps))
    counts=Counter(r['state'] for r in assessed)
    alert_segments=[s for s in segments if s['state']=='alert']
    return dict(total_frames=len(frames),duration_seconds=len(frames)/fps,fps=fps,
        alert_frames=counts['alert'],normal_prediction_frames=counts['normal'],unknown_frames=counts['unknown'],
        assessed_frames=counts['alert']+counts['normal'],assessment_coverage=(counts['alert']+counts['normal'])/len(frames),
        alert_seconds=counts['alert']/fps,unknown_seconds=counts['unknown']/fps,
        alert_periods=len(alert_segments),alert_people=sorted({p for r in assessed for p in r['alert_people']}),
        exact_intervals=segments,alert_intervals=alert_segments,seconds=bins,
        action_confirmed=False,labels_are_model_predictions=True,
        time_convention='Start inclusive, end exclusive; zero-based source frames. One-second rows summarize any alert within that interval.')
