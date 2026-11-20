"""Score reviewed retail clips with frozen A/B, isolating verified camera motion.

This reports coverage and model flags, not detection accuracy against AI notes.
Original benchmark reports, cached poses, checkpoints and videos are preserved.
"""
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/matplotlib'))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scripts.run_video_inference import load_tracks
from scripts.run_research_phase import make_model
from scripts.train_colab import score
from src.data.preprocessing import build_windows
from src.data.quality import prepare_track,accepted_window
from src.data.scene_segments import split_track
from src.experiments.quality_dataset import features,sha256
from src.experiments.artifacts import run_paths


def sequences(tracks,cfg,exclusions):
    normalized,masks,metadata=[],[],[]
    candidate=0
    for track_id,track in tracks.items():
        for scene,segment in split_track(track,exclusions):
            prepared=prepare_track(segment,cfg,coordinate_order='xy')
            for window in build_windows(prepared['frame_ids'],prepared['keypoints'],prepared['visibility_mask'],cfg['window_length'],cfg['stride']):
                candidate+=1
                start=int(np.searchsorted(prepared['frame_ids'],window['start_frame']))
                if not accepted_window(window,prepared['anchor_valid'][start:start+cfg['window_length']],cfg):continue
                normalized.append(window['keypoints']);masks.append(window['visibility_mask'])
                metadata.append(dict(track_id=track_id,scene_id=scene,start_frame=window['start_frame'],end_frame=window['end_frame'],
                    scene_last_frame=max(segment.frames)))
    arrays=dict(normalized=np.stack(normalized),visibility=np.stack(masks)) if normalized else None
    return arrays,metadata,candidate


def completed_timeline(values,metadata,tracks,frame_count,stride):
    timeline=np.full(frame_count,np.nan)
    # Each person's latest completed window expires after one stride and at a
    # scene boundary. A missing source person observation cannot retain a flag.
    for value,meta in zip(values,metadata):
        start=meta['end_frame'];end=min(start+stride,meta['scene_last_frame']+1,frame_count)
        ids=[fid for fid in range(start,end) if fid in tracks[meta['track_id']].frames]
        timeline[ids]=np.fmax(timeline[ids],float(value))
    return timeline


def main():
    torch.set_num_threads(2)
    review_folder=ROOT/'outputs/manual_validation'
    manifest=json.loads((review_folder/'review_manifest.json').read_text())
    review=json.loads((review_folder/'ai_review.json').read_text())
    if review['manifest_id']!=manifest['manifest_id'] or review['reviewer_type']!='ai_assisted':
        raise ValueError('Review provenance differs')
    private={r['case_id']:r for r in json.loads((review_folder/'_private_sources.json').read_text())}
    motion=json.loads((ROOT/'configs/retail_camera_motion.json').read_text())
    if motion['manifest_id']!=manifest['manifest_id']:raise ValueError('Camera-motion configuration belongs to different media')
    models={}
    for family in ('a','b'):
        for seed in (0,1,2):
            name=f'expanded_{family}_seed{seed}';folder,_=run_paths(name)
            saved=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
            model=make_model(family,saved['feature_dim'],saved['config']);model.load_state_dict(saved['model_state_dict'])
            threshold=float(np.percentile(np.load(folder/'validation_errors.npy',allow_pickle=False),saved['config']['threshold_percentile']))
            models[name]=(model,saved,threshold,sha256(folder/'model.pt'))
    cfg=next(iter(models.values()))[1]['config']
    if any(saved['config']!=cfg for _,saved,_,_ in models.values()):raise ValueError('Run settings differ')
    results,predictions,quality,plots=[],[],[],[]
    timelines={}
    for case in manifest['cases']:
        cid=case['case_id'];source=ROOT/private[cid]['source_path'];pose_dir=ROOT/private[cid]['pose_dir']
        if sha256(source)!=case['source_sha256'] or sha256(pose_dir/'poses.json')!=case['pose_sha256']:
            raise ValueError('Source data changed since review')
        tracks=load_tracks(pose_dir/'poses.json',source)
        intervals=motion['cases'].get(cid,{}).get('excluded_intervals',[])
        if any(not 0<=start<=end<case['frame_count'] for start,end in intervals):raise ValueError('Camera-motion bounds outside video')
        points=np.stack([frame.keypoints for track in tracks.values() for frame in track.frames.values()])
        visible=np.isfinite(points).all(-1)&(points[...,2]>=cfg['confidence_threshold'])
        quality.append(dict(case_id=cid,tracks=len(tracks),person_frames=len(points),
            mean_joint_visibility=float(visible.mean()),both_wrists_visible_fraction=float(visible[:,[9,10]].all(1).mean()),
            normalization_anchors_visible_fraction=float(visible[:,[5,6,11,12]].all(1).mean()),
            median_observations_per_track=float(np.median([len(t.frames) for t in tracks.values()])),
            note='Track counts/durations are descriptive; identity switches cannot be measured without person ground truth.'))
        baseline=sequences(tracks,cfg,[])
        guarded=sequences(tracks,cfg,intervals) if intervals else baseline
        for mode,(arrays,metadata,candidate) in [('original',baseline),('camera_guard',guarded)]:
            for name,(model,saved,threshold,model_hash) in models.items():
                if arrays is not None:
                    values,masks=features(arrays,saved['experiment'])
                    errors=score(model,values,masks,cfg['batch_size'],torch.device('cpu'))
                else:errors=np.empty(0)
                timeline=completed_timeline(errors,metadata,tracks,case['frame_count'],cfg['stride'])
                valid=np.isfinite(timeline)
                results.append(dict(case_id=cid,run=name,mode=mode,source_frames=case['frame_count'],
                    candidate_windows=candidate,accepted_windows=len(metadata),flagged_windows=int((errors>=threshold).sum()),
                    scored_frames=int(valid.sum()),score_coverage=float(valid.mean()),flagged_frames=int((valid&(timeline>=threshold)).sum()),
                    flag_fraction_of_scored_frames=float((timeline[valid]>=threshold).mean()) if valid.any() else None,
                    threshold=threshold,checkpoint_sha256=model_hash))
                for meta,error in zip(metadata,errors):
                    predictions.append(dict(case_id=cid,run=name,mode=mode,**meta,score=float(error),above_threshold=bool(error>=threshold)))
                if name.endswith('seed0'):timelines[(cid,name,mode)]=timeline/threshold
        print('Scored:',cid,'baseline windows',len(baseline[1]),'guarded windows',len(guarded[1]),flush=True)
    out=ROOT/'outputs/retail_pilot_analysis';out.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(results);table.to_csv(out/'coverage_and_flags.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'window_scores.csv',index=False)
    pd.DataFrame(quality).to_csv(out/'pose_visibility.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(13,7))
    for ax,cid in zip(axes,['RV002','RV007']):
        case=next(c for c in manifest['cases'] if c['case_id']==cid)
        for name,color in [('expanded_a_seed0','tab:blue'),('expanded_b_seed0','tab:orange')]:
            for mode,style in [('original',':'),('camera_guard','-')]:
                ax.plot(np.arange(case['frame_count'])/case['fps'],timelines[(cid,name,mode)],style,color=color,label=f"{name.split('_')[1].upper()} {mode}")
        for start,end in motion['cases'][cid]['excluded_intervals']:
            ax.axvspan(start/case['fps'],(end+1)/case['fps'],color='gray',alpha=.25,label='Verified camera motion')
        ax.axhline(1,color='black',linestyle='--',label='Frozen validation threshold')
        ax.set(title=f'{cid}: completed-window score / fixed threshold',xlabel='Source seconds',ylabel='Score / threshold')
    axes[0].legend(ncol=3,fontsize=8);fig.tight_layout()
    plot=out/'camera_motion_scores.png';fig.savefig(plot,dpi=150);plt.close(fig);plots.append(str(plot.relative_to(ROOT)))
    result=dict(manifest_id=manifest['manifest_id'],review_sha256=sha256(review_folder/'ai_review.json'),
        camera_guard_sha256=sha256(ROOT/'configs/retail_camera_motion.json'),checkpoint_sha256={k:v[3] for k,v in models.items()},
        source_cases=len(manifest['cases']),models=len(models),reviewer_type='ai_assisted',accuracy_metrics_computed=False,
        observations='Seven retail scenes reviewed from sampled source/overlay frames. All action classifications remain uncertain; no positive or negative training labels were created.',
        findings=[
            'RV002 contains a digital zoom; RV007 contains a camera pan. Guarded scoring excludes the reviewed intervals and splits tracks before interpolation/smoothing/windowing.',
            'The frozen RetailS model weights and p95 validation thresholds are unchanged. Original benchmark reports are preserved.',
            'No theft accuracy, recall or F1 can be established from uncertain AI observations on previously inspected clips.',
            'Missing wrists/hips and fragmented overhead tracks limit the usable pose evidence; visibility statistics are descriptive, not pose ground-truth accuracy.'],
        pose_quality=quality,comparisons=results,plots=plots,
        limitations=['AI sampling can miss brief actions between sampled frames.','No independent person/action labels are available.','Engineering camera exclusions were selected from visible camera motion, not model scores.'])
    (out/'report.json').write_text(json.dumps(result,indent=2))
    lines=['# Autonomous retail pilot analysis','']+['- '+x for x in result['findings']]
    lines+=['','## Visual observations','']+[f"- {r['case_id']}: {r['notes']}" for r in review['reviews']]
    lines+=['','## Coverage and model flags (seed 0)','',table[table.run.str.endswith('seed0')][['case_id','run','mode','accepted_windows','score_coverage','flagged_windows']].to_string(index=False)]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(table[table.case_id.isin(['RV002','RV007'])].to_string(index=False))


if __name__=='__main__':main()
