"""Check structural frame/label alignment without inferring temporal or person truth."""
import json
import sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.data.retails import discover_retails_root, SUBSET_DIRS, find_pose_files, gt_dir_for_subset
from src.experiments.quality_dataset import sha256


def audit_file(path, mask_path, windows):
    labels=np.load(mask_path,allow_pickle=False)
    if labels.ndim!=1 or not len(labels) or not np.isin(labels,[0,1]).all():
        raise ValueError(f'Invalid frame mask: {mask_path}')
    records=json.loads(path.read_text())
    row=dict(pose_file=path.name,mask_frames=len(labels),anomaly_frames=int(labels.sum()),
        tracks=len(records),person_frames=0,out_of_bounds_person_frames=0,malformed_poses=0,
        nonfinite_joint_observations=0,finite_confidences_outside_unit_interval=0,
        retained_windows=len(windows),window_label_mismatches=0,window_bounds_errors=0,
        windows_with_missing_source_frames=0,windows_with_missing_person=0,
        first_pose_frame=None,last_pose_frame=None)
    track_ids={}
    for person,frames in records.items():
        ids={int(k) for k in frames}; track_ids[str(person)]=ids
        for fid,record in frames.items():
            fid=int(fid); row['person_frames']+=1
            row['first_pose_frame']=fid if row['first_pose_frame'] is None else min(fid,row['first_pose_frame'])
            row['last_pose_frame']=fid if row['last_pose_frame'] is None else max(fid,row['last_pose_frame'])
            row['out_of_bounds_person_frames']+=int(not 0<=fid<len(labels))
            points=np.asarray(record.get('keypoints',[]),dtype=float)
            if points.shape!=(51,):
                row['malformed_poses']+=1; continue
            points=points.reshape(17,3)
            row['nonfinite_joint_observations']+=int((~np.isfinite(points).all(1)).sum())
            confidence=points[:,2]
            row['finite_confidences_outside_unit_interval']+=int((np.isfinite(confidence)&((confidence<0)|(confidence>1))).sum())
    for window in windows:
        start,end=int(window['start_frame']),int(window['end_frame'])
        if start<0 or end<start or end>=len(labels):
            row['window_bounds_errors']+=1; continue
        row['window_label_mismatches']+=int(int(labels[start:end+1].any())!=int(window['label']))
        person=str(window['person_id'])
        if person not in track_ids:
            row['windows_with_missing_person']+=1
        elif not set(range(start,end+1)).issubset(track_ids[person]):
            row['windows_with_missing_source_frames']+=1
    return row


def main():
    root=discover_retails_root(ROOT / 'data/retails')
    if root is None:
        raise FileNotFoundError('RetailS is missing')
    package=ROOT / 'outputs/colab_handoff/retail_training'
    rows=[]
    for subset in ['test_realworld','test_staged']:
        manifest=json.loads((package / f'data/{subset}.json').read_text())
        sources={key.replace('\\','/'):value for key,value in manifest['sources'].items()}
        grouped=defaultdict(list)
        for window in manifest['metadata']:
            grouped[window['pose_file']].append(window)
        for path in find_pose_files(root / SUBSET_DIRS[subset]):
            mask=gt_dir_for_subset(root / SUBSET_DIRS[subset]) / f'{path.stem}.npy'
            relative=str(path.relative_to(root)).replace('\\','/')
            mask_relative=str(mask.relative_to(root)).replace('\\','/')
            if sha256(path)!=sources[relative] or sha256(mask)!=sources[mask_relative]:
                raise ValueError('Pose or label source differs from the training bundle')
            rows.append(dict(subset=subset,**audit_file(path,mask,grouped.pop(path.name,[]))))
        if grouped:
            raise ValueError('Retained windows reference absent pose files')
        print('Audited:',subset,flush=True)
    df=pd.DataFrame(rows)
    counts=['person_frames','out_of_bounds_person_frames','malformed_poses','nonfinite_joint_observations',
        'finite_confidences_outside_unit_interval','retained_windows','window_label_mismatches','window_bounds_errors',
        'windows_with_missing_source_frames','windows_with_missing_person']
    totals={k:int(df[k].sum()) for k in counts}
    fatal=['out_of_bounds_person_frames','malformed_poses','window_label_mismatches','window_bounds_errors',
           'windows_with_missing_source_frames','windows_with_missing_person']
    findings=[f"Checked {len(df)} source pose/mask pairs and {totals['retained_windows']} retained windows against original source hashes.",
        'Window labels were recomputed using ANY anomalous frame in the inclusive start/end interval; person frame IDs were checked for complete temporal support.',
        f"Raw poses contain {totals['nonfinite_joint_observations']} nonfinite joint observations; the existing quality pipeline masks missing observations before normalization/loss.",
        'These checks do not establish whether the masks match visible actions in source RGB. RetailS source RGB and person-specific action labels are unavailable locally.',
        'The RetailS-specific yx axis interpretation remains an empirical assumption. No offset or coordinate changes are made by this audit.']
    out=ROOT / 'outputs/validation_alignment'; out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out / 'per_file.csv',index=False)
    result=dict(structural_checks_passed=all(totals[k]==0 for k in fatal),files=len(df),totals=totals,
        findings=findings,semantic_alignment_status='unverified_without_source_rgb',
        dataset_coordinate_order='existing explicit RetailS yx interpretation; external video poses use xy',
        original_bundle_manifest_sha256=sha256(package / 'bundle_manifest.json'))
    (out / 'report.json').write_text(json.dumps(result,indent=2))
    (out / 'report.md').write_text('\n'.join(['# Pose and label alignment audit','',
        f"Structural checks passed: {result['structural_checks_passed']}",'']+['- '+x for x in findings]+
        ['','## Counts','']+[f'- {k}: {v}' for k,v in totals.items()])+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
