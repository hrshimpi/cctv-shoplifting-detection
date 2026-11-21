"""Train frozen-image-encoder clip baselines; preserve all pose experiments."""
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The active environment has a CUDA torchvision wheel with CPU torch. The
# matching CPU wheel is isolated here; the environment itself is unchanged.
OVERLAY = ROOT / 'outputs/runtime_rgb'
if OVERLAY.is_dir(): sys.path.insert(0, str(OVERLAY))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/matplotlib'))
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from src.data.clip_classification import clip_descriptor, verify_split, portable_probability
from scripts.prepare_mnnit_clips import sha256

OUT = ROOT / 'outputs/mnnit_clip_classification'


def metrics(labels, scores, threshold):
    predictions = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0,1]).ravel()
    return dict(accuracy=float(accuracy_score(labels, predictions)), balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, zero_division=0)), recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)), roc_auc=float(roc_auc_score(labels, scores)),
        pr_auc=float(average_precision_score(labels, scores)), tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn), clips=len(labels))


def main():
    started = time.time()
    torch.set_num_threads(2); torch.manual_seed(2026)
    config_path = ROOT / 'configs/mnnit_clip_v1.json'
    cfg = json.loads(config_path.read_text())
    data = json.loads((OUT / 'data_report.json').read_text())
    if sha256(OUT/'inventory.csv') != data['inventory_sha256'] or sha256(OUT/'sampled_rgb.npz') != data['rgb_cache_sha256']:
        raise ValueError('Prepared dataset differs from verified data report')
    with (OUT/'inventory.csv').open(newline='',encoding='utf-8') as stream: inventory=list(csv.DictReader(stream))
    retained = [i for i,r in enumerate(inventory) if r['included']=='True']
    rows = [dict(inventory[i]) for i in retained]
    labels = np.array([int(r['label']) for r in rows]); groups=np.array([r['duplicate_group'] for r in rows])
    outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=cfg['seed'])
    development,test=next(outer.split(np.zeros(len(rows)),labels,groups))
    inner=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=cfg['seed'])
    train_local,val_local=next(inner.split(np.zeros(len(development)),labels[development],groups[development]))
    splits={'train':development[train_local],'validation':development[val_local],'test':test}
    for split,indices in splits.items():
        for i in indices: rows[i]['split']=split
    verify_split(rows)
    for row in rows:
        if sha256(ROOT/row['path']) != row['source_sha256']:
            raise ValueError('Original video changed')
    split_record=dict(config=cfg,config_sha256=sha256(config_path),inventory_sha256=data['inventory_sha256'],
        source_groups=groups.tolist(), rows=rows, counts={s:dict(Counter(str(labels[i]) for i in ids)) for s,ids in splits.items()},
        warning='Duplicate-aware source-file split only. Actor/session IDs are unavailable; not an independent actor or real-store benchmark.')
    split_path=OUT/'frozen_split.json'
    if split_path.exists() and json.loads(split_path.read_text())!=split_record:
        raise ValueError('Frozen split/config changed; create a new experiment instead')
    split_path.write_text(json.dumps(split_record,indent=2))
    print('Frozen split counts:',split_record['counts'],flush=True)
    weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
    torch.hub.set_dir(str(ROOT/'outputs/rgb_pretrained'))
    model=mobilenet_v3_small(weights=weights).eval()
    model.classifier=torch.nn.Identity()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    weight_path=ROOT/'outputs/rgb_pretrained/checkpoints/mobilenet_v3_small-047dcff4.pth'
    transform=weights.transforms()
    implementation={p:sha256(ROOT/p) for p in ['scripts/train_mnnit_clip_baseline.py','src/data/clip_classification.py']}
    provenance=dict(rgb_cache_sha256=data['rgb_cache_sha256'],encoder_sha256=sha256(weight_path),encoder='torchvision mobilenet_v3_small IMAGENET1K_V1',
        torch=torch.__version__,torchvision=torchvision.__version__,implementation=implementation,features='16x576 frozen embeddings -> mean/std/absolute consecutive changes')
    feature_path=OUT/'frozen_features.npz'; provenance_path=OUT/'feature_provenance.json'
    if feature_path.exists() and provenance_path.exists() and json.loads(provenance_path.read_text())==provenance:
        with np.load(feature_path,allow_pickle=False) as cached:
            all_features=cached['features']
    else:
        with np.load(OUT/'sampled_rgb.npz',allow_pickle=False) as cached:
            rgb=cached['frames']; clip_ids=cached['clip_ids']
        if clip_ids.tolist()!=[r['clip_id'] for r in inventory]: raise ValueError('RGB/metadata order differs')
        descriptors=[]
        with torch.inference_mode():
            for i,frames in enumerate(rgb):
                tensor=torch.from_numpy(frames).permute(0,3,1,2)
                embedding=model(transform(tensor)).cpu().numpy()
                descriptors.append(clip_descriptor(embedding))
                if (i+1)%20==0: print('Frozen RGB encoding',i+1,'/',len(rgb),flush=True)
        all_features=np.stack(descriptors)
        np.savez_compressed(feature_path,features=all_features,clip_ids=clip_ids)
        provenance_path.write_text(json.dumps(provenance,indent=2))
    if all_features.shape!=(len(inventory),1728) or not np.isfinite(all_features).all():
        raise ValueError('Unexpected frozen feature matrix')
    features=all_features[retained]
    reports={};prediction_rows=[]
    for name,columns in [('appearance_control',1152),('appearance_and_changes',1728)]:
        values=features[:,:columns]
        pipeline=make_pipeline(StandardScaler(),LogisticRegression(C=cfg['C'],class_weight=cfg['class_weight'],max_iter=cfg['max_iter'],random_state=cfg['seed']))
        pipeline.fit(values[splits['train']],labels[splits['train']])
        scaler,head=pipeline.steps[0][1],pipeline.steps[1][1]
        path=OUT/(name+'_model.npz')
        np.savez_compressed(path,mean=scaler.mean_,scale=scaler.scale_,coefficient=head.coef_,intercept=head.intercept_,classes=head.classes_)
        scores=portable_probability(values,scaler.mean_,scaler.scale_,head.coef_,head.intercept_)
        if not np.allclose(scores,pipeline.predict_proba(values)[:,1],atol=1e-10):raise ValueError('Portable model differs')
        reports[name]={split:metrics(labels[ids],scores[ids],cfg['probability_threshold']) for split,ids in splits.items()}
        for row,label,probability in zip(rows,labels,scores):
            prediction_rows.append(dict(model=name,clip_id=row['clip_id'],source_filename=row['source_filename'],split=row['split'],label=int(label),
                positive_score=float(probability),prediction=int(probability>=cfg['probability_threshold']),duplicate_group=row['duplicate_group']))
        print(name,json.dumps(reports[name]),flush=True)
    with (OUT/'predictions.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(prediction_rows[0]));writer.writeheader();writer.writerows(prediction_rows)
    result=dict(complete=True,task=cfg['task'],primary_model='appearance_and_changes',control_model='appearance_control',
        threshold=cfg['probability_threshold'],config=cfg,metrics=reports,split_counts=split_record['counts'],
        frozen_split_sha256=sha256(split_path),encoder_provenance=provenance,
        model_sha256={name:sha256(OUT/(name+'_model.npz')) for name in reports},
        predictions_sha256=sha256(OUT/'predictions.csv'),elapsed_seconds=time.time()-started,
        exact_duplicates_excluded=data['exact_duplicates_excluded'],test_selected_parameters=False,old_pose_models_changed=False,
        limitations=['Exploratory staged laboratory clip classification; actor/session independence cannot be established.',
            'A clip label is not an acting-person label or exact action boundary.',
            'Consecutive feature changes are appearance-embedding changes, not optical flow or physical velocity.',
            'Logistic scores are not calibrated theft probabilities. Threshold fixed at 0.5 before fitting.',
            'Do not compare these scores as improvements over RetailS: datasets, labels, splits and tasks differ.',
            'Test results are diagnostic after this run and must not drive repeated parameter tuning.'])
    (OUT/'evaluation.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE in',round(result['elapsed_seconds'],1),'seconds; no pose artifacts changed',flush=True)


if __name__=='__main__':main()
