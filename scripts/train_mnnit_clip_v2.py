"""MNNIT clip classifier v2: stronger frozen features, 5-fold CV selection, recall-tuned threshold.

Builds on Harsha's v1 (scripts/train_mnnit_clip_baseline.py) without modifying it:
- Needs scripts/prepare_mnnit_clips.py to have run (inventory.csv + extracted videos).
- Same duplicate-aware split as v1 (same seed and fold logic); train+validation = development set.
- Frames are letterboxed (aspect ratio kept, padded), so any input resolution/aspect works.
- Features: EfficientNet-B0 (ImageNet) on 32 frames -> mean/std/mean abs change, and
  R(2+1)D-18 (Kinetics-400 video backbone, real motion) on 4 x 16-frame windows -> mean/max.
  Backbones stay frozen: with ~140 development clips, fine-tuning would mostly memorise them.
- 5-fold stratified group CV on the development set picks feature set, PCA size and C (by ROC-AUC),
  then picks the threshold on out-of-fold scores to reach a target recall.
- The test split is scored exactly once with the selected model. Note v1 already reported on this
  same test split, so it is not a pristine hold-out; nothing here is selected using test labels.

    python scripts/train_mnnit_clip_v2.py [--target-recall 0.85]
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.data import video_clip_features as vcf
from src.data.clip_classification import verify_split

V1 = ROOT / 'outputs/mnnit_clip_classification'
OUT = ROOT / 'outputs/mnnit_clip_v2'
SEED = 2026
FEATURE_SETS = ['frame', 'video', 'frame_video']
PCA_SIZES = [0, 16, 64]
C_VALUES = [0.01, 0.1, 1.0]


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def metrics(labels, scores, threshold):
    pred = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return dict(threshold=float(threshold), clips=int(len(labels)), accuracy=float(accuracy_score(labels, pred)),
                balanced_accuracy=float(balanced_accuracy_score(labels, pred)),
                precision=float(precision_score(labels, pred, zero_division=0)),
                recall=float(recall_score(labels, pred, zero_division=0)), f1=float(f1_score(labels, pred, zero_division=0)),
                roc_auc=float(roc_auc_score(labels, scores)), pr_auc=float(average_precision_score(labels, scores)),
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


def fit_head(x, y, pca_size, c):
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    pca = None
    if pca_size:
        pca = PCA(n_components=min(pca_size, len(x) - 1), random_state=SEED).fit(z)
        z = pca.transform(z)
    clf = LogisticRegression(C=c, class_weight='balanced', max_iter=5000, random_state=SEED).fit(z, y)
    return dict(mean=scaler.mean_, scale=scaler.scale_,
                pca_mean=pca.mean_ if pca else np.zeros(0), pca_components=pca.components_ if pca else np.zeros((0, 0)),
                coefficient=clf.coef_, intercept=clf.intercept_)


def recall_threshold(labels, scores, target):
    """Highest threshold whose recall >= target (keeps false positives as low as possible)."""
    pos = np.sort(np.asarray(scores)[np.asarray(labels) == 1])
    k = int(np.floor((1 - target) * len(pos) + 1e-9))   # positives allowed below threshold
    return float(pos[min(k, len(pos) - 1)])


def split_like_v1(rows):
    labels = np.array([int(r['label']) for r in rows]); groups = np.array([r['duplicate_group'] for r in rows])
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    dev, test = next(outer.split(np.zeros(len(rows)), labels, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    tr, va = next(inner.split(np.zeros(len(dev)), labels[dev], groups[dev]))
    return dict(train=dev[tr], validation=dev[va], test=test), labels, groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target-recall', type=float, default=0.85)
    ap.add_argument('--no-pretrained', action='store_true', help='Random backbones (code smoke test only; metrics meaningless)')
    args = ap.parse_args()
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    with (V1 / 'inventory.csv').open(newline='', encoding='utf-8') as f:
        inventory = [r for r in csv.DictReader(f)]
    rows = [r for r in inventory if r['included'] == 'True']
    splits, labels, groups = split_like_v1(rows)
    for name, idx in splits.items():
        for i in idx:
            rows[i]['split'] = name
    verify_split(rows)
    v1_split = V1 / 'frozen_split.json'
    if v1_split.exists():
        old = {r['clip_id']: r['split'] for r in json.loads(v1_split.read_text())['rows']}
        if old != {r['clip_id']: r['split'] for r in rows}:
            raise ValueError('Split differs from v1 frozen_split.json')
        print('Split identical to v1 frozen split.', flush=True)
    print('Split counts:', {k: dict(Counter(labels[v].tolist())) for k, v in splits.items()}, flush=True)

    # ---- frozen embeddings (cached) ----
    cache = OUT / 'embeddings.npz'; prov_path = OUT / 'embedding_provenance.json'
    prov = dict(module_sha256=sha256(ROOT / 'src/data/video_clip_features.py'), pretrained=not args.no_pretrained,
                clip_ids=[r['clip_id'] for r in rows], source_sha256=[r['source_sha256'] for r in rows])
    if cache.exists() and prov_path.exists() and json.loads(prov_path.read_text()) == prov:
        with np.load(cache, allow_pickle=False) as z:
            femb, wemb = z['frame_embeddings'], z['window_embeddings']
        print('Reusing cached embeddings', femb.shape, wemb.shape, flush=True)
    else:
        nets = vcf.Backbones(pretrained=not args.no_pretrained)
        print('Backbones:', nets.names, 'on', nets.device, flush=True)
        femb, wemb = [], []
        for i, r in enumerate(rows):
            path = ROOT / r['path']
            n = int(r['frames'])
            seg = vcf.read_letterboxed(path, [(0, n)])[0]
            f, w = nets.embed(seg); femb.append(f); wemb.append(w)
            if (i + 1) % 20 == 0 or i + 1 == len(rows):
                print(f'Embedded {i + 1}/{len(rows)} clips', flush=True)
        femb, wemb = np.stack(femb), np.stack(wemb)
        np.savez_compressed(cache, frame_embeddings=femb, window_embeddings=wemb, clip_ids=np.array(prov['clip_ids']))
        prov['backbones'] = nets.names
        prov_path.write_text(json.dumps({k: v for k, v in prov.items() if k != 'backbones'}))
        (OUT / 'backbones.json').write_text(json.dumps(nets.names, indent=2))
    feats = {k: np.stack([vcf.descriptors(f, w)[k] for f, w in zip(femb, wemb)]) for k in FEATURE_SETS}

    # ---- 5-fold group CV on development set only ----
    dev = np.concatenate([splits['train'], splits['validation']]); test = splits['test']
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(dev)), labels[dev], groups[dev]))
    cv_rows, oof_store = [], {}
    for fs, p, c in product(FEATURE_SETS, PCA_SIZES, C_VALUES):
        x = feats[fs][dev]; y = labels[dev]; oof = np.zeros(len(dev)); aucs = []
        for tr, va in folds:
            head = fit_head(x[tr], y[tr], p, c)
            oof[va] = vcf.head_probability(x[va], head)
            aucs.append(roc_auc_score(y[va], oof[va]))
        cv_rows.append(dict(feature_set=fs, pca=p, C=c, cv_auc_mean=float(np.mean(aucs)), cv_auc_std=float(np.std(aucs)),
                            oof_auc=float(roc_auc_score(y, oof))))
        oof_store[(fs, p, c)] = oof
    # best mean fold AUC; ties -> simpler (smaller PCA, smaller C)
    best = max(cv_rows, key=lambda r: (round(r['cv_auc_mean'], 4), -r['pca'], -r['C']))
    key = (best['feature_set'], best['pca'], best['C'])
    oof = oof_store[key]
    thr = recall_threshold(labels[dev], oof, args.target_recall)
    print('CV best:', best, '| threshold for recall >=', args.target_recall, '=', round(thr, 4), flush=True)
    with (OUT / 'cv_results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cv_rows[0])); w.writeheader(); w.writerows(sorted(cv_rows, key=lambda r: -r['cv_auc_mean']))

    # ---- refit on all development clips; score test once ----
    head = fit_head(feats[key[0]][dev], labels[dev], key[1], key[2])
    model_path = OUT / 'model.npz'
    np.savez_compressed(model_path, threshold=np.array(thr), feature_set=np.array(key[0]), **head)
    scores = vcf.head_probability(feats[key[0]], head)
    result = dict(
        complete=True, protocol='mnnit_clip_v2', selected=dict(feature_set=key[0], pca=key[1], C=key[2]),
        target_recall=args.target_recall, threshold=thr, backbones=json.loads((OUT / 'backbones.json').read_text()),
        pretrained=not args.no_pretrained,
        preprocessing=dict(letterbox=True, frame_size=vcf.FRAME_SIZE, video_size=vcf.VIDEO_SIZE, frame_samples=vcf.FRAME_SAMPLES,
                           video_windows=vcf.VIDEO_WINDOWS, window_frames=vcf.VIDEO_WINDOW_FRAMES, window_stride=vcf.VIDEO_STRIDE),
        split_counts={k: dict(Counter(map(str, labels[v].tolist()))) for k, v in splits.items()},
        cross_validation=dict(folds=5, grid=dict(feature_set=FEATURE_SETS, pca=PCA_SIZES, C=C_VALUES), best=best,
                              out_of_fold=metrics(labels[dev], oof, thr)),
        test=metrics(labels[test], scores[test], thr), test_at_0_5=metrics(labels[test], scores[test], 0.5),
        model_sha256=sha256(model_path), elapsed_seconds=time.time() - started, test_selected_parameters=False,
        notes=['Development = v1 train+validation clips; test = v1 test clips (already reported by v1, not used for any choice here).',
               'Staged laboratory clips; actor/session independence unavailable. Clip-level label, no person/time localisation.',
               'Scores are not calibrated theft probabilities.'])
    v1_eval = V1 / 'evaluation.json'
    if v1_eval.exists():
        v1 = json.loads(v1_eval.read_text())
        result['v1_test_reference'] = v1['metrics'][v1['primary_model']]['test']
    (OUT / 'evaluation.json').write_text(json.dumps(result, indent=2))
    with (OUT / 'predictions.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['clip_id', 'source_filename', 'split', 'label', 'score', 'prediction'])
        for r, y, s in zip(rows, labels, scores):
            w.writerow([r['clip_id'], r['source_filename'], r['split'], int(y), float(s), int(s >= thr)])
    print('TEST:', json.dumps(result['test']), flush=True)
    print('Saved', OUT, f'in {result["elapsed_seconds"]:.0f}s', flush=True)


if __name__ == '__main__':
    main()
