"""research_v2: frame-level RetailS protocol with labelled dev folds, run one step at a time.

    python scripts/run_research_v2.py --step 0     # dev folds
    python scripts/run_research_v2.py --step 1     # baseline (steps 0b + 1: frame protocol + metrics)
    python scripts/run_research_v2.py --step 2     # scoring variants + per-camera normalisation
    python scripts/run_research_v2.py --step 3     # temporal smoothing
    python scripts/run_research_v2.py --step 4     # all train files + augmentation
    python scripts/run_research_v2.py --step 5     # window length
    python scripts/run_research_v2.py --step 6     # experiment F (STG-NF style), only on plateau
    python scripts/run_research_v2.py --step all

Rules enforced here:
* Only real RetailS files are read; every labelled test frame is scored (unscored = subset minimum).
* Each choice is made on one dev fold and applied to the other (cross-fitting); a video's reported
  score never depends on its own labels. Thresholds come from NORMAL training-validation data.
* A step's change is adopted on a fold only if it does not lower that fold's dev metric (mean
  per-subset frame ROC-AUC, averaged over seeds); a drop larger than the CI half-width is flagged.
* research_v1 / experiment_a-e outputs are read-only; everything is written under */research_v2/.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from src.data.retails import discover_retails_root
from src.evaluation import folds as F
from src.evaluation.frame_protocol import flatten, frame_scores
from src.evaluation.metrics import detection_metrics, fast_roc_auc, grouped_bootstrap
from src.evaluation.v2_scoring import Calibrator, window_score
from src.experiments.splits import split_by_video
from src.experiments.v2_data import WindowSet, build_windows_v2, video_cameras
from src.experiments.v2_train import ae_errors, device_and_dtype, flow_scores, make_ae, to_device, train_model

SUBSETS = ('test_realworld', 'test_staged')
OUT = ROOT / 'outputs'
CACHE = OUT / 'cache/research_v2'
CKPT = OUT / 'checkpoints/research_v2'
LOGS = OUT / 'logs/research_v2'
V1_CKPT = OUT / 'checkpoints/research_v1'
STEP_NAMES = {0: 'dev folds', 1: 'baseline (frame protocol + metrics)', 2: 'scoring variant + camera norm',
              3: 'temporal smoothing', 4: 'all train files + augmentation', 5: 'window length',
              6: 'experiment F (STG-NF flow)'}


def log(*args):
    print(*args, flush=True)


def jdump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default), encoding='utf-8')
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def h8(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8]


# =============================================================================================
# Context: config, data, folds
# =============================================================================================

class Context:
    def __init__(self, cfg_path):
        self.cfg = yaml.safe_load(Path(cfg_path).read_text())
        if self.cfg.get('version') != 'research_v2':
            raise ValueError('Config must declare version: research_v2')
        self.root = discover_retails_root(ROOT / 'data/retails')
        if self.root is None:
            raise FileNotFoundError('RetailS dataset is missing (run the notebook download cell first)')
        self.device, amp = device_and_dtype()
        self.amp = amp
        log(f'device={self.device} amp={amp} torch={torch.__version__}')
        self._windows = {}
        self._bundles = {}
        self._frames = {}

    # ---- windows -------------------------------------------------------------------------
    def windows(self, subset, length, stride, train_files=None, max_windows=None) -> WindowSet:
        key = (subset, length, stride, train_files, max_windows)
        if subset == 'train' and key not in self._windows:
            for k in [k for k in self._windows if k[0] == 'train']:   # keep one training set in RAM
                del self._windows[k]
        if key not in self._windows:
            started = time.perf_counter()
            self._windows[key] = cached_windows(self.root, subset, self.cfg, length, stride, train_files, max_windows)
            ws = self._windows[key]
            log(f'windows {subset} L={length} S={stride}: {len(ws)} kept, {int(ws.labels.sum())} positive, '
                f'counts={ws.counts} ({time.perf_counter() - started:.0f}s)')
        return self._windows[key]

    def test(self, length, stride):
        return {s: self.windows(s, length, stride) for s in SUBSETS}

    # ---- folds ---------------------------------------------------------------------------
    def videos(self):
        L, S = self.cfg['baseline_window']
        rows = []
        for subset, ws in self.test(L, S).items():
            cams = video_cameras(ws)
            for stem, mask in sorted(ws.frame_labels.items()):
                rows.append(dict(subset=subset, camera=cams[stem], stem=stem,
                                 normal=int((mask == 0).sum()), anomaly=int((mask == 1).sum())))
        return rows

    def folds(self):
        path = LOGS / 'folds.json'
        if not path.exists():
            raise FileNotFoundError('Run step 0 first')
        return json.loads(path.read_text())['assignment']


def cached_windows(root, subset, cfg, length, stride, train_files, max_windows) -> WindowSet:
    from src.experiments.v2_data import clean_key
    key = h8(dict(c=clean_key(cfg), s=subset, L=length, S=stride, f=train_files, m=max_windows,
                  seed=cfg['sample_seed'], code=_file_hash(ROOT / 'src/experiments/v2_data.py')))
    path = CACHE / 'windows' / f'{subset}_L{length}_S{stride}_{key}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        meta = json.loads(str(d.pop('meta_json')))
        frame_labels = {k[3:]: d.pop(k) for k in list(d) if k.startswith('fl_')}
        return WindowSet(**d, window_length=length, stride=stride, subset=subset, frame_labels=frame_labels,
                         counts=meta['counts'])
    ws = build_windows_v2(root, subset, cfg, CACHE, length, stride, max_files=train_files, max_windows=max_windows,
                          seed=cfg['sample_seed'], workers=cfg.get('workers'))
    arrays = {k: getattr(ws, k) for k in ('xy', 'vis', 'labels', 'pos_frac', 'pose_file', 'person', 'camera', 'video',
                                         'start', 'end', 'track')}
    arrays.update({f'fl_{k}': v for k, v in ws.frame_labels.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.partial.npz')
    np.savez(tmp, meta_json=np.array(json.dumps(dict(counts=ws.counts))), **arrays)
    tmp.replace(path)
    return ws


def _file_hash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


# =============================================================================================
# Models -> score bundles (per model key and seed; cached on disk)
# =============================================================================================

def model_spec(cfg, recipe, window, kind='ae'):
    r = cfg['recipes'][recipe]
    spec = dict(kind=kind, recipe=recipe, window=list(window), train_files=r['train_files'],
                max_train_windows=r['max_train_windows'], split_seed=cfg['split_seed'], val_fraction=cfg['val_fraction'])
    if kind == 'ae':
        spec.update(train=dict(epochs=r['epochs'], batch_size=r['batch_size'], learning_rate=r['learning_rate'],
                               cosine=r['cosine'], patience=r['patience'], amp=r['amp'], augment=r['augment'],
                               hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'], dropout=cfg['dropout']))
    else:
        spec.update(train=copy.deepcopy(cfg['flow']))
    name = f"{kind}_{recipe}_w{window[0]}s{window[1]}"
    return name + '_' + h8(spec), spec


def bundle(ctx, model_key, spec, seed):
    """Scores for test subsets + normal validation windows, cached as .npz."""
    mem = (model_key, seed)
    if mem in ctx._bundles:
        return ctx._bundles[mem]
    folder = CACHE / 'scores' / model_key / f'seed{seed}'
    path = folder / 'bundle.npz'
    L, S = spec['window']
    tests = ctx.test(L, S)
    if not path.exists():
        train_ws = ctx.windows('train', L, S, spec['train_files'], spec['max_train_windows'])
        tr_idx, va_idx, split = split_by_video(train_ws.meta_records(), spec['val_fraction'], spec['split_seed'])
        model, history, best_epoch, source = None, [], None, 'trained in research_v2'
        ck_dir = CKPT / model_key / f'seed{seed}'
        ck = ck_dir / 'model.pt'
        v1 = V1_CKPT / f'expanded_a_seed{seed}' / 'model.pt'
        is_v1_baseline = spec['kind'] == 'ae' and spec['recipe'] == 'v1' and tuple(spec['window']) == (30, 15)
        if ck.exists():
            saved = torch.load(ck, map_location='cpu', weights_only=False)
            model = _new_model(spec, ctx).to(ctx.device)
            model.load_state_dict(saved['state_dict'])
            history, best_epoch, source = saved['history'], saved['best_epoch'], saved['source']
            log(f'reusing {model_key} seed {seed} ({source})')
        elif is_v1_baseline and v1.exists():
            saved = torch.load(v1, map_location='cpu', weights_only=False)   # read-only
            model = make_ae(spec['train']).to(ctx.device)
            model.load_state_dict(saved['model_state_dict'])
            best_epoch = saved.get('best_epoch')
            source = f'research_v1 checkpoint {v1.relative_to(ROOT)} (read-only, best epoch {best_epoch})'
            log('re-scoring', source)
        else:
            log(f'training {model_key} seed {seed}: {len(tr_idx)} train / {len(va_idx)} validation windows')
            txy, tvis = to_device(train_ws, tr_idx, ctx.device)
            vxy, vvis = to_device(train_ws, va_idx, ctx.device)
            tcfg = dict(spec['train'], eval_batch_size=ctx.cfg['eval_batch_size'],
                        amp_dtype=ctx.amp if spec['train'].get('amp', spec['kind'] == 'flow') else None)
            if spec['kind'] == 'flow':
                tcfg['amp_dtype'] = None           # flow runs in fp32 (log-det stability)
            model, history, best_epoch = train_model(spec['kind'], txy, tvis, vxy, vvis, tcfg, seed, log)
            del txy, tvis
            ck_dir.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict=model.state_dict(), spec=spec, seed=seed, history=history,
                            best_epoch=best_epoch, source=source, split=split), ck.with_suffix('.partial'))
            ck.with_suffix('.partial').replace(ck)
            jdump(LOGS / 'models' / model_key / f'seed{seed}_history.json', dict(history=history, best_epoch=best_epoch, split=split))
        val_ws = train_ws.take(va_idx)
        arrays = dict(val_idx=va_idx, val_vis=val_ws.vis, val_pose_file=val_ws.pose_file, val_track=val_ws.track,
                      val_start=val_ws.start, val_end=val_ws.end, val_camera=val_ws.camera)
        vxy, vvis = to_device(train_ws, va_idx, ctx.device, fp32=True)
        arrays['val'] = _score(spec, model, vxy, vvis, ctx)
        for s, ws in tests.items():
            xy, vis = to_device(ws, np.arange(len(ws)), ctx.device, fp32=True)
            arrays[s] = _score(spec, model, xy, vis, ctx)
        del model
        if ctx.device.type == 'cuda':
            torch.cuda.empty_cache()
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / 'bundle.partial.npz'
        np.savez(tmp, meta_json=np.array(json.dumps(dict(source=source, best_epoch=best_epoch, spec=spec))), **arrays)
        tmp.replace(path)
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    meta = json.loads(str(d.pop('meta_json')))
    from types import SimpleNamespace
    val = SimpleNamespace(vis=d['val_vis'], pose_file=d['val_pose_file'], track=d['val_track'], start=d['val_start'],
                          end=d['val_end'], camera=d['val_camera'])
    b = dict(kind=spec['kind'], spec=spec, meta=meta, val=d['val'], val_ws=val,
             test={s: d[s] for s in SUBSETS}, test_ws=tests)
    ctx._bundles[mem] = b
    return b


def _new_model(spec, ctx):
    if spec['kind'] == 'ae':
        return make_ae(spec['train'])
    from src.models.stg_nf import STGNF
    return STGNF(2, spec['train']['flow_hidden'], spec['train']['flow_steps'])


def _score(spec, model, xy, vis, ctx):
    if spec['kind'] == 'ae':
        amp = ctx.amp if spec['train'].get('amp') else None
        return ae_errors(model, xy, vis, ctx.cfg['eval_batch_size'], amp)
    return flow_scores(model, xy, vis, ctx.cfg['eval_batch_size'])


# =============================================================================================
# Configs -> calibrated frame scores
# =============================================================================================

def raw_scores(b, which, ws):
    data = b['val'] if which == 'val' else b['test'][which]
    if b['kind'] == 'ae':
        return window_score(data.astype(np.float32), ws.vis, 'mean' if b['_variant'] is None else b['_variant'])
    return np.asarray(data, np.float64)


def conf_frames(ctx, conf, seed):
    """{subset: {stem: frame scores (NaN = unscored)}} + window-level scores, all shifted so that the
    normal-validation p95 is 0. Cached in memory."""
    key = (json.dumps(conf, sort_keys=True), seed)
    if key in ctx._frames:
        return ctx._frames[key]
    spec = conf['spec']
    b = dict(bundle(ctx, conf['model'], spec, seed))
    b['_variant'] = conf['variant'] if b['kind'] == 'ae' else None
    val_ws = b['val_ws']
    val_raw = raw_scores(b, 'val', val_ws)
    cal = Calibrator(log=b['kind'] == 'ae', camnorm=conf['camnorm']).fit(val_raw, val_ws.camera)
    val_cal = cal.transform(val_raw, val_ws.camera)
    pct = ctx.cfg['threshold_percentile']
    val_frames = frame_scores(val_cal, val_ws.pose_file, val_ws.track, val_ws.start, val_ws.end, sigma=conf['sigma'])
    vf = np.concatenate([v[np.isfinite(v)] for v in val_frames.values()])
    frame_thr = float(np.percentile(vf, pct))
    window_thr = float(np.percentile(val_cal, pct))
    out = dict(frames={}, windows={}, frame_thr=frame_thr, window_thr=window_thr, source=b['meta']['source'])
    for s in SUBSETS:
        ws = b['test_ws'][s]
        cal_s = cal.transform(raw_scores(b, s, ws), ws.camera)
        lengths = {k: len(v) for k, v in ws.frame_labels.items()}
        fr = frame_scores(cal_s, ws.pose_file, ws.track, ws.start, ws.end, lengths=lengths, sigma=conf['sigma'])
        out['frames'][s] = {k: v - frame_thr for k, v in fr.items()}
        out['windows'][s] = cal_s - window_thr
    ctx._frames[key] = out
    return out


def subset_frames(ctx, per_video, subset, stems):
    labels = {k: v for k, v in ctx.frame_labels[subset].items() if k in stems}
    return flatten({k: per_video[k] for k in labels}, labels, ctx.cameras[subset])


def prepare_labels(ctx):
    L, S = ctx.cfg['baseline_window']
    tests = ctx.test(L, S)
    ctx.frame_labels = {s: ws.frame_labels for s, ws in tests.items()}
    ctx.cameras = {s: video_cameras(ws) for s, ws in tests.items()}
    folds = ctx.folds()
    ctx.fold_stems = {f: {s: {k for k in ctx.frame_labels[s] if folds[F.video_key(s, ctx.cameras[s][k], k)] == f}
                          for s in SUBSETS} for f in F.FOLDS}


def dev_objective(ctx, conf, fold):
    """Mean over seeds of the mean per-subset frame ROC-AUC on one fold's videos."""
    vals = []
    for seed in ctx.cfg['seeds']:
        fr = conf_frames(ctx, conf, seed)
        per = [fast_roc_auc(**_lab_sc(subset_frames(ctx, fr['frames'][s], s, ctx.fold_stems[fold][s]))) for s in SUBSETS]
        vals.append(np.nanmean(per))
    return float(np.mean(vals))


def _lab_sc(flat):
    return dict(labels=flat['labels'], scores=flat['scores'])


def dev_halfwidth(ctx, conf, fold, draws):
    """Half-width of the 95% video-bootstrap CI of the dev objective on ``fold``."""
    rng = np.random.default_rng(ctx.cfg['bootstrap_seed'])
    flats = {seed: {s: subset_frames(ctx, conf_frames(ctx, conf, seed)['frames'][s], s, ctx.fold_stems[fold][s])
                    for s in SUBSETS} for seed in ctx.cfg['seeds']}
    groups = {}
    for s in SUBSETS:
        v = flats[ctx.cfg['seeds'][0]][s]['video']
        groups[s] = [np.flatnonzero(v == g) for g in np.unique(v)]
    samples = []
    for _ in range(draws):
        picks = {s: np.concatenate([groups[s][i] for i in rng.integers(0, len(groups[s]), len(groups[s]))]) for s in SUBSETS}
        vals = []
        for seed in ctx.cfg['seeds']:
            vals.append(np.nanmean([fast_roc_auc(flats[seed][s]['labels'][picks[s]], flats[seed][s]['scores'][picks[s]])
                                    for s in SUBSETS]))
        samples.append(np.mean(vals))
    lo, hi = np.nanquantile(samples, [0.025, 0.975])
    return float((hi - lo) / 2)


# =============================================================================================
# Reporting (cross-fitted: fold A videos scored with the fold-B choice and vice versa)
# =============================================================================================

def crossfit_flat(ctx, state, seed, subset):
    per_video = {}
    for fold in F.FOLDS:
        other = 'B' if fold == 'A' else 'A'
        fr = conf_frames(ctx, state[other], seed)['frames'][subset]   # chosen on `other`, applied to `fold`
        for k in ctx.fold_stems[fold][subset]:
            per_video[k] = fr[k]
    return flatten(per_video, ctx.frame_labels[subset], ctx.cameras[subset])


def crossfit_windows(ctx, state, seed, subset):
    out = None
    for fold in F.FOLDS:
        other = 'B' if fold == 'A' else 'A'
        conf = state[other]
        fr = conf_frames(ctx, conf, seed)
        ws = bundle(ctx, conf['model'], conf['spec'], seed)['test_ws'][subset]
        sel = np.isin(ws.pose_file, list(ctx.fold_stems[fold][subset]))
        part = dict(scores=fr['windows'][subset][sel], labels=ws.labels[sel], pos=ws.pos_frac[sel],
                    camera=ws.camera[sel], video=ws.pose_file[sel])
        out = part if out is None else {k: np.concatenate([out[k], part[k]]) for k in out}
    return out


def report_state(ctx, step, state, extra=None):
    draws = ctx.cfg['bootstrap_draws']
    rows, per_seed = [], {}
    for seed in ctx.cfg['seeds']:
        res = {}
        pooled = []
        for s in SUBSETS:
            flat = crossfit_flat(ctx, state, seed, s)
            m = detection_metrics(flat['labels'], flat['scores'], 0.0, cameras=flat['camera'], groups=flat['video'],
                                  draws=draws, seed=ctx.cfg['bootstrap_seed'])
            m['scored_frame_fraction'] = float(flat['scored'].mean())
            w = crossfit_windows(ctx, state, seed, s)
            m['window_roc_auc'] = fast_roc_auc(w['labels'], w['scores'])
            full = (w['pos'] >= 1.0) | (w['labels'] == 0)
            m['window_fully_positive_roc_auc'] = fast_roc_auc(w['labels'][full], w['scores'][full])
            m['window_counts'] = dict(normal=int((w['labels'] == 0).sum()), anomaly=int(w['labels'].sum()),
                                      fully_positive=int((w['pos'] >= 1.0).sum()))
            res[s] = m
            pooled.append((flat, s))
        labels = np.concatenate([f['labels'] for f, _ in pooled])
        scores = np.concatenate([f['scores'] for f, _ in pooled])
        res['combined (info only)'] = detection_metrics(labels, scores, 0.0, draws=0)
        per_seed[seed] = res
    for subset in list(per_seed[ctx.cfg['seeds'][0]]):
        ms = [per_seed[s][subset] for s in ctx.cfg['seeds']]
        def agg(key):
            v = np.array([m.get(key, np.nan) for m in ms], float)
            if not np.isfinite(v).any():
                return float('nan'), float('nan')
            return float(np.nanmean(v)), float(np.nanstd(v))
        row = dict(step=step, step_name=STEP_NAMES[step], subset=subset)
        for key in ('roc_auc', 'pr_auc', 'base_rate', 'balanced_accuracy', 'recall_at_fpr5', 'specificity_at_recall90',
                    'camera_auc_mean', 'camera_auc_std', 'window_roc_auc', 'window_fully_positive_roc_auc',
                    'scored_frame_fraction'):
            row[key], row[key + '_sd'] = agg(key)
        cis = [m['roc_auc_ci'] for m in ms if 'roc_auc_ci' in m]
        if cis:
            row['roc_auc_ci_low'] = float(np.mean([c[0] for c in cis]))
            row['roc_auc_ci_high'] = float(np.mean([c[1] for c in cis]))
        row['n_frames'] = ms[0]['n']
        row['anomaly_frames'] = ms[0]['num_anomaly']
        rows.append(row)
    result = dict(step=step, name=STEP_NAMES[step], state={f: _public(c) for f, c in state.items()},
                  rows=rows, per_seed=per_seed, sources={f: conf_frames(ctx, state[f], ctx.cfg['seeds'][0])['source']
                                                         for f in F.FOLDS}, **(extra or {}))
    jdump(LOGS / f'step{step}.json', result)
    write_tables(ctx)
    print_rows(rows)
    return result


def _public(conf):
    return {k: v for k, v in conf.items() if k != 'spec'} | dict(window=conf['spec']['window'], kind=conf['spec']['kind'],
                                                              recipe=conf['spec']['recipe'])


def fmt(v, sd=None, d=3):
    if v is None or not np.isfinite(v):
        return 'n/a'
    return f'{v:.{d}f}' + (f' ± {sd:.{d}f}' if sd is not None and np.isfinite(sd) else '')


def print_rows(rows):
    log('\n' + markdown_table(rows))


def markdown_table(rows):
    head = ('| step | subset | frame ROC-AUC | 95% CI | frame PR-AUC (base rate) | balanced acc | per-camera AUC mean / std '
            '| window AUC | fully-positive window AUC |\n|---|---|---|---|---|---|---|---|---|')
    lines = [head]
    for r in rows:
        ci = f"[{r['roc_auc_ci_low']:.3f}, {r['roc_auc_ci_high']:.3f}]" if 'roc_auc_ci_low' in r else 'n/a'
        lines.append(f"| {r['step']} {r['step_name']} | {r['subset']} | {fmt(r['roc_auc'], r['roc_auc_sd'])} | {ci} | "
                     f"{fmt(r['pr_auc'], r['pr_auc_sd'])} ({fmt(r['base_rate'], d=2)}) | "
                     f"{fmt(r['balanced_accuracy'], r['balanced_accuracy_sd'])} | "
                     f"{fmt(r['camera_auc_mean'])} / {fmt(r['camera_auc_std'])} | {fmt(r['window_roc_auc'], r['window_roc_auc_sd'])} | "
                     f"{fmt(r['window_fully_positive_roc_auc'])} |")
    return '\n'.join(lines)


def write_tables(ctx):
    rows, notes = [], []
    for step in range(1, 7):
        p = LOGS / f'step{step}.json'
        if p.exists():
            r = json.loads(p.read_text())
            rows += r['rows']
            notes.append(f"- **Step {step} ({r['name']})**: " + r.get('decision', 'baseline') +
                         f" Fold A choice: `{json.dumps(r['state']['A'])}`; fold B choice: `{json.dumps(r['state']['B'])}`.")
    if not rows:
        return
    import csv
    keys = sorted({k for r in rows for k in r})
    with (LOGS / 'results.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    text = ['# research_v2 results (full RetailS test set, cross-fitted over two dev folds)', '',
            'Frame-level metrics per subset; mean ± std over seeds ' + ', '.join(map(str, ctx.cfg['seeds'])) +
            '. 95% CI = video-grouped bootstrap (mean of per-seed bounds). Every labelled test frame is counted; '
            'frames with no scored person get the subset minimum. "combined" mixes two base rates and is informational only. '
            'Step 1 is the research_v1 baseline re-scored under this protocol.', '', markdown_table(rows), '', '## Decisions', ''] + notes
    (LOGS / 'results.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


# =============================================================================================
# State and steps
# =============================================================================================

def load_state(step):
    p = LOGS / f'step{step}.json'
    if not p.exists():
        raise FileNotFoundError(f'Step {step} has not been run yet')
    return json.loads(p.read_text())


def restore(ctx, public):
    """Rebuild a full conf (with model spec) from a saved public conf."""
    kind, recipe, window = public['kind'], public['recipe'], public['window']
    key, spec = model_spec(ctx.cfg, recipe, window, kind)
    if key != public['model']:
        raise ValueError(f'Config changed since this step was run ({public["model"]} -> {key}). '
                         'Use a new config version instead of editing research_v2.yaml in place.')
    return dict(model=key, spec=spec, variant=public['variant'], camnorm=public['camnorm'], sigma=public['sigma'])


def prev_state(ctx, step):
    saved = load_state(step)
    return {f: restore(ctx, saved['state'][f]) for f in F.FOLDS}


def select(ctx, candidates_by_fold, prev, step, label):
    """Pick, per fold, the candidate with the best dev objective (previous state included and preferred on ties)."""
    state, log_rows, decisions = {}, [], []
    draws = ctx.cfg['bootstrap_draws']
    gains = {}
    for fold in F.FOLDS:
        cands = candidates_by_fold[fold]
        scored = []
        for c in cands:
            obj = dev_objective(ctx, c, fold)
            scored.append((obj, c))
            log(f'  fold {fold} dev: {label(c):<45} mean per-subset frame AUC = {obj:.4f}')
            log_rows.append(dict(step=step, fold=fold, candidate=_public(c), dev_auc=obj))
        prev_obj = dev_objective(ctx, prev[fold], fold)
        best_obj, best = max(scored, key=lambda t: t[0])
        if best_obj <= prev_obj + 1e-12:
            best_obj, best = prev_obj, prev[fold]
        hw = dev_halfwidth(ctx, prev[fold], fold, draws)
        worst = min(o for o, _ in scored)
        gains[fold] = dict(prev=prev_obj, best=best_obj, gain=best_obj - prev_obj, halfwidth=hw,
                           worst_drop_beyond_ci=bool(prev_obj - worst > hw))
        changed = best is not prev[fold]
        decisions.append(f"fold {fold}: {'kept ' + label(best) if changed else 'reverted to previous (no dev gain)'} "
                         f"(dev {prev_obj:.4f} -> {best_obj:.4f}, CI half-width {hw:.4f})")
        state[fold] = best
    jdump(LOGS / f'step{step}_selection.json', dict(candidates=log_rows, gains=gains))
    return state, '; '.join(decisions) + '.', gains


def step0(ctx):
    videos = ctx.videos()
    assignment = F.make_folds(videos, ctx.cfg['fold_seed'])
    F.save_folds(LOGS / 'folds.json', videos, assignment, ctx.cfg['fold_seed'])
    rep = F.fold_report(videos, assignment)
    problems = F.check_folds(videos, assignment)
    for f in F.FOLDS:
        log(f"fold {f}: {rep[f]['videos']} videos, {rep[f]['normal_frames']} normal / {rep[f]['anomaly_frames']} anomalous frames")
    log('fold checks:', 'PASS' if not problems else problems)
    for g, v in rep['A']['per_group'].items():
        log(f"  {g}: A {v} | B {rep['B']['per_group'][g]}")


def step1(ctx):
    L, S = ctx.cfg['baseline_window']
    key, spec = model_spec(ctx.cfg, 'v1', (L, S))
    conf = dict(model=key, spec=spec, variant='mean', camnorm=False, sigma=0)
    state = {f: conf for f in F.FOLDS}
    return report_state(ctx, 1, state, dict(decision='Baseline: research_v1 expanded-A recipe, mean MSE, no smoothing.'))


def step2(ctx):
    prev = prev_state(ctx, 1)
    cands = {f: [dict(prev[f], variant=v, camnorm=c) for c in ctx.cfg['camnorm'] for v in ctx.cfg['variants']]
             for f in F.FOLDS}
    state, decision, gains = select(ctx, cands, prev, 2, lambda c: f"{c['variant']} camnorm={c['camnorm']}")
    return report_state(ctx, 2, state, dict(decision=decision, gains=gains))


def step3(ctx):
    prev = prev_state(ctx, 2)
    cands = {f: [dict(prev[f], sigma=s) for s in ctx.cfg['sigmas']] for f in F.FOLDS}
    state, decision, gains = select(ctx, cands, prev, 3, lambda c: f"sigma={c['sigma']}")
    return report_state(ctx, 3, state, dict(decision=decision, gains=gains))


def _swap_model(conf, ctx, recipe=None, window=None, kind=None):
    spec = conf['spec']
    key, new = model_spec(ctx.cfg, recipe or spec['recipe'], window or spec['window'], kind or spec['kind'])
    return dict(conf, model=key, spec=new)


def step4(ctx):
    prev = prev_state(ctx, 3)
    cands = {f: [_swap_model(prev[f], ctx, recipe='all_aug')] for f in F.FOLDS}
    state, decision, gains = select(ctx, cands, prev, 4, lambda c: f"recipe={c['spec']['recipe']}")
    return report_state(ctx, 4, state, dict(decision=decision, gains=gains))


def step5(ctx):
    prev = prev_state(ctx, 4)
    cands = {f: [_swap_model(prev[f], ctx, window=w) for w in ctx.cfg['windows']
                 if list(w) != list(prev[f]['spec']['window'])] for f in F.FOLDS}
    state, decision, gains = select(ctx, cands, prev, 5, lambda c: f"window={c['spec']['window']}")
    return report_state(ctx, 5, state, dict(decision=decision, gains=gains))


def step6(ctx, mode):
    prev = prev_state(ctx, 5)
    plateau = True
    for step in (4, 5):
        g = json.loads((LOGS / f'step{step}_selection.json').read_text())['gains']
        plateau &= all(g[f]['gain'] < g[f]['halfwidth'] for f in F.FOLDS)
    if mode == 'never' or (mode == 'auto' and not plateau):
        log(f'Experiment F skipped (mode={mode}, plateau={plateau}).')
        return None
    log(f'Experiment F running (mode={mode}, plateau after steps 4-5 = {plateau}).')
    cands = {f: [dict(_swap_model(prev[f], ctx, kind='flow'), variant='nll')] for f in F.FOLDS}
    state, decision, gains = select(ctx, cands, prev, 6, lambda c: f"{c['spec']['kind']} {c['spec']['recipe']} w={c['spec']['window']}")
    return report_state(ctx, 6, state, dict(decision=decision, gains=gains, plateau=plateau, mode=mode))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--step', default='all', help='0-6 or all')
    parser.add_argument('--config', default=str(ROOT / 'configs/research_v2.yaml'))
    parser.add_argument('--experiment-f', choices=['auto', 'always', 'never'], default=None)
    args = parser.parse_args()
    ctx = Context(args.config)
    LOGS.mkdir(parents=True, exist_ok=True)
    steps = list(range(7)) if args.step == 'all' else [int(args.step)]
    for step in steps:
        started = time.perf_counter()
        log(f'\n===== research_v2 step {step}: {STEP_NAMES[step]} =====')
        if step == 0:
            step0(ctx)
            continue
        prepare_labels(ctx)
        fn = {1: step1, 2: step2, 3: step3, 4: step4, 5: step5}.get(step)
        if fn:
            fn(ctx)
        else:
            step6(ctx, args.experiment_f or ctx.cfg.get('experiment_f', 'auto'))
        log(f'step {step} done in {time.perf_counter() - started:.0f}s')
    log('COMPLETE')


if __name__ == '__main__':
    main()
