"""Run quality-corrected matched A-E experiments and expanded 2D validation.

All artifacts are separate from legacy A/B. Decisions are fixed in the config;
test labels do not select a threshold, checkpoint, feature set or epoch.
"""
import argparse
import csv
import json
import os
import sys
import time
import subprocess
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/matplotlib'))
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from src.data.retails import discover_retails_root
from src.data.quality import masked_error
from src.experiments.quality_dataset import collect, lift, features, window_metadata, fingerprint, implementation, sha256
from src.experiments.splits import split_by_video
from src.evaluation.metrics import choose_threshold, evaluate
from src.models.bigru_autoencoder import BiGRUAutoencoder
from src.models.bigru_attention import BiGRUAttentionAutoencoder

CHECKPOINT = ROOT / 'third_party/MotionBERT/checkpoint/pose3d/MB_ft_h36m/best_epoch.bin'


def make_model(experiment, dim, cfg):
    cls = BiGRUAttentionAutoencoder if experiment == 'e' else BiGRUAutoencoder
    return cls(dim, cfg['hidden_size'], cfg['num_layers'], cfg['dropout'])


def score(model, values, masks, batch_size):
    loader = DataLoader(TensorDataset(torch.from_numpy(values), torch.from_numpy(masks)), batch_size=batch_size)
    result = []
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for x, mask in loader:
            result.append(masked_error(model(x), x, mask).numpy())
    scores = np.concatenate(result)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite model scores')
    return scores, (time.perf_counter() - started) * 1000 / len(scores)


def train(experiment, values, masks, manifest, cfg, seed, run_name):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_indices, validation_indices, split = split_by_video(window_metadata(manifest), cfg['val_fraction'], cfg['split_seed'])
    trainer_hash = sha256(Path(__file__))
    signature = fingerprint(dict(data=manifest['fingerprint'], config=cfg, experiment=experiment, seed=seed,
                                 trainer_sha256=trainer_hash))
    folder = ROOT / 'outputs/checkpoints/research_v1' / run_name
    log = ROOT / 'outputs/logs/research_v1' / run_name
    folder.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    path = folder / 'model.pt'
    model = make_model(experiment, values.shape[-1], cfg)
    if path.exists() and (folder / 'validation_errors.npy').exists():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['fingerprint'] != signature:
            raise ValueError(f'Run signature changed; preserve existing run and use a new version: {run_name}')
        model.load_state_dict(saved['model_state_dict'])
        print('Reusing verified checkpoint:', run_name, flush=True)
        return model, saved, folder
    generator = torch.Generator().manual_seed(seed)
    training_loader = DataLoader(TensorDataset(torch.from_numpy(values[train_indices]), torch.from_numpy(masks[train_indices])),
        batch_size=cfg['batch_size'], shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    history, best_loss, started = [], float('inf'), time.perf_counter()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total, count = 0., 0
        for x, mask in training_loader:
            optimizer.zero_grad()
            loss = masked_error(model(x), x, mask).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        validation_errors, _ = score(model, values[validation_indices], masks[validation_indices], cfg['batch_size'])
        validation_loss = float(validation_errors.mean())
        history.append(dict(epoch=epoch, train_loss=total / count, validation_loss=validation_loss,
                            elapsed_seconds=time.perf_counter() - started))
        if validation_loss < best_loss:
            best_loss = validation_loss
            saved = dict(model_state_dict=model.state_dict(), config=cfg, experiment=experiment, seed=seed,
                fingerprint=signature, trainer_sha256=trainer_hash, training_data_fingerprint=manifest['fingerprint'], split=split,
                feature_dim=values.shape[-1], best_epoch=epoch, validation_loss=best_loss,
                source_coordinate_order=cfg['coordinate_order'], inference_coordinate_order='xy',
                threshold_protocol='p95 of normal validation windows; no test-label tuning',
                implementation=implementation(), motionbert_sha256=sha256(CHECKPOINT) if experiment in 'cde' else None)
            temporary = path.with_suffix('.partial.pt')
            torch.save(saved, temporary)
            temporary.replace(path)
        with (log / 'training_log.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader(); writer.writerows(history)
        print(f'{run_name} epoch {epoch}/{cfg["epochs"]}: train={total/count:.6f} validation={validation_loss:.6f} ({time.perf_counter()-started:.0f}s)', flush=True)
    saved = torch.load(path, weights_only=False, map_location='cpu')
    model.load_state_dict(saved['model_state_dict'])
    validation_errors, _ = score(model, values[validation_indices], masks[validation_indices], cfg['batch_size'])
    np.save(folder / 'validation_errors.npy', validation_errors)
    (log / 'split_manifest.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    return model, saved, folder


def evaluate_run(model, saved, folder, datasets, lifted, cfg, run_name):
    experiment = saved['experiment']
    threshold = choose_threshold(np.load(folder / 'validation_errors.npy'), cfg['threshold_percentile'])
    results, rows, all_labels, all_scores = {}, [], [], []
    for subset in ('test_realworld', 'test_staged'):
        arrays, manifest = datasets[subset]
        values, masks = features(arrays, experiment, lifted.get(subset))
        scores, timing = score(model, values, masks, cfg['batch_size'])
        labels = arrays['labels']
        metrics = evaluate(labels, scores, threshold).as_dict()
        predicted = scores >= threshold
        confusion = dict(TN=int(((labels == 0) & ~predicted).sum()), FP=int(((labels == 0) & predicted).sum()),
                         FN=int(((labels == 1) & ~predicted).sum()), TP=int(((labels == 1) & predicted).sum()))
        results[subset] = dict(**metrics, confusion=confusion, coverage=manifest['counts'],
                              inference_ms_per_window=timing, data_fingerprint=manifest['fingerprint'])
        for meta, label, value, pred in zip(manifest['metadata'], labels, scores, predicted):
            rows.append(dict(subset=subset, **meta, score=float(value), prediction=int(pred), threshold=threshold,
                outcome=('TP' if pred else 'FN') if label else ('FP' if pred else 'TN')))
        all_labels.append(labels); all_scores.append(scores)
    labels, scores = np.concatenate(all_labels), np.concatenate(all_scores)
    results['combined'] = evaluate(labels, scores, threshold).as_dict()
    results['combined']['anomaly_fraction'] = float(labels.mean())
    report = dict(run=run_name, seed=saved['seed'], experiment=experiment, config=cfg,
        feature_dim=saved['feature_dim'], best_epoch=saved['best_epoch'], split=saved['split'],
        checkpoint_sha256=sha256(folder / 'model.pt'), metrics=results,
        protocol=saved['threshold_protocol'], filtered_evaluation=True,
        caution='Metrics are conditional on the retained windows; rejected windows are unscored, not normal. Video masks are not person annotations.')
    log = ROOT / 'outputs/logs/research_v1' / run_name
    with (log / 'window_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (log / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(run_name, json.dumps(results['combined']), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['matched', 'expanded', 'all'], default='all')
    parser.add_argument('--expanded-seed', type=int, default=None, help='Run a single declared expanded seed (worker mode)')
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / 'configs/research_v1.yaml').read_text())
    root = discover_retails_root(ROOT / 'data/retails')
    if root is None:
        raise FileNotFoundError('RetailS dataset is missing')
    datasets = {s: collect(root, s, cfg, None) for s in ('test_realworld', 'test_staged')}
    results = []
    if args.stage in ('matched', 'all'):
        from src.motionbert.lift import load_motionbert
        torch.set_num_threads(cfg['lift_threads'])
        model_3d = load_motionbert(CHECKPOINT)
        training = collect(root, 'train', cfg, cfg['matched_train_files'])
        lifted = {s: lift(a, m, CHECKPOINT, model_3d) for s, (a, m) in {**datasets, 'train': training}.items()}
        torch.set_num_threads(2)
        for experiment in 'abcde':
            name = f'matched_{experiment}_seed0'
            values, masks = features(training[0], experiment, lifted['train'])
            model, saved, folder = train(experiment, values, masks, training[1], cfg, 0, name)
            results.append(evaluate_run(model, saved, folder, datasets, lifted, cfg, name))
        del model_3d, lifted
    if args.stage in ('expanded', 'all'):
        training = collect(root, 'train', cfg, cfg['expanded_train_files'])
        # Predeclared A family; expanded runs are not selected by inspecting test scores.
        if args.expanded_seed is not None:
            if args.expanded_seed not in cfg['expanded_seeds']:
                raise ValueError('Seed must be declared in research_v1.yaml')
            values, masks = features(training[0], 'a')
            seed = args.expanded_seed
            name = f'expanded_a_seed{seed}'
            model, saved, folder = train('a', values, masks, training[1], cfg, seed, name)
            results.append(evaluate_run(model, saved, folder, datasets, {}, cfg, name))
        else:
            # Each process has its own RNG. Read shared, verified caches instead
            # of duplicating a lift or racing to create the training cache.
            del training
            jobs = ROOT / 'outputs/logs/research_v1/jobs'
            jobs.mkdir(parents=True, exist_ok=True)
            workers, streams, last_lines = {}, [], {}
            try:
                for seed in cfg['expanded_seeds']:
                    log_path = jobs / f'expanded_seed{seed}.txt'
                    stream = log_path.open('w', encoding='utf-8')
                    streams.append(stream)
                    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--stage', 'expanded', '--expanded-seed', str(seed)]
                    process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    workers[seed] = process, log_path
                while any(p.poll() is None for p, _ in workers.values()):
                    for seed, (p, log_path) in workers.items():
                        lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
                        last = lines[-1] if lines else 'starting'
                        if last_lines.get(seed) != last:
                            print(f'seed {seed}: {last}', flush=True)
                            last_lines[seed] = last
                    time.sleep(10)
                for seed, (process, log_path) in workers.items():
                    if process.returncode:
                        raise RuntimeError(f'Expanded seed {seed} failed: inspect {log_path}')
                    report = ROOT / f'outputs/logs/research_v1/expanded_a_seed{seed}/evaluation.json'
                    results.append(json.loads(report.read_text()))
            except BaseException:
                for process, _ in workers.values():
                    if process.poll() is None:
                        process.terminate()
                for process, _ in workers.values():
                    process.wait()
                raise
            finally:
                for stream in streams:
                    stream.close()
    output = ROOT / 'outputs/logs/research_v1'
    output.mkdir(parents=True, exist_ok=True)
    tag = f'{args.stage}_seed{args.expanded_seed}' if args.expanded_seed is not None else args.stage
    (output / f'{tag}_runs.json').write_text(json.dumps([r['run'] for r in results], indent=2), encoding='utf-8')
    print('COMPLETE:', args.stage, flush=True)


if __name__ == '__main__':
    main()
