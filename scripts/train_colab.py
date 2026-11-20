"""Train prepared expanded A/B features on CUDA, with epoch-boundary resume."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.bigru_autoencoder import BiGRUAutoencoder
from src.evaluation.metrics import evaluate


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_bundle(root):
    manifest = json.loads((root / 'bundle_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f'Bundle integrity failure: {name}')
    return manifest


def masked_error(predicted, target, mask):
    count = mask.sum((1, 2))
    if (count == 0).any():
        raise ValueError('Unsupported window')
    return (((predicted - target) ** 2) * mask).sum((1, 2)) / count


def loader(x, masks, batch, generator=None):
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(masks)),
                      batch_size=batch, shuffle=generator is not None, generator=generator)


def score(model, x, masks, batch, device):
    model.eval()
    values = []
    with torch.no_grad():
        for xb, mb in loader(x, masks, batch):
            xb, mb = xb.to(device), mb.to(device)
            values.append(masked_error(model(xb), xb, mb).cpu().numpy())
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite scores')
    return result


def atomic_save(value, path):
    temporary = path.with_suffix('.partial.pt')
    torch.save(value, temporary)
    temporary.replace(path)


def train_seed(root, output, seed, device):
    manifest = verify_bundle(root)
    cfg = manifest['config']
    experiment = manifest.get('experiment', 'a')
    if experiment not in ('a', 'b'):
        raise ValueError('Only prepared A/B features are supported')
    run = f'expanded_{experiment}_seed{seed}'
    folder = output / run
    folder.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps(dict(bundle=manifest, seed=seed,
        trainer=digest(Path(__file__))), sort_keys=True).encode()).hexdigest()
    if (folder / 'complete.json').exists():
        complete = json.loads((folder / 'complete.json').read_text())
        if complete['signature'] != signature:
            raise ValueError('Completed run differs; use a new output directory')
        for name, sha in complete['files'].items():
            if digest(folder / name) != sha:
                raise ValueError(f'Completed result changed: {name}')
        print(f'{run}: already complete', flush=True)
        return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    with np.load(root / 'data/train.npz', allow_pickle=False) as data:
        x, masks = data['features'], data['masks']
        ti, vi = data['train_indices'], data['validation_indices']
        if np.any(data['labels']) or set(ti) & set(vi) or len(ti) + len(vi) != len(x):
            raise ValueError('Invalid normal training split')
    model = BiGRUAutoencoder(x.shape[-1], cfg['hidden_size'], cfg['num_layers'], cfg['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    generator = torch.Generator().manual_seed(seed)
    history, best, best_state, best_epoch, start = [], float('inf'), None, 0, 1
    resume = folder / 'resume.pt'
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['signature'] != signature:
            raise ValueError('Resume config/data/code differs; use a new output directory')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        for parameters in optimizer.state.values():
            for key, value in parameters.items():
                if torch.is_tensor(value):
                    parameters[key] = value.to(device)
        history, best, best_state, best_epoch = state['history'], state['best'], state['best_state'], state['best_epoch']
        start = state['epoch'] + 1
        torch.set_rng_state(state['torch_rng'])
        generator.set_state(state['loader_rng'])
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
        if device.type == 'cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        print(f'{run}: resuming at epoch {start}', flush=True)
    training = loader(x[ti], masks[ti], cfg['batch_size'], generator)
    for epoch in range(start, cfg['epochs'] + 1):
        model.train(); total, count = 0., 0
        for xb, mb in training:
            xb, mb = xb.to(device), mb.to(device)
            optimizer.zero_grad()
            loss = masked_error(model(xb), xb, mb).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); total += loss.item() * len(xb); count += len(xb)
        validation = score(model, x[vi], masks[vi], cfg['batch_size'], device)
        value = float(validation.mean())
        if value < best:
            best, best_epoch = value, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(dict(epoch=epoch, train_loss=total/count, validation_loss=value))
        atomic_save(dict(signature=signature, epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
            history=history, best=best, best_state=best_state, best_epoch=best_epoch,
            torch_rng=torch.get_rng_state(), loader_rng=generator.get_state(), numpy_rng=np.random.get_state(),
            python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None), resume)
        (folder / 'history.json').write_text(json.dumps(history, indent=2))
        print(f'{run} {epoch}/{cfg["epochs"]}: train={total/count:.6f} validation={value:.6f}', flush=True)
    model.load_state_dict(best_state)
    validation = score(model, x[vi], masks[vi], cfg['batch_size'], device)
    np.save(folder / 'validation_errors.npy', validation)
    threshold = float(np.percentile(validation, cfg['threshold_percentile']))
    saved = dict(model_state_dict=best_state, config=cfg, experiment=experiment, seed=seed,
        fingerprint=signature, trainer_sha256=digest(Path(__file__)),
        training_data_fingerprint=manifest['datasets']['train']['fingerprint'], split=manifest['split'],
        feature_dim=x.shape[-1], best_epoch=best_epoch, validation_loss=best,
        source_coordinate_order=cfg['coordinate_order'], inference_coordinate_order='xy',
        threshold_protocol='p95 of normal validation windows; no test-label tuning',
        implementation=manifest['datasets']['train']['implementation'], motionbert_sha256=None,
        backend=dict(device=str(device), torch=torch.__version__, cuda=torch.version.cuda),
        bundle_manifest_sha256=digest(root / 'bundle_manifest.json'))
    atomic_save(saved, folder / 'model.pt')
    metrics, all_scores, all_labels = {}, [], []
    for subset in ('test_realworld', 'test_staged'):
        with np.load(root / f'data/{subset}.npz', allow_pickle=False) as test:
            scores = score(model, test['features'], test['masks'], cfg['batch_size'], device)
            labels = test['labels']
        np.savez_compressed(folder / f'{subset}_scores.npz', scores=scores, labels=labels)
        metrics[subset] = evaluate(labels, scores, threshold).as_dict()
        all_scores.append(scores); all_labels.append(labels)
    metrics['combined'] = evaluate(np.concatenate(all_labels), np.concatenate(all_scores), threshold).as_dict()
    report = dict(run=run, seed=seed, experiment=experiment, best_epoch=best_epoch, config=cfg,
        checkpoint_sha256=digest(folder / 'model.pt'), metrics=metrics, split=manifest['split'],
        caution='Conditional on retained windows. Video anomaly labels are not person-level shoplifting labels.')
    (folder / 'evaluation.json').write_text(json.dumps(report, indent=2))
    names = ['model.pt', 'validation_errors.npy', 'history.json', 'evaluation.json',
             'test_realworld_scores.npz', 'test_staged_scores.npz']
    (folder / 'complete.json').write_text(json.dumps(dict(signature=signature,
        files={name: digest(folder / name) for name in names}), indent=2))
    print(json.dumps(metrics['combined'], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime in Colab before training')
    if args.device == 'cpu':
        torch.set_num_threads(2)
    manifest = verify_bundle(args.data_root)
    if not set(args.seeds).issubset(manifest['config']['expanded_seeds']):
        raise ValueError('Undeclared seed')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'bundle_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Output directory belongs to different data/code; choose a separate directory')
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for seed in args.seeds:
        train_seed(args.data_root, args.output, seed, torch.device(args.device))
    print('COMPLETE: all requested seeds', flush=True)


if __name__ == '__main__':
    main()
