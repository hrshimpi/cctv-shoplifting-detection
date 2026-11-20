"""Validate Colab results and preserve them separately from local checkpoints."""
import argparse
import io
import json
import sys
import uuid
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.train_colab import digest
from src.models.bigru_autoencoder import BiGRUAutoencoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--package', type=Path, default=ROOT / 'outputs/colab_handoff/retail_training', help='Prepared package used for this training run')
    args = parser.parse_args()
    package = args.package.resolve()
    manifest = json.loads((package / 'bundle_manifest.json').read_text())
    experiment = manifest.get('experiment', 'a')
    if experiment not in ('a','b'):
        raise ValueError('Unsupported experiment')
    dimension = {'a':34, 'b':58}[experiment]
    import hashlib
    with zipfile.ZipFile(args.archive) as archive:
        if archive.testzip() is not None:
            raise ValueError('Corrupt ZIP')
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Duplicate ZIP entries')
        returned = json.loads(archive.read('bundle_manifest.json'))
        if returned != manifest:
            raise ValueError('Result bundle differs from local exported data/code')
        permitted = {'bundle_manifest.json', *(f'data/{s}.json' for s in ['train', 'test_realworld', 'test_staged'])}
        for subset in ('train', 'test_realworld', 'test_staged'):
            if hashlib.sha256(archive.read(f'data/{subset}.json')).hexdigest() != manifest['files'][f'data/{subset}.json']:
                raise ValueError('Changed source metadata')
        cfg = manifest['config']
        for seed in cfg['expanded_seeds']:
            prefix = f'expanded_{experiment}_seed{seed}'
            complete = json.loads(archive.read(f'{prefix}/complete.json'))
            expected_signature = hashlib.sha256(json.dumps(dict(bundle=manifest, seed=seed,
                trainer=digest(package / 'scripts/train_colab.py')), sort_keys=True).encode()).hexdigest()
            if complete['signature'] != expected_signature:
                raise ValueError('Wrong training signature')
            expected_names = {'model.pt', 'validation_errors.npy', 'history.json', 'evaluation.json',
                              'test_realworld_scores.npz', 'test_staged_scores.npz'}
            if set(complete['files']) != expected_names:
                raise ValueError('Missing or unexpected result files')
            permitted.add(f'{prefix}/complete.json')
            for name, expected in complete['files'].items():
                permitted.add(f'{prefix}/{name}')
                if hashlib.sha256(archive.read(f'{prefix}/{name}')).hexdigest() != expected:
                    raise ValueError(f'Corrupt result: {prefix}/{name}')
            history = json.loads(archive.read(f'{prefix}/history.json'))
            if [row['epoch'] for row in history] != list(range(1, cfg['epochs'] + 1)):
                raise ValueError('Incomplete training history')
            if not np.isfinite([[row['train_loss'], row['validation_loss']] for row in history]).all():
                raise ValueError('Nonfinite training history')
            saved = torch.load(io.BytesIO(archive.read(f'{prefix}/model.pt')), map_location='cpu', weights_only=False)
            if saved['fingerprint'] != expected_signature or saved['config'] != cfg or saved['seed'] != seed or saved['split'] != manifest['split']:
                raise ValueError('Checkpoint identity differs')
            if saved['experiment'] != experiment or saved['feature_dim'] != dimension:
                raise ValueError('Checkpoint feature family differs')
            model = BiGRUAutoencoder(dimension, cfg['hidden_size'], cfg['num_layers'], cfg['dropout'])
            model.load_state_dict(saved['model_state_dict'], strict=True)
            if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
                raise ValueError('Nonfinite model weights')
            calibration = np.load(io.BytesIO(archive.read(f'{prefix}/validation_errors.npy')), allow_pickle=False)
            if calibration.shape != (manifest['split']['validation_windows'],) or not np.isfinite(calibration).all() or (calibration < 0).any():
                raise ValueError('Invalid calibration scores')
            from src.evaluation.metrics import evaluate
            labels_all, scores_all = [], []
            report = json.loads(archive.read(f'{prefix}/evaluation.json'))
            threshold = float(np.percentile(calibration, cfg['threshold_percentile']))
            for subset in ('test_realworld', 'test_staged'):
                with np.load(io.BytesIO(archive.read(f'{prefix}/{subset}_scores.npz')), allow_pickle=False) as scores, np.load(package / f'data/{subset}.npz', allow_pickle=False) as original:
                    if not np.array_equal(scores['labels'], original['labels']) or scores['scores'].shape != original['labels'].shape or not np.isfinite(scores['scores']).all():
                        raise ValueError('Invalid test scores/labels')
                    labels_all.append(scores['labels']); scores_all.append(scores['scores'])
            computed = evaluate(np.concatenate(labels_all), np.concatenate(scores_all), threshold).as_dict()
            if not all(np.isclose(computed[k], report['metrics']['combined'][k]) for k in computed):
                raise ValueError('Evaluation report differs from raw scores')
        if set(archive.namelist()) != permitted:
            raise ValueError('Unexpected archive contents')
        target = ROOT / 'outputs/colab_results' / uuid.uuid4().hex[:12]
        target.mkdir(parents=True, exist_ok=False)
        archive.extractall(target)  # Every member is from the fixed permitted set above.
    (target / 'import_validation.json').write_text(json.dumps(dict(status='passed',
        archive_sha256=digest(args.archive), bundle_sha256=digest(package / 'bundle_manifest.json'),
        package_dir=str(package.relative_to(ROOT))), indent=2))
    print('VALIDATED AND IMPORTED:', target)
    print('Local research checkpoints are preserved. Next: integrate results and run frame-level diagnostics.')


if __name__ == '__main__':
    main()
