"""Register verified Colab models and derive reports compatible with local analysis."""
import argparse
import csv
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from src.evaluation.metrics import evaluate
from src.experiments.quality_dataset import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('import_folder', type=Path)
    args = parser.parse_args()
    source = args.import_folder.resolve()
    if not source.is_relative_to((ROOT / 'outputs/colab_results').resolve()):
        raise ValueError('Use the validated local import folder')
    validation = json.loads((source / 'import_validation.json').read_text())
    package = ROOT / validation.get('package_dir', 'outputs/colab_handoff/retail_training')
    if not package.resolve().is_relative_to((ROOT / 'outputs').resolve()):
        raise ValueError('Prepared package must be inside outputs')
    original_manifest = package / 'bundle_manifest.json'
    if validation['status'] != 'passed' or sha256(original_manifest) != validation['bundle_sha256']:
        raise ValueError('Import provenance failed')
    manifest = json.loads((source / 'bundle_manifest.json').read_text())
    if manifest != json.loads(original_manifest.read_text()):
        raise ValueError('Imported manifest content differs from exported manifest')
    registry_path = ROOT / 'outputs/research_runs.json'
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    for seed in manifest['config']['expanded_seeds']:
        name = f'expanded_{manifest.get("experiment", "a")}_seed{seed}'
        folder = source / name
        complete = json.loads((folder / 'complete.json').read_text())
        for file, expected in complete['files'].items():
            if sha256(folder / file) != expected:
                raise ValueError(f'Changed imported result: {name}/{file}')
        if name in registry and (ROOT / registry[name]['checkpoint_dir']).resolve() != folder:
            raise ValueError(f'A different import is already registered for {name}')
        saved = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
        calibration = np.load(folder / 'validation_errors.npy', allow_pickle=False)
        threshold = float(np.percentile(calibration, saved['config']['threshold_percentile']))
        original = json.loads((folder / 'evaluation.json').read_text())
        results, rows, labels_all, scores_all = {}, [], [], []
        for subset in ['test_realworld', 'test_staged']:
            metadata_path = source / f'data/{subset}.json'
            if sha256(metadata_path) != manifest['files'][f'data/{subset}.json']:
                raise ValueError('Changed imported window metadata')
            metadata = json.loads(metadata_path.read_text())
            with np.load(folder / f'{subset}_scores.npz', allow_pickle=False) as arrays:
                scores, labels = arrays['scores'], arrays['labels']
            if not np.array_equal(labels, [m['label'] for m in metadata['metadata']]):
                raise ValueError('Scores and window identities differ')
            predicted = scores >= threshold
            metrics = evaluate(labels, scores, threshold).as_dict()
            if not all(np.isclose(v, original['metrics'][subset][k]) for k, v in metrics.items()):
                raise ValueError('Imported metrics differ')
            confusion = {key:int(mask.sum()) for key, mask in dict(TN=(labels == 0) & ~predicted,
                FP=(labels == 0) & predicted, FN=(labels == 1) & ~predicted, TP=(labels == 1) & predicted).items()}
            results[subset] = dict(**metrics, confusion=confusion, coverage=metadata['counts'], data_fingerprint=metadata['fingerprint'])
            for meta, label, score, pred in zip(metadata['metadata'], labels, scores, predicted):
                rows.append(dict(subset=subset, **meta, score=float(score), prediction=int(pred), threshold=threshold,
                                 outcome=('TP' if pred else 'FN') if label else ('FP' if pred else 'TN')))
            labels_all.append(labels); scores_all.append(scores)
        results['combined'] = evaluate(np.concatenate(labels_all), np.concatenate(scores_all), threshold).as_dict()
        results['combined']['anomaly_fraction'] = float(np.concatenate(labels_all).mean())
        report = dict(**{k:original[k] for k in ['run','seed','experiment','config','best_epoch','split','checkpoint_sha256']},
            feature_dim=saved['feature_dim'], metrics=results, protocol=saved['threshold_protocol'],
            filtered_evaluation=True, caution=original['caution'], backend=saved['backend'],
            provenance=dict(import_dir=str(source.relative_to(ROOT)), trainer_sha256=saved['trainer_sha256'],
                            bundle_sha256=validation['bundle_sha256'], source='Colab GPU training; CPU matched runs remain separate'))
        derived = source / 'research_logs' / name
        derived.mkdir(parents=True, exist_ok=True)
        (derived / 'evaluation.json').write_text(json.dumps(report, indent=2))
        with (derived / 'window_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        pd.DataFrame(json.loads((folder / 'history.json').read_text())).to_csv(derived / 'training_log.csv', index=False)
        (derived / 'split_manifest.json').write_text(json.dumps(saved['split'], indent=2))
        paths = [folder / 'model.pt', folder / 'validation_errors.npy', derived / 'evaluation.json',
                 derived / 'window_predictions.csv', derived / 'training_log.csv']
        registry[name] = dict(checkpoint_dir=str(folder.relative_to(ROOT)), log_dir=str(derived.relative_to(ROOT)),
            files={str(p.relative_to(ROOT)):sha256(p) for p in paths}, backend=saved['backend'])
        print('Registered:', name, 'best epoch:', saved['best_epoch'])
    temporary = registry_path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(registry, indent=2)); temporary.replace(registry_path)


if __name__ == '__main__':
    main()
