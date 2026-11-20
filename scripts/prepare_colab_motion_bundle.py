"""Freeze expanded B inputs with the exact cohort/split of imported Colab A."""
import json
import shutil
import sys
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nbformat
import numpy as np
from scripts.prepare_colab_bundle import notebook
from scripts.train_colab import verify_bundle
from src.experiments.artifacts import run_paths
from src.experiments.quality_dataset import features, implementation, sha256


def make_notebook():
    result = notebook()
    replacements = {
        'Retail_Shoplifting_Colab_Training.zip':'Retail_Shoplifting_Colab_Training_B.zip',
        'Retail_Shoplifting_Colab_Results.zip':'Retail_Shoplifting_Colab_Results_B.zip',
        'Retail_Shoplifting_Training_v1':'Retail_Shoplifting_Training_B_v1',
        'Retail_Shoplifting_GPU_Training.ipynb':'Retail_Shoplifting_GPU_Training_B.ipynb',
        'expanded_a_seed':'expanded_b_seed', 'COLAB_TRAINING.md':'COLAB_EXPERIMENT_B.md',
        '/content/retail_training':'/content/retail_training_b',
        '/content/drive/MyDrive/Retail_':'/content/drive/MyDrive/Shoplifting/Retail_',
    }
    for cell in result.cells:
        for old,new in replacements.items():
            cell.source = cell.source.replace(old,new)
        if cell.cell_type == 'code' and 'for module, package in' in cell.source:
            cell.source = cell.source.replace('("matplotlib", "matplotlib")', '("matplotlib", "matplotlib"), ("pandas", "pandas")')
        if cell.cell_type == 'markdown' and cell.source.startswith('## 3.'):
            cell.source = ('## 3. Train Experiment B — three seeds\n\nRun fresh B models with 58 features per frame: 34 normalized positions and 24 joint displacements. Keep 15 epochs, batch size 32, Adam learning rate 0.001, hidden size 128, two layers and dropout 0.2. Seeds are 0/1/2. Training/validation windows and labels match A exactly.\n\nCheck each seed reaches 15/15. Epoch checkpoints save to the separate B output directory on Drive. Rerun this cell after reconnecting to resume saved epochs. Do not point B to the A output directory. Best epochs and p95 thresholds use normal validation only. Loss scales differ because B includes additional features; compare detection metrics rather than raw A/B loss values.')
        if cell.cell_type == 'code' and 'rows = []' in cell.source:
            cell.source += '\nassert all(json.loads((OUT / f"expanded_b_seed{s}" / "evaluation.json").read_text())["experiment"] == "b" for s in (0,1,2))\nbaseline = json.loads((ROOT / "baseline_a.json").read_text())\na = pd.DataFrame(baseline["rows"])\nprint("A versus B on identical retained test windows (means across three seeds):")\ndisplay(pd.DataFrame({"A_positions": a[["precision","recall","f1","roc_auc","pr_auc"]].mean(), "B_positions_velocity": table[["precision","recall","f1","roc_auc","pr_auc"]].mean()}))\nprint("This is an exploratory follow-up to inspected test results; improvement needs fresh confirmation data.")'
    result.cells[0].source = ('# Retail shoplifting — Experiment B GPU training\n\nTest whether explicit joint motion helps compared with Experiment A. Use normalized 2D positions plus per-frame displacements of shoulders, elbows, wrists, hips, knees and ankles: **58 features per frame**. This is the existing BiGRU Autoencoder with the same hidden size and depth; its input/output parameter count grows with the added features.\n\nThe package preserves A’s 100-file sample, 35,557 training windows, 6,284 normal validation windows and 1,856 retained test windows. Only the feature family changes. No new videos or pose extraction are needed.\n\nUpload `Retail_Shoplifting_Colab_Training_B.zip` to **My Drive → Shoplifting**. Select a **GPU runtime** (T4 when available), then run cells one at a time. Keep A’s existing files. B outputs use a separate directory.\n\nThis exploratory experiment follows inspection of test results; it cannot establish generalization without fresh confirmation data. No guarantee of improvement is made.')
    result.metadata['colab']['name'] = 'Retail_Shoplifting_GPU_Training_B.ipynb'
    nbformat.validate(result)
    for cell in result.cells:
        if cell.cell_type == 'code':
            compile(cell.source,'colab_b','exec')
    return result


def main():
    baseline_package = ROOT / 'outputs/colab_handoff/retail_training'
    baseline = verify_bundle(baseline_package)
    destination = ROOT / 'outputs/colab_handoff/experiment_b'
    archive_path = destination / 'Retail_Shoplifting_Colab_Training_B.zip'
    if archive_path.exists():
        raise FileExistsError('B handoff is already frozen; preserve it and version a new experiment before rebuilding')
    package = destination / 'retail_training_b'
    (package / 'data').mkdir(parents=True,exist_ok=True)
    cfg = baseline['config']
    datasets, checks = {}, {}
    for subset in ['train','test_realworld','test_staged']:
        source_manifest = json.loads((baseline_package / f'data/{subset}.json').read_text())
        if source_manifest['implementation'] != implementation():
            raise ValueError('Preprocessing changed since A; do not mix cohorts')
        fingerprint = source_manifest['fingerprint']
        limit = cfg['expanded_train_files'] if subset == 'train' else None
        cache = ROOT / f'outputs/cache/quality_v1/{subset}_{limit}_{fingerprint[:16]}.npz'
        if sha256(cache) != source_manifest['array_sha256']:
            raise ValueError('Source pose cache changed')
        with np.load(cache,allow_pickle=False) as arrays:
            values,masks = features({k:arrays[k] for k in arrays.files},'b')
        with np.load(baseline_package / f'data/{subset}.npz',allow_pickle=False) as original:
            if not np.array_equal(values[...,:34],original['features']) or not np.array_equal(masks[...,:34],original['masks']):
                raise ValueError('B position features/masks differ from A')
            payload = dict(features=values,masks=masks,labels=original['labels'])
            if subset == 'train':
                payload.update(train_indices=original['train_indices'],validation_indices=original['validation_indices'])
            if masks[:,0,34:].any() or values[:,0,34:].any() or values.shape[-1] != 58:
                raise ValueError('Invalid B velocity masks/features')
        np.savez_compressed(package / f'data/{subset}.npz',**payload)
        shutil.copy2(baseline_package / f'data/{subset}.json',package / f'data/{subset}.json')
        datasets[subset] = baseline['datasets'][subset]
        checks[subset] = dict(shape=list(values.shape),position_features_identical_to_a=True,
            position_masks_identical_to_a=True,labels_identical_to_a=True,window_metadata_identical_to_a=True,
            split_indices_copied_from_a=subset=='train')
        print(subset,values.shape,flush=True)
    baseline_rows, hashes = [], {}
    for seed in cfg['expanded_seeds']:
        model,log = run_paths(f'expanded_a_seed{seed}')
        report = json.loads((log / 'evaluation.json').read_text())
        saved_hash = sha256(model / 'model.pt')
        if saved_hash != report['checkpoint_sha256'] or report['config'] != cfg or report['split'] != baseline['split']:
            raise ValueError('Baseline A protocol differs')
        hashes[f'expanded_a_seed{seed}'] = saved_hash
        baseline_rows.append(dict(seed=seed,**report['metrics']['combined']))
    (package / 'baseline_a.json').write_text(json.dumps(dict(rows=baseline_rows,checkpoint_sha256=hashes),indent=2))
    protocol = dict(experiment='b',baseline_experiment='a',config=cfg,split=baseline['split'],
        feature_dim=58,features='34 normalized 2D positions + 24 displacements of 12 joints; first-frame velocity masked',
        units='Body-relative displacement per source frame, not metres/second',
        fixed='Dataset cohort, window identities, splits, masks for positions, hidden/depth settings, optimizer, epochs, seeds and threshold protocol',
        changed='Feature vector and associated model input/output parameter count',
        selection='Best normal-validation loss; threshold normal-validation p95; no test-based tuning',
        interpretation='Exploratory follow-up after test inspection; fresh confirmation required; additional features alter loss scale',
        baseline_manifest_sha256=sha256(baseline_package / 'bundle_manifest.json'),baseline_checkpoints=hashes,
        checks=checks)
    (package / 'experiment_protocol.json').write_text(json.dumps(protocol,indent=2))
    for name in ['scripts/train_colab.py','src/models/bigru_autoencoder.py','src/evaluation/metrics.py']:
        target = package / name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT / name,target)
    for name in ['src/__init__.py','src/models/__init__.py','src/evaluation/__init__.py']:
        (package / name).write_text('')
    notebook_path = ROOT / 'notebooks/Retail_Shoplifting_GPU_Training_B.ipynb'
    nb = make_notebook(); nbformat.write(nb,notebook_path)
    nbformat.write(nb,package / notebook_path.name)
    shutil.copy2(ROOT / 'COLAB_EXPERIMENT_B.md',package / 'COLAB_EXPERIMENT_B.md')
    files = {str(p.relative_to(package)).replace('\\','/'):sha256(p) for p in package.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.name != 'bundle_manifest.json'}
    manifest = dict(schema=1,experiment='b',feature_dim=58,config=cfg,split=baseline['split'],datasets=datasets,files=files,
        protocol_sha256=sha256(package / 'experiment_protocol.json'),scope='Prepared expanded B features only; A files remain preserved.')
    (package / 'bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
    temporary = archive_path.with_suffix('.partial.zip')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted([*files,'bundle_manifest.json']):
            archive.write(package / name,name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
    temporary.replace(archive_path)
    record = dict(path=str(archive_path),bytes=archive_path.stat().st_size,sha256=sha256(archive_path),checks=checks)
    (destination / 'handoff.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
