"""Export verified cached A features and a portable Google Colab training ZIP."""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nbformat
import numpy as np
import yaml
from src.experiments.quality_dataset import features, implementation, sha256, window_metadata
from src.experiments.splits import split_by_video


def notebook():
    cells = []
    md = lambda text: cells.append(nbformat.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbformat.v4.new_code_cell(text))
    md('# Retail shoplifting model training on GPU\n\nTrain the remaining expanded 2D BiGRU experiment: 100 normal RetailS pose files, seeds 0/1/2, 15 epochs each. The five matched A–E runs are already preserved locally. This notebook uses the same prepared features, masks, source-video split, architecture and normal-validation p95 threshold. GPU arithmetic can differ from CPU results.\n\n**Before running:** select **Runtime → Change runtime type → GPU** (for example T4). Upload `Retail_Shoplifting_Colab_Training.zip` to the top level of **My Drive**. Run cells in order. Keep the Drive output directory to resume after a disconnect.')
    code('from google.colab import drive\ndrive.mount("/content/drive")\nfrom pathlib import Path\nZIP = Path("/content/drive/MyDrive/Retail_Shoplifting_Colab_Training.zip")\nOUT = Path("/content/drive/MyDrive/Retail_Shoplifting_Training_v1")\nROOT = Path("/content/retail_training")\nassert ZIP.is_file(), f"Upload the training ZIP to My Drive: {ZIP}"')
    md('## 1. Extract and verify inputs\nCheck: verification reports 41,841 normal windows, a disjoint source-video split, and 1,856 retained test windows. The ZIP contains prepared pose tensors, not RGB videos. It preserves the current RetailS-specific coordinate interpretation and quality gates; this does not establish pose or detection accuracy.')
    code('import zipfile, hashlib, json\nwith zipfile.ZipFile(ZIP) as archive:\n    for entry in archive.infolist():\n        target = (ROOT / entry.filename).resolve()\n        assert target.is_relative_to(ROOT.resolve()), "Unsafe archive path"\n    archive.extractall(ROOT)\nmanifest = json.loads((ROOT / "bundle_manifest.json").read_text())\nfor name, expected in manifest["files"].items():\n    with (ROOT / name).open("rb") as f:\n        assert hashlib.file_digest(f, "sha256").hexdigest() == expected, name\nprint("Integrity check passed")\nprint(json.dumps({"split": manifest["split"], "counts": {k: v["counts"] for k,v in manifest["datasets"].items()}}, indent=2))')
    md('## 2. Check the GPU and dependencies\nCheck: CUDA must be available. Keep Colab’s installed PyTorch build; the notebook only installs missing supporting packages.')
    code('import importlib.util, subprocess, sys\nfor module, package in [("numpy", "numpy"), ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib")]:\n    if importlib.util.find_spec(module) is None:\n        subprocess.check_call([sys.executable, "-m", "pip", "install", package])\nimport torch\nassert torch.cuda.is_available(), "Select a GPU runtime, reconnect, then rerun these cells"\nprint("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))')
    md('## 3. Train three seeds\nCheck: each seed reaches epoch 15/15 and produces `complete.json`. Best weights are selected using normal validation loss. Test labels are used only for evaluation. Checkpoints are saved to Drive after each epoch; rerun this cell to resume at the next saved epoch. A disconnect can lose the current unfinished epoch.\n\nThe stopped local runs did not save optimizer/RNG state, so these GPU runs start afresh. Completed Colab epochs include that state. Keep the same GPU/runtime software where possible; changing hardware can change floating-point results.')
    code('subprocess.run([sys.executable, "-u", str(ROOT / "scripts/train_colab.py"), "--data-root", str(ROOT), "--output", str(OUT), "--device", "cuda", "--seeds", "0", "1", "2"], check=True)')
    md('## 4. Verify and inspect results\nCheck: three completed runs, 15 logged epochs per run, finite metrics. Precision alone can be misleading because the retained test windows contain many anomalies. Report recall, F1, ROC-AUC, PR-AUC and rejected-window coverage as well. These are video-mask-derived anomaly labels, not person-level theft annotations.')
    code('import pandas as pd\nrows = []\nfor seed in (0, 1, 2):\n    folder = OUT / f"expanded_a_seed{seed}"\n    complete = json.loads((folder / "complete.json").read_text())\n    for name, expected in complete["files"].items():\n        with (folder / name).open("rb") as f:\n            assert hashlib.file_digest(f, "sha256").hexdigest() == expected, name\n    assert len(json.loads((folder / "history.json").read_text())) == 15\n    report = json.loads((folder / "evaluation.json").read_text())\n    rows.append({"seed": seed, "best_epoch": report["best_epoch"], **report["metrics"]["combined"]})\ntable = pd.DataFrame(rows)\ndisplay(table)\ndisplay(table[["precision", "recall", "f1", "roc_auc", "pr_auc"]].agg(["mean", "std"]))')
    md('## 5. Export results for the local repository\nDownload the results ZIP, place it in the local repository’s `outputs/colab_handoff/` directory, and follow `COLAB_TRAINING.md`. Keep Drive checkpoints until local validation is complete. The export includes the CPU-loadable best weights, scores, normal calibration errors, logs, provenance and checksums. Full resume states remain on Drive.')
    code('RESULT = OUT.parent / "Retail_Shoplifting_Colab_Results.zip"\nwith zipfile.ZipFile(RESULT, "w", zipfile.ZIP_DEFLATED) as archive:\n    archive.write(OUT / "bundle_manifest.json", "bundle_manifest.json")\n    for subset in ("train", "test_realworld", "test_staged"):\n        archive.write(ROOT / f"data/{subset}.json", f"data/{subset}.json")\n    for seed in (0, 1, 2):\n        folder = OUT / f"expanded_a_seed{seed}"\n        complete = json.loads((folder / "complete.json").read_text())\n        for name in ["complete.json", *complete["files"]]:\n            archive.write(folder / name, f"{folder.name}/{name}")\nprint(RESULT, round(RESULT.stat().st_size / 1024**2, 1), "MiB")\nfrom google.colab import files\nfiles.download(str(RESULT))')
    result = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name':'python3', 'display_name':'Python 3'}, 'colab': {'name':'Retail_Shoplifting_GPU_Training.ipynb'}, 'accelerator':'GPU'})
    nbformat.validate(result)
    return result


def main():
    cfg = yaml.safe_load((ROOT / 'configs/research_v1.yaml').read_text())
    destination = ROOT / 'outputs/colab_handoff'
    package = destination / 'retail_training'
    (package / 'data').mkdir(parents=True, exist_ok=True)
    datasets = {}
    split = None
    for subset, pattern in [('train', 'train_100_*.json'), ('test_realworld', 'test_realworld_None_*.json'), ('test_staged', 'test_staged_None_*.json')]:
        candidates = list((ROOT / 'outputs/cache/quality_v1').glob(pattern))
        candidates = [p for p in candidates if json.loads(p.read_text())['implementation'] == implementation()]
        if len(candidates) != 1:
            raise ValueError(f'Expected one current cache: {subset}, got {len(candidates)}')
        source = candidates[0]
        manifest = json.loads(source.read_text())
        if sha256(source.with_suffix('.npz')) != manifest['array_sha256']:
            raise ValueError(f'Cache checksum failed: {source}')
        with np.load(source.with_suffix('.npz'), allow_pickle=False) as cached:
            arrays = {k: cached[k] for k in cached.files}
        values, masks = features(arrays, 'a')
        payload = dict(features=values, masks=masks, labels=arrays['labels'])
        if subset == 'train':
            ti, vi, split = split_by_video(window_metadata(manifest), cfg['val_fraction'], cfg['split_seed'])
            payload.update(train_indices=ti, validation_indices=vi)
        np.savez_compressed(package / f'data/{subset}.npz', **payload)
        manifest.pop('array_path', None)
        (package / f'data/{subset}.json').write_text(json.dumps(manifest, indent=2))
        datasets[subset] = {k: manifest[k] for k in ['fingerprint', 'counts', 'implementation']}
        print(subset, values.shape, flush=True)
    names = ['scripts/train_colab.py', 'src/models/bigru_autoencoder.py', 'src/evaluation/metrics.py']
    for name in names:
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for name in ['src/__init__.py', 'src/models/__init__.py', 'src/evaluation/__init__.py']:
        (package / name).write_text('')
    nb = notebook()
    nbpath = ROOT / 'notebooks/Retail_Shoplifting_GPU_Training.ipynb'
    nbformat.write(nb, nbpath)
    nbformat.write(nb, package / nbpath.name)
    shutil.copy2(ROOT / 'COLAB_TRAINING.md', package / 'COLAB_TRAINING.md')
    files = {str(p.relative_to(package)).replace('\\', '/'): sha256(p) for p in package.rglob('*') if p.is_file() and p.name != 'bundle_manifest.json' and '__pycache__' not in p.parts}
    (package / 'bundle_manifest.json').write_text(json.dumps(dict(schema=1, config=cfg, experiment='a',
        datasets=datasets, split=split, files=files,
        scope='Prepared expanded 2D experiment only; no raw videos, images or 3D retraining.',
        caveat='RetailS yx interpretation retained; anomaly labels are video-level; metrics conditional on quality gates.'), indent=2))
    archive_path = destination / 'Retail_Shoplifting_Colab_Training.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted([*files, 'bundle_manifest.json']):
            archive.write(package / name, name)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity failed')
    record = dict(path=str(archive_path), bytes=archive_path.stat().st_size, sha256=sha256(archive_path), datasets=datasets, split=split)
    (destination / 'handoff.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
