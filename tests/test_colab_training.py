"""Tiny synthetic checks for the portable trainer; no research training."""
import json
import tempfile
import unittest
import shutil
import zipfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from scripts.train_colab import train_seed, digest, verify_bundle, atomic_save


class ColabTrainingTests(unittest.TestCase):
    def test_train_finish_resume_and_integrity(self):
        for experiment in ('a', 'b'):
            with self.subTest(experiment=experiment):
                self.run_case(experiment)

    def run_case(self, experiment):
        torch.set_num_threads(1)
        dimension = {'a':34, 'b':58}[experiment]
        run = f'expanded_{experiment}_seed0'
        temporary_root = Path(__file__).resolve().parents[1] / 'outputs/test_tmp'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            root = Path(temporary).resolve()  # outputs/ may be a symlink (e.g. to Google Drive on Colab)
            (root / 'data').mkdir()
            rng = np.random.default_rng(0)
            x = rng.normal(size=(8, 3, dimension)).astype('float32')
            np.savez(root / 'data/train.npz', features=x, masks=np.ones_like(x, dtype=bool),
                labels=np.zeros(8, dtype=int), train_indices=np.arange(6), validation_indices=np.arange(6, 8))
            for subset in ('test_realworld', 'test_staged'):
                np.savez(root / f'data/{subset}.npz', features=x[:2], masks=np.ones_like(x[:2], dtype=bool), labels=np.array([0, 1]))
            for subset in ('train', 'test_realworld', 'test_staged'):
                metadata = [dict(pose_file='1_video.json', person_id='1', camera_id='1', video_id='video',
                    start_frame=i*3, end_frame=i*3+2, label=i) for i in (0,1)]
                info = dict(metadata=metadata, fingerprint='synthetic', counts=dict(candidate=2,retained=2,rejected=0,
                    candidate_anomaly=1,rejected_anomaly=0))
                (root / f'data/{subset}.json').write_text(json.dumps(info))
            cfg = dict(hidden_size=4, num_layers=1, dropout=0., learning_rate=.001, batch_size=2,
                epochs=2, threshold_percentile=95, coordinate_order='yx', expanded_seeds=[0])
            manifest = dict(experiment=experiment, config=cfg, datasets={'train':dict(fingerprint='synthetic', implementation={})},
                split={'validation_windows':2}, files={str(p.relative_to(root)).replace('\\', '/'):digest(p) for p in (root / 'data').iterdir()})
            (root / 'bundle_manifest.json').write_text(json.dumps(manifest))
            out = root / 'results'
            train_seed(root, out, 0, torch.device('cpu'))
            folder = out / run
            expected = digest(folder / 'model.pt')
            # Re-enter after last saved epoch but before final export.
            (folder / 'complete.json').unlink()
            train_seed(root, out, 0, torch.device('cpu'))
            self.assertEqual(expected, digest(folder / 'model.pt'))
            self.assertEqual(2, len(json.loads((folder / 'history.json').read_text())))
            train_seed(root, out, 0, torch.device('cpu'))  # Compatible completed run is skipped.
            interrupted = root / 'interrupted'
            def stop_after_save(value, path):
                atomic_save(value, path)
                if path.name == 'resume.pt' and value['epoch'] == 1:
                    raise RuntimeError('synthetic disconnect')
            with patch('scripts.train_colab.atomic_save', side_effect=stop_after_save):
                with self.assertRaisesRegex(RuntimeError, 'synthetic disconnect'):
                    train_seed(root, interrupted, 0, torch.device('cpu'))
            train_seed(root, interrupted, 0, torch.device('cpu'))
            resumed = torch.load(interrupted / run / 'model.pt', map_location='cpu', weights_only=False)
            reference = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
            for key in reference['model_state_dict']:
                self.assertTrue(torch.equal(reference['model_state_dict'][key], resumed['model_state_dict'][key]))
            # Validate an exported result through the actual local importer.
            from scripts import import_colab_results
            from scripts import train_colab
            local = root / 'local'
            package = local / 'outputs/colab_handoff/retail_training'
            shutil.copytree(root / 'data', package / 'data')
            (package / 'scripts').mkdir()
            shutil.copy2(train_colab.__file__, package / 'scripts/train_colab.py')
            shutil.copy2(root / 'bundle_manifest.json', package / 'bundle_manifest.json')
            archive_path = root / 'results.zip'
            complete = json.loads((folder / 'complete.json').read_text())
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.write(root / 'bundle_manifest.json', 'bundle_manifest.json')
                for subset in ('train', 'test_realworld', 'test_staged'):
                    archive.write(root / f'data/{subset}.json', f'data/{subset}.json')
                for name in ['complete.json', *complete['files']]:
                    archive.write(folder / name, f'{run}/{name}')
            with patch.object(import_colab_results, 'ROOT', local), patch('sys.argv', ['import_colab_results.py', str(archive_path)]):
                import_colab_results.main()
            imported = list((local / 'outputs/colab_results').glob('*/import_validation.json'))
            self.assertEqual(len(imported), 1)
            from scripts import integrate_colab_results
            from src.experiments.artifacts import run_paths
            import_folder = imported[0].parent
            # Windows-exported JSON can have CRLF while Colab writes LF.
            manifest_path = package / 'bundle_manifest.json'
            manifest_path.write_bytes(manifest_path.read_bytes().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))
            validation = json.loads(imported[0].read_text())
            validation['bundle_sha256'] = digest(manifest_path)
            imported[0].write_text(json.dumps(validation))
            with patch.object(integrate_colab_results,'ROOT',local), patch('sys.argv',['integrate_colab_results.py',str(import_folder)]):
                integrate_colab_results.main()
            model_folder, log_folder = run_paths(run,local)
            self.assertEqual(model_folder, import_folder / run)
            normalized = json.loads((log_folder / 'evaluation.json').read_text())
            self.assertEqual(normalized['feature_dim'],dimension)
            self.assertEqual(len((log_folder / 'window_predictions.csv').read_text().splitlines()),5)
            (root / 'data/train.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                verify_bundle(root)


if __name__ == '__main__':
    unittest.main()
