import json
import tempfile
import unittest
from pathlib import Path
from src.experiments.artifacts import run_paths, research_runs
from src.experiments.quality_dataset import sha256


class ResearchArtifactTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / 'outputs/test_tmp'
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temporary.name)
        (self.root / 'outputs').mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def test_unregistered_local_paths_are_preserved(self):
        model, log = run_paths('matched_a_seed0', self.root)
        self.assertEqual(model, self.root / 'outputs/checkpoints/research_v1/matched_a_seed0')
        self.assertEqual(log, self.root / 'outputs/logs/research_v1/matched_a_seed0')

    def test_import_hashes_are_checked_before_use(self):
        model = self.root / 'outputs/colab_results/import/seed0'
        model.mkdir(parents=True)
        path = model / 'model.pt'
        path.write_bytes(b'original')
        record = dict(checkpoint_dir=str(model.relative_to(self.root)), log_dir='outputs/derived',
                      files={str(path.relative_to(self.root)):sha256(path)})
        (self.root / 'outputs/research_runs.json').write_text(json.dumps({'expanded_a_seed0':record}))
        self.assertEqual(run_paths('expanded_a_seed0',self.root)[0],model)
        path.write_bytes(b'modified')
        with self.assertRaisesRegex(ValueError,'Changed registered artifact'):
            run_paths('expanded_a_seed0', self.root)

    def test_registry_paths_cannot_escape_outputs(self):
        record = dict(checkpoint_dir='src', log_dir='outputs/logs', files={})
        (self.root / 'outputs/research_runs.json').write_text(json.dumps({'expanded_a_seed0':record}))
        with self.assertRaisesRegex(ValueError,'inside outputs'):
            run_paths('expanded_a_seed0', self.root)

    def test_b_registration_requires_all_three_seed_reports(self):
        self.assertEqual(len(research_runs(self.root)),8)
        (self.root / 'outputs/research_runs.json').write_text(json.dumps({'expanded_b_seed0':{}}))
        names=research_runs(self.root)
        self.assertEqual(len(names),11)
        self.assertTrue(all(f'expanded_b_seed{s}' in names for s in (0,1,2)))
