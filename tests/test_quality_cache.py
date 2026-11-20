import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from src.experiments import quality_dataset


class QualityCacheTests(unittest.TestCase):
    def test_config_source_changes_and_corrupt_cache_are_detected(self):
        scratch = Path(__file__).resolve().parents[1] / 'outputs/test_tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            data = root / 'data'
            source = data / 'RetailS_train/pose/train/1_1_1.json'
            source.parent.mkdir(parents=True)
            points = np.ones((17, 3), dtype=float)
            points[:, :2] = [100, 200]
            points[[5, 6], 0] = 50
            records = {'1': {str(i): {'keypoints': points.ravel().tolist(), 'scores': None} for i in range(31)}}
            source.write_text(json.dumps(records))
            cfg = dict(coordinate_order='yx', confidence_threshold=.3, max_interpolation_gap=5,
                       smoothing_window=3, window_length=30, stride=15, max_normalized_coordinate=10,
                       min_visible_fraction=.5, sample_seed=0)
            with patch.object(quality_dataset, 'ROOT', root), patch.object(quality_dataset, 'implementation', return_value={'test': 'fixed'}):
                arrays, first = quality_dataset.collect(data, 'train', cfg, 1)
                repeated, same = quality_dataset.collect(data, 'train', cfg, 1)
                np.testing.assert_array_equal(arrays['normalized'], repeated['normalized'])
                self.assertEqual(first['fingerprint'], same['fingerprint'])
                _, changed_cfg = quality_dataset.collect(data, 'train', {**cfg, 'confidence_threshold': .4}, 1)
                self.assertNotEqual(first['fingerprint'], changed_cfg['fingerprint'])
                cache_path = Path(first['array_path'])
                cache_path.write_bytes(cache_path.read_bytes() + b'corrupted')
                with self.assertRaisesRegex(ValueError, 'integrity'):
                    quality_dataset.collect(data, 'train', cfg, 1)
                records['1']['0']['keypoints'][0] += 1
                source.write_text(json.dumps(records))
                _, changed_source = quality_dataset.collect(data, 'train', cfg, 1)
                self.assertNotEqual(first['fingerprint'], changed_source['fingerprint'])


if __name__ == '__main__':
    unittest.main()
