import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.acquire_action_label_sample import validate_clip


class ActionLabelSampleTests(unittest.TestCase):
    def make_clip(self, path, ids, dtype=np.uint8):
        np.savez_compressed(path, frames=np.zeros((16, 8, 8, 3), dtype=dtype), frame_indices=ids)

    def test_original_frame_indices_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'clip.npz'
            ids = np.linspace(100, 159, 16, dtype=np.int64)
            self.make_clip(path, ids)
            frames, actual, key = validate_clip(path, 100, 160)
            np.testing.assert_array_equal(actual, ids)
            self.assertEqual(frames.shape, (16, 8, 8, 3))
            self.assertEqual(key, 'frames')

    def test_wrong_pixel_type_and_outside_or_reversed_times_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'clip.npz'
            for ids, dtype in [(np.arange(16), np.float32), (np.arange(16) + 100, np.uint8),
                               (np.arange(16)[::-1], np.uint8), (np.arange(16) + .5, np.uint8)]:
                self.make_clip(path, ids, dtype)
                with self.assertRaises(ValueError):
                    validate_clip(path, 0, 16)

    def test_missing_original_timing_is_not_invented(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'clip.npz'
            np.savez_compressed(path, frames=np.zeros((16, 8, 8, 3), dtype=np.uint8))
            with self.assertRaises(ValueError):
                validate_clip(path, 0, 16)
