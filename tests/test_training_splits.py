import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.experiments.splits import split_by_video
from src.experiments.dataset import build_windows_for_subset


class TrainingSplitTests(unittest.TestCase):
    def test_overlapping_windows_and_parts_of_one_video_stay_together(self):
        meta = [SimpleNamespace(camera_id=c, video_id=v, pose_file=f)
                for c, v, f in [("01", "10", "1_10_1.json"), ("1", "10", "1_10_2.json"),
                                ("1", "20", "1_20_1.json"), ("2", "30", "2_30_1.json")]]
        train, val, report = split_by_video(meta, .34, 7)
        self.assertEqual(0 in train, 1 in train)
        self.assertEqual(0 in val, 1 in val)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(range(4)))
        self.assertFalse(set(map(tuple, report["train_groups"])) & set(map(tuple, report["validation_groups"])))
        again = split_by_video(meta, .34, 7)
        np.testing.assert_array_equal(train, again[0])
        np.testing.assert_array_equal(val, again[1])

    def test_one_video_cannot_be_split_without_leakage(self):
        with self.assertRaises(ValueError):
            split_by_video([SimpleNamespace(camera_id="1", video_id="10")]*20, .1, 0)

    def test_missing_test_mask_cannot_silently_become_normal(self):
        with patch("src.experiments.dataset.find_pose_files", return_value=[Path("1_10.json")]):
            with self.assertRaisesRegex(FileNotFoundError, "Missing test ground truth"):
                build_windows_for_subset(Path("missing_fixture_root"), "test_staged", {}, False, False)


if __name__ == "__main__":
    unittest.main()
