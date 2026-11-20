import unittest
import numpy as np
from src.evaluation.label_audit import positive_intervals, window_label_support


class LabelAuditTests(unittest.TestCase):
    def test_inclusive_boundaries_and_separate_intervals(self):
        self.assertEqual(positive_intervals([1, 1, 0, 1]), [(0, 1), (3, 3)])
        self.assertEqual(positive_intervals([0, 0]), [])
        self.assertEqual(positive_intervals([1]), [(0, 0)])

    def test_any_frame_label_is_not_person_identity(self):
        row = window_label_support([0, 1, 0], 0, 2, [1, 2, 1])
        self.assertEqual(row['positive_frames'], 1)
        self.assertAlmostEqual(row['positive_fraction'], 1 / 3)
        self.assertEqual(row['positive_frames_with_multiple_people'], 1)
        self.assertFalse(row['actor_identity_verified'])

    def test_invalid_masks_counts_and_bounds_fail(self):
        for labels in [[], [0, 2], [[0, 1]], [np.nan]]:
            with self.assertRaises(ValueError):
                positive_intervals(labels)
        for start, end, people in [(-1, 1, [1, 1]), (0, 2, [1, 1]),
                                    (1, 0, [1, 1]), (0, 1, [1]),
                                    (0, 1, [1, -1]), (0, 1, [1, np.nan])]:
            with self.assertRaises(ValueError):
                window_label_support([0, 1], start, end, people)
