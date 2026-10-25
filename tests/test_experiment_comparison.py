import unittest

import numpy as np

from scripts.compare_2d_experiments import classify_errors


class ExperimentComparisonTests(unittest.TestCase):
    def test_threshold_boundary_and_confusion_partition(self):
        outcomes, counts = classify_errors([0, 0, 1, 1], [.1, .5, .49, .8], .5)
        self.assertEqual(outcomes.tolist(), ["TN", "FP", "FN", "TP"])
        self.assertEqual(counts, {"TN": 1, "FP": 1, "FN": 1, "TP": 1})

    def test_invalid_scores_or_labels_fail(self):
        for labels, scores in [([0], [np.nan]), ([2], [.1]), ([0, 1], [.1]), ([], [])]:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                classify_errors(labels, scores, .5)


if __name__ == "__main__":
    unittest.main()
