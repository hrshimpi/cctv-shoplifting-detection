import unittest
import numpy as np
from src.evaluation.frame_scores import aggregate_windows


class FrameScoreTests(unittest.TestCase):
    def test_max_over_overlap_and_people_preserves_unscored_frames(self):
        metadata = [dict(pose_file='video.json', start_frame=1, end_frame=3),
                    dict(pose_file='video.json', start_frame=3, end_frame=4)]
        result = aggregate_windows(np.array([.2, .8]), metadata, {'video.json': 6})['video.json']
        self.assertTrue(np.isnan(result[[0, 5]]).all())
        np.testing.assert_allclose(result[1:5], [.2, .2, .8, .8])

    def test_pose_file_parts_are_separate_timelines(self):
        metadata = [dict(pose_file='1_1_1.json', start_frame=0, end_frame=1),
                    dict(pose_file='1_1_2.json', start_frame=0, end_frame=1)]
        result = aggregate_windows(np.array([.1, .9]), metadata)
        self.assertAlmostEqual(float(result['1_1_1.json'][0]), .1)
        self.assertAlmostEqual(float(result['1_1_2.json'][0]), .9)

    def test_invalid_frames_or_nonfinite_scores_fail(self):
        metadata = [dict(pose_file='video.json', start_frame=0, end_frame=3)]
        with self.assertRaises(ValueError):
            aggregate_windows(np.array([.1]), metadata, {'video.json': 3})
        with self.assertRaises(ValueError):
            aggregate_windows(np.array([np.nan]), metadata)


if __name__ == '__main__':
    unittest.main()
