import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.run_video_inference import active_score, load_tracks
from src.utils.joint_mapping import coco17_to_h36m17, DIRECT_H36M_TO_COCO


class ResearchInferenceTests(unittest.TestCase):
    def test_no_score_before_completion_or_after_a_gap(self):
        windows = [dict(start_frame=0, end_frame=29, score=.1), dict(start_frame=100, end_frame=129, score=.2)]
        self.assertIsNone(active_score(windows, 28, 15))
        self.assertEqual(active_score(windows, 29, 15)['score'], .1)
        self.assertIsNone(active_score(windows, 44, 15))
        self.assertIsNone(active_score(windows, 100, 15))
        self.assertEqual(active_score(windows, 129, 15)['score'], .2)

    def test_retail_video_pose_schema_and_source_frame_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'poses.json'
            path.write_text(json.dumps({'7': {'31': {'keypoints': list(range(51)), 'scores': None}}}))
            track = load_tracks(path, Path('retail.mp4'))[7]
            self.assertEqual(track.frame_ids_sorted, [31])
            self.assertEqual(track.frames[31].keypoints.shape, (17, 3))

    def test_mapping_preserves_limbs_and_does_not_invent_confidence(self):
        points = np.arange(2 * 17 * 3, dtype=np.float32).reshape(2, 17, 3)
        points[..., 2] = 1
        points[:, 11, 2] = 0
        mapped = coco17_to_h36m17(points)
        for target, source in DIRECT_H36M_TO_COCO.items():
            np.testing.assert_array_equal(mapped[:, target], points[:, source])
        self.assertTrue((mapped[:, 0, 2] == 0).all())
        self.assertTrue((mapped[:, 7, 2] == 0).all())


if __name__ == '__main__':
    unittest.main()
