import unittest
from src.evaluation.video_status import active_window, model_status, PRIMARY_RUN


class VideoStatusTests(unittest.TestCase):
    def test_scores_begin_at_completed_window_and_expire(self):
        window={'end_frame':29,'scene_last_frame':100}
        self.assertIsNone(active_window([window],28,15))
        self.assertIs(active_window([window],29,15),window)
        self.assertIs(active_window([window],43,15),window)
        self.assertIsNone(active_window([window],44,15))

    def test_scene_boundary_and_latest_window(self):
        first={'end_frame':29,'scene_last_frame':32}
        second={'end_frame':31,'scene_last_frame':32}
        self.assertIs(active_window([first,second],32,15),second)
        self.assertIsNone(active_window([first,second],33,15))

    def test_camera_motion_overrides_scores(self):
        self.assertIn('PAUSED',model_status({PRIMARY_RUN:10},{PRIMARY_RUN:1},True))

    def test_primary_threshold_and_unknown_are_distinct(self):
        self.assertIn('UNSCORED',model_status({},{}))
        self.assertIn('BELOW',model_status({PRIMARY_RUN:.5,'expanded_b_seed0':10},{PRIMARY_RUN:1}))
        self.assertIn('UNVERIFIED',model_status({PRIMARY_RUN:1},{PRIMARY_RUN:1}))

    def test_invalid_values_rejected(self):
        for score,threshold in [(float('nan'),1),(float('inf'),1),(1,0),(1,-1),(1,float('nan'))]:
            with self.assertRaises(ValueError):
                model_status({PRIMARY_RUN:score},{PRIMARY_RUN:threshold})
