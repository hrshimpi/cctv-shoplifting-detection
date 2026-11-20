import unittest
from pathlib import Path
import numpy as np
from src.data.retails import PersonTrack,PoseFrame
from src.data.scene_segments import split_track
from scripts.score_review_sequences import sequences,completed_timeline


class SceneSegmentTests(unittest.TestCase):
    def track(self):
        points=np.column_stack([np.arange(17),np.arange(17),np.ones(17)]).astype('float32')
        return PersonTrack('1','external','video','1',Path('video.mp4'),
                           {i:PoseFrame(i,points.copy(),None) for i in range(120)})

    def test_split_preserves_ids_and_excludes_camera_motion(self):
        original=self.track();segments=split_track(original,[(40,49)])
        self.assertEqual([scene for scene,_ in segments],[0,1])
        self.assertEqual(sorted(segments[0][1].frames),list(range(40)))
        self.assertEqual(sorted(segments[1][1].frames),list(range(50,120)))
        self.assertEqual(len(original.frames),120)

    def test_preprocessed_windows_cannot_span_motion(self):
        cfg=dict(coordinate_order='yx',confidence_threshold=.3,max_interpolation_gap=5,smoothing_window=3,
            max_normalized_coordinate=10,window_length=30,stride=15,min_visible_fraction=.5)
        _,metadata,_=sequences({1:self.track()},cfg,[(40,49)])
        self.assertTrue(metadata)
        self.assertTrue(all(m['end_frame']<40 or m['start_frame']>=50 for m in metadata))

    def test_completed_scores_expire_at_scene_boundary(self):
        tracks={1:self.track()}
        metadata=[dict(track_id=1,end_frame=39,scene_last_frame=39)]
        timeline=completed_timeline(np.array([.5]),metadata,tracks,120,15)
        self.assertEqual(timeline[39],.5)
        self.assertTrue(np.isnan(timeline[40:]).all())

    def test_overlapping_or_reversed_intervals_are_rejected(self):
        for intervals in [[(40,30)],[(20,40),(35,45)]]:
            with self.assertRaises(ValueError):split_track(self.track(),intervals)
