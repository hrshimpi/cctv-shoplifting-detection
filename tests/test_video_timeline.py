import unittest
from src.evaluation.video_status import PRIMARY_RUN
from src.evaluation.video_timeline import assess_frame,summarize_timeline


def frame(fid,scores,moving=False,fps=30):
    people=[dict(track_id=i+1,scores={PRIMARY_RUN:v}) for i,v in enumerate(scores)]
    return dict(frame_id=fid,seconds=fid/fps,people=people,camera_motion=moving,
                primary_model_alert=not moving and any(v is not None and v>=1 for v in scores))


class VideoTimelineTests(unittest.TestCase):
    def test_unknown_is_not_normal_and_camera_pause_overrides_scores(self):
        self.assertEqual(assess_frame(frame(0,[]),1)['reason'],'no_person_tracked')
        self.assertEqual(assess_frame(frame(0,[None]),1)['state'],'unknown')
        self.assertEqual(assess_frame(frame(0,[5],True),1)['reason'],'camera_motion')
        self.assertEqual(assess_frame(frame(0,[.5,None]),1)['unassessed_people'],[2])

    def test_alert_tracks_and_stored_flags_are_checked(self):
        self.assertEqual(assess_frame(frame(0,[.5,1.1,None]),1)['alert_people'],[2])
        value=frame(0,[2]);value['primary_model_alert']=False
        with self.assertRaises(ValueError):assess_frame(value,1)

    def test_time_intervals_cover_every_frame_without_smoothing_short_alerts(self):
        values=[frame(0,[None]),frame(1,[.5]),frame(2,[1]),frame(3,[.2]),frame(4,[None])]
        report=summarize_timeline(values,30,1)
        self.assertEqual(report['alert_periods'],1)
        self.assertEqual(report['alert_frames'],1)
        self.assertEqual(report['normal_prediction_frames'],2)
        self.assertEqual(report['unknown_frames'],2)
        alert=report['alert_intervals'][0]
        self.assertAlmostEqual(alert['start_seconds'],2/30)
        self.assertAlmostEqual(alert['end_seconds_exclusive'],3/30)
        self.assertAlmostEqual(sum(x['duration_seconds'] for x in report['exact_intervals']),5/30)
        self.assertEqual(report['seconds'][0]['unknown_frames'],2)
        self.assertEqual(report['seconds'][0]['alert_frames'],1)
        self.assertFalse(report['action_confirmed'])

    def test_integer_and_fractional_fps_bins_have_no_lost_frames(self):
        for fps in [30,29.97]:
            report=summarize_timeline([frame(i,[.2],fps=fps) for i in range(70)],fps,1)
            self.assertEqual(sum(r['frames'] for r in report['seconds']),70)
            self.assertEqual(report['alert_periods'],0)
            self.assertEqual(report['unknown_frames'],0)

    def test_noncontiguous_frames_or_invalid_scores_fail(self):
        with self.assertRaises(ValueError):summarize_timeline([frame(1,[.5])],30,1)
        wrong=frame(0,[.5]);wrong['seconds']=5
        with self.assertRaises(ValueError):summarize_timeline([wrong],30,1)
        for threshold in [0,float('nan')]:
            with self.assertRaises(ValueError):assess_frame(frame(0,[.5]),threshold)
        with self.assertRaises(ValueError):assess_frame(frame(0,[float('nan')]),1)
