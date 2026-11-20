import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.audit_pose_label_alignment import audit_file
from src.evaluation.manual_review import validate_review
from scripts.serve_validation_review import byte_range


class ManualReviewTests(unittest.TestCase):
    def setUp(self):
        self.manifest=dict(manifest_id='example',cases=[dict(case_id='RV001',source_sha256='hash',frame_count=30,track_ids=['1'])])
        self.payload=dict(schema=1,manifest_id='example',reviewer='reviewer',reviews=[dict(case_id='RV001',source_sha256='hash',
            status='pending',reviewed_whole_video=False,retail_scene='',tracking_quality='',notes='',events=[])])

    def complete(self,status='no_visible_anomaly'):
        self.payload['reviews'][0].update(status=status,reviewed_whole_video=True,retail_scene='yes',tracking_quality='usable')
        return self.payload['reviews'][0]

    def event(self):
        return dict(start_frame=0,end_frame=29,track_id='1',label='observable_anomaly',confidence='high',observation='Observed action description')

    def test_pending_is_not_normal_or_complete(self):
        result=validate_review(self.payload,self.manifest)
        self.assertEqual(result['pending_cases'],1)
        self.assertEqual(result['no_visible_anomaly_cases'],0)
        self.assertFalse(result['complete'])

    def test_inclusive_first_and_last_event_frames(self):
        review=self.complete('events_present'); review['events']=[self.event()]
        result=validate_review(self.payload,self.manifest)
        self.assertEqual(result['annotated_intervals'],1)
        self.assertEqual(result['person_linked_intervals'],1)
        review['events'][0]['end_frame']=30
        with self.assertRaisesRegex(ValueError,'inside'):
            validate_review(self.payload,self.manifest)

    def test_normal_cannot_contradict_an_event(self):
        review=self.complete(); review['events']=[self.event()]
        with self.assertRaisesRegex(ValueError,'contradicts'):
            validate_review(self.payload,self.manifest)

    def test_unknown_person_and_changed_source_are_rejected(self):
        review=self.complete('events_present'); review['events']=[self.event()]
        review['events'][0]['track_id']='999'
        with self.assertRaisesRegex(ValueError,'Track ID'):
            validate_review(self.payload,self.manifest)
        review['events'][0]['track_id']=None
        self.assertEqual(validate_review(self.payload,self.manifest)['person_linked_intervals'],0)
        review['source_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'changed source'):
            validate_review(self.payload,self.manifest)

    def test_uncertain_is_not_an_asserted_anomaly(self):
        review=self.complete('uncertain'); review['events']=[self.event()]
        with self.assertRaisesRegex(ValueError,'Uncertain'):
            validate_review(self.payload,self.manifest)
        review['events'][0]['label']='uncertain'
        result=validate_review(self.payload,self.manifest)
        self.assertEqual(result['events_present_cases'],0)
        self.assertEqual(result['uncertain_cases'],1)

    def test_whole_video_review_is_required(self):
        review=self.complete(); review['reviewed_whole_video']=False
        with self.assertRaisesRegex(ValueError,'Complete-video'):
            validate_review(self.payload,self.manifest)

    def test_ai_review_is_attributed_and_never_reference_ground_truth(self):
        review=self.complete('uncertain')
        self.payload['reviewer_type']='ai_assisted'
        review.update(reviewed_whole_video=False,review_method='sampled_frames',evidence_frame_ids=[0,15,29])
        result=validate_review(self.payload,self.manifest)
        self.assertTrue(result['complete'])
        self.assertEqual(result['reviewer_type'],'ai_assisted')
        self.assertFalse(result['verified_ground_truth'])
        self.assertFalse(result['eligible_for_supervised_training'])
        review['status']='no_visible_anomaly'
        with self.assertRaisesRegex(ValueError,'absence'):
            validate_review(self.payload,self.manifest)

    def test_ai_sampling_cannot_claim_full_frame_review(self):
        review=self.complete('uncertain');self.payload['reviewer_type']='ai_assisted'
        review.update(review_method='sampled_frames',evidence_frame_ids=[0,29])
        with self.assertRaisesRegex(ValueError,'sample evidence'):
            validate_review(self.payload,self.manifest)


class AlignmentAuditTests(unittest.TestCase):
    def test_labels_use_inclusive_bounds_and_original_person_frames(self):
        parent=Path(__file__).resolve().parents[1]/'outputs/test_tmp'; parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            root=Path(temporary); pose=root/'1_1.json'; mask=root/'1_1.npy'
            pose.write_text(json.dumps({'1':{str(i):dict(keypoints=[1.,2.,1.]*17) for i in range(4)}}))
            np.save(mask,np.array([0,0,0,1]))
            window=dict(person_id='1',start_frame=0,end_frame=3,label=1)
            result=audit_file(pose,mask,[window])
            self.assertEqual(result['window_label_mismatches'],0)
            self.assertEqual(result['out_of_bounds_person_frames'],0)
            window['label']=0
            self.assertEqual(audit_file(pose,mask,[window])['window_label_mismatches'],1)
            window.update(person_id='2')
            self.assertEqual(audit_file(pose,mask,[window])['windows_with_missing_person'],1)

    def test_missing_track_frame_and_nonfinite_joint_are_reported(self):
        parent=Path(__file__).resolve().parents[1]/'outputs/test_tmp'; parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            root=Path(temporary); pose=root/'1_1.json'; mask=root/'1_1.npy'
            pose.write_text(json.dumps({'1':{'0':dict(keypoints=[1.,2.,float('nan')]*17),'2':dict(keypoints=[1.,2.,1.]*17)}}))
            np.save(mask,np.array([0,0,0]))
            result=audit_file(pose,mask,[dict(person_id='1',start_frame=0,end_frame=2,label=0)])
            self.assertEqual(result['windows_with_missing_source_frames'],1)
            self.assertEqual(result['nonfinite_joint_observations'],17)


class ReviewMediaRangeTests(unittest.TestCase):
    def test_regular_open_ended_and_suffix_ranges(self):
        self.assertEqual(byte_range('bytes=0-99',200),(0,99))
        self.assertEqual(byte_range('bytes=100-',200),(100,199))
        self.assertEqual(byte_range('bytes=-50',200),(150,199))

    def test_invalid_and_outside_ranges_are_rejected(self):
        for header in ['bytes=200-','bytes=50-10','bytes=-0','bytes=0-10,20-30','invalid']:
            with self.subTest(header=header), self.assertRaises(ValueError):
                byte_range(header,200)
